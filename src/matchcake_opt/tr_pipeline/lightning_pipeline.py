import argparse
import json
import os
import shutil
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Type, Union

import torch
from lightning import Trainer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from ..datamodules.datamodule import DataModule
from ..modules.base_model import BaseModel
from ..utils.model_checkpoint import ModelCheckpoint
from ..utils.progress_bar import EpochProgressBar


class LightningPipeline:
    DEFAULT_MAX_EPOCHS = 1024
    DEFAULT_MAX_TIME = "90:00:00:00"  # DD:HH:MM:SS
    DEFAULT_OVERWRITE_FIT = True

    @classmethod
    def add_specific_args(cls, parent_parser: Optional[argparse.ArgumentParser] = None):
        if parent_parser is None:
            parent_parser = argparse.ArgumentParser()
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--max_epochs",
            type=int,
            default=cls.DEFAULT_MAX_EPOCHS,
            help="Maximum number of epochs to train the model",
        )
        parser.add_argument(
            "--max_time",
            type=str,
            default=cls.DEFAULT_MAX_TIME,
            help="Maximum time to train the model in DD:HH:MM:SS format",
        )
        parser.add_argument(
            "--overwrite_fit",
            type=bool,
            default=cls.DEFAULT_OVERWRITE_FIT,
            action=argparse.BooleanOptionalAction,
            help="Overwrite the fit checkpoints and fit the model again",
        )
        return parent_parser

    def __init__(
        self,
        model_cls: Type[BaseModel],
        datamodule: DataModule,
        *,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        max_time: str = DEFAULT_MAX_TIME,
        checkpoint_folder: Optional[Union[str, Path]] = None,
        monitor: str = "val_loss",
        monitor_mode: str = "min",
        overwrite_fit: bool = DEFAULT_OVERWRITE_FIT,
        verbose: bool = False,
        accelerator: str = "auto",
        trainer_args: Optional[Dict[str, Any]] = None,
        **model_args,
    ):
        torch.set_float32_matmul_precision("medium")

        self._model_cls = model_cls
        self._datamodule = datamodule
        self._max_epochs = max_epochs
        self._max_time = max_time
        self._model_args = model_args
        if checkpoint_folder is None:
            checkpoint_folder = f"checkpoints/{model_cls.MODEL_NAME}"
        self._checkpoint_folder = Path(checkpoint_folder)
        self._monitor = monitor
        self._monitor_mode = monitor_mode
        self._overwrite_fit = overwrite_fit
        self._verbose = verbose
        self._accelerator = accelerator
        self._trainer_args = trainer_args if trainer_args is not None else {}
        self._set_trainer_args_defaults()

        self.checkpoint_callback = ModelCheckpoint(
            save_last=True,
            dirpath=self.checkpoint_folder,
            filename="{epoch:02d}-{step:04d}",
            monitor=self.monitor,
            mode=self.monitor_mode,
            save_best_to=self.checkpoint_folder / f"{self.model_cls.MODEL_NAME}.ckpt",
        )
        self.progress_bar = EpochProgressBar()
        self.checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ""
        self.model = self._build_model_instance()
        self.trainer = self._build_trainer_instance()

    def setup(self):
        if self.overwrite_fit:
            shutil.rmtree(self.checkpoint_folder, ignore_errors=True)
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)
        if self.verbose:  # pragma: no cover
            print(f"Checkpoint folder: {self.checkpoint_folder}")
            print(f"Model: {self.model_cls.MODEL_NAME}\n{self.model}")
            print(f"Max epochs: {self.max_epochs}")
            print(f"Max time: {self.max_time}")
            print(f"Monitor: {self.monitor} ({self.monitor_mode})")
            print(f"Overwrite fit: {self.overwrite_fit}")
        return self

    def run(self) -> Dict[str, Any]:
        self.setup()
        start_time = time.perf_counter()
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=(None if self.overwrite_fit else "last"),
        )
        end_time = time.perf_counter()
        metrics: Dict[str, Any] = self.run_validation()
        metrics["training_time"] = end_time - start_time
        self.save_metrics_to_checkpoint_folder(metrics, name="validation_metrics")
        return metrics

    def run_validation(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        metrics: List[Dict[str, Any]] = self.trainer.validate(  # type: ignore
            model=self.model,
            datamodule=self.datamodule,
            verbose=self.verbose,
            ckpt_path="best",
        )
        if len(metrics) == 0:
            return {}
        metrics_0: Dict[str, Any] = metrics[0]
        end_time = time.perf_counter()
        metrics_0["validation_time"] = end_time - start_time
        return metrics_0

    def run_test(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        metrics: Dict[str, Any] = self.trainer.test(  # type: ignore
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path="best",
        )[0]
        end_time = time.perf_counter()
        metrics["test_time"] = end_time - start_time
        self.save_metrics_to_checkpoint_folder(metrics, name="test_metrics")
        return metrics

    def setup_model_args(self):
        self.model_args["input_shape"] = self.datamodule.input_shape
        self.model_args["output_shape"] = self.datamodule.output_shape

    def _build_model_instance(self):
        self.setup_model_args()
        return self.model_cls(**self.model_args)

    def _build_trainer_instance(self):
        logger = TensorBoardLogger(
            os.path.join(self.checkpoint_folder, "tb_logs"), name=self.model_cls.MODEL_NAME  # type: ignore
        )
        trainer = Trainer(
            accelerator=self.accelerator,
            max_epochs=self.max_epochs,  # type: ignore
            max_time=self.max_time,  # type: ignore
            callbacks=self.get_callbacks(),
            logger=logger,
            enable_progress_bar=self.verbose,
            enable_model_summary=self.verbose,
            **self.trainer_args,  # type: ignore
        )
        return trainer

    def get_callbacks(self) -> List[Callback]:
        callbacks: List[Callback] = [
            self.checkpoint_callback,
        ]
        if self.verbose and self.max_epochs > 1:
            callbacks.append(self.progress_bar)
        return callbacks

    def _set_trainer_args_defaults(self):
        profiler = SimpleProfiler(dirpath=self.checkpoint_folder, filename="perf_logs")
        self.trainer_args.setdefault("precision", "32-true")
        self.trainer_args.setdefault("gradient_clip_val", 0.5)
        self.trainer_args.setdefault("gradient_clip_algorithm", "norm")
        self.trainer_args.setdefault("profiler", profiler)
        return self

    def save_metrics_to_checkpoint_folder(self, metrics: Dict[str, Any], name: str = "metrics"):
        metrics_path = self.checkpoint_folder / f"{name}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True, default=str)
        if self.verbose:  # pragma: no cover
            print(f"Metrics saved to {metrics_path}")
        return metrics_path

    @property
    def model_cls(self):
        return self._model_cls

    @property
    def datamodule(self):
        return self._datamodule

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_time(self):
        return self._max_time

    @property
    def model_args(self):
        return self._model_args

    @property
    def checkpoint_folder(self):
        return self._checkpoint_folder

    @property
    def monitor(self):
        return self._monitor

    @property
    def monitor_mode(self):
        return self._monitor_mode

    @property
    def overwrite_fit(self):
        return self._overwrite_fit

    @property
    def verbose(self):
        return self._verbose

    @property
    def accelerator(self):
        return self._accelerator

    @property
    def trainer_args(self):
        return self._trainer_args
