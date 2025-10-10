import argparse
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pythonbasictools as pbt
from ax.api.client import Client
from pythonbasictools.collections_tools import list_insert_replace_at, ravel_dict
from tqdm import tqdm

from ..datasets.datamodule import DataModule
from ..modules.base_model import BaseModel
from .lightning_pipeline import LightningPipeline


class AutoMLPipeline:
    DEFAULT_AUTOML_ITERATIONS = 32
    DEFAULT_INNER_MAX_EPOCHS = 10_000
    DEFAULT_INNER_MAX_TIME = "90:00:00:00"
    DEFAULT_AUTOML_OVERWRITE_FIT = False

    @classmethod
    def add_specific_args(cls, parent_parser: Optional[argparse.ArgumentParser] = None):
        if parent_parser is None:
            parent_parser = argparse.ArgumentParser()
        parent_parser = LightningPipeline.add_specific_args(parent_parser)
        parser = parent_parser.add_argument_group(cls.__name__)  # type: ignore
        parser.add_argument(
            "--automl_iterations",
            type=int,
            default=cls.DEFAULT_AUTOML_ITERATIONS,
        )
        parser.add_argument(
            "--inner_max_epochs",
            type=int,
            default=cls.DEFAULT_INNER_MAX_EPOCHS,
        )
        parser.add_argument(
            "--inner_max_time",
            type=str,
            default=cls.DEFAULT_INNER_MAX_TIME,
        )
        parser.add_argument(
            "--automl_overwrite_fit",
            type=bool,
            default=cls.DEFAULT_AUTOML_OVERWRITE_FIT,
            action=argparse.BooleanOptionalAction,
        )
        return parent_parser

    def __init__(
        self,
        model_cls: Type[BaseModel],
        datamodule: DataModule,
        *,
        automl_iterations: int = DEFAULT_AUTOML_ITERATIONS,
        inner_max_epochs: int = DEFAULT_INNER_MAX_EPOCHS,
        inner_max_time: str = DEFAULT_INNER_MAX_TIME,
        automl_overwrite_fit: bool = DEFAULT_AUTOML_OVERWRITE_FIT,
        checkpoint_folder: str = "ax_ckpts",
        history_filename: str = "history.json",
        ax_state_filename: str = "ax_state.json",
        **pipeline_args,
    ):
        self.model_cls = model_cls
        self.datamodule = datamodule
        self.automl_iterations = automl_iterations
        self.inner_max_epochs = inner_max_epochs
        self.inner_max_time = inner_max_time
        self.automl_overwrite_fit = automl_overwrite_fit
        self.checkpoint_folder = Path(checkpoint_folder)
        self.history_filename = history_filename
        self.ax_state_filename = ax_state_filename
        self._hp_list = [c.name for c in self.model_cls.HP_CONFIGS]
        self.monitor = pipeline_args.get("monitor", "val_loss")
        self.monitor_mode = pipeline_args.get("monitor_mode", "min")

        self.client = Client()
        self.client.configure_experiment(
            parameters=self.model_cls.HP_CONFIGS,
        )
        self.client.experiment_name = f"{self.model_cls.MODEL_NAME}_experiment"  # type: ignore
        self.client.configure_optimization(
            objective=self.monitor if self.monitor_mode == "max" else f"-{self.monitor}",
        )
        self.pipeline_args = pipeline_args
        self.history: List[Dict[str, Any]] = []
        if not self.automl_overwrite_fit:
            self.maybe_load()

    def __getstate__(self):
        state = {
            "history": self.history,
            "best_checkpoint": self.best_checkpoint,
        }
        return state

    def __setstate__(self, state):
        self.history = state["history"]
        return

    def get_best_params(self):
        return self.client.get_best_parameterization()[0]

    def get_best_pipeline_params(self):
        return {**self.pipeline_args, **self.get_best_params()}

    def run_pipeline(self, **pipeline_args) -> Tuple[LightningPipeline, Dict[str, Any]]:
        params_hash = pbt.hash_dict({k: v for k, v in pipeline_args.items() if k in self._hp_list})
        pipeline_args["checkpoint_folder"] = self.checkpoint_folder / params_hash
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = LightningPipeline(
                model_cls=self.model_cls,
                datamodule=self.datamodule,
                **pipeline_args,
            )
            metrics = pipeline.run()
        return pipeline, metrics

    def run_best_pipeline(self, **additional_params):
        additional_params.setdefault("verbose", True)
        return self.run_pipeline(**{**self.get_best_pipeline_params(), **additional_params})

    def __call__(self, params, idx):
        pipeline, metrics = self.run_pipeline(
            **{
                **self.pipeline_args,
                **params,
                **{
                    "max_epochs": self.inner_max_epochs,
                    "max_time": self.inner_max_time,
                    "overwrite_fit": False,
                    "verbose": False,
                },
            }
        )
        metrics = ravel_dict(metrics)
        metrics["checkpoint_folder"] = Path(pipeline.checkpoint_folder).relative_to(self.checkpoint_folder)
        self.history = list_insert_replace_at(self.history, idx, metrics)
        return metrics

    def run(self):
        p_bar = tqdm(initial=len(self.history), total=self.automl_iterations, desc="AutoML iterations")
        for iteration in range(p_bar.n, p_bar.total):
            trials = self.client.get_next_trials(max_trials=1)
            for trial_index, parameters in trials.items():
                new_y = self(parameters, iteration)
                self.client.complete_trial(trial_index, {self.monitor: new_y[self.monitor]})
            p_bar.set_postfix({"best": self.client.get_best_parameterization()})
            self.save()
            self.delete_not_best_checkpoints()
            self.copy_best_checkpoint()
            p_bar.update()
        p_bar.close()
        self.save()
        return self

    def save(self):
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)
        json.dump(
            self.__getstate__(),
            open(self.checkpoint_folder / self.history_filename, "w+"),
            indent=4,
            default=str,
        )
        self.client.save_to_json_file((self.checkpoint_folder / self.ax_state_filename).resolve())
        return self

    def maybe_load(self):
        save_path = self.checkpoint_folder / self.history_filename
        if os.path.exists(save_path):
            data = json.load(open(save_path, "r"))
            try:
                self.__setstate__(data)
            except Exception as e:
                warnings.warn(f"Failed to load {save_path}: {e}")
        ax_state_file = (self.checkpoint_folder / self.ax_state_filename).resolve()
        if ax_state_file.exists():
            self.client = self.client.load_from_json_file(ax_state_file)
        else:
            warnings.warn("AX state file not found, starting a new experiment.")
        return self

    def delete_not_best_checkpoints(self):
        best_y = self.client.get_best_parameterization()[1][self.monitor][0]
        for history_item in self.history:
            checkpoint_folder = history_item.get("checkpoint_folder", None)
            if checkpoint_folder is None:
                continue
            checkpoint_folder = (self.checkpoint_folder / checkpoint_folder).resolve()
            y = history_item[self.monitor]
            if not np.isclose(y, best_y, atol=1e-5):
                try:
                    shutil.rmtree(checkpoint_folder, ignore_errors=True)
                except Exception as e:
                    pass
        return self

    def copy_best_checkpoint(self):
        best_checkpoint_folder = self.best_checkpoint
        if best_checkpoint_folder is None:
            return None
        best_checkpoint = best_checkpoint_folder / f"{self.model_cls.MODEL_NAME}.ckpt"
        if not best_checkpoint.exists():
            return None
        target_path = self.checkpoint_folder / f"{self.model_cls.MODEL_NAME}.ckpt"
        shutil.copyfile(best_checkpoint, target_path)
        return target_path

    @property
    def best_checkpoint(self) -> Optional[Path]:
        best_y = self.client.get_best_parameterization()[1][self.monitor][0]  # type: ignore
        for history_item in self.history:
            checkpoint_folder = history_item.get("checkpoint_folder", None)
            if checkpoint_folder is None:
                continue
            checkpoint_folder = (self.checkpoint_folder / checkpoint_folder).resolve()
            if not checkpoint_folder.exists():
                continue
            y = history_item[self.monitor]
            if np.isclose(y, best_y):
                return checkpoint_folder
        return None
