import shutil
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional, Union

import lightning
import lightning as pl
import torch


class ModelCheckpoint(lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint):
    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Union[bool, Literal["link"]]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
        *,
        save_best_to: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        if save_best_to is not None:
            save_best_to = Path(save_best_to)
        self._save_best_to = save_best_to

    def _update_best_and_save(
        self, current: torch.Tensor, trainer: "pl.Trainer", monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        super()._update_best_and_save(current, trainer, monitor_candidates)
        if self._save_best_to is None:  # pragma: no cover
            return
        best_model_path = Path(self.best_model_path)
        if not best_model_path.exists():
            return
        if best_model_path.is_file():
            shutil.copyfile(best_model_path, self._save_best_to)
        return

    @property
    def save_best_to(self) -> Optional[Path]:
        return self._save_best_to
