import sys

import lightning as pl
from lightning.pytorch.callbacks import ProgressBar
from tqdm import tqdm


class EpochProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.p_bar = None
        self._enabled = True
        self._current_postfix = {}

    def on_train_start(self, trainer, pl_module):
        self.p_bar = tqdm(
            desc="Epoch Progress",
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            total=trainer.max_epochs,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.p_bar:  # pragma: no cover
            return
        self.p_bar.update(1)
        self.p_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.p_bar:  # pragma: no cover
            return
        self.p_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, trainer, pl_module):
        if not self.p_bar:  # pragma: no cover
            return
        self.p_bar.close()

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled
