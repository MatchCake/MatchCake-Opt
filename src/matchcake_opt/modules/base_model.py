import argparse
import warnings
from typing import Any, Dict, Optional

import lightning
import torch
from ax import RangeParameterConfig
from torchmetrics import MetricCollection


class BaseModel(lightning.LightningModule):
    MODEL_NAME = "BaseModel"
    DEFAULT_OPTIMIZER = "AdamW"
    DEFAULT_LEARNING_RATE = 2e-4

    HP_CONFIGS = [
        RangeParameterConfig(
            name="learning_rate",
            parameter_type="float",
            bounds=(1e-5, 0.5),
        ),
    ]

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: Optional[argparse.ArgumentParser] = None
    ) -> argparse.ArgumentParser:
        if parent_parser is None:
            parent_parser = argparse.ArgumentParser()
        parser = parent_parser.add_argument_group(f"{cls.__name__} Model")
        parser.add_argument(
            "--optimizer",
            type=str,
            default=cls.DEFAULT_OPTIMIZER,
            help="Optimizer to use for training",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=cls.DEFAULT_LEARNING_RATE,
            help="Learning rate for the optimizer",
        )
        return parent_parser

    def __init__(
        self,
        input_shape: Optional[tuple[int, ...]],
        output_shape: Optional[tuple[int, ...]],
        optimizer: str = DEFAULT_OPTIMIZER,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        **kwargs,
    ):
        super().__init__()
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self.save_hyperparameters("optimizer", "learning_rate", "input_shape", "output_shape")
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.val_loss: torch.nn.Module = torch.nn.MSELoss()
        self.train_loss: torch.nn.Module = torch.nn.MSELoss()
        self._metrics: MetricCollection = self.configure_metrics()

        self.train_metrics = self._metrics.clone(prefix="train_")
        self.val_metrics = self._metrics.clone(prefix="val_")
        self.test_metrics = self._metrics.clone(prefix="test_")

    def configure_metrics(self) -> MetricCollection:
        raise NotImplementedError("configure_metrics method must be implemented in subclasses.")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.train_loss(output, target)
        with torch.no_grad():
            self.log("train_loss", loss.detach().cpu(), prog_bar=True)
            self._metrics.update(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        with torch.no_grad():
            output = self(inputs)
            loss = self.val_loss(output, target)
            self.log(f"val_loss", loss, prog_bar=True)
            self.val_metrics.update(output, target)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        with torch.no_grad():
            output = self(inputs)
            loss = self.val_loss(output, target)
            self.log(f"test_loss", loss, prog_bar=True)
            self.test_metrics.update(output, target)
        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self._optimizer)(self.parameters(), lr=self._learning_rate)
        return optimizer

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output for the given input tensor.
        :param x: Input tensor.
        :return: Output tensor.
        """
        self.eval()
        with torch.no_grad():
            output = self(x)
            output = torch.softmax(output, dim=1)
        return output

    def on_train_epoch_start(self) -> None:
        self.train_metrics.reset()
        return

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        return

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()
        return

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        return

    def on_test_epoch_start(self) -> None:
        self.test_metrics.reset()
        return

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), prog_bar=True)
        return

    @property
    def optimizer(self) -> str:
        return self._optimizer

    @property
    def learning_rate(self) -> float:
        return self._learning_rate
