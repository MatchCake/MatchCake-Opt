from collections import Counter
from typing import Sequence

import numpy as np
import torch
from matchcake.utils.torch_utils import to_numpy
from torch_geometric.data import Data
from torchmetrics import MetricCollection

from .base_model import BaseModel


class MaxcutModel(BaseModel):
    MODEL_NAME = "MaxcutModel"

    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        optimizer: str = BaseModel.DEFAULT_OPTIMIZER,
        learning_rate: float = BaseModel.DEFAULT_LEARNING_RATE,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            optimizer=optimizer,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.train_loss = torch.nn.Identity()
        self.val_loss = torch.nn.Identity()

    def configure_metrics(self) -> MetricCollection:
        return MetricCollection([], prefix="train_")

    def sample(self, x) -> torch.Tensor:
        raise NotImplementedError("Child classes must implement the sample method that generates the cut solution.")

    def training_step(self, batch: Data, batch_idx):
        inputs = batch
        output = self(inputs)
        loss = self.train_loss(output)
        with torch.no_grad():
            self.log("train_loss", loss.detach().cpu(), prog_bar=True)
            self.train_metrics.update(inputs, output)
        return loss

    def validation_step(self, batch: Data, batch_idx):
        inputs = batch
        with torch.no_grad():
            output = self(inputs)
            loss = self.val_loss(output)
            self.log(f"val_loss", loss, prog_bar=True)
            self.val_metrics.update(inputs, output)
        return loss

    def test_step(self, batch: Data, batch_idx):
        inputs = batch
        with torch.no_grad():
            output = self(inputs)
            loss = self.val_loss(output)
            self.log(f"test_loss", loss, prog_bar=True)
            self.test_metrics.update(inputs, output)
        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output for the given input tensor.
        :param x: Input tensor.
        :return: Output tensor.
        """
        self.eval()
        with torch.no_grad():
            samples = self.sample(x)
        return samples
