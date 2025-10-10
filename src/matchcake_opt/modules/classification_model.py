import argparse
import warnings
from typing import Any, Dict, Optional

import lightning
import torch
from ax import RangeParameterConfig
from matchcake.utils import torch_utils
from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection, Precision, Recall

from .base_model import BaseModel


class ClassificationModel(BaseModel):
    MODEL_NAME = "ClassificationModel"

    def __init__(
        self,
        input_shape: Optional[tuple[int, ...]],
        output_shape: Optional[tuple[int, ...]],
        **kwargs,
    ):
        super().__init__(input_shape, output_shape, **kwargs)
        self.val_loss: torch.nn.Module = torch.nn.NLLLoss()
        self.train_loss: torch.nn.Module = torch.nn.NLLLoss()

    def configure_metrics(self) -> MetricCollection:
        num_classes = self.output_shape[-1] if self.output_shape else 1
        return MetricCollection(
            Accuracy("multiclass", num_classes=num_classes),
            F1Score("multiclass", num_classes=num_classes),
            Recall("multiclass", num_classes=num_classes),
            Precision("multiclass", num_classes=num_classes),
            AUROC("multiclass", num_classes=num_classes),
            prefix="train_",
        )

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        logits = torch.log_softmax(output, dim=1)
        loss = self.train_loss(logits, target)
        with torch.no_grad():
            self.log("train_loss", loss.detach().cpu(), prog_bar=True)
            probs = torch.softmax(output, dim=1)
            self.train_metrics.update(probs, target)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        with torch.no_grad():
            output = self(inputs)
            logits = torch.log_softmax(output, dim=1)
            probs = torch.softmax(output, dim=1)
            loss = self.val_loss(logits, target)
            self.log(f"val_loss", loss, prog_bar=True)
            self.val_metrics.update(probs, target)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        with torch.no_grad():
            output = self(inputs)
            logits = torch.log_softmax(output, dim=1)
            probs = torch.softmax(output, dim=1)
            loss = self.val_loss(logits, target)
            self.log(f"test_loss", loss, prog_bar=True)
            self.test_metrics.update(probs, target)
        return loss

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
