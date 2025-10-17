from collections import Counter

import numpy as np
import torch
from matchcake.utils.torch_utils import to_numpy
from torch_geometric.data import Data
from torchmetrics import MetricCollection

from .base_model import BaseModel


class MaxcutModel(BaseModel):
    MODEL_NAME = "MaxcutModel"

    @staticmethod
    def bitstring_arr_to_str(bit_string_sample):
        return "".join(str(bs) for bs in np.asarray(bit_string_sample, dtype=int).ravel())

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
            self.train_metrics.update(output)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        with torch.no_grad():
            output = self(inputs)
            loss = self.val_loss(output)
            self.log(f"val_loss", loss, prog_bar=True)
            self.val_metrics.update(output)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch
        with torch.no_grad():
            output = self(inputs)
            loss = self.val_loss(output)
            self.log(f"test_loss", loss, prog_bar=True)
            self.test_metrics.update(output)
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

    def compute_metrics_from_samples(self, samples: torch.Tensor) -> dict:
        samples_arr = to_numpy(samples).astype(int)
        samples_str = np.asarray(list(map(self.bitstring_arr_to_str, samples_arr)))
        counts_str = Counter(samples_str)
        counts_str_sorted_keys = list(sorted(list(counts_str.keys())))
        counts_str_arr = np.asarray([counts_str[key] for key in counts_str_sorted_keys])
        probs = counts_str_arr / np.sum(counts_str_arr)
        bitstrings_arr = self.bitstrings_to_arr(counts_str_sorted_keys)

        n_cut_edges = np.asarray([
            len(self.get_cut_edges_from_sets(
                np.arange(self.n_qubits)[key == 0],
                np.arange(self.n_qubits)[key == 1]
            ))
            for key in bitstrings_arr
        ])
        components = {
            "n_cut_edges": n_cut_edges.tolist(),
            "probs": probs.tolist(),
            "bit_strings": counts_str_sorted_keys,
            "max_cut": int(np.max(n_cut_edges)),
            "max_cut_state": counts_str_sorted_keys[np.argmax(n_cut_edges)],
        }
        return components
