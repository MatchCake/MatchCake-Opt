import argparse
from copy import deepcopy
from typing import Optional

import lightning
import psutil
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from torch_geometric.loader import DataLoader

from ..datasets.maxcut_dataset import MaxcutDataset
from .datamodule import DataModule


class MaxcutDataModule(DataModule):
    @classmethod
    def add_specific_args(cls, parent_parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
        if parent_parser is None:
            parent_parser = argparse.ArgumentParser()
        parser = parent_parser.add_argument_group(f"{cls.__name__} Arguments")
        return parent_parser

    @classmethod
    def from_dataset_name(
            cls,
            dataset_name: str,
            fold_id: int,
            batch_size: int = 0,
            random_state: int = 0,
            num_workers: int = 0,
    ) -> "DataModule":
        raise NotImplementedError("MaxcutDataModule does not support from_dataset_name method.")

    def __init__(
        self,
        train_dataset: MaxcutDataset,
        test_dataset: Optional[MaxcutDataset] = None,
    ):
        if test_dataset is None:
            test_dataset = deepcopy(train_dataset)
            train_dataset.train = False
        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            fold_id=0,
            batch_size=1,
            random_state=0,
            num_workers=0
        )

    def _split_train_val_dataset(self, dataset: MaxcutDataset):
        return dataset, None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 0,
            pin_memory=True,
        )

    @property
    def input_shape(self):
        return self.test_dataset.get_input_shape()

    @property
    def output_shape(self):
        return self.test_dataset.get_output_shape()

    @property
    def train_dataset(self) -> MaxcutDataset:
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self) -> MaxcutDataset:
        return self._test_dataset
