import argparse
from typing import Any, Optional, Tuple

import lightning
import psutil
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from ..datasets.base_dataset import BaseDataset


class DataModule(lightning.LightningDataModule):
    """
    Handles the loading, splitting, and management of datasets used for training,
    validation, and testing in a PyTorch Lightning workflow.

    The `DataModule` class provides a standardized interface for working with
    datasets, including handling train/validation splits, data loaders, and other
    dataset-specific configurations.
    """

    DEFAULT_RANDOM_STATE = 0
    DEFAULT_TRAIN_VAL_SPLIT = 0.85
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_NUM_WORKERS = min(2, psutil.cpu_count(logical=True) - 1)

    @classmethod
    def from_dataset_name(
        cls,
        dataset_name: str,
        split_id: int,
        *,
        train_val_split: float = DEFAULT_TRAIN_VAL_SPLIT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> "DataModule":
        from ..datasets import get_dataset_cls_by_name

        return cls(
            train_dataset=get_dataset_cls_by_name(dataset_name)(train=True),
            test_dataset=get_dataset_cls_by_name(dataset_name)(train=False),
            split_id=split_id,
            train_val_split=train_val_split,
            batch_size=batch_size,
            random_state=random_state,
            num_workers=num_workers,
        )

    @classmethod
    def add_specific_args(cls, parent_parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
        if parent_parser is None:
            parent_parser = argparse.ArgumentParser()
        parser = parent_parser.add_argument_group(f"{cls.__name__} Arguments")
        parser.add_argument("--batch_size", type=int, default=cls.DEFAULT_BATCH_SIZE)
        parser.add_argument("--fold_id", type=int, default=0, help="Fold ID for cross-validation")
        parser.add_argument("--random_state", type=int, default=cls.DEFAULT_RANDOM_STATE)
        parser.add_argument("--num_workers", type=int, default=cls.DEFAULT_NUM_WORKERS)
        return parent_parser

    def __init__(
        self,
        train_dataset: BaseDataset,
        test_dataset: BaseDataset,
        split_id: int,
        *,
        train_val_split: float = DEFAULT_TRAIN_VAL_SPLIT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ):
        """
        Initializes the class with the provided training and testing datasets, split
        information, and relevant parameters.

        :param train_dataset: The dataset to be used for training the model.
        :type train_dataset: BaseDataset
        :param test_dataset: The dataset to be used for testing the model.
        :type test_dataset: BaseDataset
        :param split_id: An identifier for tracking or differentiating dataset splits.
        :type split_id: int
        :param train_val_split: Proportion of the training dataset to be used
            for validation. Defaults to DEFAULT_TRAIN_VAL_SPLIT. Must be between 0 and 1.
        :type train_val_split: float, optional
        :param batch_size: The size of each batch used during data loading.
            Defaults to DEFAULT_BATCH_SIZE. Must be a positive integer.
        :type batch_size: int, optional
        :param random_state: Determines the randomness for reproducibility during
            dataset splitting. Defaults to DEFAULT_RANDOM_STATE.
        :type random_state: int, optional
        :param num_workers: Number of workers to use for data loading.
            Defaults to DEFAULT_NUM_WORKERS.
        :type num_workers: int, optional
        """
        super().__init__()
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        assert train_val_split > 0, f"Train split must be positive, got {train_val_split}"
        assert train_val_split <= 1, f"Train split must be at most 1, got {train_val_split}"
        self._train_val_split = train_val_split
        self._batch_size = batch_size
        self._random_state = random_state
        self._split_id = split_id
        self._given_train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._num_workers = num_workers
        self._train_dataset: Optional[Subset] = None
        self._val_dataset: Optional[Subset] = None

    def prepare_data(self) -> None:
        self._given_train_dataset.prepare_data()
        self._test_dataset.prepare_data()
        self._train_dataset, self._val_dataset = self._split_train_val_dataset(self._given_train_dataset)
        return

    def _split_train_val_dataset(self, dataset: Dataset) -> Tuple[Subset, Subset]:
        train_subset, val_subset = random_split(
            dataset,
            lengths=[self._train_val_split, 1 - self._train_val_split],
            generator=torch.Generator().manual_seed(self._split_id),
        )
        return train_subset, val_subset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
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
    def train_dataset(self) -> Optional[Subset]:
        return self._train_dataset

    @property
    def val_dataset(self) -> Optional[Subset]:
        return self._val_dataset

    @property
    def test_dataset(self) -> BaseDataset:
        return self._test_dataset
