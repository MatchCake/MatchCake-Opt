import argparse
from typing import Optional

import lightning
import psutil
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from ..datasets.base_dataset import BaseDataset


class DataModule(lightning.LightningDataModule):
    DEFAULT_RANDOM_STATE = 0
    N_FOLDS = 5
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_NUM_WORKERS = min(2, psutil.cpu_count(logical=True) - 1)

    @classmethod
    def from_dataset_name(
        cls,
        dataset_name: str,
        fold_id: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> "DataModule":
        from ..datasets import get_dataset_cls_by_name

        return cls(
            train_dataset=get_dataset_cls_by_name(dataset_name)(train=True),
            test_dataset=get_dataset_cls_by_name(dataset_name)(train=False),
            fold_id=fold_id,
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
        fold_id: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ):
        super().__init__()
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        self._batch_size = batch_size
        self._random_state = random_state
        assert 0 <= fold_id < self.N_FOLDS, f"Fold id {fold_id} is out of range [0, {self.N_FOLDS})"
        self._fold_id = fold_id
        self._train_dataset, self._val_dataset = self._split_train_val_dataset(train_dataset)
        self._test_dataset = test_dataset
        self._num_workers = num_workers

    def _split_train_val_dataset(self, dataset: Dataset):
        fold_ratio = 1 / self.N_FOLDS
        subsets = random_split(
            dataset,
            lengths=[fold_ratio for _ in range(self.N_FOLDS)],
            generator=torch.Generator().manual_seed(self._random_state),
        )
        val_subset = subsets[self._fold_id]
        train_subset_indexes = [i for i in range(self.N_FOLDS) if i != self._fold_id]
        train_subset: torch.utils.data.Dataset = torch.utils.data.ConcatDataset(
            [subsets[i] for i in train_subset_indexes]
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
    def train_dataset(self) -> ConcatDataset:
        return self._train_dataset

    @property
    def val_dataset(self) -> Subset:
        return self._val_dataset

    @property
    def test_dataset(self) -> BaseDataset:
        return self._test_dataset
