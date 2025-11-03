from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from medmnist import PathMNIST
from torch.utils.data import ConcatDataset
from torchvision.transforms import v2

from .base_dataset import BaseDataset


class PathMNISTDataset(BaseDataset):
    DATASET_NAME = "PathMNIST"

    @staticmethod
    def to_scalar_tensor(y: Sequence[int]):
        return torch.tensor(y).item()

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.238, 0.238, 0.238), (0.358, 0.309, 0.352)),
            ]
        )
        target_transform = v2.Compose([self.to_scalar_tensor, v2.ToDtype(torch.long)])
        self._data = PathMNIST(
            root=self.data_dir,
            split="train" if self.train else "test",
            download=True,
            transform=transform,
            target_transform=target_transform,
        )
        self._n_classes = np.unique(self._data.labels).size
        if self.train:
            val_dataset = PathMNIST(
                root=self.data_dir, split="val", download=True, transform=transform, target_transform=target_transform
            )
            self._data = ConcatDataset([self._data, val_dataset])

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_output_shape(self) -> tuple:
        return (self._n_classes,)
