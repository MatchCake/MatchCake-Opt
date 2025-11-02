from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from medmnist import RetinaMNIST
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Compose

from .base_dataset import BaseDataset


class RetinaMNISTDataset(BaseDataset):
    DATASET_NAME = "RetinaMNIST"

    @staticmethod
    def to_scalar_tensor(y: Sequence[int]):
        return torch.tensor(y).item()

    @staticmethod
    def to_long_tensor(y: Sequence[int]) -> torch.Tensor:
        return torch.tensor(y, dtype=torch.long)

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        transform = Compose(
            [
                ToTensor(),
                transforms.Normalize(0.0, 1.0),
            ]
        )
        target_transform = Compose([self.to_scalar_tensor, self.to_long_tensor])
        self._data = RetinaMNIST(
            root=self.data_dir,
            split="train" if self.train else "test",
            download=True,
            transform=transform,
            target_transform=target_transform,
        )
        self._n_classes = np.unique(self._data.labels).size
        if self.train:
            val_dataset = RetinaMNIST(
                root=self.data_dir, split="val", download=True, transform=transform, target_transform=target_transform
            )
            self._data = ConcatDataset([self._data, val_dataset])

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_output_shape(self) -> tuple:
        return (self._n_classes,)
