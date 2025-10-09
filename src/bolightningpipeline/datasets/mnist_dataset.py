from pathlib import Path
from typing import Sequence, Union

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Compose

from .base_dataset import BaseDataset


class MNISTDataset(BaseDataset):
    DATASET_NAME = "MNIST"

    @staticmethod
    def to_long_tensor(y: Sequence[int]) -> torch.Tensor:
        """Convert a Sequence of integers to a long tensor."""
        return torch.tensor(y, dtype=torch.long)

    def __init__(
            self,
            data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME,
            train: bool = True,
            **kwargs
    ):
        super().__init__(data_dir, train, **kwargs)
        self._data = MNIST(
            self.data_dir,
            train=self.train,
            download=True,
            transform=Compose([
                ToTensor(),
                transforms.Normalize(0.0, 1.0),
                # transforms.RandomCrop(32, pad_if_needed=True),  # Randomly crop a 32x32 patch
                # transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
                # transforms.RandomRotation(10),  # Randomly rotate up to 10 degrees
                # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            ]),
            target_transform=Compose([self.to_long_tensor])
        )

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_output_shape(self) -> tuple:
        return (10,)
