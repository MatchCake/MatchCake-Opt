from pathlib import Path
from typing import Sequence, Union

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from .base_dataset import BaseDataset


class MNISTDataset(BaseDataset):
    DATASET_NAME = "MNIST"

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        self._data = MNIST(
            self.data_dir,
            train=self.train,
            download=True,
            transform=v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize((0.1307,), (0.3081,)),
                    # TODO: Add a param to add other transforms?
                    # transforms.RandomCrop(32, pad_if_needed=True),  # Randomly crop a 32x32 patch
                    # transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
                    # transforms.RandomRotation(10),  # Randomly rotate up to 10 degrees
                    # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                ]
            ),
            target_transform=v2.Compose([v2.ToDtype(torch.long)]),
        )

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_output_shape(self) -> tuple:
        return (10,)
