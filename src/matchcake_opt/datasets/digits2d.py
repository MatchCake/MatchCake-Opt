from pathlib import Path
from typing import Union

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base_dataset import BaseDataset


class Digits2D(BaseDataset):
    DATASET_NAME = "digits2d"
    TRAIN_TEST_SPLIT = 0.9

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        x, y = self._load_data()
        x, y = self._preprocess(x, y)
        self._data = torch.utils.data.TensorDataset(x, y)

    def _load_data(self):
        x, y = load_digits(return_X_y=True, as_frame=False)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=self.TRAIN_TEST_SPLIT,
            random_state=0,
            shuffle=False,
        )
        if self.train:
            return x_train, y_train
        return x_test, y_test

    def _preprocess(self, x, y):
        x = StandardScaler().fit_transform(x)
        x = torch.from_numpy(x).float().reshape(-1, 8, 8)
        y = torch.from_numpy(y).long()
        return x, y

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_output_shape(self) -> tuple:
        return (10,)
