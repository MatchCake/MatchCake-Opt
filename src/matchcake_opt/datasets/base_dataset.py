from pathlib import Path
from typing import Union

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    DATASET_NAME = "base_dataset"

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._train = train
        self._kwargs = kwargs

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def get_input_shape(self) -> tuple:
        return tuple(self[0][0].shape)  # pragma: no cover

    def get_output_shape(self) -> tuple:
        return tuple(self[0][-1].shape)  # pragma: no cover

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def train(self) -> bool:
        return self._train
