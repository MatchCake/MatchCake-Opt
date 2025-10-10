from pathlib import Path
from typing import Union

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    DATASET_NAME = "base_dataset"

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.train = train

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def get_input_shape(self) -> tuple:
        return tuple(self[0][0].shape)

    def get_output_shape(self) -> tuple:
        return tuple(self[0][-1].shape)
