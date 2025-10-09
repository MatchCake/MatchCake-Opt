from typing import Dict, Type, Union

from .datamodule import DataModule

from ..utils import get_all_subclasses
from .base_dataset import BaseDataset
from .digits2d import Digits2D
from .mnist_dataset import MNISTDataset
from .cifar10_dataset import Cifar10Dataset
from .pathmnist_dataset import PathMNISTDataset


dataset_name_to_type_map: Dict[str, Type[BaseDataset]] = {
    _cls.DATASET_NAME: _cls
    for _cls in get_all_subclasses(BaseDataset, include_base_cls=False)
    if hasattr(_cls, "DATASET_NAME")
}
dataset_name_to_type_map.update(
    {
        _cls.DATASET_NAME.lower(): _cls
        for _cls in get_all_subclasses(BaseDataset, include_base_cls=False)
        if hasattr(_cls, "DATASET_NAME")
    }
)


def get_dataset_cls_by_name(dataset: Union[str, Type[BaseDataset]]) -> Type[BaseDataset]:
    if isinstance(dataset, str):
        if dataset.lower() not in dataset_name_to_type_map:
            raise ValueError(f"Dataset {dataset} ({dataset.lower()}) not found in {dataset_name_to_type_map.keys()}")
        return dataset_name_to_type_map[dataset.lower()]
    return dataset
