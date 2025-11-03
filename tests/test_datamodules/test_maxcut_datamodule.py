import argparse

import pytest
from torch_geometric.loader import DataLoader

from matchcake_opt.datamodules.maxcut_datamodule import MaxcutDataModule
from matchcake_opt.datasets.maxcut_dataset import MaxcutDataset


class TestMaxcutDataModule:
    @pytest.fixture
    def datamodule_instance(self):
        dataset = MaxcutDataset(4, "regular", d=3)
        datamodule = MaxcutDataModule(dataset)
        datamodule.prepare_data()
        return datamodule

    def test_add_specific_args(self):
        parser = MaxcutDataModule.add_specific_args()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_split_train_val_dataset(self, datamodule_instance):
        assert datamodule_instance._split_train_val_dataset(datamodule_instance.train_dataset) == (
            datamodule_instance.train_dataset,
            None,
        )

    def test_train_dataloader(self, datamodule_instance):
        assert isinstance(datamodule_instance.train_dataloader(), DataLoader)

    def test_val_dataloader(self, datamodule_instance):
        assert isinstance(datamodule_instance.val_dataloader(), DataLoader)

    def test_test_dataloader(self, datamodule_instance):
        assert isinstance(datamodule_instance.test_dataloader(), DataLoader)
