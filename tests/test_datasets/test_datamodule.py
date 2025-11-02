import argparse
from unittest.mock import MagicMock

import numpy as np
import pytest
from torch.utils.data import DataLoader

from matchcake_opt.datamodules.datamodule import DataModule
from matchcake_opt.datasets.digits2d import Digits2D


class TestDataModule:
    @pytest.fixture
    def datamodule_instance(self, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr("matchcake_opt.datasets.digits2d.load_digits", mock)
        mock.return_value = (np.zeros((10, 8 * 8)), np.zeros((10,), dtype=int))
        datamodule = DataModule.from_dataset_name("digits2d", 0)
        return datamodule

    def test_from_dataset_name(self, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr("matchcake_opt.datasets.digits2d.load_digits", mock)
        mock.return_value = (np.zeros((10, 8 * 8)), np.zeros((10,), dtype=int))

        datamodule = DataModule.from_dataset_name("digits2d", 0)
        assert isinstance(datamodule.test_dataset, Digits2D)

    def test_train_dataloader(self, datamodule_instance):
        assert isinstance(datamodule_instance.train_dataloader(), DataLoader)

    def test_val_dataloader(self, datamodule_instance):
        assert isinstance(datamodule_instance.val_dataloader(), DataLoader)

    def test_test_dataloader(self, datamodule_instance):
        assert isinstance(datamodule_instance.test_dataloader(), DataLoader)

    def test_input_shape(self, datamodule_instance):
        assert datamodule_instance.input_shape == datamodule_instance.test_dataset.get_input_shape()

    def test_output_shape(self, datamodule_instance):
        assert datamodule_instance.output_shape == datamodule_instance.test_dataset.get_output_shape()

    def test_add_specific_args(self):
        parser = DataModule.add_specific_args()
        assert isinstance(parser, argparse.ArgumentParser)
