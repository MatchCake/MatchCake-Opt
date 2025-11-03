import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from matchcake_opt.datasets.cifar10_dataset import Cifar10Dataset


class TestCifar10Dataset:
    MOCK_LEN = 10

    @pytest.fixture(scope="class")
    def data_dir(self):
        path = Path(".tmp") / "data_dir" / "cifar10"
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def cifar10_mock(self, monkeypatch):
        cifar10_cls_mock = MagicMock()
        monkeypatch.setattr("matchcake_opt.datasets.cifar10_dataset.CIFAR10", cifar10_cls_mock)
        mock = MagicMock()
        cifar10_cls_mock.return_value = mock
        mock.__getitem__.return_value = (torch.zeros(28, 28), torch.zeros(1).long())
        mock.__len__.return_value = self.MOCK_LEN
        return mock

    def test_init(self, cifar10_mock, data_dir):
        dataset = Cifar10Dataset(data_dir=data_dir, train=True)
        assert dataset._data == cifar10_mock

    def test_getitem(self, cifar10_mock, data_dir):
        dataset = Cifar10Dataset(data_dir=data_dir, train=True)
        datum = dataset[0]
        assert isinstance(datum, tuple)
        assert isinstance(datum[0], torch.Tensor)
        assert isinstance(datum[1], torch.Tensor)
        assert datum[1].dtype == torch.long
        cifar10_mock.__getitem__.assert_called_once_with(0)

    def test_len(self, cifar10_mock, data_dir):
        dataset = Cifar10Dataset(data_dir=data_dir, train=True)
        assert len(dataset) == self.MOCK_LEN

    def test_output_shape(self, cifar10_mock, data_dir):
        dataset = Cifar10Dataset(data_dir=data_dir, train=True)
        assert dataset.get_output_shape() == (10,)
