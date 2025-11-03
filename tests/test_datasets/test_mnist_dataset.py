import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from matchcake_opt.datasets.mnist_dataset import MNISTDataset


class TestMNISTDataset:
    MOCK_LEN = 10

    @pytest.fixture(scope="class")
    def data_dir(self):
        path = Path(".tmp") / "data_dir" / "mnist"
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def data_mock(self, monkeypatch):
        cls_mock = MagicMock()
        monkeypatch.setattr("matchcake_opt.datasets.mnist_dataset.MNIST", cls_mock)
        mock = MagicMock()
        cls_mock.return_value = mock
        mock.__getitem__.return_value = (torch.zeros(28, 28), torch.zeros(1).long())
        mock.__len__.return_value = self.MOCK_LEN
        return mock

    @pytest.fixture
    def dataset_instance(self, data_mock, data_dir):
        return MNISTDataset(data_dir=data_dir, train=True)

    def test_init(self, data_mock, data_dir):
        dataset = MNISTDataset(data_dir=data_dir, train=True)
        assert dataset._data == data_mock

    def test_getitem(self, data_mock, dataset_instance):
        datum = dataset_instance[0]
        assert isinstance(datum, tuple)
        assert isinstance(datum[0], torch.Tensor)
        assert isinstance(datum[1], torch.Tensor)
        assert datum[1].dtype == torch.long
        data_mock.__getitem__.assert_called_once_with(0)

    def test_len(self, dataset_instance):
        assert len(dataset_instance) == self.MOCK_LEN

    def test_output_shape(self, dataset_instance):
        assert dataset_instance.get_output_shape() == (10,)
