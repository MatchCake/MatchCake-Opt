import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from matchcake_opt.datasets.digits2d import Digits2D


class TestDigits2D:
    MOCK_LEN = 10

    @pytest.fixture(scope="class")
    def data_dir(self):
        path = Path(".tmp") / "data_dir" / "digits2d"
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def data_mock(self, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr("matchcake_opt.datasets.digits2d.load_digits", mock)
        mock.return_value = (np.zeros((self.MOCK_LEN, 8 * 8)), np.zeros((self.MOCK_LEN,), dtype=int))
        return mock

    @pytest.fixture
    def dataset_instance(self, data_mock, data_dir):
        return Digits2D(data_dir=data_dir, train=True)

    def test_init(self, data_mock, data_dir):
        dataset = Digits2D(data_dir=data_dir, train=True)
        assert isinstance(dataset._data, torch.utils.data.Dataset)

    def test_getitem(self, dataset_instance):
        datum = dataset_instance[0]
        assert isinstance(datum, tuple)
        assert isinstance(datum[0], torch.Tensor)
        assert isinstance(datum[1], torch.Tensor)
        assert datum[1].dtype == torch.long

    def test_len(self, dataset_instance):
        assert len(dataset_instance) == int(dataset_instance.TRAIN_TEST_SPLIT * self.MOCK_LEN)

    def test_output_shape(self, dataset_instance):
        assert dataset_instance.get_output_shape() == (10,)
