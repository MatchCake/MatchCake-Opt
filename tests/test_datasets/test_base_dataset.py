import shutil
from pathlib import Path

import pytest

from matchcake_opt.datasets.base_dataset import BaseDataset


class TestBaseDataset:
    @pytest.fixture
    def data_dir(self):
        path = Path(".tmp") / "data_dir"
        yield path
        shutil.rmtree(path, ignore_errors=True)

    def test_init(self, data_dir):
        dataset = BaseDataset(data_dir=data_dir, train=True)
        assert dataset.data_dir == data_dir
        assert data_dir.exists()
        assert dataset.train
