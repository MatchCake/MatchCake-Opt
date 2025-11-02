import pytest
from torch_geometric.data import Data

from matchcake_opt.datasets.maxcut_dataset import MaxcutDataset


@pytest.mark.parametrize(
    "graph_type",
    [
        "circular",
        "erdos_renyi",
        "regular",
    ],
)
class TestMaxcutDataset:
    def test_prepare_data(self, graph_type):
        dataset = MaxcutDataset(3, graph_type)
        dataset.prepare_data()
        assert dataset._built_flag
        assert isinstance(dataset._data, Data)

    def test_getitem(self, graph_type):
        dataset = MaxcutDataset(3, graph_type)
        dataset.prepare_data()
        assert isinstance(dataset[0], Data)

    def test_len(self, graph_type):
        dataset = MaxcutDataset(3, graph_type)
        dataset.prepare_data()
        assert len(dataset) == 1

    @pytest.mark.parametrize("n_nodes", [3, 4, 5])
    def test_get_input_shape(self, n_nodes, graph_type):
        dataset = MaxcutDataset(3, graph_type)
        dataset.prepare_data()
        assert dataset.get_input_shape() == (n_nodes,)

    def test_train_setter(self, n_nodes, graph_type):
        dataset = MaxcutDataset(3, graph_type, train=True)
        dataset.prepare_data()
        assert dataset.train
        dataset.train = False
        assert not dataset.train
