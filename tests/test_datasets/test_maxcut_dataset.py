import pytest
from torch_geometric.data import Data

from matchcake_opt.datasets.maxcut_dataset import MaxcutDataset


@pytest.mark.parametrize(
    "graph_type, graph_params",
    [
        ("circular", {"n_nodes": 3}),
        ("erdos_renyi", {"n_nodes": 3, "p": 0.2}),
        ("regular", {"n_nodes": 4, "d": 3}),
    ],
)
class TestMaxcutDataset:
    @pytest.fixture
    def dataset(self, graph_type, graph_params):
        dataset = MaxcutDataset(graph_type=graph_type, **graph_params)
        dataset.prepare_data()
        return dataset

    def test_prepare_data(self, graph_type, graph_params):
        dataset = MaxcutDataset(graph_type=graph_type, **graph_params)
        dataset.prepare_data()
        assert dataset._built_flag
        assert isinstance(dataset._data, Data)

    def test_getitem(self, dataset):
        assert isinstance(dataset[0], Data)

    def test_len(self, dataset):
        assert len(dataset) == 1

    @pytest.mark.parametrize("n_nodes", [4, 6, 8])
    def test_get_input_shape(self, n_nodes, graph_type, graph_params):
        graph_params["n_nodes"] = n_nodes
        dataset = MaxcutDataset(graph_type=graph_type, **graph_params)
        dataset.prepare_data()
        assert dataset.get_input_shape() == (n_nodes,)

    def test_train_setter(self, dataset):
        assert dataset.train
        dataset.train = False
        assert not dataset.train

    def test_verify_graph_params(self, graph_type, graph_params):
        if len(graph_params) > 1:
            with pytest.raises(ValueError):
                MaxcutDataset(4, graph_type=graph_type)
        else:
            MaxcutDataset(4, graph_type=graph_type)

    def test_output_shape(self, dataset):
        assert dataset.get_output_shape() == (1,)
