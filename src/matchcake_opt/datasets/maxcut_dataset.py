from pathlib import Path
from typing import Literal, Optional, Union

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from .base_dataset import BaseDataset


class MaxcutDataset(BaseDataset):
    DATASET_NAME = "Maxcut"
    GRAPH_TYPES_TO_PARAMS = {
        "regular": ["d"],
        "erdos_renyi": ["p"],
        "circular": [],
    }

    def __init__(
        self,
        n_nodes: int,
        graph_type: Literal["regular", "erdos_renyi", "circular"],
        seed: int = 0,
        data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME,
        train: bool = True,
        **kwargs,
    ):
        super().__init__(data_dir, train=train, **kwargs)
        self._n_nodes = n_nodes
        self._graph_type = graph_type
        self._seed = seed
        self._verify_graph_params(self._graph_type, **kwargs)
        self._nx_graph: Optional[nx.Graph] = None
        self._data: Optional[Data] = None
        self._built_flag = False

    def _verify_graph_params(self, graph_type: str, **graph_params):
        expected_params = self.GRAPH_TYPES_TO_PARAMS.get(graph_type)
        if expected_params is None:
            raise ValueError(f"Unsupported graph type: {graph_type}")
        missing_params = [param for param in expected_params if param not in graph_params]
        if missing_params:
            raise ValueError(f"Missing parameters for graph type '{graph_type}': {', '.join(missing_params)}")

    def prepare_data(self) -> None:
        if self._graph_type == "regular":
            self._nx_graph = self._build_regular_graph()
        elif self._graph_type == "erdos_renyi":
            self._nx_graph = self._build_erdos_renyi_graph()
        elif self._graph_type == "circular":
            self._nx_graph = self._build_circular_graph()
        else:
            raise ValueError(f"Unsupported graph type: {self._graph_type}")
        self._data = from_networkx(self._nx_graph)
        self._data.y = torch.tensor([self._get_lower_energy_bound(), self._get_upper_n_cut_bound()], dtype=torch.float)
        self._built_flag = True
        return

    def __getitem__(self, item):
        if not self._built_flag:
            self.prepare_data()
        return self._data

    def __len__(self):
        return 1

    def get_input_shape(self) -> tuple:
        return (self._n_nodes,)

    def get_output_shape(self) -> tuple:
        return (1,)

    def _build_regular_graph(self) -> nx.Graph:
        return nx.random_regular_graph(self._kwargs["d"], self._n_nodes, seed=self._seed)

    def _build_erdos_renyi_graph(self) -> nx.Graph:
        return nx.erdos_renyi_graph(self._n_nodes, self._kwargs["p"], seed=self._seed)

    def _build_circular_graph(self) -> nx.Graph:
        return nx.circulant_graph(self._n_nodes, [1])

    def _get_lower_energy_bound(self) -> float:
        r"""
        Returns a lower bound on the energy of the Max-Cut problem for the current graph.
        ... Math::
            E_{min} >= \sum_{(i,j) \in E} -w_{ij}
        """
        if self._nx_graph is None:
            raise ValueError("Graph has not been built yet.")
        weights = nx.get_edge_attributes(self._nx_graph, "weight")
        lower_bound = -sum(weights.values()) if weights else -self._nx_graph.number_of_edges()
        return lower_bound

    def _get_upper_n_cut_bound(self) -> float:
        r"""
        Returns an upper bound on the number of cuts of the Max-Cut problem for the current graph.
        ... Math::
            C_{max} <= \sum_{(i,j) \in E} w_{ij}
        """
        if self._nx_graph is None:
            raise ValueError("Graph has not been built yet.")
        weights = nx.get_edge_attributes(self._nx_graph, "weight")
        upper_bound = sum(weights.values()) if weights else self._nx_graph.number_of_edges()
        return upper_bound

    @property
    def train(self) -> bool:
        return self._train

    @train.setter
    def train(self, value: bool):
        self._train = value
