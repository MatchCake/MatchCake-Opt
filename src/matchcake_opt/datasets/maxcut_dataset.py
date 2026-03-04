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
        "weighted_erdos_renyi": ["p"],
        "circular": [],
    }

    def __init__(
        self,
        n_nodes: int,
        graph_type: Literal["regular", "erdos_renyi", "circular", "weighted_erdos_renyi"],
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
        elif self._graph_type == "weighted_erdos_renyi":
            self._nx_graph = self._build_weighted_erdos_renyi_graph()
        elif self._graph_type == "circular":
            self._nx_graph = self._build_circular_graph()
        else:
            raise ValueError(f"Unsupported graph type: {self._graph_type}")  # pragma: no cover
        self._data = from_networkx(self._nx_graph)
        self._data.y = torch.tensor([self._get_lower_energy_bound(), self._get_upper_n_cut_bound()], dtype=torch.float)
        self._built_flag = True
        return

    def __getitem__(self, item):
        if not self._built_flag:
            self.prepare_data()  # pragma: no cover
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

    def _build_weighted_erdos_renyi_graph(self) -> nx.Graph:
        """
        Builds an Erdos-Renyi graph where each existing edge (i, j) is assigned a weight.

        Parameters (via **kwargs):
            p: float
            weight_distribution: {"uniform", "normal", "exponential"} (default: "uniform")

            If weight_distribution == "uniform":
                weight_low: float (default: 0.0)
                weight_high: float (default: 1.0)

            If weight_distribution == "normal":
                weight_mean: float (default: 0.0)
                weight_std: float (default: 1.0)

            If weight_distribution == "exponential":
                weight_rate: float (default: 1.0)   # rate = 1/scale

        Notes:
            - Weights are stored as the edge attribute "weight".
            - Sampling is reproducible w.r.t. self._seed.
        """
        g = nx.erdos_renyi_graph(self._n_nodes, self._kwargs["p"], seed=self._seed)

        gen = torch.Generator()
        gen.manual_seed(self._seed)

        dist = str(self._kwargs.get("weight_distribution", "uniform")).lower()

        for edge in g.edges():
            if dist == "uniform":
                low = float(self._kwargs.get("weight_low", -1.0))
                high = float(self._kwargs.get("weight_high", 1.0))
                if high <= low:
                    raise ValueError("For uniform weights, require weight_high > weight_low.")
                w = low + (high - low) * torch.rand((), generator=gen)

            elif dist == "normal":
                mean = float(self._kwargs.get("weight_mean", 0.0))
                std = float(self._kwargs.get("weight_std", 1.0))
                if std <= 0.0:
                    raise ValueError("For normal weights, require weight_std > 0.")
                w = torch.normal(mean=mean, std=std, size=(), generator=gen)

            elif dist == "exponential":
                rate = float(self._kwargs.get("weight_rate", 1.0))
                if rate <= 0.0:
                    raise ValueError("For exponential weights, require weight_rate > 0.")
                u01 = torch.rand((), generator=gen)
                w = -torch.log1p(-u01) / rate  # inverse-CDF sampling

            else:
                raise ValueError(
                    f"Unsupported weight_distribution '{dist}'. " "Expected one of: 'uniform', 'normal', 'exponential'."
                )

            g.edges[edge]["weight"] = float(w.item())

        return g

    def _build_circular_graph(self) -> nx.Graph:
        return nx.circulant_graph(self._n_nodes, [1])

    def _get_lower_energy_bound(self) -> float:
        r"""
        Returns a lower bound on the energy of the Max-Cut problem for the current graph.
        ... Math::
            E_{min} >= \sum_{(i,j) \in E} -w_{ij}
        """
        if self._nx_graph is None:
            raise ValueError("Graph has not been built yet.")  # pragma: no cover
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
            raise ValueError("Graph has not been built yet.")  # pragma: no cover
        weights = nx.get_edge_attributes(self._nx_graph, "weight")
        upper_bound = sum(weights.values()) if weights else self._nx_graph.number_of_edges()
        return upper_bound

    @property
    def train(self) -> bool:
        return self._train

    @train.setter
    def train(self, value: bool):
        self._train = value
