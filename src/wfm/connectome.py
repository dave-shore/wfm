import rustworkx as rx
import torch.nn as nn
import torch
import numpy as np
import jax.numpy as jnp
from numpy.random import randn
from scipy.special import chebyt
from concurrent.futures import ProcessPoolExecutor
import os
from time import sleep

class ConnectomeGraph(rx.PyDiGraph):

    def __init__(self, number_of_nodes: int, input_dim: int, output_dim: int, window_size: int = 1, num_workers: int | None = None, seed_size: int = 10, chebyshev_filter_degree: int = 3, edge_filter_threshold: float = 0.5, seed_noise_strength: float = 1.0, reduction: str = "mean", library: str = "torch", rnn_type: str = "GRU"):

        super().__init__(multiclass=False)

        self.extend_from_weighted_edge_list(
            [(i,j,{"weight": randn()}) for i in range(number_of_nodes) for j in range(number_of_nodes) if j != i]
        )
        self.full_edge_list = self.edges()

        self.node_bams = nn.Parameter(
            torch.randn(1, number_of_nodes, input_dim, output_dim)
        )

        self.seed_logits = nn.Parameter(
            torch.randn(number_of_nodes)
        )

        self.rescaling_lambda = nn.Parameter(
            torch.ones(1)
        )

        self.rnn = getattr(nn, rnn_type)(output_dim, output_dim, batch_first = True)

        self.chebyshev_filter_degree = chebyshev_filter_degree
        self.window_size = window_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() * 2 // 5

        self.seed_size = seed_size if isinstance(seed_size, int) else int(number_of_nodes * seed_size)
        assert self.seed_size <= number_of_nodes, f"Seed size must be less than or equal to the number of nodes, got {self.seed_size} and {number_of_nodes}."

        self.eps = torch.finfo(torch.get_default_dtype()).eps
        self.edge_filter_threshold = edge_filter_threshold
        self.reduction_fn = getattr(
            torch, 
            reduction.lower(),
            getattr(torch.nn.functional, reduction.lower(), None)
        )

        self.library = library
        self.I = torch.eye(number_of_nodes)
        self.b = torch.zeros(number_of_nodes)


    def _update_pruned_graph_copy(self):

        edge_indices = self.filter_edges(lambda e: 1 / (1+np.exp(-e['weight'])) > self.edge_filter_threshold)

        self.pruned_graph_copy = self.edge_subgraph([e for i,e in self.full_edge_list if i in edge_indices])
        self.pruned_graph_copy.add_nodes_from([n for n in self.nodes() if n not in self.pruned_graph_copy.nodes()])


    def _induce_seed_noise(self):

        self.seed_logits = self.seed_logits + torch.randn_like(self.seed_logits) * self.seed_noise_strength

    
    def _window_processing(self, V: torch.Tensor):

        batch_size, window_size, input_dim = V.shape
        V = V.unsqueeze(1)

        seed_nodes = torch.nn.functional.softmax(self.seed_logits).topk(k = self.seed_size, dim = 0).indices.tolist()

        seed_values = V.expand(-1, self.seed_size, -1, -1) @ self.node_bams[:,seed_nodes].expand(batch_size, -1, -1, -1)
        # seed_values.shape = (batch_size, seed_size, seq_len, output_dim)

        expanded_seed = self.pruned_graph_copy.subgraph(seed_nodes).nodes()

        for k in range(1, self.chebyshev_filter_degree + 1):

            new_nodes = [n for n in expanded_seed if n not in seed_nodes]

            if len(new_nodes) == 0:
                break

            adjacency = rx.adjacency_matrix(self.pruned_graph_copy, weight_label = "weight", node_list = expanded_seed)
            adjacency = torch.from_numpy(adjacency)

            transition = adjacency / (adjacency.sum(axis = 1) + self.eps) + self.eps

            principal_eigenvector = torch.linalg.solve(transition - self.I, self.b)
            Phi = torch.diag(principal_eigenvector)**(0.5)

            L = self.I - 0.5 * (Phi @ transition @ Phi.reciprocal() + Phi.reciprocal() @ transition.T @ Phi)
            # L.shape = (expanded_seed_size, expanded_seed_size)

            L_rescaled = 2*L / self.rescaling_lambda - self.I
            # L_rescaled.shape = (expanded_seed_size, expanded_seed_size)

            chebyshev_filter = chebyt(k)(L_rescaled)
            chebyshev_filter = torch.from_numpy(chebyshev_filter)
            # chebyshev_filter.shape = (expanded_seed_size, expanded_seed_size)

            expanded_values = V.expand(-1, len(new_nodes), -1, -1) @ self.node_bams[:,new_nodes].expand(batch_size, -1, -1, -1)
            reorder = [seed_nodes.index(n) if n in seed_nodes else len(seed_nodes) + expanded_seed.index(n) for n in expanded_seed]
            expanded_values = torch.cat([seed_values, expanded_values], dim = 1)[:, reorder].contiguous()
            expanded_values = torch.einsum("ijkl,hj->ihkl", expanded_values, chebyshev_filter)
            # expanded_values.shape = (batch_size, len(expanded_seed), window_size, output_dim)

            seed_values = expanded_values.clone()
            seed_nodes = expanded_seed
            expanded_seed = self.pruned_graph_copy.subgraph(seed_nodes).nodes()

        output = self.reduction_fn(expanded_values, dim = 1)
        # output.shape = (batch_size, window_size, output_dim)
        return output


    def forward(self, V: torch.Tensor):

        batch_size, seq_len, input_dim = V.shape

        self._update_pruned_graph_copy()
        self._induce_seed_noise()

        split_V = V.split(self.window_size, dim = 1)
        results = []

        with ProcessPoolExecutor(max_workers = self.num_workers) as executor:
            futures = [executor.submit(self._window_processing, v) for v in split_V]
            for future in futures:
                while not future.done():
                    sleep(1e-3)
                results.append(future.result())

        output = torch.cat(results, dim = 1)

        return self.rnn(output)
        


