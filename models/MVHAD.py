import torch
from torch import Tensor, nn

from data_types import EdgeType, NodeInformation
from .GraphLayer import GraphLayer
from .OutputLayer import OutputLayer


class MVHAD(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        d_hidden: int,
        d_output_hidden: int,
        num_heads: int,
        num_output_layer: int,
        k_dict: dict[EdgeType, int],
        *,
        node_config: dict[str, NodeInformation],
        edge_types: list[EdgeType],
        dtype=None,
        device=None
    ):
        super().__init__()

        node_indices = {k: v['index'] for k, v in node_config.items()}

        self.d_hidden = d_hidden
        self.num_nodes = sum(len(v) for v in node_indices.values())
        self.num_nodes_dict = {node_type: len(indices) for node_type, indices in node_indices.items()}
        self.node_indices = {node_type: torch.tensor(indices, device=device, dtype=torch.int32) for node_type, indices in node_indices.items()}
        self.node_types = list(node_indices.keys())
        self.edge_types = edge_types
        self.k_dict = k_dict

        self.embedding_layer = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=d_hidden),
            nn.BatchNorm1d(d_hidden)
        )
        self.x_proj_layer_dict = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(sequence_len, d_hidden),
                    nn.BatchNorm1d(d_hidden)
                )
                for node_type in self.node_types
            }
        )
        self.v_proj_layer_dict = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(d_hidden, d_hidden),
                    nn.BatchNorm1d(d_hidden)
                )
                for node_type in self.node_types
            }
        )
        self.graph_layer = GraphLayer(d_hidden, num_heads, node_indices=node_indices, edge_types=edge_types)
        self.graph_output_process_layer = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            nn.ReLU()
        )
        self.process_layer = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            nn.ReLU()
        )
        self.sensor_output_layer = OutputLayer(d_input=d_hidden, d_hidden=d_output_hidden, d_output=1, num_layers=num_output_layer)
        self.actuator_output_layer = OutputLayer(d_input=d_hidden, d_hidden=d_output_hidden, d_output=3, num_layers=num_output_layer)

        self.dtype = dtype
        self.device = device

        self.to(dtype)
        self.to(device)

    @staticmethod
    def __flatten(x: Tensor, v: Tensor, node_indices: dict[str, Tensor]) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        batch_size, num_nodes, sequence_len = x.shape
        _, d_hidden = v.shape

        x_flatten = x.reshape(-1, sequence_len)
        v_flatten = v.repeat(batch_size, 1, 1).reshape(-1, d_hidden)

        steps = torch.arange(batch_size).to(x.device) * num_nodes

        node_indices_flatten = {}
        for node_type, indices in node_indices.items():
            node_indices_flatten[node_type] = indices.repeat(batch_size) + steps.repeat_interleave(indices.shape[0])

        return x_flatten, v_flatten, node_indices_flatten

    @staticmethod
    def __cos_similarity(x: Tensor, y: Tensor) -> Tensor:
        x_norm = x.norm(dim=-1).unsqueeze(-1) + 1e-8
        y_norm = y.norm(dim=-1).unsqueeze(-1) + 1e-8

        return (x @ y.T) / (x_norm @ y_norm.T)

    def __get_edges(self, x: Tensor, y: Tensor, k: int) -> Tensor:
        similarity = self.__cos_similarity(x, y)

        _, indices = torch.topk(similarity, k, dim=-1)

        source_nodes = torch.arange(x.shape[0], device=self.device).repeat_interleave(k)
        target_nodes = indices.reshape(-1)

        edges = torch.stack([source_nodes, target_nodes], dim=0)

        return edges

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, num_nodes, _ = x.shape

        v = self.embedding_layer(torch.arange(self.num_nodes).to(self.device))  # [num_nodes, d_hidden]

        x_flatten, v_flatten, node_indices_flatten = self.__flatten(x, v, self.node_indices)
        x_proj_dict = {node_type: self.x_proj_layer_dict[node_type](x_flatten[node_indices_flatten[node_type]]) for node_type in self.node_types}
        v_proj_dict = {node_type: self.v_proj_layer_dict[node_type](v_flatten[node_indices_flatten[node_type]]) for node_type in self.node_types}

        edge_index_dict = {}
        step_basic_dict = {node_type: torch.arange(batch_size, device=self.device) * self.num_nodes_dict[node_type] for node_type in self.node_types}
        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type

            k = self.k_dict[edge_type]

            edges = self.__get_edges(v[self.node_indices[src_type]], v[self.node_indices[dst_type]], k)

            edges = edges.repeat(1, batch_size)

            edges[0] += step_basic_dict[src_type].repeat_interleave(edges.shape[1] // batch_size)
            edges[1] += step_basic_dict[dst_type].repeat_interleave(edges.shape[1] // batch_size)

            edge_index_dict[edge_type] = edges

        z_dict = self.graph_layer(x_proj_dict, v_proj_dict, edge_index_dict)

        z_flatten = torch.zeros([batch_size * self.num_nodes, self.d_hidden], dtype=self.dtype, device=self.device)
        for node_type, indices in node_indices_flatten.items():
            z_flatten[indices] = z_dict[node_type]
        z_flatten = self.graph_output_process_layer(z_flatten)

        p_flatten = torch.zeros([batch_size * self.num_nodes, self.d_hidden], dtype=self.dtype, device=self.device)
        for node_type, indices in node_indices_flatten.items():
            p_flatten[indices] = z_flatten[indices] * v_proj_dict[node_type]
        p_flatten = self.process_layer(p_flatten)

        p = p_flatten.reshape(batch_size, num_nodes, -1)
        sensor_output = self.sensor_output_layer(p[:, self.node_indices['sensor'], :])
        actuator_output = self.actuator_output_layer(p[:, self.node_indices['actuator'], :])

        return sensor_output, actuator_output

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(x)
