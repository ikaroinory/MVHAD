from collections import defaultdict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from data_types import EdgeType


class GraphLayer(MessagePassing):
    def __init__(
        self,
        d_output: int,
        num_heads: int,
        *,
        node_indices: dict[str, list[int]],
        edge_types: list[EdgeType]
    ):
        super().__init__(aggr='add', node_dim=0)

        self.node_types = list(node_indices.keys())
        self.d_output = d_output
        self.num_heads = num_heads

        self.w_pi = nn.ParameterDict({'->'.join(edge_type): nn.Parameter(torch.zeros([1, num_heads, (d_output // num_heads) * 4])) for edge_type in edge_types})
        self.process_layer_dict = nn.ModuleDict(
            {
                '->'.join(edge_type): nn.Sequential(
                    nn.BatchNorm1d(d_output),
                    nn.ReLU(),
                )
                for edge_type in edge_types
            }
        )
        self.semantic_attention_layer = nn.Sequential(
            nn.Linear(d_output, d_output),
            nn.Tanh(),
            nn.Linear(d_output, 1, bias=False),
            nn.LeakyReLU(),
            nn.Softmax(dim=0)
        )

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x_dict: dict[str, Tensor], v_dict: dict[str, Tensor], edge_index_dict: dict[EdgeType, Tensor]) -> dict[str, Tensor]:
        z_list_dict: dict[str, list[Tensor]] = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type

            # print(g_dict[src_type][edge_index[0]])  # xj
            # print(g_dict[dst_type][edge_index[1]])  # xi
            z: Tensor = self.propagate(
                edge_index,
                x_proj=(x_dict[src_type], x_dict[dst_type]),
                v_proj=(v_dict[src_type], v_dict[dst_type]),
                edge_type=edge_type
            )
            z = self.process_layer_dict['->'.join(edge_type)](z)  # [num_nodes, d_output * 2]

            z_list_dict[dst_type].append(z)

        z_dict = {}
        for node_type, z_list in z_list_dict.items():
            z_all = torch.stack(tuple(z_list), dim=0)

            beta = self.semantic_attention_layer(z_all)

            output = torch.sum(beta.expand(-1, -1, self.d_output) * z_all, dim=0)

            z_dict[node_type] = F.relu(output)

        return z_dict

    def message(self, x_proj_j: Tensor, x_proj_i: Tensor, v_proj_j: Tensor, v_proj_i: Tensor, edge_index_i: Tensor, edge_type: EdgeType) -> Tensor:
        x_i_heads = x_proj_i.reshape(-1, self.num_heads, self.d_output // self.num_heads)
        x_j_heads = x_proj_j.reshape(-1, self.num_heads, self.d_output // self.num_heads)
        v_i_heads = v_proj_i.reshape(-1, self.num_heads, self.d_output // self.num_heads)
        v_j_heads = v_proj_j.reshape(-1, self.num_heads, self.d_output // self.num_heads)

        g_i = torch.cat([v_i_heads, x_i_heads], dim=-1)  # [num_nodes, num_heads, (d_output // num_heads) * 2]
        g_j = torch.cat([v_j_heads, x_j_heads], dim=-1)
        g = torch.cat([g_i, g_j], dim=-1)  # [num_nodes, num_heads, (d_output // num_heads) * 4]

        edge_type_str = '->'.join(edge_type)

        pi = torch.einsum('nhd,nhd->nh', g, self.w_pi[edge_type_str])
        pi = self.leaky_relu(pi)
        alpha = softmax(pi, index=edge_index_i)

        return (alpha.view(-1, self.num_heads, 1) * x_j_heads).reshape(-1, self.d_output)

    def __call__(self, x_dict: dict[str, Tensor], v_dict: dict[str, Tensor], edge_index_dict: dict[tuple[str, str, str], Tensor]) -> dict[str, Tensor]:
        return super().__call__(x_dict, v_dict, edge_index_dict)
