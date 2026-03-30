from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GraphBatch:
    node_features: torch.Tensor  # [N, node_dim]
    edge_index: torch.Tensor  # [2, E]
    edge_attr: torch.Tensor  # [E, 3] => [length, speed, capacity]


class HeuristicGNN(nn.Module):
    """Simple message-passing network that outputs an edge heuristic matrix."""

    def __init__(self, node_dim: int = 3, edge_dim: int = 3, hidden_dim: int = 64, depth: int = 3):
        super().__init__()
        self.depth = depth
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = self.node_encoder(batch.node_features)
        e = self.edge_encoder(batch.edge_attr)
        src, dst = batch.edge_index[0], batch.edge_index[1]

        for _ in range(self.depth):
            msg = x[src] + e
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, msg)
            x = self.update(agg, x)

        logits = self.edge_decoder(torch.cat([x[src], x[dst], e], dim=-1)).squeeze(-1)
        return logits

    @staticmethod
    def edge_logits_to_matrix(num_nodes: int, edge_index: torch.Tensor, edge_logits: torch.Tensor) -> torch.Tensor:
        heu = torch.zeros((num_nodes, num_nodes), device=edge_logits.device)
        src, dst = edge_index[0], edge_index[1]
        heu[src, dst] = edge_logits
        return heu
