from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from build_graph_data import CordonGraphData


class GraphEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        x = self.norm1(x + h)
        h = self.ffn(x)
        x = self.norm2(x + h)
        return x


class GlobalCrossInject(nn.Module):
    """
    Inject one global token into graph node embeddings using cross attention.
    """

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, node_x: torch.Tensor, global_token: torch.Tensor) -> torch.Tensor:
        # node_x: [N, H] -> [1, N, H]
        # global_token: [H] or [1, H] -> [1, 1, H]
        q = node_x.unsqueeze(0)
        if global_token.dim() == 1:
            kv = global_token.view(1, 1, -1)
        else:
            kv = global_token.view(1, 1, -1)

        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = self.norm(q + attn_out)
        h = self.ffn(x)
        x = self.norm2(x + h)
        return x.squeeze(0)


class PolicyAwareHeuristicNet(nn.Module):
    """
    Input
    -----
    1) Road graph:
       - node feature: only virtual-node flag
       - edge feature: [length, free_time, capacity]
    2) OD graph:
       - node feature: only virtual-node flag
       - edge feature: [demand]
    3) Global feature:
       - policy type embedding (toll / speed_limit)

    Pipeline
    --------
    policy embedding -> cross-attn inject into road graph and OD graph separately
                   -> fuse two graph node embeddings
                   -> dense pair scorer
                   -> heuristic matrix [N, N]
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        road_layers: int = 2,
        od_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        pair_hidden_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        # nodes have almost no information; only virtual-flag is kept
        self.road_node_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.od_node_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.road_edge_norm = nn.LayerNorm(3)
        self.od_edge_norm = nn.LayerNorm(1)

        self.road_blocks = nn.ModuleList(
            [GraphEncoderBlock(hidden_dim, edge_dim=3, heads=heads, dropout=dropout) for _ in range(road_layers)]
        )
        self.od_blocks = nn.ModuleList(
            [GraphEncoderBlock(hidden_dim, edge_dim=1, heads=heads, dropout=dropout) for _ in range(od_layers)]
        )

        self.policy_emb = nn.Embedding(2, hidden_dim)
        self.road_global_inject = GlobalCrossInject(hidden_dim, heads=heads, dropout=dropout)
        self.od_global_inject = GlobalCrossInject(hidden_dim, heads=heads, dropout=dropout)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dst_proj = nn.Linear(hidden_dim, hidden_dim)

        self.pair_mlp = nn.Sequential(
            nn.Linear(4, pair_hidden_dim),
            nn.GELU(),
            nn.Linear(pair_hidden_dim, pair_hidden_dim),
            nn.GELU(),
            nn.Linear(pair_hidden_dim, 1),
        )

        self.out_act = nn.Softplus()

    def encode_road(self, data: CordonGraphData) -> torch.Tensor:
        x = self.road_node_proj(data.road_x)
        edge_attr = self.road_edge_norm(data.road_edge_attr)
        for blk in self.road_blocks:
            x = blk(x, data.road_edge_index, edge_attr)
        return x

    def encode_od(self, data: CordonGraphData) -> torch.Tensor:
        x = self.od_node_proj(data.od_x)
        edge_attr = self.od_edge_norm(data.od_edge_attr)
        for blk in self.od_blocks:
            x = blk(x, data.od_edge_index, edge_attr)
        return x

    def forward(self, data: CordonGraphData) -> torch.Tensor:
        """
        Returns
        -------
        heuristic_matrix: [N, N], strictly positive
        """
        road_h = self.encode_road(data)
        od_h = self.encode_od(data)

        policy_token = self.policy_emb(data.policy_type_id.view(-1))[0]

        road_h = self.road_global_inject(road_h, policy_token)
        od_h = self.od_global_inject(od_h, policy_token)

        h = self.fuse(torch.cat([road_h, od_h], dim=-1))

        src = self.src_proj(h)                      # [N, H]
        dst = self.dst_proj(h)                      # [N, H]
        bilinear = (src @ dst.transpose(0, 1)) / math.sqrt(self.hidden_dim)  # [N, N]

        pair_bias = self.pair_mlp(data.pair_feat).squeeze(-1)                # [N, N]

        logits = bilinear + pair_bias
        heur = self.out_act(logits) + 1e-8

        # discourage self-loop selection
        eye = torch.eye(heur.size(0), dtype=heur.dtype, device=heur.device)
        heur = heur * (1.0 - eye) + eye * 1e-8
        return heur
