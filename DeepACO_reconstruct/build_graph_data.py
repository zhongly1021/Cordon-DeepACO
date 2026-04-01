from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import networkx as nx
import torch


POLICY_TYPE_TO_ID = {
    "toll": 0,
    "speed_limit": 1,
}


@dataclass
class CordonGraphData:
    """
    Model input container.

    Attributes
    ----------
    road
        [N, 1] node feature for road graph. We only keep the virtual-node flag.
        [2, E_r] directed edge index used by the road-graph encoder.
        [E_r, 3] = [length, free_time, capacity].
    od_x:
        [N, 1] node feature for OD graph. We only keep the virtual-node flag.
        [2, E_od] directed OD edges.
        [E_od, 1] = [demand].
    pair_feat:
        [N, N, 4] dense pair feature tensor:
            [:,:,0] road length
            [:,:,1] road free time
            [:,:,2] road capacity
            [:,:,3] OD demand
    policy_type_id:
        scalar long tensor, 0=toll, 1=speed_limit.
    """

    road_x: torch.Tensor
    road_edge_index: torch.Tensor
    road_edge_attr: torch.Tensor
    od_x: torch.Tensor
    od_edge_index: torch.Tensor
    od_edge_attr: torch.Tensor
    pair_feat: torch.Tensor
    policy_type_id: torch.Tensor
    node_ids: list[int]
    node2idx: dict[int, int]
    idx2node: dict[int, int]
    virtual_node: int

    @property
    def num_nodes(self) -> int:
        return int(self.road_x.size(0))

    def to(self, device: torch.device | str) -> "CordonGraphData":
        return replace(
            self,
            road_x=self.road_x.to(device),
            road_edge_index=self.road_edge_index.to(device),
            road_edge_attr=self.road_edge_attr.to(device),
            od_x=self.od_x.to(device),
            od_edge_index=self.od_edge_index.to(device),
            od_edge_attr=self.od_edge_attr.to(device),
            pair_feat=self.pair_feat.to(device),
            policy_type_id=self.policy_type_id.to(device),
        )


def _sorted_unique_nodes(links: Sequence[Tuple], od_demand: Mapping[Tuple[int, int], float]) -> list[int]:
    nodes = set()
    for u, v, *_ in links:
        nodes.add(int(u))
        nodes.add(int(v))
    for o, d in od_demand.keys():
        nodes.add(int(o))
        nodes.add(int(d))
    return sorted(nodes)


def _build_env_graph(links: Sequence[Tuple]) -> nx.Graph:
    """
    Build the undirected road graph used by CordonEnv.
    The virtual node is NOT added here.
    """
    g = nx.Graph()
    for raw in links:
        if len(raw) != 5:
            raise ValueError("Each link must be (u, v, length, free_time, capacity).")
        u, v, length, free_time, capacity = raw
        g.add_edge(
            int(u),
            int(v),
            length=float(length),
            free_time=float(free_time),
            capacity=float(capacity),
        )
    return g


def _make_bidirectional_road_edges(
    env_graph: nx.Graph,
    node2idx: Dict[int, int],
    virtual_node: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    road_edges: list[tuple[int, int]] = []
    road_attr: list[list[float]] = []

    # real-real road edges from the undirected environment graph
    for u, v, data in env_graph.edges(data=True):
        feat = [
            float(data.get("length", 0.0)),
            float(data.get("free_time", 0.0)),
            float(data.get("capacity", 0.0)),
        ]
        road_edges.append((int(u), int(v)))
        road_attr.append(feat)
        road_edges.append((int(v), int(u)))
        road_attr.append(feat)

    # virtual edges both directions, all features set to -1 as requested
    for n in env_graph.nodes:
        n = int(n)
        road_edges.append((virtual_node, n))
        road_attr.append([-1.0, -1.0, -1.0])
        road_edges.append((n, virtual_node))
        road_attr.append([-1.0, -1.0, -1.0])

    edge_index = torch.tensor(
        [[node2idx[u], node2idx[v]] for u, v in road_edges],
        dtype=torch.long,
    ).t().contiguous()
    edge_attr = torch.tensor(road_attr, dtype=torch.float32)

    # dense pair feature for pair scorer
    n_nodes = len(node2idx)
    pair_feat = torch.zeros((n_nodes, n_nodes, 4), dtype=torch.float32)
    for (u, v), feat in zip(road_edges, road_attr):
        ui = node2idx[u]
        vi = node2idx[v]
        pair_feat[ui, vi, 0] = feat[0]
        pair_feat[ui, vi, 1] = feat[1]
        pair_feat[ui, vi, 2] = feat[2]

    return edge_index, edge_attr, pair_feat


def _make_od_edges(
    od_demand: Mapping[Tuple[int, int], float],
    real_nodes: Sequence[int],
    node2idx: Dict[int, int],
    virtual_node: int,
    pair_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    od_edges: list[tuple[int, int]] = []
    od_attr: list[list[float]] = []

    for (o, d), demand in od_demand.items():
        od_edges.append((int(o), int(d)))
        od_attr.append([float(demand)])

    # virtual OD edges both directions, demand = -1 as requested
    for n in real_nodes:
        n = int(n)
        od_edges.append((virtual_node, n))
        od_attr.append([-1.0])
        od_edges.append((n, virtual_node))
        od_attr.append([-1.0])

    edge_index = torch.tensor(
        [[node2idx[u], node2idx[v]] for u, v in od_edges],
        dtype=torch.long,
    ).t().contiguous()
    edge_attr = torch.tensor(od_attr, dtype=torch.float32)

    for (u, v), feat in zip(od_edges, od_attr):
        ui = node2idx[u]
        vi = node2idx[v]
        pair_feat[ui, vi, 3] = feat[0]

    return edge_index, edge_attr, pair_feat


def build_cordon_graph_data(
    links: Iterable[Tuple],
    od_demand: Dict[Tuple[int, int], float],
    policy_type: str,
    virtual_node: int = -1,
) -> Dict[str, Any]:
    """
    Build both:
    1) actual solver inputs for the MSA reward function: links + od_demand + env graph
    2) virtual-augmented model inputs for the heuristic network

    Parameters
    ----------
    links:
        iterable of (u, v, length, free_time, capacity)
    od_demand:
        dict {(o, d): demand}
    policy_type:
        "toll" or "speed_limit"
    virtual_node:
        id of virtual node, default -1

    Returns
    -------
    A dict with keys:
        - links
        - od_demand
        - road_graph
        - model_data
        - node2idx
        - idx2node
    """
    links = [tuple(x) for x in links] # 这里是一个?
    od_demand = {tuple(k): float(v) for k, v in od_demand.items()} 

    if policy_type not in POLICY_TYPE_TO_ID:
        raise ValueError(f"Unknown policy_type={policy_type}. Expected one of {list(POLICY_TYPE_TO_ID.keys())}.")

    real_nodes = _sorted_unique_nodes(links, od_demand)
    if virtual_node in real_nodes:
        raise ValueError("virtual_node id collides with a real network node id.")

    env_graph = _build_env_graph(links)

    node_ids = [int(virtual_node)] + [int(n) for n in real_nodes]
    node2idx = {n: i for i, n in enumerate(node_ids)}
    idx2node = {i: n for n, i in node2idx.items()}

    n_nodes = len(node_ids)
    is_virtual = torch.zeros((n_nodes, 1), dtype=torch.float32)
    is_virtual[node2idx[virtual_node], 0] = 1.0

    road_edge_index, road_edge_attr, pair_feat = _make_bidirectional_road_edges(
        env_graph=env_graph,
        node2idx=node2idx,
        virtual_node=virtual_node,
    )

    od_edge_index, od_edge_attr, pair_feat = _make_od_edges(
        od_demand=od_demand,
        real_nodes=real_nodes,
        node2idx=node2idx,
        virtual_node=virtual_node,
        pair_feat=pair_feat,
    )

    model_data = CordonGraphData(
        road_x=is_virtual.clone(),
        road_edge_index=road_edge_index,
        road_edge_attr=road_edge_attr,
        od_x=is_virtual.clone(),
        od_edge_index=od_edge_index,
        od_edge_attr=od_edge_attr,
        pair_feat=pair_feat,
        policy_type_id=torch.tensor(POLICY_TYPE_TO_ID[policy_type], dtype=torch.long),
        node_ids=node_ids,
        node2idx=node2idx,
        idx2node=idx2node,
        virtual_node=int(virtual_node),
    )

    return {
        "links": links,
        "od_demand": od_demand,
        "road_graph": env_graph,
        "model_data": model_data,
        "node2idx": node2idx,
        "idx2node": idx2node,
    }
