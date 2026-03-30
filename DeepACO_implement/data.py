from __future__ import annotations

from typing import Dict, Iterable, Tuple

import networkx as nx
import torch

from .model import GraphBatch


def build_training_graph(
    links: Iterable[Tuple[int, int, float, float, float]],
    coords: Dict[int, Tuple[float, float]],
    virtual_node: int = -1,
) -> tuple[nx.DiGraph, GraphBatch, dict[int, int]]:
    """
    Build graph and tensors for training.

    links format: (u, v, length, speed, capacity)
    """
    g = nx.DiGraph()
    for u, v, length, speed, capacity in links:
        g.add_edge(u, v, length=float(length), speed=float(speed), capacity=float(capacity))

    if virtual_node in g.nodes:
        raise ValueError("virtual_node id already exists in graph")

    g.add_node(virtual_node, pos=(0.0, 0.0), is_virtual=True)
    for n in list(g.nodes):
        if n == virtual_node:
            continue
        g.add_edge(virtual_node, n, length=0.0, speed=1.0, capacity=1e9)

    node_list = sorted(g.nodes)
    node2idx = {n: i for i, n in enumerate(node_list)}

    node_features = []
    for n in node_list:
        x, y = coords.get(n, (0.0, 0.0))
        is_virtual = 1.0 if n == virtual_node else 0.0
        node_features.append([float(x), float(y), is_virtual])

    edge_src, edge_dst, edge_attr = [], [], []
    for u, v, d in g.edges(data=True):
        edge_src.append(node2idx[u])
        edge_dst.append(node2idx[v])
        edge_attr.append([d["length"], d["speed"], d["capacity"]])

    graph_batch = GraphBatch(
        node_features=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    return g, graph_batch, node2idx
