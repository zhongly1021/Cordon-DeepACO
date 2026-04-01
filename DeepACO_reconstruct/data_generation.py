from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
import random

import numpy as np
import networkx as nx

from build_graph_data import build_cordon_graph_data


def build_grid_graph_with_pos(
    n_rows: int = 4,
    n_cols: int = 4,
    seed: int | None = None,
    length_range: tuple[float, float] = (0.8, 2.0),
    speed_range: tuple[float, float] = (25.0, 60.0),
    capacity_range: tuple[float, float] = (100.0, 500.0),
):
    rng = np.random.default_rng(seed)

    G = nx.grid_2d_graph(n_rows, n_cols)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    pos_map = {}
    coords = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    for idx, (i, j) in enumerate(coords):
        pos_map[idx] = (j, -i)
        G.nodes[idx]["pos"] = (j, -i)

    for u, v in G.edges():
        length = float(rng.uniform(*length_range))
        speed = float(rng.uniform(*speed_range))
        free_time = float(length / max(speed, 1e-8))
        capacity = float(rng.uniform(*capacity_range))

        G.edges[u, v]["length"] = length
        G.edges[u, v]["free_time"] = free_time
        G.edges[u, v]["capacity"] = capacity

    return G, pos_map


def graph_to_directed_links(G: nx.Graph) -> list[tuple[int, int, float, float, float]]:
    links = []
    for u, v, data in G.edges(data=True):
        feat = (
            int(u),
            int(v),
            float(data["length"]),
            float(data["free_time"]),
            float(data["capacity"]),
        )
        links.append(feat)
        links.append((feat[1], feat[0], feat[2], feat[3], feat[4]))
    return links


def generate_random_od_demand(
    nodes: Sequence[int],
    n_pairs: int | None = None,
    demand_range: tuple[float, float] = (30.0, 200.0),
    seed: int | None = None,
) -> dict[tuple[int, int], float]:
    """
    Random OD matrix in dict form:
        {(o, d): demand}
    """
    rng = np.random.default_rng(seed)
    nodes = list(nodes)

    all_pairs = [(o, d) for o in nodes for d in nodes if o != d]
    if n_pairs is None:
        n_pairs = max(1, len(all_pairs) // 4)

    n_pairs = min(int(n_pairs), len(all_pairs))
    chosen_idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)

    od = {}
    for idx in chosen_idx:
        o, d = all_pairs[int(idx)]
        od[(int(o), int(d))] = float(rng.uniform(*demand_range))
    return od


def default_policy_value(policy_type: str, seed: int | None = None) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    if policy_type == "toll":
        return {
            "inside": float(rng.uniform(3.0, 8.0)),
            "outside": float(rng.uniform(0.0, 3.0)),
        }

    if policy_type == "speed_limit":
        return {
            "inside": float(rng.uniform(20.0, 40.0)),
            "outside": float(rng.uniform(40.0, 70.0)),
        }

    raise ValueError(f"Unknown policy_type={policy_type}")


def generate_single_grid_instance(
    n_rows: int = 4,
    n_cols: int = 4,
    policy_type: str | None = None,
    virtual_node: int = -1,
    n_od_pairs: int | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    One synthetic instance.

    Outputs:
        - links / od_demand / road_graph for reward + env
        - pyg_data for model forward
    """
    rng = np.random.default_rng(seed)

    if policy_type is None:
        policy_type = random.choice(["toll", "speed_limit"])
        # 分别训练speed limit/toll

    G, pos_map = build_grid_graph_with_pos(
        n_rows=n_rows,
        n_cols=n_cols,
        seed=seed,
    )
    links = graph_to_directed_links(G)
    od_demand = generate_random_od_demand(
        nodes=list(G.nodes),
        n_pairs=n_od_pairs,
        seed=None if seed is None else seed + 7,
    )

    pack = build_cordon_graph_data(
        links=links,
        od_demand=od_demand,
        policy_type=policy_type,
        virtual_node=virtual_node,
    )

    instance = {
        "grid_shape": (int(n_rows), int(n_cols)),
        "pos_map": pos_map,
        "policy_type": policy_type,
        "policy_value": default_policy_value(policy_type, seed=None if seed is None else seed + 13),
        "virtual_node": int(virtual_node),
        "links": pack["links"],
        "od_demand": pack["od_demand"],
        "road_graph": pack["road_graph"],
        "pyg_data": pack["model_data"],   # alias for training
        "model_data": pack["model_data"],
        "node2idx": pack["node2idx"],
        "idx2node": pack["idx2node"],
    }
    return instance


def generate_dataset(
    num_instances: int = 50,
    grid_shapes: Sequence[tuple[int, int]] = ((3, 3), (3, 4), (4, 4)),
    seed: int = 42,
    virtual_node: int = -1,
) -> list[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    dataset = []

    for k in range(num_instances):
        n_rows, n_cols = grid_shapes[int(rng.integers(0, len(grid_shapes)))]
        policy_type = random.choice(["toll", "speed_limit"])
        # 不能随便generate
        inst = generate_single_grid_instance(
            n_rows=n_rows,
            n_cols=n_cols,
            policy_type=policy_type,
            virtual_node=virtual_node,
            seed=seed + 1000 + k,
        )
        dataset.append(inst)

    return dataset


def split_dataset(
    dataset: Sequence[Dict[str, Any]],
    val_ratio: float = 0.2,
    seed: int = 42,
):
    dataset = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(dataset)

    n_val = int(len(dataset) * val_ratio)
    val_set = dataset[:n_val]
    train_set = dataset[n_val:]
    return train_set, val_set