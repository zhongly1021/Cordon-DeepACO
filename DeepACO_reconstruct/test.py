from __future__ import annotations

from typing import Any, Dict

import torch

from build_graph_data import build_cordon_graph_data
from model import PolicyAwareHeuristicNet
from reward_function_new import MSARewardFunction
from cordon_environment import CordonEnv
from deepaco import DeepACOAgent


def load_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_dim = int(ckpt.get("hidden_dim", 64))

    model = PolicyAwareHeuristicNet(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def evaluate_real_network(
    links,
    od_demand,
    policy_type: str,
    policy_value: dict[str, float],
    checkpoint_path: str,
    virtual_node: int = -1,
    device: str = "cpu",
    n_ants: int = 32,
    n_rounds: int = 5,
    reward_weights: Dict[str, float] | None = None,
    solver_kwargs: Dict[str, Any] | None = None,
):
    #这里需要导入真实的数据
    pack = build_cordon_graph_data(
        links=links,
        od_demand=od_demand,
        policy_type=policy_type,
        virtual_node=virtual_node,
    )

    model, ckpt = load_model(checkpoint_path, device=device)

    data = pack["model_data"].to(device)
    with torch.no_grad():
        heu_mat = model(data)

    reward_fn = MSARewardFunction(
        links=pack["links"],
        od_demand=pack["od_demand"],
        virtual_node=virtual_node,
        reward_weights=reward_weights,
        solver_kwargs=solver_kwargs,
    )
    reward_fn.compute_initial(
        zone_nodes_or_state=None,
        policy_kind="none",
        policy_value=None,
    )

    env_factory = lambda: CordonEnv(
        road_graph=pack["road_graph"],
        max_steps=pack["road_graph"].number_of_nodes() + 2,
        virtual_node=virtual_node,
    )

    agent = DeepACOAgent(
        node2idx=data.node2idx,
        n_ants=n_ants,
        alpha=1.0,
        beta=1.0,
        decay=0.9,
        max_steps=pack["road_graph"].number_of_nodes() + 2,
        virtual_node=virtual_node,
        elitist=False,
        min_max=False,
        device=device,
    )

    out = agent.run(
        env_factory=env_factory,
        heu_mat=heu_mat,
        reward_fn=reward_fn,
        n_rounds=n_rounds,
        inference=False,
        update_pheromone=True,
        terminal_reward_only=True,
        policy_kind=policy_type,
        policy_value=policy_value,
    )

    return {
        "best_reward": out["best_reward"],
        "best_path": out["best_path"],
        "best_round": out["best_round"],
        "history": out["history"],
        "checkpoint_meta": {
            "best_val_reward": ckpt.get("best_val_reward", None),
            "hidden_dim": ckpt.get("hidden_dim", None),
        },
    }


if __name__ == "__main__":
    # 这里只是示意：你后面把真实的 links / od_demand 填进来即可
    links = [
        (0, 1, 1.0, 0.03, 200.0),
        (1, 0, 1.0, 0.03, 200.0),
        (1, 2, 1.2, 0.04, 180.0),
        (2, 1, 1.2, 0.04, 180.0),
        (0, 2, 1.8, 0.05, 160.0),
        (2, 0, 1.8, 0.05, 160.0),
    ]
    od_demand = {
        (0, 2): 120.0,
        (2, 0): 90.0,
    }

    result = evaluate_real_network(
        links=links,
        od_demand=od_demand,
        policy_type="toll",
        policy_value={"inside": 5.0, "outside": 1.0},
        checkpoint_path="reinforce_best.pt",
        virtual_node=-1,
        device="cpu",
        n_ants=16,
        n_rounds=5,
    )

    print(result)