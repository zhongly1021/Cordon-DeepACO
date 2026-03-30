from __future__ import annotations

import torch

from .deepaco import DeepACOAgent
from .model import HeuristicGNN


def reinforce_train(
    model: HeuristicGNN,
    graph_batch,
    env_factory,
    reward_fn,
    episodes: int = 50,
    max_steps: int = 20,
    lr: float = 1e-3,
    gamma: float = 0.99,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    agent = DeepACOAgent(model)
    history = []

    for _ in range(episodes):
        edge_logits = model(graph_batch)
        heu_mat = model.edge_logits_to_matrix(graph_batch.node_features.shape[0], graph_batch.edge_index, edge_logits)

        env = env_factory()
        rollout = agent.rollout(env, heu_mat, reward_fn, max_steps=max_steps)
        returns = agent.discounted_returns(rollout.rewards, gamma=gamma)

        if len(rollout.log_probs) == 0:
            continue

        log_probs = torch.stack(rollout.log_probs)
        loss = -(log_probs * returns.to(log_probs.device)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append({"loss": float(loss.item()), "reward": float(sum(rollout.rewards))})

    return history


def grpo_train(
    model: HeuristicGNN,
    graph_batch,
    env_factory,
    reward_fn,
    episodes: int = 50,
    group_size: int = 4,
    max_steps: int = 20,
    lr: float = 1e-3,
):
    """Group Relative Policy Optimization style training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    agent = DeepACOAgent(model)
    history = []

    for _ in range(episodes):
        edge_logits = model(graph_batch)
        heu_mat = model.edge_logits_to_matrix(graph_batch.node_features.shape[0], graph_batch.edge_index, edge_logits)

        group_rollouts = []
        for _ in range(group_size):
            env = env_factory()
            group_rollouts.append(agent.rollout(env, heu_mat, reward_fn, max_steps=max_steps))

        group_rewards = torch.tensor([sum(r.rewards) for r in group_rollouts], dtype=torch.float32)
        advantage = group_rewards - group_rewards.mean()

        loss = torch.tensor(0.0)
        for k, r in enumerate(group_rollouts):
            if not r.log_probs:
                continue
            log_prob_sum = torch.stack(r.log_probs).sum()
            loss = loss - log_prob_sum * advantage[k]

        loss = loss / max(group_size, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append({"loss": float(loss.item()), "group_mean_reward": float(group_rewards.mean().item())})

    return history
