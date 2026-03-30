from __future__ import annotations

from typing import Callable

import torch

from .deepaco import DeepACOAgent
from .model import HeuristicGNN


def _build_heuristic(model: HeuristicGNN, graph_batch):
    edge_logits = model(graph_batch)
    return model.edge_logits_to_matrix(graph_batch.node_features.shape[0], graph_batch.edge_index, edge_logits)


def _normalize_group(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    return (values - values.mean()) / (values.std(unbiased=False) + eps)


def reinforce_train(
    model: HeuristicGNN,
    graph_batch,
    env_factory: Callable,
    reward_fn: Callable,
    node2idx: dict[int, int],
    episodes: int = 50,
    n_ants: int = 20,
    max_steps: int = 20,
    lr: float = 1e-3,
    entropy_coef: float = 1e-3,
):
    """
    REINFORCE style aligned with notebook:
    - no discounted return
    - per-ant episode reward as objective signal
    - normalized advantage within each ant group
    - entropy regularization
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    agent = DeepACOAgent(model, node2idx=node2idx)
    history = []

    for _ in range(episodes):
        heu_mat = _build_heuristic(model, graph_batch)
        rollouts = [agent.rollout(env_factory(), heu_mat, reward_fn, max_steps=max_steps) for _ in range(n_ants)]

        rewards = torch.tensor([sum(r.rewards) for r in rollouts], dtype=torch.float32, device=heu_mat.device)
        adv = _normalize_group(rewards)

        ant_log_probs = []
        ant_entropies = []
        for r in rollouts:
            if r.log_probs:
                ant_log_probs.append(torch.stack(r.log_probs).mean())
                ant_entropies.append(torch.stack(r.entropies).mean())
            else:
                ant_log_probs.append(torch.zeros((), device=heu_mat.device))
                ant_entropies.append(torch.zeros((), device=heu_mat.device))

        ant_log_probs = torch.stack(ant_log_probs)
        entropy_mean = torch.stack(ant_entropies).mean()

        # maximize E[A * log pi], optimized via gradient descent on negative objective
        policy_loss = -(adv.detach() * ant_log_probs).mean()
        loss = policy_loss - entropy_coef * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        history.append(
            {
                "loss": float(loss.item()),
                "mean_reward": float(rewards.mean().item()),
                "entropy": float(entropy_mean.item()),
            }
        )

    return history


def grpo_train(
    model: HeuristicGNN,
    graph_batch,
    env_factory: Callable,
    reward_fn: Callable,
    node2idx: dict[int, int],
    episodes: int = 50,
    group_size: int = 8,
    max_steps: int = 20,
    lr: float = 1e-4,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
):
    """
    GRPO objective (single-update variant):
      max E[min(r*A, clip(r,1-e,1+e)*A) - beta * KL]
    where A is group-normalized reward.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    agent = DeepACOAgent(model, node2idx=node2idx)
    history = []

    for _ in range(episodes):
        heu_mat = _build_heuristic(model, graph_batch)
        rollouts = [agent.rollout(env_factory(), heu_mat, reward_fn, max_steps=max_steps) for _ in range(group_size)]

        rewards = torch.tensor([sum(r.rewards) for r in rollouts], dtype=torch.float32, device=heu_mat.device)
        adv = _normalize_group(rewards)

        policy_terms = []
        kl_terms = []

        for i, r in enumerate(rollouts):
            if not r.actions:
                continue

            env_replay = env_factory()
            env_replay.reset()
            step_new = []
            step_old = []

            for t, action in enumerate(r.actions):
                new_lp, _ = agent.evaluate_action_log_prob(env_replay, heu_mat, action)
                step_new.append(new_lp)
                step_old.append(r.old_log_probs[t])
                env_replay.step(action)

            new_lp = torch.stack(step_new).mean()
            old_lp = torch.stack(step_old).mean()
            ratio = torch.exp(new_lp - old_lp)

            unclipped = ratio * adv[i]
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv[i]
            policy_terms.append(torch.min(unclipped, clipped))
            kl_terms.append((new_lp - old_lp) ** 2)

        if not policy_terms:
            history.append({"loss": 0.0, "group_mean_reward": float(rewards.mean().item()), "kl": 0.0})
            continue

        policy_obj = torch.stack(policy_terms).mean()
        kl_obj = torch.stack(kl_terms).mean() if kl_terms else torch.zeros((), device=heu_mat.device)
        loss = -(policy_obj - kl_beta * kl_obj)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        history.append(
            {
                "loss": float(loss.item()),
                "group_mean_reward": float(rewards.mean().item()),
                "kl": float(kl_obj.item()),
            }
        )

    return history
