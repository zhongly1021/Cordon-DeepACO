from __future__ import annotations

import torch

from .deepaco import DeepACOAgent
from .model import HeuristicGNN


def _build_heuristic(model: HeuristicGNN, graph_batch):
    edge_logits = model(graph_batch)
    return model.edge_logits_to_matrix(graph_batch.node_features.shape[0], graph_batch.edge_index, edge_logits)


def reinforce_train(
    model: HeuristicGNN,
    graph_batch,
    env_factory,
    reward_fn,
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
    - per-ant total reward as objective signal
    - normalized advantage within each ant group
    - entropy regularization
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    agent = DeepACOAgent(model, node2idx=node2idx)
    history = []

    for _ in range(episodes):
        heu_mat = _build_heuristic(model, graph_batch)

        rollouts = [agent.rollout(env_factory(), heu_mat, reward_fn, max_steps=max_steps) for _ in range(n_ants)]
        costs = torch.tensor([sum(r.rewards) for r in rollouts], dtype=torch.float32, device=heu_mat.device)
        adv = (costs - costs.mean()) / (costs.std() + 1e-6)

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

        loss = torch.sum(adv.detach() * ant_log_probs) / max(n_ants, 1)
        loss = 10.0 * loss - entropy_coef * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        history.append(
            {
                "loss": float(loss.item()),
                "mean_reward": float(costs.mean().item()),
                "entropy": float(entropy_mean.item()),
            }
        )
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
    node2idx: dict[int, int],
    episodes: int = 50,
    group_size: int = 8,
    max_steps: int = 20,
    lr: float = 1e-4,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
):
    """
    Original-style GRPO objective:
      L = E[min(r*A, clip(r,1-e,1+e)*A) - beta * KL(pi || pi_ref)]
    where A is group-normalized reward.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    agent = DeepACOAgent(model, node2idx=node2idx)
    history = []

    for _ in range(episodes):
        heu_mat = _build_heuristic(model, graph_batch)
        rollouts = [agent.rollout(env_factory(), heu_mat, reward_fn, max_steps=max_steps) for _ in range(group_size)]

        rewards = torch.tensor([sum(r.rewards) for r in rollouts], dtype=torch.float32, device=heu_mat.device)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

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

            # scalar surrogate KL with detached reference log-prob
            kl_terms.append((new_lp - old_lp) ** 2)

        if not policy_terms:
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
