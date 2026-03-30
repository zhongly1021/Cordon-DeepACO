from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch.distributions import Categorical

from .cordon_environment import CordonEnv


@dataclass
class Rollout:
    path: List[int]
    actions: List[int]
    rewards: List[float]
    log_probs: List[torch.Tensor]
    old_log_probs: List[torch.Tensor]
    entropies: List[torch.Tensor]
    log_probs: List[torch.Tensor]
    rewards: List[float]


class DeepACOAgent:
    """DeepACO-style agent adapted for the virtual-node CordonEnv."""

    def __init__(self, model, node2idx: dict[int, int], device: str = "cpu"):
        self.model = model
        self.node2idx = node2idx
        self.device = device

    def _action_distribution(self, env: CordonEnv, heuristic_matrix: torch.Tensor) -> tuple[list[int], Categorical]:
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def select_action(self, env: CordonEnv, heuristic_matrix: torch.Tensor) -> tuple[int, torch.Tensor]:
        actions = env.available_actions()
        if not actions:
            raise RuntimeError("No available actions")

        cur_idx = self.node2idx[env.current_node]
        act_idx = torch.tensor([self.node2idx[a] for a in actions], dtype=torch.long, device=heuristic_matrix.device)
        row_scores = heuristic_matrix[cur_idx, act_idx]
        probs = torch.softmax(row_scores, dim=0)
        return actions, Categorical(probs=probs)

    def select_action(self, env: CordonEnv, heuristic_matrix: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        actions, dist = self._action_distribution(env, heuristic_matrix)
        idx = dist.sample()
        return actions[int(idx.item())], dist.log_prob(idx), dist.entropy()

    def evaluate_action_log_prob(self, env: CordonEnv, heuristic_matrix: torch.Tensor, action: int) -> tuple[torch.Tensor, torch.Tensor]:
        actions, dist = self._action_distribution(env, heuristic_matrix)
        pos = actions.index(action)
        idx = torch.tensor(pos, dtype=torch.long, device=heuristic_matrix.device)
        return dist.log_prob(idx), dist.entropy()

    def rollout(self, env: CordonEnv, heuristic_matrix: torch.Tensor, reward_fn, max_steps: int) -> Rollout:
        env.reset()
        path = [env.virtual_node]
        actions: List[int] = []
        rewards: List[float] = []
        log_probs: List[torch.Tensor] = []
        old_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        current = env.current_node
        if current == env.virtual_node:
            current = actions[0]

        scores = []
        for a in actions:
            i = current if current >= 0 else 0
            scores.append(heuristic_matrix[i, a])
        probs = torch.softmax(torch.stack(scores), dim=0)
        dist = Categorical(probs=probs)
        idx = dist.sample()
        return actions[int(idx.item())], dist.log_prob(idx)

    def rollout(self, env: CordonEnv, heuristic_matrix: torch.Tensor, reward_fn, max_steps: int) -> Rollout:
        env.reset()
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []

        for _ in range(max_steps):
            if not env.available_actions():
                break

            action, log_prob, entropy = self.select_action(env, heuristic_matrix)
            out = env.step(action)
            reward = reward_fn(out.state)

            actions.append(action)
            path = out.state
            rewards.append(float(reward))
            log_probs.append(log_prob)
            old_log_probs.append(log_prob.detach())
            entropies.append(entropy)

            if out.done:
                break

        return Rollout(
            path=path,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            entropies=entropies,
        )

    @staticmethod
    def normalize(values: Sequence[float], eps: float = 1e-6) -> torch.Tensor:
        x = torch.tensor(values, dtype=torch.float32)
        return (x - x.mean()) / (x.std() + eps)
            action, log_prob = self.select_action(env, heuristic_matrix)
            out = env.step(action)
            reward = reward_fn(out.state)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            if out.done:
                break

        return Rollout(path=env.path.copy(), log_probs=log_probs, rewards=rewards)

    @staticmethod
    def discounted_returns(rewards: Sequence[float], gamma: float = 0.99) -> torch.Tensor:
        g = 0.0
        out = []
        for r in reversed(rewards):
            g = r + gamma * g
            out.append(g)
        out.reverse()
        return torch.tensor(out, dtype=torch.float32)
