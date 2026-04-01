from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import torch
from torch.distributions import Categorical, kl_divergence

from cordon_environment import CordonEnv

'''
@dataclass
class Rollout:
    path: List[int]
    actions: List[int]
    rewards: List[float]
    log_probs: List[torch.Tensor]
    old_log_probs: List[torch.Tensor]
    entropies: List[torch.Tensor]
    total_reward: float
'''

@dataclass
class Rollout:
    path: List[int]
    actions: List[int]
    support_edges: List[Tuple[int, int]]   # 新增
    rewards: List[float]
    log_probs: List[torch.Tensor]
    old_log_probs: List[torch.Tensor]
    entropies: List[torch.Tensor]
    total_reward: float

class DeepACOAgent:
    """
    Simplified DeepACO rewritten for the user's CordonEnv.
    Probability rule:
        score(a) = sum_{s in state} [tau(s,a)^alpha * eta(s,a)^beta]
    Then softmax over candidate actions.
    """

    def __init__(
        self,
        node2idx: dict[int, int],
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay: float = 0.9,
        max_steps: int | None = None,
        virtual_node: int = -1,
        elitist: bool = False,
        min_max: bool = False,
        pheromone_init: float = 1.0,
        heuristic_floor: float = 1e-12,
        device: str = "cpu",
    ):
        self.node2idx = dict(node2idx)
        self.node_list = list(node2idx.keys())
        self.n_nodes = len(self.node_list)

        self.n_ants = int(n_ants)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.decay = float(decay)
        self.max_steps = max_steps
        self.virtual_node = int(virtual_node)
        self.elitist = bool(elitist)# 不管
        self.min_max = bool(min_max)# 不管
        self.pheromone_init = float(pheromone_init) #?
        self.heuristic_floor = float(heuristic_floor)
        self.device = device

        self.lowest_cost = float("inf")
        self.shortest_path: List[int] | None = None

        self.pheromone = torch.full(
            (self.n_nodes, self.n_nodes),
            float(self.pheromone_init),
            dtype=torch.float32,
            device=self.device,
        ) #

        self.min_pheromone = 0.1
        self.max_pheromone: float | None = None

    def _node_idx(self, node: int) -> int:
        if node not in self.node2idx:
            raise KeyError(f"Node {node} not found in node2idx.")
        return int(self.node2idx[node])

    def _real_zone_nodes(self, env):
        return [n for n in env.path if n != env.virtual_node]

    def _support_sources_for_action(self, env: CordonEnv, action: int) -> List[int]:
        state = list(env.path)
        # 第一步：只有 virtual -> real
        if len(state) == 1 and state[0] == self.virtual_node:
            if action == self.virtual_node:
                return []
            return [self.virtual_node]

        real_zone_nodes = [n for n in state if n != self.virtual_node]
        if action == self.virtual_node:
            return real_zone_nodes
        srcs = []
        for u in real_zone_nodes:
            if env.graph.has_edge(u, action):
                srcs.append(u)
        return srcs

    def _choose_support_edge(
        self,
        env: CordonEnv,
        heu_mat: torch.Tensor,
        action: int,) -> tuple[int, int]:
        srcs = self._support_sources_for_action(env, action)
        if len(srcs) == 0:
            if len(env.path) == 1:
                return (self.virtual_node, action)
            real_zone_nodes = [n for n in env.path if n != self.virtual_node]
            if len(real_zone_nodes) > 0:
                return (real_zone_nodes[-1], action)
            return (self.virtual_node, action)

        contribs = []
        for u in srcs:
            # scrs只有有edge的
            contribs.append(self._pair_score(heu_mat, int(u), int(action)))

        contribs_t = torch.stack(contribs, dim=0)

        if torch.sum(contribs_t) <= 0:
            probs = torch.ones_like(contribs_t) / len(contribs_t)
        else:
            #probs = torch.softmax(contribs_t, dim=0)
            probs = contribs_t/contribs.sum()

        idx = Categorical(probs=probs).sample()
        src = int(srcs[int(idx.item())])
        return (src, int(action))

    def _call_reward_fn(
        self,
        reward_fn,
        state: Sequence[int],
        policy_kind: str | None = None,
        policy_value: Any | None = None,
    ) -> float:
        
        out = reward_fn.evaluate(
                    zone_state=state,
                    policy_kind=policy_kind,
                    policy_value=policy_value,
                )

        if isinstance(out, dict):
            return float(out.get("reward", 0.0))
        return float(out)

    def _state_available_actions(self, env: CordonEnv) -> List[int]:
        state = list(env.path)
        if len(state) == 0:
            return []
        if len(state) == 1:
            visited = set(state)
            candidates = set()
            candidates.update(env.graph.neighbors(state[0]))
            return sorted(n for n in candidates if (n not in visited) and (n != self.virtual_node))
        
        visited = set(state)
        candidates = set()
        for s in state:
            if s in env.graph:
                if s == self.virtual_node:
                    continue
                else:
                    candidates.update(env.graph.neighbors(s))
        available = sorted(
            n for n in candidates
            if (n not in visited) and (n != self.virtual_node)
        )
        if len(state) > 1:
            available.append(self.virtual_node)
        return available

    def _pair_score(self, heu_mat: torch.Tensor, src: int, dst: int) -> torch.Tensor:
        i = self._node_idx(src)
        j = self._node_idx(dst)

        tau = self.pheromone[i, j].clamp_min(self.heuristic_floor) # all set to 1 during the training
        eta = heu_mat[i, j].clamp_min(self.heuristic_floor)
        return (tau ** self.alpha) * (eta ** self.beta)

    def _candidate_scores(self, env: CordonEnv, heu_mat: torch.Tensor, actions: Sequence[int]) -> torch.Tensor:
        state = list(env.path)
        scores = []

        for a in actions:
            # 这是对所有state -- action的概率的汇集 但是state不一定能和
            total = torch.zeros((), dtype=heu_mat.dtype, device=heu_mat.device)
            for s in state:
                if env.graph.has_edge(int(s), int(a)):
                    total = total + self._pair_score(heu_mat, int(s), int(a))
                else:
                    continue
            scores.append(total)

        return torch.stack(scores, dim=0)

    # =========================================================
    # action selection
    # =========================================================
    '''
    def select_action(
        self,
        env: CordonEnv,
        heu_mat: torch.Tensor,
        inference: bool = False,
    ) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
        actions = self._state_available_actions(env)
        if len(actions) == 0:
            raise RuntimeError("No available actions.")
        #====
        raw_scores = self._candidate_scores(env, heu_mat, actions)
        if torch.sum(raw_scores) <= 0:
            probs = torch.ones_like(raw_scores) / len(raw_scores)
        else:
            probs = torch.softmax(raw_scores, dim=0)

        dist = Categorical(probs=probs)

        if inference:
            idx = torch.argmax(probs)
            return int(actions[int(idx.item())]), None, None

        idx = dist.sample()
        action = int(actions[int(idx.item())])
        log_prob = dist.log_prob(idx)
        entropy = dist.entropy()
        return action, log_prob, entropy
    '''
    
    def _action_distribution(
        self,
        env: CordonEnv,
        heu_mat: torch.Tensor,
    ):
        actions = self._state_available_actions(env)
        if len(actions) == 0:
            raise RuntimeError("No available actions.")
        raw_scores = self._candidate_scores(env, heu_mat, actions)
        raw_scores = raw_scores.clamp_min(1e-12)
        if torch.sum(raw_scores) <= 0:
            probs = torch.ones_like(raw_scores) / len(raw_scores)
        else:
            probs = raw_scores / raw_scores.sum()
        dist = Categorical(probs=probs)
        return actions, probs, dist


    def select_action(
        self,
        env: CordonEnv,
        heu_mat: torch.Tensor,
        inference: bool = False,
    ) -> tuple[int, tuple[int, int], torch.Tensor | None, torch.Tensor | None]:

        actions, probs, dist = self._action_distribution(env, heu_mat)

        if inference:
            idx = torch.argmax(probs)
            action = int(actions[int(idx.item())])
            support_edge = self._choose_support_edge(env, heu_mat, action)
            return action, support_edge, None, None

        idx = dist.sample()
        action = int(actions[int(idx.item())])
        support_edge = self._choose_support_edge(env, heu_mat, action)
        log_prob = dist.log_prob(idx)
        entropy = dist.entropy()
        return action, support_edge, log_prob, entropy


    def evaluate_action_log_prob(
        self,
        env: CordonEnv,
        heu_mat: torch.Tensor,
        action: int,) -> tuple[torch.Tensor, torch.Tensor]:

        actions = self._state_available_actions(env)
        if action not in actions:
            raise ValueError(f"Action {action} not in legal actions {actions}")
        raw_scores = self._candidate_scores(env, heu_mat, actions)
        if torch.sum(raw_scores) <= 0:
            probs = torch.ones_like(raw_scores) / len(raw_scores)
        else:
            #probs = torch.softmax(raw_scores, dim=0)
            probs = raw_scores/raw_scores.sum()

        dist = Categorical(probs=probs)
        pos = actions.index(action)
        idx = torch.tensor(pos, dtype=torch.long, device=heu_mat.device)
        return dist.log_prob(idx), dist.entropy()

    # =========================================================
    # rollout / sampling
    # =========================================================
    '''
    def rollout(
        self,
        env: CordonEnv,
        heu_mat: torch.Tensor,
        reward_fn,
        max_steps: int | None = None,
        terminal_reward_only: bool = True, ###
        inference: bool = False,
        policy_kind: str | None = None,
        policy_value: Any | None = None,) -> Rollout:

        env.reset()
        max_steps = int(max_steps if max_steps is not None else (self.max_steps or env.max_steps))
        actions: List[int] = []
        rewards: List[float] = []
        log_probs: List[torch.Tensor] = []
        old_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        for _ in range(max_steps):
            legal = self._state_available_actions(env)
            if len(legal) == 0:
                break
            action, log_prob, entropy = self.select_action(env, heu_mat, inference=inference) 
            # 这里需要保存到底是使用的哪一条边 然后再后面update的时候具体使用
            out = env.step(action)
            actions.append(int(action))
            if not inference:
                assert log_prob is not None and entropy is not None
                log_probs.append(log_prob)
                old_log_probs.append(log_prob.detach())
                entropies.append(entropy)
            rewards.append(0.0)
            if out.done or action == self.virtual_node:
                break

        path = env.path.copy()

        total_reward = self._call_reward_fn(
                reward_fn,
                path,
                policy_kind=policy_kind,
                policy_value=policy_value,)
        
        return Rollout(
            path=path,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            entropies=entropies,
            total_reward=float(total_reward),)
    '''
    def rollout(
        self,
        env: CordonEnv,
        heu_mat: torch.Tensor,
        reward_fn,
        max_steps: int | None = None,
        terminal_reward_only: bool = True,
        inference: bool = False,
        policy_kind: str | None = None,
        policy_value: Any | None = None,) -> Rollout:

        env.reset()
        max_steps = int(max_steps if max_steps is not None else (self.max_steps or env.max_steps))
        actions: List[int] = []
        support_edges: List[tuple[int, int]] = []   # 新增
        rewards: List[float] = []
        log_probs: List[torch.Tensor] = []
        old_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        for _ in range(max_steps):
            legal = self._state_available_actions(env)
            if len(legal) == 0:
                break

            action, support_edge, log_prob, entropy = self.select_action(
                env, heu_mat, inference=inference
            )
            out = env.step(action)

            actions.append(int(action))
            support_edges.append((int(support_edge[0]), int(support_edge[1])))

            if not inference:
                assert log_prob is not None and entropy is not None
                log_probs.append(log_prob)
                old_log_probs.append(log_prob.detach())
                entropies.append(entropy)
            rewards.append(0.0)
            if out.done or action == self.virtual_node:
                break

        path = env.path.copy()

        total_reward = self._call_reward_fn(
                reward_fn,
                path,
                policy_kind=policy_kind,
                policy_value=policy_value,)

        return Rollout(
            path=path,
            actions=actions,
            support_edges=support_edges,   # 新增
            rewards=total_reward,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            entropies=entropies,
            total_reward=float(total_reward),
        )

    def sample(
        self,
        env_factory,
        heu_mat: torch.Tensor,
        reward_fn,
        inference: bool = False,
        terminal_reward_only: bool = True,
        policy_kind: str | None = None,
        policy_value: Any | None = None,
    ):
        rollouts = []
        for _ in range(self.n_ants):
            env = env_factory()
            r = self.rollout(
                env=env,
                heu_mat=heu_mat,
                reward_fn=reward_fn,
                max_steps=self.max_steps,
                terminal_reward_only=terminal_reward_only,
                inference=inference,
                policy_kind=policy_kind,
                policy_value=policy_value,
            )
            rollouts.append(r)
        costs = [float(r.total_reward) for r in rollouts]
        paths = [r.path for r in rollouts]
        log_probs = [r.log_probs for r in rollouts]
        entropies = [r.entropies for r in rollouts]

        if inference:
            return costs, None, paths
        return costs, log_probs, paths, entropies, rollouts

    # =========================================================
    # pheromone update
    # =========================================================
    '''
    def update_pheromone(self, paths: Sequence[Sequence[int]], costs: Sequence[float]) -> None:
        self.pheromone *= self.decay
        if len(paths) == 0:
            return
        costs_t = torch.as_tensor(costs, dtype=torch.float32, device=self.pheromone.device)

        if self.elitist:
            best_idx = int(torch.argmax(costs_t).item())
            best_path = paths[best_idx]
            best_reward = float(costs_t[best_idx].item())
            contrib = max(best_reward, 0.0)

            uniq_edges = set((int(u), int(v)) for u, v in zip(best_path[:-1], best_path[1:]))
            for u, v in uniq_edges:
                ui = self._node_idx(u)
                vi = self._node_idx(v)
                self.pheromone[ui, vi] += contrib
        else:
            for path, reward in zip(paths, costs):
                contrib = max(float(reward), 0.0)
                uniq_edges = set((int(u), int(v)) for u, v in zip(path[:-1], path[1:]))
                for u, v in uniq_edges:
                    ui = self._node_idx(u)
                    vi = self._node_idx(v)
                    self.pheromone[ui, vi] += contrib

        if self.min_max:
            if self.max_pheromone is None:
                positive = costs_t[costs_t > 0]
                if len(positive) > 0:
                    self.max_pheromone = float(positive.max().item()) * max(self.n_nodes, 1)
                else:
                    self.max_pheromone = 10.0
            self.pheromone.clamp_(min=self.min_pheromone, max=self.max_pheromone)
    '''
    
    def update_pheromone(self, rollouts: Sequence[Rollout]) -> None:
        self.pheromone *= self.decay

        if len(rollouts) == 0:
            return

        rewards_t = torch.tensor(
            [float(r.total_reward) for r in rollouts],
            dtype=torch.float32,
            device=self.pheromone.device,
        )
        if self.elitist:
            best_idx = int(torch.argmax(rewards_t).item())
            best_rollout = rollouts[best_idx]
            contrib = max(float(best_rollout.total_reward), 0.0)

            uniq_edges = set((int(u), int(v)) for (u, v) in best_rollout.support_edges)
            for u, v in uniq_edges:
                ui = self._node_idx(u)
                vi = self._node_idx(v)
                self.pheromone[ui, vi] += contrib
        else:
            for r in rollouts:
                contrib = max(float(r.total_reward), 0.0)
                uniq_edges = set((int(u), int(v)) for (u, v) in r.support_edges)
                for u, v in uniq_edges:
                    ui = self._node_idx(u)
                    vi = self._node_idx(v)
                    self.pheromone[ui, vi] += contrib
        '''
        if self.min_max:
            if self.max_pheromone is None:
                positive = rewards_t[rewards_t > 0]
                if len(positive) > 0:
                    self.max_pheromone = float(positive.max().item()) * max(self.n_nodes, 1)
                else:
                    self.max_pheromone = 10.0
            self.pheromone.clamp_(min=self.min_pheromone, max=self.max_pheromone)
        '''
    
    def run(
        self,
        env_factory,
        heu_mat: torch.Tensor,
        reward_fn,
        n_rounds: int = 1,
        inference: bool = False,
        update_pheromone: bool = True,
        terminal_reward_only: bool = True,
        policy_kind: str | None = None,
        policy_value: Any | None = None,
    ):
        history = []
        best_reward = float("-inf")
        best_path = None
        best_round = -1
        best_rollout = None
        last_rollouts = None

        for rd in range(int(n_rounds)):
            rollouts = []
            for _ in range(self.n_ants): # 
                # 这里其实可以并行----
                env = env_factory()
                r = self.rollout(
                    env=env,
                    heu_mat=heu_mat,
                    reward_fn=reward_fn,
                    max_steps=self.max_steps,
                    terminal_reward_only=terminal_reward_only,
                    inference=inference,
                    policy_kind=policy_kind,
                    policy_value=policy_value,
                )
                rollouts.append(r)

            rewards = [float(r.total_reward) for r in rollouts]
            paths = [r.path for r in rollouts]

            if update_pheromone and (not inference):
                self.update_pheromone(paths, rewards)

            round_best_idx = int(torch.tensor(rewards).argmax().item())
            round_best_reward = rewards[round_best_idx]
            round_best_path = paths[round_best_idx]

            history.append(
                {
                    "round": rd,
                    "mean_reward": float(sum(rewards) / max(len(rewards), 1)),
                    "best_reward": float(round_best_reward),
                    "best_path": round_best_path,
                }
            )

            if round_best_reward > best_reward:
                best_reward = float(round_best_reward)
                best_path = list(round_best_path)
                best_round = rd
                best_rollout = rollouts[round_best_idx]

            last_rollouts = rollouts

        return {
            "best_reward": float(best_reward),
            "best_path": best_path,
            "best_round": int(best_round),
            "best_rollout": best_rollout,
            "last_rollouts": last_rollouts,
            "history": history,
        }


    # ==================================== =====================
    # utility
    # =========================================================
    @staticmethod
    def normalize(values: Sequence[float], eps: float = 1e-6) -> torch.Tensor:
        x = torch.tensor(list(values), dtype=torch.float32)
        return (x - x.mean()) / (x.std() + eps)

    @staticmethod
    def discounted_returns(rewards: Sequence[float], gamma: float = 0.99) -> torch.Tensor:
        g = 0.0
        out = []
        for r in reversed(list(rewards)):
            g = float(r) + float(gamma) * g
            out.append(g)
        out.reverse()
        return torch.tensor(out, dtype=torch.float32)
