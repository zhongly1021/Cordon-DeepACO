from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx


@dataclass
class StepResult:
    state: List[int]
    reward: float | None
    done: bool
    info: Dict


class CordonEnv:
    """
    Cordon environment with a virtual node M.

    Design rule requested by user:
    1) Start from virtual node M.
    2) First action can choose any real node x.
    3) After choosing x, available actions become neighbors of M or x.
    4) Repeat the same logic with visited path order preserved.
    """

    def __init__(self, road_graph: nx.Graph, max_steps: int = 20, virtual_node: int = -1):
        if road_graph.number_of_nodes() == 0:
            raise ValueError("road_graph must contain at least one node")

        self.base_graph = road_graph.copy()
        self.virtual_node = virtual_node
        self.max_steps = max_steps

        if self.virtual_node in self.base_graph.nodes:
            raise ValueError("virtual_node id must not collide with existing road node ids")

        self.graph = self.base_graph.copy()
        self.graph.add_node(self.virtual_node, is_virtual=True)
        for n in self.base_graph.nodes:
            self.graph.add_edge(self.virtual_node, n, is_virtual_edge=True)

        self.path: List[int] = []
        self.visited = set()
        self.current_node = self.virtual_node
        self.steps = 0

    def reset(self) -> List[int]:
        self.path = [self.virtual_node]
        self.visited = {self.virtual_node}
        self.current_node = self.virtual_node
        self.steps = 0
        return self.path.copy()

    def available_actions(self) -> List[int]:
        if self.current_node == self.virtual_node:
            # First move: all real nodes are legal.
            return sorted(n for n in self.graph.neighbors(self.virtual_node) if n not in self.visited)

        candidates = set(self.graph.neighbors(self.virtual_node))
        candidates.update(self.graph.neighbors(self.current_node))
        # state must keep ordered unique visit path; prevent revisits.
        return sorted(n for n in candidates if n not in self.visited and n != self.virtual_node)

    def step(self, action: int) -> StepResult:
        legal_actions = self.available_actions()
        if action not in legal_actions:
            raise ValueError(f"Illegal action {action}; legal actions are {legal_actions}")

        self.current_node = action
        self.path.append(action)
        self.visited.add(action)
        self.steps += 1

        done = self.steps >= self.max_steps or len(self.available_actions()) == 0
        # Reward left blank intentionally, as requested.
        reward = None
        info = {
            "current_node": self.current_node,
            "available_actions": self.available_actions(),
        }
        return StepResult(state=self.path.copy(), reward=reward, done=done, info=info)

    def get_state(self) -> Tuple[int, ...]:
        return tuple(self.path)
