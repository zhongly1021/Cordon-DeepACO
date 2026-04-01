from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Mapping, Sequence, Tuple
import math
import networkx as nx


MetricName = Literal["efficiency", "environment", "accessibility", "equity"]
PolicyKind = Literal["none", "toll", "speed_limit"]


@dataclass
class Metrics:
    efficiency: float
    environment: float
    accessibility: float
    equity: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "efficiency": float(self.efficiency),
            "environment": float(self.environment),
            "accessibility": float(self.accessibility),
            "equity": float(self.equity),
        }


class MSAStaticAssignmentSolver:
    """
    Fixed-demand lower-level MSA solver for bilevel cordon design.
    Required links format:
        (u, v, length, free_time, capacity)
    where:
        - length: physical link length
        - free_time: uncongested travel time on the link
        - capacity: link capacity
    Policy:
        - "none"
        - "toll"
        - "speed_limit"
    For toll:
        policy_value = {"inside": toll_in, "outside": toll_out}
    For speed_limit:
        policy_value = {"inside": v_in, "outside": v_out}
    """

    def __init__(
        self,
        links: Iterable[Tuple],
        od_demand: Dict[Tuple[int, int], float],
        zone_nodes: set[int],
        policy_kind: PolicyKind = "none",
        policy_value: Any | None = None,
        max_iter: int = 100,
        tol: float = 1e-5,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
        value_of_time: float = 1.0,
        emission_power: float = 2.0,
        eps: float = 1e-9,
    ):
        self.links = list(links)
        self.od = {k: float(v) for k, v in od_demand.items() if float(v) > 0.0}
        self.zone_nodes = set(zone_nodes) # 这里用env.get_state()就可以获得

        self.policy_kind = policy_kind
        self.policy_value = policy_value

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.bpr_alpha = float(bpr_alpha)
        self.bpr_beta = float(bpr_beta)
        self.value_of_time = float(value_of_time)
        self.emission_power = float(emission_power)
        self.eps = float(eps)

        self.graph = nx.DiGraph()
        self._build_graph()
        self._update_link_costs()

    # =========================================================
    # graph construction
    # =========================================================
    def _build_graph(self) -> None:
        for raw in self.links: # tuple
            # input 都必须包含 u, v, length, free time, capacity------
            if len(raw) != 5:
                raise ValueError(
                    "Each link must be a 5-tuple: (u, v, length, free_time, capacity)."
                )

            u, v, length, free_time, capacity = raw
            u = int(u)
            v = int(v)
            length = max(float(length), self.eps)
            free_time = max(float(free_time), self.eps)
            capacity = max(float(capacity), self.eps)

            inside_zone = (u in self.zone_nodes) and (v in self.zone_nodes)

            self.graph.add_edge(
                u,
                v,
                length=length,
                base_free_time=free_time,
                capacity=capacity,
                inside_zone=inside_zone,
                flow=0.0,
                free_time=free_time,
                time=free_time,   # physical travel time
                cost=free_time,   # generalized route-choice cost
                toll=0.0,
                applied_speed_limit=None,
            )

    def _get_region_value(self, data: dict, field_name: str) -> float | None:
        """
        policy_value should be:
            {"inside": x_in, "outside": x_out}
        """
        if self.policy_value is None:
            return None
        key = "inside" if data["inside_zone"] else "outside"
        val = self.policy_value.get(key, None)
        if val is None:
            return None
        val = float(val)
        return val

    def _get_applicable_toll(self, data: dict) -> float:
        if self.policy_kind != "toll":
            return 0.0
        toll = self._get_region_value(data, "toll")
        if toll is None:
            return 0.0
        return float(toll)
    
    def _get_applicable_speed_limit(self, data: dict) -> float | None:
        if self.policy_kind != "speed_limit":
            return None
        vlim = self._get_region_value(data, "speed_limit")
        if vlim is None or vlim <= 0:
            return None
        return float(vlim)

    def _policy_additive_cost(self, data: dict) -> float:
        if self.policy_kind != "toll":
            return 0.0
        toll = self._get_applicable_toll(data)
        return self.value_of_time * float(toll)

    # =========================================================
    # msa core
    # =========================================================
    def _update_link_costs(self) -> None:
        for _, _, data in self.graph.edges(data=True):
            flow = max(float(data["flow"]), 0.0)
            cap = max(float(data["capacity"]), self.eps)
            length = max(float(data["length"]), self.eps)
            t0 = max(float(data["base_free_time"]), self.eps)

            t_bpr = t0 * (1.0 + self.bpr_alpha * (flow / cap) ** self.bpr_beta)
            vlim = self._get_applicable_speed_limit(data)
            if vlim is not None:
                limited_time = length / max(float(vlim), self.eps)
                congested_time = max(t_bpr, limited_time)
                data["applied_speed_limit"] = float(vlim)
            else:
                congested_time = t_bpr
                data["applied_speed_limit"] = None

            additive_cost = self._policy_additive_cost(data)
            generalized_cost = congested_time + additive_cost # additive cost 需要修改 usage-tolling
            data["free_time"] = t0
            data["time"] = float(congested_time)      # physical time
            data["cost"] = float(generalized_cost)    # route-choice cost
            data["toll"] = float(additive_cost)

    def _all_or_nothing(self) -> Dict[Tuple[int, int], float]:
        aon = {(u, v): 0.0 for u, v in self.graph.edges()}
        for (o, d), demand in self.od.items():
            path = nx.shortest_path(self.graph, source=o, target=d, weight="cost")
            for i in range(len(path) - 1):
                aon[(path[i], path[i + 1])] += demand
        return aon

    def solve(self) -> Metrics:
        prev_tsst = math.inf
        for it in range(1, self.max_iter + 1):
            aon = self._all_or_nothing()
            step = 1.0 / float(it)
            for u, v, data in self.graph.edges(data=True):
                old_flow = float(data["flow"])
                new_flow = aon[(u, v)]
                data["flow"] = (1.0 - step) * old_flow + step * new_flow

            self._update_link_costs()

            tsst = self.total_system_travel_time()
            if abs(prev_tsst - tsst) <= self.tol:
                break
            prev_tsst = tsst

        return self.compute_metrics()

    # =========================================================
    # metrics
    # =========================================================
    def total_system_travel_time(self) -> float:
        return float(
            sum(
                float(data["time"]) * float(data["flow"])
                for _, _, data in self.graph.edges(data=True)
            )
        )

    def total_emission_proxy(self) -> float:# 这里可以修改为和速度\时间等相关的变量
        return float(
            sum(
                (float(data["time"]) ** self.emission_power) * float(data["flow"])
                for _, _, data in self.graph.edges(data=True)
            )
        )

    def _od_generalized_path_costs(self) -> list[tuple[float, float]]:
        out = []
        for (o, d), demand in self.od.items():
            c = nx.shortest_path_length(self.graph, source=o, target=d, weight="cost")
            out.append((float(c), float(demand)))
        return out

    def average_accessibility(self) -> float:
        vals = []
        weights = []
        for c, demand in self._od_generalized_path_costs():
            if demand <= 0:
                continue
            if math.isinf(c):
                vals.append(0.0)
            else:
                vals.append(1.0 / (1.0 + max(c, self.eps)))
            weights.append(demand)
        if not weights:
            return 0.0
        wsum = sum(weights)
        return float(sum(v * w for v, w in zip(vals, weights)) / max(wsum, self.eps))

    def equity_proxy(self) -> float:
        costs = []
        weights = []
        for c, demand in self._od_generalized_path_costs():
            if demand <= 0 or math.isinf(c):
                continue
            costs.append(float(c))
            weights.append(float(demand))
        if not weights:
            return 0.0

        wsum = sum(weights)
        mean_c = sum(c * w for c, w in zip(costs, weights)) / max(wsum, self.eps)
        var_c = sum(w * (c - mean_c) ** 2 for c, w in zip(costs, weights)) / max(wsum, self.eps)

        return float(1.0 / (1.0 + var_c))

    def compute_metrics(self) -> Metrics:
        tsst = self.total_system_travel_time()
        emi = self.total_emission_proxy()
        acc = self.average_accessibility()
        eq = self.equity_proxy()

        # transform all metrics so that "larger is better"
        return Metrics(
            efficiency=float(1.0 / (1.0 + tsst)),
            environment=float(1.0 / (1.0 + emi)),
            accessibility=float(acc),
            equity=float(eq),
        )


class MSARewardFunction:
    """
    Reward wrapper for upper-level bilevel decisions.

    Inputs:
        - zone_nodes_or_state # 
        - policy_kind # 
        - policy_value # 

    Reward:
        ratio_i = new_metric_i / initial_metric_i
        reward  = weighted average of the 4 ratios
    """

    def __init__(
        self,
        links: Iterable[Tuple],
        od_demand: Dict[Tuple[int, int], float],
        virtual_node: int = -1,
        reward_weights: Mapping[MetricName, float] | None = None,
        solver_kwargs: Dict[str, Any] | None = None,
        eps: float = 1e-9,
    ):
        self.links = list(links)
        self.od_demand = dict(od_demand)
        self.virtual_node = int(virtual_node)
        self.solver_kwargs = solver_kwargs or {}
        self.eps = float(eps)

        default_weights = {
            "efficiency": 1.0,
            "environment": 1.0,
            "accessibility": 1.0,
            "equity": 1.0,
        }
        if reward_weights is None:
            self.reward_weights = default_weights
        else:
            self.reward_weights = {**default_weights, **dict(reward_weights)}

        self.initial_metrics: Metrics | None = None
        # 要先initial_metrics ....

    def _extract_zone_nodes(self, zone_state: Sequence[int] | set[int] | None) -> set[int]:
        if zone_state is None:
            return set()
        zone_nodes = set()
        for x in zone_state:
            x = int(x)
            if x != self.virtual_node:
                zone_nodes.add(x)
        return zone_nodes

    def _solve(
        self,
        zone_nodes: set[int],
        policy_kind: PolicyKind = "none",
        policy_value: Any | None = None,
    ) -> Metrics:
        solver = MSAStaticAssignmentSolver(
            links=self.links,
            od_demand=self.od_demand,
            zone_nodes=zone_nodes,
            policy_kind=policy_kind,
            policy_value=policy_value,
            **self.solver_kwargs,
        )
        return solver.solve()

    def compute_initial(
        self,
        zone_state: Sequence[int] | set[int] | None = None,
        policy_kind: PolicyKind = "none",
        policy_value: Any | None = None,
    ) -> Metrics:
        # initial zone nodes 全都是------
        # policy kind/policy value都不设置限制
        zone_nodes = self._extract_zone_nodes(zone_state)
        self.initial_metrics = self._solve(
            zone_nodes=zone_nodes,
            policy_kind=policy_kind,
            policy_value=policy_value,
        )
        return self.initial_metrics

    def evaluate(
        self,
        zone_state: Sequence[int] | set[int],
        policy_kind: PolicyKind,
        policy_value: Any | None,
    ) -> Dict[str, Any]:
        if self.initial_metrics is None:
            raise RuntimeError("You must call compute_initial(...) first.")

        zone_nodes = self._extract_zone_nodes(zone_state)
        new_metrics = self._solve(
            zone_nodes=zone_nodes,
            policy_kind=policy_kind,
            policy_value=policy_value,
        )

        init_map = self.initial_metrics.as_dict()
        new_map = new_metrics.as_dict()

        ratios = {
            k: float(new_map[k] / max(init_map[k], self.eps))
            for k in ("efficiency", "environment", "accessibility", "equity")
        }

        weight_sum = max(sum(float(v) for v in self.reward_weights.values()), self.eps)
        reward = sum(self.reward_weights[k] * ratios[k] for k in ratios) / weight_sum

        return {
            "reward": float(reward),
            "metrics": new_map,
            "ratios": ratios,
            "zone_nodes": sorted(zone_nodes),
            "policy_kind": policy_kind,
            "policy_value": policy_value,
        }

    def __call__(
        self,
        zone_state: Sequence[int] | set[int],
        policy_kind: PolicyKind,
        policy_value: Any | None,
    ) -> float:
        out = self.evaluate(
            zone_nodes_or_state=zone_state,
            policy_kind=policy_kind,
            policy_value=policy_value,
        )
        return float(out["reward"])