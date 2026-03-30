from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import math
import networkx as nx


@dataclass
class Metrics:
    efficiency: float
    environment: float
    accessibility: float
    equity: float


class MSAElasticWithTolls:
    """
    Lightweight MSA solver reconstructed from `DeepACO.ipynb`.

    links format: (u, v, free_time, capacity)
    od_init format: {(origin, destination): demand}
    toll_dict format: {(u, v): toll_multiplier}
    """

    def __init__(
        self,
        links: Iterable[Tuple[int, int, float, float]],
        od_init: Dict[Tuple[int, int], float],
        toll_dict: Dict[Tuple[int, int], float],
        zone_nodes: set[int],
        max_iter: int = 120,
        tol: float = 1e-4,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.bpr_alpha = bpr_alpha
        self.bpr_beta = bpr_beta
        self.zone_nodes = set(zone_nodes)

        self.graph = nx.DiGraph()
        for u, v, fft, cap in links:
            inside_zone = (u in self.zone_nodes) and (v in self.zone_nodes)
            self.graph.add_edge(
                u,
                v,
                free_time=float(fft),
                capacity=max(float(cap), 1e-6),
                toll=float(toll_dict.get((u, v), 0.0)),
                inside_zone=inside_zone,
                flow=0.0,
                time=float(fft),
                cost=float(fft),
            )

        self.od = {k: float(v) for k, v in od_init.items() if float(v) > 0.0}
        self._update_costs()

    def _od_type(self, o: int, d: int) -> str:
        if o in self.zone_nodes:
            return "internal"
        return "inbound"

    def _weight_for_od_type(self, od_type: str):
        def w(u, v, data):
            t = data["time"]
            if od_type == "inbound" and data.get("inside_zone", False):
                return t * (1.0 + data["toll"])
            return t

        return w

    def _update_costs(self):
        for _, _, data in self.graph.edges(data=True):
            f = max(data["flow"], 0.0)
            cap = data["capacity"]
            fft = data["free_time"]
            t = fft * (1.0 + self.bpr_alpha * (f / cap) ** self.bpr_beta)
            data["time"] = t
            if data["inside_zone"]:
                data["cost"] = t * (1.0 + data["toll"])
            else:
                data["cost"] = t

    def _all_or_nothing(self) -> Dict[Tuple[int, int], float]:
        aon = {(u, v): 0.0 for u, v in self.graph.edges()}
        for (o, d), demand in self.od.items():
            od_type = self._od_type(o, d)
            weight_fn = self._weight_for_od_type(od_type)
            try:
                path = nx.shortest_path(self.graph, source=o, target=d, weight=weight_fn)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for i in range(len(path) - 1):
                aon[(path[i], path[i + 1])] += demand
        return aon

    def solve(self) -> Mapping[str, float]:
        prev_total_cost = math.inf
        for it in range(1, self.max_iter + 1):
            aon = self._all_or_nothing()
            step = 1.0 / it
            for u, v, data in self.graph.edges(data=True):
                data["flow"] = (1.0 - step) * data["flow"] + step * aon[(u, v)]
            self._update_costs()

            total_cost = self.total_system_travel_time()
            if abs(prev_total_cost - total_cost) <= self.tol:
                break
            prev_total_cost = total_cost

        return self.compute_metrics()

    def total_system_travel_time(self) -> float:
        return float(sum(data["time"] * data["flow"] for _, _, data in self.graph.edges(data=True)))

    def total_emission_proxy(self) -> float:
        return float(sum((data["time"] ** 2) * data["flow"] for _, _, data in self.graph.edges(data=True)))

    def average_accessibility(self) -> float:
        acc_values = []
        for (o, d), demand in self.od.items():
            if demand <= 0:
                continue
            od_type = self._od_type(o, d)
            weight_fn = self._weight_for_od_type(od_type)
            try:
                c = nx.shortest_path_length(self.graph, source=o, target=d, weight=weight_fn)
                acc_values.append(1.0 / max(c, 1e-6))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                acc_values.append(0.0)
        return float(sum(acc_values) / max(len(acc_values), 1))

    def equity_proxy(self) -> float:
        trip_costs = []
        for (o, d), demand in self.od.items():
            if demand <= 0:
                continue
            od_type = self._od_type(o, d)
            weight_fn = self._weight_for_od_type(od_type)
            try:
                c = nx.shortest_path_length(self.graph, source=o, target=d, weight=weight_fn)
                trip_costs.append(c)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if not trip_costs:
            return 0.0
        mean = sum(trip_costs) / len(trip_costs)
        variance = sum((x - mean) ** 2 for x in trip_costs) / len(trip_costs)
        # higher is better, so invert dispersion
        return float(1.0 / (1.0 + variance))

    def compute_metrics(self) -> Mapping[str, float]:
        tsst = self.total_system_travel_time()
        return {
            "efficiency": float(1.0 / (1.0 + tsst)),
            "environment": float(1.0 / (1.0 + self.total_emission_proxy())),
            "accessibility": float(self.average_accessibility()),
            "equity": float(self.equity_proxy()),
        }


class MSARewardFunction:
    """
    Reward = sum(metric_new - metric_initial).

    This class now includes actual MSA solving instead of only accepting a mock callable.
    """

    def __init__(
        self,
        links: Iterable[Tuple[int, int, float, float]],
        od_init: Dict[Tuple[int, int], float],
        zone_nodes: set[int],
        solver_kwargs: Dict | None = None,
    ):
        self.links = list(links)
        self.od_init = dict(od_init)
        self.zone_nodes = set(zone_nodes)
        self.solver_kwargs = solver_kwargs or {}
        self.initial_metrics: Metrics | None = None

    def _solve(self, toll_dict: Dict[Tuple[int, int], float]) -> Metrics:
        solver = MSAElasticWithTolls(
            links=self.links,
            od_init=self.od_init,
            toll_dict=toll_dict,
            zone_nodes=self.zone_nodes,
            **self.solver_kwargs,
        )
        out = solver.solve()
        return Metrics(
            efficiency=float(out["efficiency"]),
            environment=float(out["environment"]),
            accessibility=float(out["accessibility"]),
            equity=float(out["equity"]),
        )

    def compute_initial(self, toll_dict: Dict[Tuple[int, int], float] | None = None) -> Metrics:
        self.initial_metrics = self._solve(toll_dict or {})
        return self.initial_metrics

    def __call__(self, toll_dict: Dict[Tuple[int, int], float]) -> float:
        if self.initial_metrics is None:
            raise RuntimeError("Call compute_initial(...) before reward evaluation.")
        new_metrics = self._solve(toll_dict)
        return float(
            (new_metrics.efficiency - self.initial_metrics.efficiency)
            + (new_metrics.environment - self.initial_metrics.environment)
            + (new_metrics.accessibility - self.initial_metrics.accessibility)
            + (new_metrics.equity - self.initial_metrics.equity)
        )
