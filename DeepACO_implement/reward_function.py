from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Tuple

import math
import networkx as nx


def demand_func(D0: dict, c: float, u: dict, w: tuple, C0: dict) -> float:
    """Elastic demand function from notebook."""
    return D0[w] * math.exp(u[w] * (1.0 - c / max(C0[w], 1e-9)))


def inv_demand_func(D0: dict, d: float, u: dict, w: tuple, C0: dict) -> float:
    """Inverse demand (kept for compatibility with notebook API)."""
    return C0[w] - (C0[w] / max(u[w], 1e-9)) * math.log(max(d, 1e-12) / max(D0[w], 1e-12))


@dataclass
class Metrics:
    efficiency: float
    environment: float
    accessibility: float
    equity: float


class MSAElasticWithTolls:
    """
    MSA elastic assignment solver rebuilt from `DeepACO.ipynb` logic.

    - inbound OD (origin not in zone) pays toll only on inside-zone edges
    - internal OD (origin in zone) uses travel time only
    - flow decomposition: flow / flow_in / flow_out
    """

    def __init__(
        self,
        links: Iterable[Tuple[int, int, float, float]],
        od_init: Dict[Tuple[int, int], float],
        toll_dict: Dict[Tuple[int, int], float],
        zone_nodes: set[int],
        demand_fn: Callable = demand_func,
        inv_demand_fn: Callable = inv_demand_func,
        max_iter: int = 200,
        tol: float = 1e-4,
        step_rule: str = "harmonic",
        alpha0: float = 0.3,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
        u_default: float = 0.3,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.D = demand_fn
        self.Dinv = inv_demand_fn
        self.step_rule = step_rule
        self.alpha0 = alpha0
        self.bpr_a = bpr_alpha
        self.bpr_b = bpr_beta
        self.zone_nodes = set(zone_nodes)

        self.G = nx.DiGraph()
        for u, v, fft, cap in links:
            inside = (u in self.zone_nodes) and (v in self.zone_nodes)
            self.G.add_edge(
                u,
                v,
                free_time=float(fft),
                capacity=max(float(cap), 1e-9),
                toll=float(toll_dict.get((u, v), 0.0)),
                inside_zone=bool(inside),
                flow=0.0,
                flow_in=0.0,
                flow_out=0.0,
                time=float(fft),
                cost=float(fft),
            )

        self.od = {k: float(v) for k, v in od_init.items() if float(v) > 0.0}
        self.D0 = {k: float(v) for k, v in od_init.items()}

        self.C0, self.u = {}, {}
        for w in self.od:
            o, d = w
            try:
                self.C0[w] = nx.shortest_path_length(self.G, o, d, weight="free_time")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self.C0[w] = 1e6
            self.u[w] = float(u_default)

        self._init_assignment()

    def _od_type(self, o: int, d: int) -> str:
        _ = d
        return "internal" if (o in self.zone_nodes) else "inbound"

    def _weight_for_odtype(self, od_type: str):
        def _w(u, v, data):
            t = data["time"]
            if od_type == "inbound" and data.get("inside_zone", False):
                return t * (1.0 + data["toll"])
            return t

        return _w

    def _update_costs(self):
        for _, _, data in self.G.edges(data=True):
            fft = data["free_time"]
            cap = max(data["capacity"], 1e-9)
            f = max(data["flow"], 0.0)
            t = fft * (1.0 + self.bpr_a * (f / cap) ** self.bpr_b)
            data["time"] = t
            data["cost"] = t * (1.0 + data["toll"]) if data["inside_zone"] else t

    def _init_assignment(self):
        for _, _, data in self.G.edges(data=True):
            t = data["free_time"]
            data["time"] = t
            data["cost"] = t * (1.0 + data["toll"]) if data["inside_zone"] else t

        aux_total = {(u, v): 0.0 for u, v in self.G.edges()}
        aux_in = {(u, v): 0.0 for u, v in self.G.edges()}
        aux_out = {(u, v): 0.0 for u, v in self.G.edges()}

        for w in list(self.od.keys()):
            o, d = w
            odt = self._od_type(o, d)
            wfun = self._weight_for_odtype(odt)
            try:
                c_sp = nx.shortest_path_length(self.G, o, d, weight=wfun)
                d_w = max(self.D(self.D0, c_sp, self.u, w, self.C0), 0.0)
                path = nx.shortest_path(self.G, o, d, weight=wfun)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d_w = 0.0
                path = [o]

            self.od[w] = d_w
            for a, b in zip(path, path[1:]):
                aux_total[(a, b)] += d_w
                if odt == "internal":
                    aux_in[(a, b)] += d_w
                else:
                    aux_out[(a, b)] += d_w

        for (u, v), g in aux_total.items():
            self.G[u][v]["flow"] = g
            self.G[u][v]["flow_in"] = aux_in[(u, v)]
            self.G[u][v]["flow_out"] = aux_out[(u, v)]

    def _all_or_nothing(self):
        aux_total = {(u, v): 0.0 for u, v in self.G.edges()}
        aux_in = {(u, v): 0.0 for u, v in self.G.edges()}
        aux_out = {(u, v): 0.0 for u, v in self.G.edges()}
        aux_demand = {}

        for w in list(self.od.keys()):
            o, d = w
            odt = self._od_type(o, d)
            wfun = self._weight_for_odtype(odt)
            try:
                c_sp = nx.shortest_path_length(self.G, o, d, weight=wfun)
                h_w = max(self.D(self.D0, c_sp, self.u, w, self.C0), 0.0)
                path = nx.shortest_path(self.G, o, d, weight=wfun)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                h_w = 0.0
                path = [o]
            aux_demand[w] = h_w

            for a, b in zip(path, path[1:]):
                aux_total[(a, b)] += h_w
                if odt == "internal":
                    aux_in[(a, b)] += h_w
                else:
                    aux_out[(a, b)] += h_w

        return aux_total, aux_in, aux_out, aux_demand

    def _step_size(self, k: int) -> float:
        if self.step_rule == "harmonic":
            return 1.0 / max(k, 1)
        if self.step_rule == "sqrt":
            return 1.0 / math.sqrt(max(k, 1))
        if self.step_rule == "const":
            return float(self.alpha0)
        return 1.0 / max(k, 1)

    def _gap(self, aux_total: Mapping, aux_demand: Mapping) -> float:
        denom = max(sum(self.od.values()), 1e-12)
        g = 0.0
        for (u, v), gk in aux_total.items():
            f = self.G[u][v]["flow"]
            g += abs(gk - f)
        g /= denom

        for w, d_act in self.od.items():
            o, d = w
            try:
                kappa = nx.shortest_path_length(self.G, o, d, weight=lambda u, v, data: data["cost"])
                d_est = max(self.D(self.D0, kappa, self.u, w, self.C0), 0.0)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d_est = 0.0
            g += abs(d_act - d_est) / denom
        return g

    def run(self):
        for it in range(1, self.max_iter + 1):
            self._update_costs()
            aux_total, aux_in, aux_out, aux_demand = self._all_or_nothing()
            lam = self._step_size(it)

            for u, v, data in self.G.edges(data=True):
                f, g = data["flow"], aux_total[(u, v)]
                f_in, g_in = data["flow_in"], aux_in[(u, v)]
                f_out, g_out = data["flow_out"], aux_out[(u, v)]
                data["flow"] = f + lam * (g - f)
                data["flow_in"] = f_in + lam * (g_in - f_in)
                data["flow_out"] = f_out + lam * (g_out - f_out)

            for w, d_old in list(self.od.items()):
                self.od[w] = max(d_old + lam * (aux_demand[w] - d_old), 0.0)

            if self._gap(aux_total, aux_demand) < self.tol:
                break

        flow_res = {(u, v): data["flow"] for u, v, data in self.G.edges(data=True)}
        return flow_res, self.od

    # --- metrics for reward ---
    def total_system_travel_time(self) -> float:
        self._update_costs()
        return float(sum(data["time"] * data["flow"] for _, _, data in self.G.edges(data=True)))

    def total_emission_proxy(self) -> float:
        self._update_costs()
        return float(sum((data["time"] ** 2) * data["flow"] for _, _, data in self.G.edges(data=True)))

    def average_accessibility(self) -> float:
        vals = []
        for (o, d), demand in self.od.items():
            if demand <= 0:
                continue
            odt = self._od_type(o, d)
            wfun = self._weight_for_odtype(odt)
            try:
                c = nx.shortest_path_length(self.G, source=o, target=d, weight=wfun)
                vals.append(1.0 / max(c, 1e-9))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                vals.append(0.0)
        return float(sum(vals) / max(len(vals), 1))

    def equity_proxy(self) -> float:
        vals = []
        for (o, d), demand in self.od.items():
            if demand <= 0:
                continue
            odt = self._od_type(o, d)
            wfun = self._weight_for_odtype(odt)
            try:
                vals.append(nx.shortest_path_length(self.G, source=o, target=d, weight=wfun))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        return float(1.0 / (1.0 + var))

    def compute_metrics(self) -> Metrics:
        tsst = self.total_system_travel_time()
        eff = 1.0 / (1.0 + tsst)
        env = 1.0 / (1.0 + self.total_emission_proxy())
        acc = self.average_accessibility()
        eq = self.equity_proxy()
        return Metrics(efficiency=float(eff), environment=float(env), accessibility=float(acc), equity=float(eq))


class MSARewardFunction:
    """Reward computed by comparing MSA metrics against an initial baseline solution."""

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

    def _solve_metrics(self, toll_dict: Dict[Tuple[int, int], float]) -> Metrics:
        solver = MSAElasticWithTolls(
            links=self.links,
            od_init=self.od_init,
            toll_dict=toll_dict,
            zone_nodes=self.zone_nodes,
            **self.solver_kwargs,
        )
        solver.run()
        return solver.compute_metrics()

    def compute_initial(self, toll_dict: Dict[Tuple[int, int], float] | None = None) -> Metrics:
        self.initial_metrics = self._solve_metrics(toll_dict or {})
        return self.initial_metrics

    def __call__(self, toll_dict: Dict[Tuple[int, int], float]) -> float:
        if self.initial_metrics is None:
            raise RuntimeError("Initial solution not computed. Please call compute_initial(...) first.")
        new_metrics = self._solve_metrics(toll_dict)
        return float(
            (new_metrics.efficiency - self.initial_metrics.efficiency)
            + (new_metrics.environment - self.initial_metrics.environment)
            + (new_metrics.accessibility - self.initial_metrics.accessibility)
            + (new_metrics.equity - self.initial_metrics.equity)
        )
