from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping


@dataclass
class Metrics:
    efficiency: float
    environment: float
    accessibility: float
    equity: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "efficiency": self.efficiency,
            "environment": self.environment,
            "accessibility": self.accessibility,
            "equity": self.equity,
        }


class MSARewardFunction:
    """
    Reward built from MSA metrics compared with an initial solution.

    Reward = sum(new_metric - initial_metric) over
    [efficiency, environment, accessibility, equity].
    """

    def __init__(self, msa_solver: Callable[..., Mapping[str, float]], solver_kwargs: Dict | None = None):
        self.msa_solver = msa_solver
        self.solver_kwargs = solver_kwargs or {}
        self.initial_metrics: Metrics | None = None

    def compute_initial(self, cordon_solution) -> Metrics:
        metric_map = self.msa_solver(cordon_solution, **self.solver_kwargs)
        self.initial_metrics = self._to_metrics(metric_map)
        return self.initial_metrics

    def __call__(self, cordon_solution) -> float:
        if self.initial_metrics is None:
            raise RuntimeError("Initial metrics not set. Call compute_initial(...) first.")

        new_map = self.msa_solver(cordon_solution, **self.solver_kwargs)
        new_metrics = self._to_metrics(new_map)

        reward = (
            (new_metrics.efficiency - self.initial_metrics.efficiency)
            + (new_metrics.environment - self.initial_metrics.environment)
            + (new_metrics.accessibility - self.initial_metrics.accessibility)
            + (new_metrics.equity - self.initial_metrics.equity)
        )
        return float(reward)

    @staticmethod
    def _to_metrics(metric_map: Mapping[str, float]) -> Metrics:
        required = ["efficiency", "environment", "accessibility", "equity"]
        missing = [k for k in required if k not in metric_map]
        if missing:
            raise KeyError(f"MSA output missing keys: {missing}")
        return Metrics(
            efficiency=float(metric_map["efficiency"]),
            environment=float(metric_map["environment"]),
            accessibility=float(metric_map["accessibility"]),
            equity=float(metric_map["equity"]),
        )
