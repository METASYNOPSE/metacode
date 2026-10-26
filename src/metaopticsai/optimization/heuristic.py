"""Heuristic optimizer — uses the analytical backend to drive a coarse search.
"""
from __future__ import annotations

import numpy as np

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.results import FOMResult, RCWASweepResult
from metaopticsai.optimization.base import Optimizer


class HeuristicOptimizer(Optimizer):
    """Analytical FOM = mean_amplitude² · min(1, phase_coverage).

    Doesn't actually search the parameter space — it computes a Strehl
    estimate from a pre-computed RCWA sweep. For real gradient-based
    search use `GradientOptimizer`.
    """

    name = "heuristic"

    def optimize(
        self,
        params: MetalensDesignParams,
        *,
        target_fom: float,
        max_iters: int,
        sweep_result: RCWASweepResult | None = None,
        **kwargs,
    ) -> tuple[MetalensDesignParams, FOMResult, list[float]]:
        if sweep_result is None:
            raise ValueError(
                "HeuristicOptimizer requires `sweep_result` from a prior "
                "RCWA sweep — it does not run RCWA itself."
            )

        history: list[float] = []
        best = 0.0
        for it in range(max_iters):
            coverage = sweep_result.phase_coverage
            mean_amp = sweep_result.mean_transmission
            strehl = float(mean_amp ** 2 * min(1.0, coverage))
            history.append(strehl)
            best = max(best, strehl)
            if best >= target_fom:
                break

        fom = FOMResult(
            value=best,
            name="strehl_ratio",
            backend=sweep_result.backend,
            iteration=len(history),
            metadata={"method": self.name},
        )
        return params, fom, history