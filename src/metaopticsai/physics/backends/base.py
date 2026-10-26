"""Simulation backend abstraction.

This is the load-bearing seam of the architecture. metabox3, the analytical
heuristic, and (future) S4 all implement this same interface, so everything
above `physics/backends/` is backend-agnostic.

Backends MUST:
  - be safe to construct without their heavy dependencies installed
    (TF, S4, etc.) — defer imports to `is_available()` or to the
    methods themselves;
  - convert all return values to numpy / Python builtins before returning
    (no TF/torch tensors leak upward);
  - never log secrets or hardcode paths.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.incidence import IncidenceSpec
from metaopticsai.domain.results import (
    UnitCellResult, RCWASweepResult, FieldResult,
)


class SimulationBackend(ABC):
    """Common interface for electromagnetic simulation backends."""

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """True iff the backend's dependencies are importable and any
        required resources (e.g. material CSV dir) are present."""

    @abstractmethod
    def simulate_unit_cell(
        self,
        params: MetalensDesignParams,
        incidence: IncidenceSpec,
    ) -> UnitCellResult: ...

    @abstractmethod
    def sweep_unit_cells(
        self,
        params: MetalensDesignParams,
        incidence: IncidenceSpec,
        sweep_param: str,
        sweep_values: np.ndarray,
    ) -> RCWASweepResult: ...

    @abstractmethod
    def propagate_field(
        self,
        phase_mask: np.ndarray,
        wavelength_m: float,
        pixel_size_m: float,
        distance_m: float,
        refractive_index: float = 1.0,
    ) -> FieldResult: ...


class BackendRegistry:
    """Resolves a backend name to an instance. Used by tools, workflows, and
    the optimizer to obtain a `SimulationBackend` without hard-importing one.
    """

    def __init__(self):
        self._backends: dict[str, SimulationBackend] = {}

    def register(self, backend: SimulationBackend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"Backend {backend.name!r} already registered.")
        self._backends[backend.name] = backend

    def get(self, name: str = "auto") -> SimulationBackend:
        """Return a backend by name. `"auto"` selects metabox."""
        if name == "auto":
            b = self._backends.get("metabox")
            if b is not None and b.is_available():
                return b
            raise RuntimeError(
                "MetaBox RCWA backend unavailable. "
                "Install TensorFlow with `pip install .[tf]`."
            )
        b = self._backends.get(name)
        if b is None:
            raise KeyError(
                f"Backend {name!r} not registered. Known: {list(self._backends)}"
            )
        if not b.is_available():
            raise RuntimeError(f"Backend {name!r} not available.")
        return b

    def available(self) -> list[str]:
        """Return names of all currently-available backends."""
        return [n for n, b in self._backends.items() if b.is_available()]

    def list_all(self) -> list[str]:
        """Return all registered backend names, available or not."""
        return list(self._backends)