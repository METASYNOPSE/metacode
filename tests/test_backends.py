"""Tests: backend availability, MetaBox RCWA, and clear error when unavailable."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pytest

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.incidence import IncidenceSpec
from metaopticsai.physics.backends import default_registry
from metaopticsai.physics.backends.base import BackendRegistry
from metaopticsai.physics.backends.metabox import MetaboxBackend


# ── Test: registry only contains metabox ────────────────────────────────────


def test_default_registry_contains_only_metabox():
    """default_registry registers exactly one backend named 'metabox'."""
    reg = default_registry()
    assert reg.list_all() == ["metabox"]


def test_metabox_backend_availability():
    """MetaboxBackend.is_available() returns a bool without raising."""
    b = MetaboxBackend()
    available = b.is_available()
    assert isinstance(available, bool)


# ── Test: clear error when MetaBox unavailable ───────────────────────────────


def test_auto_raises_when_metabox_unavailable():
    """BackendRegistry.get('auto') raises RuntimeError with a clear message
    when the MetaBox backend is not available."""
    reg = BackendRegistry()

    class _FakeMetabox(MetaboxBackend):
        def is_available(self):
            return False

    reg.register(_FakeMetabox())
    with pytest.raises(RuntimeError, match="MetaBox RCWA backend unavailable"):
        reg.get("auto")


def test_named_backend_raises_when_unavailable():
    """BackendRegistry.get('metabox') raises when the backend is not available."""
    reg = BackendRegistry()

    class _FakeMetabox(MetaboxBackend):
        def is_available(self):
            return False

    reg.register(_FakeMetabox())
    with pytest.raises(RuntimeError):
        reg.get("metabox")


# ── Test: canonical tool registry builds cleanly ─────────────────────────────


def test_build_tool_registry():
    """build_tool_registry returns a registry with all expected tool names."""
    from metaopticsai.domain.handles import Artifact
    from metaopticsai.tools import build_tool_registry

    class _MockStore:
        def __init__(self):
            self._d = {}
        def put(self, kind, payload, **meta):
            art = Artifact.new(kind, payload, **meta)
            self._d[art.handle] = art
            return art.handle
        def get(self, handle, *, expected_kind=None):
            return self._d[handle]
        def list(self, kind=None):
            return [a.summary() for a in self._d.values()
                    if kind is None or a.kind == kind]

    store = _MockStore()
    reg = build_tool_registry(store, default_registry())
    names = reg.names()

    for expected in (
        "run_rcwa_sweep", "validate_design", "list_templates",
        "get_template", "get_workflow_guide", "get_optimization_tips",
        "generate_phase_mask", "analyze_psf", "export_gds",
    ):
        assert expected in names, f"Missing tool: {expected}"


# ── Test: MetaBox small RCWA (skipped if TF unavailable) ────────────────────


def _metabox_available() -> bool:
    try:
        return MetaboxBackend().is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _metabox_available(), reason="MetaBox/TensorFlow not available")
def test_metabox_small_sweep():
    """Small MetaBox RCWA sweep (5 samples) — physics correctness check."""
    reg = default_registry()
    backend = reg.get("metabox")
    params = MetalensDesignParams(
        wavelength_nm=532.0,
        pillar_height_nm=600.0,
        min_diameter_nm=80.0,
        max_diameter_nm=300.0,
        periodicity_nm=350.0,
        xy_harmonics=(3, 3),
    )
    inc = IncidenceSpec.at_wavelength_nm(532.0)
    sweep_vals = np.linspace(80.0, 300.0, 5)
    result = backend.sweep_unit_cells(params, inc, "diameter", sweep_vals)

    assert result.sweep_values.shape == (5,)
    assert result.amplitudes.shape == (5,)
    assert result.phases_rad.shape == (5,)
    assert result.backend == "metabox"
    assert np.all(result.amplitudes >= 0)
    assert np.all(result.amplitudes <= 1.01)


@pytest.mark.skipif(not _metabox_available(), reason="MetaBox/TensorFlow not available")
def test_metabox_simulate_unit_cell():
    """Single unit-cell simulation returns valid UnitCellResult."""
    backend = MetaboxBackend()
    params = MetalensDesignParams(
        wavelength_nm=532.0,
        pillar_height_nm=600.0,
        min_diameter_nm=80.0,
        max_diameter_nm=300.0,
        periodicity_nm=350.0,
    )
    inc = IncidenceSpec.at_wavelength_nm(532.0)
    result = backend.simulate_unit_cell(params, inc)

    assert 0.0 <= result.amplitude <= 1.01
    assert isinstance(result.phase_rad, float)
    assert result.backend == "metabox"
    assert isinstance(result.tx_complex, complex)
