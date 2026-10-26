"""Tests 3, 12-13: canonical tool registry, mocked LLM, no orphaned imports."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest

from metaopticsai.domain.handles import Artifact
from metaopticsai.physics.backends import default_registry
from metaopticsai.tools import build_tool_registry
from metaopticsai.tools.base import ToolRegistry


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


@pytest.fixture
def registry():
    return build_tool_registry(_MockStore(), default_registry())


def test_validate_design_valid(registry):
    tool = registry.get("validate_design")
    result = tool.call_sync({
        "params": {
            "wavelength_nm": 532.0,
            "pillar_material": "TiO2",
            "pillar_height_nm": 600.0,
            "periodicity_nm": 350.0,
            "min_diameter_nm": 80.0,
            "max_diameter_nm": 300.0,
        }
    })
    assert result["valid"] is True
    assert isinstance(result["derived"]["numerical_aperture"], float)


def test_validate_design_invalid(registry):
    tool = registry.get("validate_design")
    result = tool.call_sync({
        "params": {
            "min_diameter_nm": 300.0,
            "max_diameter_nm": 100.0,  # invalid: min > max
        }
    })
    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_list_templates(registry):
    tool = registry.get("list_templates")
    result = tool.call_sync({})
    assert result["count"] >= 3
    names = [t["name"] for t in result["templates"]]
    assert "tio2_metalens_532nm" in names


def test_get_template(registry):
    tool = registry.get("get_template")
    result = tool.call_sync({"name": "tio2_metalens_532nm"})
    assert "params" in result["template"]
    assert result["template"]["wavelength_nm"] == 532.0


def test_get_template_unknown(registry):
    tool = registry.get("get_template")
    with pytest.raises(ValueError):
        tool.call_sync({"name": "no_such_template"})


def test_get_workflow_guide(registry):
    tool = registry.get("get_workflow_guide")
    result = tool.call_sync({})
    assert "guide" in result
    assert len(result["guide"]) > 100


def test_get_optimization_tips_low_coverage(registry):
    tool = registry.get("get_optimization_tips")
    result = tool.call_sync({
        "result": {"phase_coverage_2pi_fraction": 0.5, "mean_transmission": 0.8}
    })
    assert len(result["tips"]) > 0
    assert any("phase coverage" in tip.lower() for tip in result["tips"])


def test_get_optimization_tips_good(registry):
    tool = registry.get("get_optimization_tips")
    result = tool.call_sync({
        "result": {"phase_coverage_2pi_fraction": 0.97, "mean_transmission": 0.90}
    })
    assert any("strong" in tip.lower() or "ready" in tip.lower()
               for tip in result["tips"])


# ── Test 12: mocked LLM workflow ────────────────────────────────────────────


def test_agent_mocked_llm():
    """Test 12 — agent completes without real API calls."""
    from metaopticsai.orchestration.agent import run_agent, build_registry_and_store

    registry, store = build_registry_and_store()

    # Mock the Groq client so no real HTTP is made
    mock_msg = MagicMock()
    mock_msg.content = "The TiO2 metalens design is ready."
    mock_msg.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("metaopticsai.orchestration.agent.build_registry_and_store",
               return_value=(registry, store)):
        with patch("metaopticsai.orchestration.groq_client.get_client",
                   return_value=mock_client):
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                result = run_agent(
                    "Design a TiO2 metalens.",
                    registry=registry,
                )

    assert "ready" in result.final_text.lower()
    assert result.stopped_on == "final_text"
    assert result.iterations == 1


# ── Test 13: no orphaned imports ─────────────────────────────────────────────


def test_no_mcp_server_imports():
    """Test 13 — no remaining imports of deleted mcp_server module."""
    import importlib
    import inspect
    import pkgutil
    import metaopticsai

    for _importer, modname, _ in pkgutil.walk_packages(
        path=metaopticsai.__path__,
        prefix=metaopticsai.__name__ + ".",
        onerror=lambda x: None,
    ):
        if "rcwa_engine" in modname:
            continue
        try:
            mod = importlib.import_module(modname)
            source = inspect.getsource(mod)
        except Exception:
            continue
        assert "from mcp_server" not in source, f"{modname} still imports from mcp_server"
        assert "import mcp_server" not in source, f"{modname} still imports from mcp_server"


def test_no_llm_package_imports():
    """Test 13b — no remaining imports of deleted llm/ top-level package."""
    import importlib
    import inspect
    import pkgutil
    import metaopticsai

    for _importer, modname, _ in pkgutil.walk_packages(
        path=metaopticsai.__path__,
        prefix=metaopticsai.__name__ + ".",
        onerror=lambda x: None,
    ):
        if "rcwa_engine" in modname:
            continue
        try:
            mod = importlib.import_module(modname)
            source = inspect.getsource(mod)
        except Exception:
            continue
        assert "from llm." not in source, f"{modname} still imports from llm/"
        assert "import llm." not in source, f"{modname} still imports from llm/"


def test_no_vendor_metabox3_imports():
    """Test 13c — no remaining imports of the old vendor.metabox3 path."""
    import importlib
    import inspect
    import pkgutil
    import metaopticsai

    for _importer, modname, _ in pkgutil.walk_packages(
        path=metaopticsai.__path__,
        prefix=metaopticsai.__name__ + ".",
        onerror=lambda x: None,
    ):
        if "rcwa_engine" in modname:
            continue
        try:
            mod = importlib.import_module(modname)
            source = inspect.getsource(mod)
        except Exception:
            continue
        assert "vendor.metabox3" not in source, (
            f"{modname} still references vendor.metabox3"
        )
        assert "from metaopticsai.vendor" not in source, (
            f"{modname} still imports from metaopticsai.vendor"
        )
