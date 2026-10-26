"""Tests 1-2: package import and public API without LLM credentials."""
import sys
from pathlib import Path

# Ensure src/ is on the path for editable-without-install scenarios.
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_package_imports():
    """Test 1 — import metaopticsai and core submodules."""
    import metaopticsai
    assert hasattr(metaopticsai, "__version__")

    from metaopticsai.domain.design import MetalensDesignParams
    from metaopticsai.domain.incidence import IncidenceSpec
    from metaopticsai.domain.results import UnitCellResult, RCWASweepResult
    from metaopticsai.domain.templates import list_templates, get_template
    from metaopticsai.configs.settings import settings
    from metaopticsai.physics.backends.base import BackendRegistry
    from metaopticsai.physics.backends.metabox import MetaboxBackend
    from metaopticsai.physics.rcwa import sweep_diameter
    from metaopticsai.phase.profiles import fresnel_zone_lens
    from metaopticsai.analysis.psf import compute_psf
    from metaopticsai.tools.base import BaseTool, ToolRegistry
    assert True


def test_no_llm_credentials_required():
    """Test 2 — core API importable without GROQ_API_KEY."""
    import os
    os.environ.pop("GROQ_API_KEY", None)

    # These must import cleanly with no credentials
    from metaopticsai.domain import MetalensDesignParams
    from metaopticsai.physics.backends import default_registry
    from metaopticsai.tools import build_tool_registry
    assert True
