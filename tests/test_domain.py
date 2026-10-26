"""Tests 4-5: design parameter validation and template loading."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest
from pydantic import ValidationError

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.incidence import IncidenceSpec
from metaopticsai.domain.templates import list_templates, get_template, TEMPLATES


def test_design_params_defaults():
    """Test 4a — default MetalensDesignParams validates."""
    p = MetalensDesignParams()
    assert p.wavelength_nm == 532.0
    assert p.pillar_material == "TiO2"
    assert 0 < p.numerical_aperture < 1
    assert p.f_number > 0
    assert p.mean_diameter_nm == (p.min_diameter_nm + p.max_diameter_nm) / 2


def test_design_params_constraints():
    """Test 4b — cross-field validator catches bad geometry."""
    with pytest.raises(ValidationError):
        MetalensDesignParams(min_diameter_nm=200, max_diameter_nm=100)

    with pytest.raises(ValidationError):
        # max_diameter >= periodicity - 20
        MetalensDesignParams(max_diameter_nm=340, periodicity_nm=350)


def test_design_params_from_dict():
    """Test 4c — tolerant from_dict ignores extra keys."""
    d = {
        "wavelength_nm": 850.0,
        "pillar_material": "Si",
        "extra_key": "ignored",
    }
    p = MetalensDesignParams.from_dict(d)
    assert p.wavelength_nm == 850.0
    assert p.pillar_material == "Si"


def test_templates():
    """Test 4d — all templates load and pass Pydantic validation."""
    tpl = list_templates()
    assert len(tpl) >= 3

    for t in tpl:
        full = get_template(t["name"])
        params = full["params"]
        p = MetalensDesignParams(**params)
        assert p.wavelength_nm == full["wavelength_nm"]

    with pytest.raises(KeyError):
        get_template("nonexistent_template")


def test_incidence_spec():
    """Test 4e — IncidenceSpec normal-incidence construction."""
    inc = IncidenceSpec.at_wavelength_nm(532.0)
    assert abs(inc.wavelength_m[0] - 532e-9) < 1e-15
    assert inc.theta_deg == [0.0]
