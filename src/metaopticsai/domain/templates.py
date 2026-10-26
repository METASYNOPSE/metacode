"""Verified meta-atom design templates.

Each template is a fully-validated starting point for a design task. Parameters
map directly onto `MetalensDesignParams`, which re-validates before any physics call.

Adding a new template: append a dict with the required keys. It is guaranteed to
load through `MetalensDesignParams(**template["params"])`.
"""
from __future__ import annotations

from typing import Any


TEMPLATES: dict[str, dict[str, Any]] = {
    "tio2_metalens_532nm": {
        "name": "tio2_metalens_532nm",
        "summary": "TiO2 nanopillar metalens, green visible (532 nm).",
        "wavelength_nm": 532.0,
        "material": "TiO2",
        "shape": "cylinder",
        "notes": (
            "Direct derivative of Khorasaninejad et al. 2016 (Science 352). "
            "Height 600 nm, period 350 nm. Diameter sweep 80–300 nm gives "
            "full 2*pi phase coverage."
        ),
        "params": {
            "wavelength_nm": 532.0,
            "focal_length_mm": 2.0,
            "diameter_mm": 0.5,
            "pillar_material": "TiO2",
            "pillar_height_nm": 600.0,
            "min_diameter_nm": 80.0,
            "max_diameter_nm": 300.0,
            "periodicity_nm": 350.0,
        },
    },

    "si_nir_metalens_850nm": {
        "name": "si_nir_metalens_850nm",
        "summary": "Silicon cylinder metalens, near-IR (850 nm).",
        "wavelength_nm": 850.0,
        "material": "Si",
        "shape": "cylinder",
        "notes": (
            "Silicon high-index disks; height 475 nm, period 390 nm. "
            "Compatible with CMOS process. Diameter range 60-130 nm "
            "provides 2*pi coverage."
        ),
        "params": {
            "wavelength_nm": 850.0,
            "focal_length_mm": 3.0,
            "diameter_mm": 1.0,
            "pillar_material": "Si",
            "pillar_height_nm": 475.0,
            "min_diameter_nm": 60.0,
            "max_diameter_nm": 130.0,
            "periodicity_nm": 390.0,
        },
    },

    "gan_uv_metalens_405nm": {
        "name": "gan_uv_metalens_405nm",
        "summary": "GaN pillar metalens for near-UV (405 nm).",
        "wavelength_nm": 405.0,
        "material": "GaN",
        "shape": "cylinder",
        "notes": (
            "GaN is transparent down to ~360 nm and has n~2.5 at 405 nm. "
            "Short wavelength requires small periodicity (~200 nm) and "
            "pillar heights ~400-500 nm."
        ),
        "params": {
            "wavelength_nm": 405.0,
            "focal_length_mm": 1.0,
            "diameter_mm": 0.3,
            "pillar_material": "GaN",
            "pillar_height_nm": 450.0,
            "min_diameter_nm": 40.0,
            "max_diameter_nm": 170.0,
            "periodicity_nm": 200.0,
        },
    },
}


def list_templates() -> list[dict[str, Any]]:
    """Compact catalog for browsing available templates."""
    return [
        {
            "name": t["name"],
            "summary": t["summary"],
            "wavelength_nm": t["wavelength_nm"],
            "material": t["material"],
            "shape": t["shape"],
        }
        for t in TEMPLATES.values()
    ]


def get_template(name: str) -> dict[str, Any]:
    """Full template body (parameters + provenance) by name."""
    if name not in TEMPLATES:
        raise KeyError(
            f"Unknown template {name!r}. Known: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[name]
