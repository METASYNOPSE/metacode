"""Material refractive-index lookup. Thin facade over the rcwa_engine CSV tables.

Imports from `physics.rcwa_engine` are lazy so the module loads without TF.
"""
from __future__ import annotations

import logging
from pathlib import Path

from metaopticsai.configs.settings import settings
from metaopticsai.domain.materials import MaterialIndex

log = logging.getLogger(__name__)


def list_materials() -> list[str]:
    """List every refractive-index CSV bundled in `metabox3/material_data`."""
    return sorted(p.stem for p in settings.paths.material_dir.glob("*.csv"))


def get_material_index(name: str, wavelength_nm: float) -> MaterialIndex:
    """Cubic-interpolated `n + ik` at a given wavelength.

    Raises ValueError if `wavelength_nm` is outside the CSV's tabulated range.
    """
    # Lazy import — keeps cold-start fast and lets us swap implementations.
    from metaopticsai.physics.rcwa_engine.rcwa import Material  # type: ignore

    mat = Material(name, custom_csv_dir=str(settings.paths.material_dir))
    wl_m = wavelength_nm * 1e-9
    idx = mat.index_at(wl_m)
    return MaterialIndex(
        name=name,
        wavelength_nm=wavelength_nm,
        n=float(idx.real),
        k=float(idx.imag),
        valid_range_um=(float(mat.min_wl_n * 1e6), float(mat.max_wl_n * 1e6)),
    )


def get_material_csv_path(name: str) -> Path:
    """Return the path to the raw CSV — used by MCP resources."""
    p = settings.paths.material_dir / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Material {name!r} not found. Available: {list_materials()}"
        )
    return p