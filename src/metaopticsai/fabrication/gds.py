"""GDSII layout generation. Wraps gdsfactory.

gdsfactory is imported lazily inside `GDSEngine.generate()` so that importing
this module (and downstream tools) is cheap and doesn't fail in environments
without gdsfactory installed.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import interp1d


def _import_gdsfactory():
    """Return the gdsfactory module, raising a clear error if missing."""
    try:
        import gdsfactory as gf
    except ImportError as e:
        raise ImportError(
            "GDS export requires gdsfactory. Install with:\n"
            "    pip install gdsfactory\n"
            f"(original error: {e})"
        ) from e
    # Activate the generic PDK; safe if it fails.
    try:
        gf.gpdk.PDK.activate()
    except Exception:  # noqa: BLE001
        pass
    return gf


class GDSEngine:
    """Convert a quantized phase mask into a fabrication-ready GDSII file."""

    def __init__(self) -> None:
        self._progress: Optional[Callable[[float], None]] = None

    def set_progress_callback(self, cb: Callable[[float], None]) -> None:
        self._progress = cb

    def _report(self, pct: float) -> None:
        if self._progress is not None:
            self._progress(min(pct, 100.0))

    # ── shape helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _cross_points(length_um: float, width_um: float) -> list[tuple[float, float]]:
        w, l = width_um / 2.0, length_um / 2.0
        return [
            (-w, -l), (w, -l), (w, -w), (l, -w),
            (l, w), (w, w), (w, l), (-w, l),
            (-w, w), (-l, w), (-l, -w), (-w, -w),
        ]

    @staticmethod
    def _fin_points(
        theta_deg: float, fin_length_um: float, fin_width_um: float
    ) -> list[tuple[float, float]]:
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        fl, fw = fin_length_um / 2.0, fin_width_um / 2.0

        def rot(x, y):
            return (x * cos_t + y * sin_t, -x * sin_t + y * cos_t)

        return [rot(-fl, -fw), rot(fl, -fw), rot(fl, fw), rot(-fl, fw)]

    # ── main entry point ──────────────────────────────────────────────────

    def generate(
        self,
        *,
        quantized_mask: np.ndarray,
        bins: list[float],
        dimensions: list[float],
        phases: list[float],
        period: float,                  # nm
        pixel_size_um: float,
        unitcells_per_pixel: int,
        shape: str,                     # "Cylinder" | "Cross" | "Fin"
        is_circular: bool,
        output_path: str,
        cross_width: float = 0.0,
        fin_width: float = 0.0,
        fin_length: float = 0.0,
    ) -> dict[str, float | str]:
        gf = _import_gdsfactory()
        t_start = time.time()

        ph_arr = np.array(phases, dtype=float)
        dim_arr = np.array(dimensions, dtype=float)
        sort_idx = np.argsort(ph_arr)
        ph_sorted = ph_arr[sort_idx]
        dim_sorted = dim_arr[sort_idx]
        # Deduplicate — newer scipy raises on repeated x values in cubic interp.
        _, unique_idx = np.unique(ph_sorted, return_index=True)
        f_interp = interp1d(
            ph_sorted[unique_idx], dim_sorted[unique_idx],
            kind="linear" if len(unique_idx) < 4 else "cubic",
            fill_value="extrapolate",
        )
        ph_min, ph_max = float(ph_arr.min()), float(ph_arr.max())

        rows, cols = quantized_mask.shape
        m = min(rows, cols)
        top = gf.Component(f"METALENS_{uuid.uuid4().hex[:6]}")
        cell_cache: dict[float, "gf.Component"] = {}
        period_um = period / 1000.0

        for i in range(rows):
            self._report(i / rows * 100)
            for j in range(cols):
                if is_circular:
                    ci, cj = i - m / 2, j - m / 2
                    if ci * ci + cj * cj >= (m / 2) ** 2:
                        continue

                level = quantized_mask[rows - 1 - i, j]
                phase_val = bins[level - 1] if level <= len(bins) else bins[-1]
                phase_val = float(np.clip(phase_val, ph_min, ph_max))
                dim_nm = round(float(f_interp(phase_val)), 1)
                dim_um = dim_nm / 1000.0

                if dim_nm not in cell_cache:
                    unit = gf.Component(f"u_{dim_nm}")
                    if shape == "Cylinder" and dim_um > 0.001:
                        unit.add_ref(gf.components.circle(radius=dim_um, layer=(1, 0)))
                    elif shape == "Cross":
                        pts = self._cross_points(dim_um, cross_width / 1000.0)
                        if len(pts) > 2:
                            unit.add_polygon(pts, layer=(1, 0))
                    elif shape == "Fin":
                        pts = self._fin_points(dim_nm, fin_length / 1000.0, fin_width / 1000.0)
                        if len(pts) > 2:
                            unit.add_polygon(pts, layer=(1, 0))

                    arr = gf.Component(f"a_{dim_nm}")
                    for ai in range(unitcells_per_pixel):
                        for aj in range(unitcells_per_pixel):
                            ref = arr.add_ref(unit)
                            ref.dmove((ai * period_um, aj * period_um))
                    cell_cache[dim_nm] = arr

                ref = top.add_ref(cell_cache[dim_nm])
                ref.dmove((j * pixel_size_um, i * pixel_size_um))

        self._report(100)
        if not output_path.endswith(".gds"):
            output_path += ".gds"
        top.write_gds(output_path)

        elapsed = time.time() - t_start
        size_kb = os.path.getsize(output_path) / 1024 if os.path.exists(output_path) else 0
        return {"time_seconds": elapsed, "file_size_kb": size_kb, "filepath": output_path}