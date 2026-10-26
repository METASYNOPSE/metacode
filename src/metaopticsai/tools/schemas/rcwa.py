"""Schemas for RCWA tools."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RCWASweepInput(BaseModel):
    wavelength_nm: float = Field(..., gt=200, lt=20000,
        description="Design wavelength in nm, e.g. 532 for green.")
    pillar_material: Literal["TiO2", "Si", "GaN", "SiN"] = Field("TiO2",
        description="Meta-atom material. TiO2=visible, Si=NIR, GaN=UV/blue, SiN=CMOS visible.")
    pillar_height_nm: float = Field(600.0, gt=50, lt=3000,
        description="Pillar height in nm. Increase to widen phase coverage.")
    period_nm: float = Field(350.0, gt=100, lt=2000,
        description="Square unit-cell period. 0.6–1.1× wavelength is the safe regime.")
    min_diameter_nm: float = Field(50.0, gt=10, lt=2000,
        description="Smallest pillar diameter to sweep (fab limit ~80 nm).")
    max_diameter_nm: float = Field(160.0, gt=20, lt=2000,
        description="Largest pillar diameter; should be ≤ 0.85 × period.")
    n_samples: int = Field(20, ge=5, le=100,
        description="Number of points in the diameter sweep.")
    xy_harmonics: int = Field(3, ge=1, le=11,
        description="RCWA harmonics per axis. 3 is fast; 5–7 for high-NA designs.")
    backend: Literal["auto", "metabox"] = Field("auto",
        description="Simulation backend. Only 'metabox' (RCWA) is supported.")


class RCWASweepOutput(BaseModel):
    handle: str
    phase_coverage_2pi_fraction: float
    mean_transmission: float
    elapsed_s: float
    n_samples: int
    backend: str
    advice: str
