"""Tests 9-10: phase profile generation and PSF/MTF/Strehl analysis."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pytest

from metaopticsai.phase.profiles import fresnel_zone_lens, axicon, spiral_phase_plate
from metaopticsai.analysis.psf import compute_psf


# ── Test 9: phase profiles ───────────────────────────────────────────────────


def test_fresnel_zone_lens_shape():
    mask = fresnel_zone_lens(
        wavelength_nm=532.0,
        focal_length_um=2000.0,
        diameter_um=500.0,
        pixel_size_um=0.35,
    )
    assert mask.ndim == 2
    assert mask.shape[0] == mask.shape[1]
    # All values in [0, 2π]
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 2 * np.pi + 1e-10


def test_axicon_shape():
    mask = axicon(
        wavelength_nm=532.0,
        cone_angle_deg=2.0,
        diameter_um=300.0,
        pixel_size_um=0.35,
    )
    assert mask.ndim == 2
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 2 * np.pi + 1e-10


def test_spiral_phase_plate():
    mask = spiral_phase_plate(
        charge=1,
        diameter_um=300.0,
        pixel_size_um=0.5,
    )
    assert mask.ndim == 2
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 2 * np.pi + 1e-10


# ── Test 10: PSF/MTF/Strehl ───────────────────────────────────────────────────


def test_psf_flat_phase():
    """Flat phase mask → PSF should peak at centre."""
    mask = np.zeros((32, 32))
    psf, x, y = compute_psf(mask, wavelength_nm=532.0, pixel_size_um=0.35,
                             focal_length_um=1000.0, pad_factor=4)
    assert psf.shape == (128, 128)
    # Normalized to 1
    assert abs(psf.max() - 1.0) < 1e-9
    # Peak should be at or very near centre
    cy, cx = np.unravel_index(psf.argmax(), psf.shape)
    assert abs(cy - 64) <= 2
    assert abs(cx - 64) <= 2


def test_psf_lens_mask():
    """Focusing FZL mask → PSF has a well-defined peak."""
    mask = fresnel_zone_lens(
        wavelength_nm=532.0,
        focal_length_um=2000.0,
        diameter_um=300.0,
        pixel_size_um=0.35,
    )
    psf, x, y = compute_psf(mask, wavelength_nm=532.0, pixel_size_um=0.35,
                             focal_length_um=2000.0)
    assert psf.max() == pytest.approx(1.0, abs=1e-9)
    assert len(x) == psf.shape[1]


def test_strehl_ratio():
    from metaopticsai.analysis.strehl import compute_strehl_ratio
    # Flat phase → Strehl = 1.0
    mask = np.zeros((32, 32))
    strehl, _, _ = compute_strehl_ratio(mask)
    assert strehl == pytest.approx(1.0, abs=1e-6)


def test_mtf_analytical():
    from metaopticsai.analysis.mtf import compute_mtf
    psf_2d = np.zeros((64, 64))
    psf_2d[32, 32] = 1.0
    freqs, mtf, _ = compute_mtf(psf_2d, pixel_size_um=0.35, wavelength_nm=532.0,
                                  focal_length_um=1000.0)
    assert len(freqs) == len(mtf)
    assert mtf[0] == pytest.approx(1.0, abs=0.05)
