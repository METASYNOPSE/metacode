"""MetaboxBackend — the only EM simulation backend in this project.

All imports from `physics.rcwa_engine` are lazy, gated behind `is_available()`.
This keeps cold import fast and lets the package load without TensorFlow.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from metaopticsai.configs.settings import settings
from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.incidence import IncidenceSpec
from metaopticsai.domain.results import (
    FieldResult, RCWASweepResult, UnitCellResult,
)
from metaopticsai.physics.backends.base import SimulationBackend

log = logging.getLogger(__name__)


# Default refractive indices for the four supported pillar materials.
# For wavelength-dependent values, use `physics.materials.get_material_index`.
_DEFAULT_N = {"TiO2": 2.35, "Si": 3.48, "GaN": 2.32, "SiN": 2.0}


class MetaboxBackend(SimulationBackend):
    """Adapter around vendored metabox3. Domain in, domain out."""

    name = "metabox"

    def __init__(self):
        self._available: bool | None = None

    # ── availability ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import tensorflow as tf  # noqa: F401
                from metaopticsai.physics import rcwa_engine  # noqa: F401
                self._available = True
            except Exception as e:  # noqa: BLE001
                log.info("MetaboxBackend unavailable: %s", e)
                self._available = False
        return self._available

    # ── public API ──────────────────────────────────────────────────────────

    def simulate_unit_cell(self, params, incidence):
        self._require()
        from metaopticsai.physics.rcwa_engine import rcwa, utils
        import tensorflow as tf

        proto = self._build_protocell(params)
        inc = utils.Incidence(wavelength=tuple(incidence.wavelength_m))
        cfg = self._build_simconfig(params, n_samples=1)

        diam_m = params.mean_diameter_nm * 1e-9
        param_tensor = tf.constant(
            np.array([[diam_m]], dtype=np.float32)
        )

        t0 = time.perf_counter()
        out = rcwa.simulate_parameterized_unit_cells(param_tensor, proto, inc, cfg)
        # Shape: (n_wavelengths, n_samples, 2 polarizations)
        tx = out[0, 0, 0].numpy()  # x-pol complex t
        ty_arr = out[0, 0, 1].numpy() if out.shape[-1] > 1 else 0+0j

        return UnitCellResult(
            tx_complex=complex(tx),
            ty_complex=complex(ty_arr),
            amplitude=float(np.abs(tx)),
            phase_rad=float(np.angle(tx)),
            elapsed_s=time.perf_counter() - t0,
            backend=self.name,
        )

    def sweep_unit_cells(self, params, incidence, sweep_param, sweep_values):
        self._require()
        from metaopticsai.physics.rcwa_engine import rcwa, utils
        import tensorflow as tf

        if sweep_param != "diameter":
            raise NotImplementedError(
                f"Sweep over {sweep_param!r} not yet supported for MetaboxBackend."
            )

        # Build a fresh protocell whose diameter feature spans the sweep range.
        sweep_min = float(np.min(sweep_values))
        sweep_max = float(np.max(sweep_values))
        proto = self._build_protocell(
            params,
            min_diameter_nm=sweep_min,
            max_diameter_nm=sweep_max,
        )
        inc = utils.Incidence(wavelength=tuple(incidence.wavelength_m))
        cfg = self._build_simconfig(params, n_samples=len(sweep_values))

        param_tensor = tf.constant(
            (sweep_values * 1e-9).astype(np.float32).reshape(1, -1)
        )

        t0 = time.perf_counter()
        out = rcwa.simulate_parameterized_unit_cells(param_tensor, proto, inc, cfg)
        elapsed = time.perf_counter() - t0

        # Extract x-pol complex transmission: shape (1, N, 2) → (N,)
        tx = out[0, :, 0].numpy()
        phases = np.unwrap(np.angle(tx))
        phases = phases - phases[0]  # zero-align
        amps = np.abs(tx)

        return RCWASweepResult(
            sweep_param=sweep_param,
            sweep_values=np.asarray(sweep_values, dtype=float),
            amplitudes=amps,
            phases_rad=phases,
            wavelength_m=incidence.wavelength_m[0],
            phase_coverage=float((phases.max() - phases.min()) / (2 * np.pi)),
            mean_transmission=float(amps.mean()),
            elapsed_s=elapsed,
            backend=self.name,
        )

    def propagate_field(
        self, phase_mask, wavelength_m, pixel_size_m, distance_m,
        refractive_index=1.0,
    ):
        self._require()
        from metaopticsai.physics.rcwa_engine import propagation
        import tensorflow as tf

        n = phase_mask.shape[0]
        field_tensor = tf.cast(
            np.exp(1j * phase_mask)[np.newaxis, :, :], tf.complex64
        )
        field2d = propagation.Field2D(
            tensor=field_tensor, n_pixels=n,
            wavelength=[wavelength_m], theta=[0.0], phi=[0.0],
            period=pixel_size_m,
            upsampling=1, use_padding=True, use_antialiasing=True,
        )
        H = propagation.get_transfer_function(
            field_like=field2d, ref_idx=refractive_index, prop_dist=distance_m,
        )
        out = propagation.propagate(field2d, H)
        intensity = out.get_intensity().numpy()[0, 0]
        peak = float(intensity.max())
        normed = intensity / max(peak, 1e-30)

        return FieldResult(
            intensity_2d=normed,
            peak_intensity=peak,
            shape=tuple(normed.shape),  # type: ignore[arg-type]
            wavelength_m=wavelength_m,
            backend=self.name,
        )

    # ── internal builders ───────────────────────────────────────────────────

    def _require(self) -> None:
        if not self.is_available():
            raise RuntimeError(
                "MetaboxBackend unavailable. Install with `pip install .[tf]`."
            )

    def _build_protocell(
        self,
        params: MetalensDesignParams,
        *,
        min_diameter_nm: float | None = None,
        max_diameter_nm: float | None = None,
    ) -> Any:
        from metaopticsai.physics.rcwa_engine import rcwa, utils

        n_index = _DEFAULT_N.get(params.pillar_material)
        if n_index is None:
            raise ValueError(f"Unknown pillar material {params.pillar_material!r}")

        d_min = (min_diameter_nm or params.min_diameter_nm) * 1e-9
        d_max = (max_diameter_nm or params.max_diameter_nm) * 1e-9

        diam_feat = utils.Feature(
            vmin=d_min, vmax=d_max, name="diameter", sampling=20,
        )
        pillar = rcwa.Circle(material=n_index, radius=diam_feat)
        layer = rcwa.Layer(
            material=1.0,
            thickness=params.pillar_height_nm * 1e-9,
            shapes=(pillar,),
        )
        uc = rcwa.UnitCell(
            layers=[layer],
            periodicity=(params.periodicity_nm * 1e-9,) * 2,
            refl_index=1.0,
            tran_index=params.substrate_index,
        )
        return rcwa.ProtoUnitCell(uc)

    def _build_simconfig(self, params: MetalensDesignParams, n_samples: int):
        from metaopticsai.physics.rcwa_engine import rcwa
        return rcwa.SimConfig(
            xy_harmonics=params.xy_harmonics,
            resolution=settings.rcwa.resolution,
            minibatch_size=min(max(n_samples, 1), settings.rcwa.minibatch_size),
            return_tensor=True,
            return_zeroth_order=True,
            use_transmission=True,
            include_z_comp=False,
        )