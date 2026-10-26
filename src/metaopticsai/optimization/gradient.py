"""Gradient-based optimizer — TensorFlow gradient tape through metabox3.

Builds a `LensAssembly` with a single `Metasurface` whose meta-atom diameter
is marked variable, then runs `optimize_single_lens_assembly` with
FOM = LOG_STREHL_RATIO.  Adam is the default optimizer; the LR is expressed
in SI metres because metabox3 stores diameters in metres.

MetaBox assembly API (physics/rcwa_engine/assembly.py):
  Metasurface(diameter, refractive_index, thickness, proto_unit_cell, ...)
  LensAssembly(surfaces, incidence, figure_of_merit)
  optimize_single_lens_assembly(lens_assembly, optimizer, n_iter)
"""
from __future__ import annotations

import logging

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.results import FOMResult
from metaopticsai.optimization.base import Optimizer
from metaopticsai.physics.backends.base import BackendRegistry

log = logging.getLogger(__name__)


class GradientOptimizer(Optimizer):
    """Differentiable optimizer running through the metabox3 assembly module."""

    name = "gradient"

    def __init__(self, backends: BackendRegistry):
        self.backends = backends

    def optimize(
        self,
        params: MetalensDesignParams,
        *,
        target_fom: float,
        max_iters: int,
        learning_rate: float = 2e-9,
        **kwargs,
    ) -> tuple[MetalensDesignParams, FOMResult, list[float]]:
        try:
            backend = self.backends.get("metabox")
            if not backend.is_available():
                raise RuntimeError("metabox backend not available")
        except Exception as e:
            raise RuntimeError(
                "GradientOptimizer requires the metabox backend (TensorFlow)."
            ) from e

        try:
            import tensorflow as tf
            from metaopticsai.physics.rcwa_engine import rcwa, utils, assembly
        except ImportError as e:
            raise RuntimeError(
                "GradientOptimizer requires `tensorflow` and the metabox3 "
                "RCWA engine (`pip install .[tf]`)."
            ) from e

        # ── Build proto unit cell with a variable diameter feature ────────────
        n_index_map = {"TiO2": 2.35, "Si": 3.48, "GaN": 2.32, "SiN": 2.0}
        n_pillar = n_index_map[params.pillar_material]

        diam_feature = utils.Feature(
            vmin=params.min_diameter_nm * 1e-9,
            vmax=params.max_diameter_nm * 1e-9,
            name="diameter",
            sampling=int(kwargs.get("n_sweep", 30)),
        )
        pillar = rcwa.Circle(material=n_pillar, radius=diam_feature)
        layer = rcwa.Layer(
            material=1.0,
            thickness=params.pillar_height_nm * 1e-9,
            shapes=(pillar,),
        )
        period_m = params.periodicity_nm * 1e-9
        uc = rcwa.UnitCell(
            layers=[layer],
            periodicity=(period_m, period_m),
            refl_index=1.0,
            tran_index=params.substrate_index,
        )
        proto = rcwa.ProtoUnitCell(uc)
        incidence = utils.Incidence(wavelength=[params.wavelength_nm * 1e-9])

        # ── Build a Metasurface using the correct assembly.Metasurface API ────
        #
        # assembly.Metasurface inherits from Surface(diameter, refractive_index,
        # thickness) where:
        #   diameter          = aperture in metres
        #   refractive_index  = exit medium (1.0 = air)
        #   thickness         = propagation distance to focal plane in metres
        try:
            ms = assembly.Metasurface(
                diameter=params.diameter_mm * 1e-3,
                refractive_index=1.0,
                thickness=params.focal_length_mm * 1e-3,
                proto_unit_cell=proto,
                set_structures_variable=True,
            )
            # figure_of_merit is set on LensAssembly, not on optimize_single_lens_assembly
            lens_assembly = assembly.LensAssembly(
                surfaces=[ms],
                incidence=incidence,
                figure_of_merit=assembly.FigureOfMerit.LOG_STREHL_RATIO,
            )
        except Exception as e:
            log.warning(
                "metabox3 assembly construction failed (%s) — returning unoptimized.", e,
            )
            return params, FOMResult(
                value=0.0, name="strehl_ratio", backend="metabox",
                iteration=0, metadata={"method": self.name, "error": str(e)},
            ), []

        # ── Run the gradient loop ─────────────────────────────────────────────
        tf_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        try:
            loss_history = assembly.optimize_single_lens_assembly(
                lens_assembly,
                optimizer=tf_optimizer,
                n_iter=max_iters,
                verbose=0,
            )
        except Exception as e:
            log.exception("metabox3 optimization failed")
            return params, FOMResult(
                value=0.0, name="strehl_ratio", backend="metabox",
                iteration=0, metadata={"method": self.name, "error": str(e)},
            ), []

        history = [
            float(tf.exp(l).numpy()) if hasattr(l, "numpy") else float(l)
            for l in loss_history
        ]
        final = history[-1] if history else 0.0

        return params, FOMResult(
            value=final,
            name="strehl_ratio",
            backend="metabox",
            iteration=len(history),
            metadata={"method": self.name, "learning_rate": learning_rate},
        ), history
