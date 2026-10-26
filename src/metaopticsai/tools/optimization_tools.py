"""Optimization-related tools: optimize_metalens, train_metamodel,
get_job, list_artifacts."""
from __future__ import annotations

import logging
from pydantic import BaseModel
from typing import Literal

from metaopticsai.configs.settings import settings
from metaopticsai.optimization.heuristic import HeuristicOptimizer
from metaopticsai.physics.backends.base import BackendRegistry
from metaopticsai.tools.base import BaseTool
from metaopticsai.tools.schemas.optimization import (
    OptimizeInput, OptimizeOutput,
    TrainMetamodelInput, JobSubmitOutput,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# OptimizeMetalensTool — analytical heuristic by default
# ─────────────────────────────────────────────────────────────────────────

class OptimizeMetalensTool(BaseTool[OptimizeInput, OptimizeOutput]):
    name = "optimize_metalens"
    description = (
        "Iteratively tune a metalens design to reach the target Strehl ratio. "
        "method='heuristic' is fast; 'gradient' runs the TF-backed path "
        "via metabox3 RCWA."
    )
    Input = OptimizeInput
    Output = OptimizeOutput
    needs_store = True

    def __init__(self, store, backends: BackendRegistry | None = None):
        super().__init__(store=store)
        self.backends = backends   # required only for method=gradient

    def _run(self, p: OptimizeInput) -> OptimizeOutput:
        sweep_art = self.store.get(p.sweep_handle, expected_kind="rcwa_sweep")
        sweep = sweep_art.payload

        # Reconstruct an RCWASweepResult-shaped object for the optimizer.
        # Minimal: only the fields the heuristic optimizer reads.
        from metaopticsai.domain.results import RCWASweepResult
        import numpy as np
        sweep_result = RCWASweepResult(
            sweep_param="diameter",
            sweep_values=np.asarray(sweep["diameters_nm"]),
            amplitudes=np.asarray(sweep["amplitudes"]),
            phases_rad=np.asarray(sweep["phases_rad"]),
            wavelength_m=sweep["wavelength_nm"] * 1e-9,
            phase_coverage=float((max(sweep["phases_rad"]) - min(sweep["phases_rad"])) / (2 * 3.14159265358979)),
            mean_transmission=float(np.mean(sweep["amplitudes"])),
            elapsed_s=0.0,
            backend=sweep_art.metadata.get("backend", "unknown"),
        )

        # Dummy params with the wavelength embedded
        from metaopticsai.domain.design import MetalensDesignParams
        params = MetalensDesignParams(wavelength_nm=sweep["wavelength_nm"])

        if p.method == "heuristic":
            optimizer = HeuristicOptimizer()
        elif p.method == "gradient":
            if self.backends is None:
                log.warning("backends not provided; falling back to heuristic.")
                optimizer = HeuristicOptimizer()
            else:
                try:
                    from metaopticsai.optimization.gradient import GradientOptimizer
                    optimizer = GradientOptimizer(self.backends)
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "Gradient optimizer unavailable (%s); falling back.", e,
                    )
                    optimizer = HeuristicOptimizer()
        elif p.method == "surrogate":
            log.warning("surrogate method requires a trained metamodel; "
                        "falling back to heuristic.")
            optimizer = HeuristicOptimizer()
        else:
            log.warning("Unknown method %r; falling back to heuristic.", p.method)
            optimizer = HeuristicOptimizer()

        _, fom, history = optimizer.optimize(
            params,
            target_fom=p.target_strehl,
            max_iters=p.max_iters,
            learning_rate=p.learning_rate,
            sweep_result=sweep_result,
        )

        handle = self.store.put(
            "metalens_design",
            {
                "sweep_handle": p.sweep_handle,
                "achieved_strehl": fom.value,
                "target_strehl": p.target_strehl,
                "fom_history": history,
                "method": optimizer.name,
            },
            achieved_strehl=fom.value,
            wavelength_nm=sweep["wavelength_nm"],
        )

        return OptimizeOutput(
            handle=handle,
            achieved_strehl=fom.value,
            iterations_run=fom.iteration,
            target_met=fom.value >= p.target_strehl * 0.99,
            method=optimizer.name,
        )


# ─────────────────────────────────────────────────────────────────────────
# Long-running jobs: train_metamodel, submit_job (AutoML), get_job
# ─────────────────────────────────────────────────────────────────────────

class TrainMetamodelTool(BaseTool):
    """Submits a metamodel-training background job. Returns job_id immediately."""
    name = "train_metamodel"
    description = (
        "Train a small neural surrogate of the parameterised RCWA simulator. "
        "Returns a job_id — poll with `get_job`. Heavy: requires TensorFlow."
    )
    Input = TrainMetamodelInput
    Output = JobSubmitOutput

    def __init__(self, store, backends: BackendRegistry, job_manager):
        super().__init__(store=store)
        self.backends = backends
        self.job_manager = job_manager
        self.needs_store = True  # set per-instance; class default is False

    def _run(self, p: TrainMetamodelInput) -> JobSubmitOutput:
        if not self.backends.get("metabox").is_available():
            raise RuntimeError("train_metamodel requires the metabox backend (TF).")

        job_id = self.job_manager.submit(
            "train_metamodel",
            _train_metamodel_worker,
            p.model_dump(),
            metadata={"wavelength_nm": p.wavelength_nm, "material": p.pillar_material},
        )
        return JobSubmitOutput(
            job_id=job_id, status="queued",
            extra={"wavelength_nm": p.wavelength_nm, "material": p.pillar_material},
        )


def _train_metamodel_worker(
    payload: dict, *, _manager, _job_id, **_kwargs,
) -> dict:
    """Worker function — runs inside a background thread.

    Imports metabox3 lazily so a bare submit doesn't hard-fail at import time.
    """
    from metaopticsai.physics.rcwa_engine import rcwa, utils, modeling
    import numpy as np

    _manager.update_status(_job_id, "sampling")

    p = payload
    n_index = {"TiO2": 2.35, "Si": 3.48, "GaN": 2.32, "SiN": 2.0}[p["pillar_material"]]
    diam_feat = utils.Feature(
        vmin=p["min_diameter_nm"] * 1e-9,
        vmax=p["max_diameter_nm"] * 1e-9,
        name="diameter",
        sampling=p["n_samples"],
    )
    pillar = rcwa.Circle(material=n_index, radius=diam_feat)
    layer = rcwa.Layer(material=1.0, thickness=p["pillar_height_nm"] * 1e-9, shapes=(pillar,))
    uc = rcwa.UnitCell(
        layers=[layer],
        periodicity=(p["period_nm"] * 1e-9, p["period_nm"] * 1e-9),
        refl_index=1.0, tran_index=1.46,
    )
    proto = rcwa.ProtoUnitCell(uc)
    incidence = utils.Incidence(wavelength=[p["wavelength_nm"] * 1e-9])
    cfg = rcwa.SimConfig(
        xy_harmonics=(settings.rcwa.harmonics, settings.rcwa.harmonics),
        resolution=settings.rcwa.resolution,
        minibatch_size=settings.rcwa.minibatch_size,
        return_tensor=True, return_zeroth_order=True,
        use_transmission=True, include_z_comp=False,
    )

    sim_lib = modeling.sample_protocell(proto, incidence, cfg)
    _manager.update_status(_job_id, "training")
    mm = modeling.create_and_train_model(sim_lib, n_epochs=p["n_epochs"], verbose=0)

    save_dir = settings.paths.output_dir / "metamodels"
    save_dir.mkdir(parents=True, exist_ok=True)
    name = f"mm_{p['pillar_material']}_{int(p['wavelength_nm'])}nm_{_job_id}"
    mm.save(name=name, path=str(save_dir), overwrite=True)

    return {
        "metamodel_path": str(save_dir / name),
        "wavelength_nm": p["wavelength_nm"],
        "material": p["pillar_material"],
    }


# ─────────────────────────────────────────────────────────────────────────
# Job polling
# ─────────────────────────────────────────────────────────────────────────

class _JobInput(BaseModel):
    job_id: str


class GetJobTool(BaseTool):
    name = "get_job"
    description = (
        "Poll a submitted job by id. status ∈ {queued, sampling, training, "
        "running, done, failed}."
    )
    Input = _JobInput

    def __init__(self, store, job_manager):
        super().__init__(store=store)
        self.job_manager = job_manager
        self.needs_store = False

    def _run(self, p: _JobInput):
        return self.job_manager.get(p.job_id).to_dict()


# ─────────────────────────────────────────────────────────────────────────
# Artifact listing
# ─────────────────────────────────────────────────────────────────────────

class _ListArtifactsInput(BaseModel):
    kind: Literal[
        "phase_mask", "quantized_mask", "rcwa_sweep", "psf", "mtf",
        "zernike", "gds_file", "metalens_design", "field_2d",
        "metamodel", "job_result",
    ] | None = None


class ListArtifactsTool(BaseTool):
    name = "list_artifacts"
    description = (
        "List handles currently in the in-process store, optionally filtered "
        "by kind. Useful for recovering state after a context reset."
    )
    Input = _ListArtifactsInput
    needs_store = True

    def _run(self, p: _ListArtifactsInput):
        return self.store.list(p.kind)
