"""Tool layer — single composition point for scientific tools.

Use:
    from metaopticsai.tools import build_tool_registry
    registry = build_tool_registry(store, backends, job_manager)

Consumed by direct programmatic access (tests, notebooks) and any external
caller that injects a store implementation.

NOTE: Concrete tool classes are imported lazily inside `build_tool_registry`
so that `from metaopticsai.tools import BaseTool` stays cheap and doesn't
drag in matplotlib/gdspy/scipy until you actually compose a registry.
"""
from __future__ import annotations

from metaopticsai.physics.backends.base import BackendRegistry
from metaopticsai.tools.base import ArtifactStore, BaseTool, ToolRegistry


def build_tool_registry(
    store: ArtifactStore,
    backends: BackendRegistry,
    job_manager=None,
) -> ToolRegistry:
    """Single composition root for the full tool catalog.

    Args:
        store: The ArtifactStore for handle-based payload sharing.
        backends: The simulation backend registry.
        job_manager: Optional JobManager for heavy/long-running tools.
            When None, train_metamodel is not registered.
    """
    # Lazy imports — keep `from metaopticsai.tools import BaseTool` cheap.
    from metaopticsai.tools.rcwa_tools import RunRCWASweepTool, GetFDTDLibraryTool
    from metaopticsai.tools.phase_tools import GeneratePhaseMaskTool
    from metaopticsai.tools.analysis_tools import (
        AnalyzePSFTool, AnalyzeMTFTool, ZernikeDecomposeTool,
    )
    from metaopticsai.tools.propagation_tools import (
        PropagateFieldTool, ComputeMaxIntensityTool,
        ComputeCenterIntensityTool, ComputeMTFVolumeTool,
    )
    from metaopticsai.tools.materials_tools import (
        ListMaterialsTool, GetMaterialIndexTool,
    )
    from metaopticsai.tools.gds_tools import ExportGDSTool
    from metaopticsai.tools.optimization_tools import (
        OptimizeMetalensTool, TrainMetamodelTool,
        GetJobTool, ListArtifactsTool,
    )
    from metaopticsai.tools.design_tools import (
        ValidateDesignTool, ListTemplatesTool, GetTemplateTool,
        GetWorkflowGuideTool, GetOptimizationTipsTool,
    )

    reg = ToolRegistry()

    # ── 0. Design workflow helpers ────────────────────────────────────────
    reg.register(ListTemplatesTool())
    reg.register(GetTemplateTool())
    reg.register(GetWorkflowGuideTool())
    reg.register(ValidateDesignTool())
    reg.register(GetOptimizationTipsTool())

    # ── 1. RCWA & FDTD library ────────────────────────────────────────────
    reg.register(GetFDTDLibraryTool())
    reg.register(RunRCWASweepTool(store, backends))

    # ── 2. Materials ─────────────────────────────────────────────────────
    reg.register(ListMaterialsTool())
    reg.register(GetMaterialIndexTool())

    # ── 3. Phase masks ───────────────────────────────────────────────────
    reg.register(GeneratePhaseMaskTool(store))

    # ── 4. Analysis & propagation ────────────────────────────────────────
    reg.register(AnalyzePSFTool(store))
    reg.register(AnalyzeMTFTool(store))
    reg.register(ZernikeDecomposeTool(store))
    reg.register(PropagateFieldTool(store, backends))
    reg.register(ComputeMaxIntensityTool(store))
    reg.register(ComputeCenterIntensityTool(store))
    reg.register(ComputeMTFVolumeTool(store))

    # ── 5. GDS export ────────────────────────────────────────────────────
    reg.register(ExportGDSTool(store))

    # ── 6. Optimization (light + gradient) ───────────────────────────────
    reg.register(OptimizeMetalensTool(store, backends))

    # ── 7. Heavy jobs (only if job_manager provided) ─────────────────────
    if job_manager is not None:
        reg.register(TrainMetamodelTool(store, backends, job_manager))
        reg.register(GetJobTool(store, job_manager))

    # ── 8. Bookkeeping ───────────────────────────────────────────────────
    reg.register(ListArtifactsTool(store))

    return reg


__all__ = ["build_tool_registry", "BaseTool", "ToolRegistry"]