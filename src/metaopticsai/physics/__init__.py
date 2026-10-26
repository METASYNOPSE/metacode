"""Physics layer — RCWA simulation backend + high-level facades.

Public API:
    from metaopticsai.physics.rcwa import sweep_diameter, get_registry
    from metaopticsai.physics.propagation import propagate
    from metaopticsai.physics.materials import list_materials, get_material_index
"""
from metaopticsai.physics import rcwa, propagation, materials
from metaopticsai.physics.backends import (
    SimulationBackend, BackendRegistry, default_registry,
    MetaboxBackend,
)

__all__ = [
    "rcwa", "propagation", "materials",
    "SimulationBackend", "BackendRegistry", "default_registry",
    "MetaboxBackend",
]
