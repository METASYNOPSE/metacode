"""Simulation backends — MetaBox RCWA is the only EM simulator."""
from metaopticsai.physics.backends.base import SimulationBackend, BackendRegistry
from metaopticsai.physics.backends.metabox import MetaboxBackend


def default_registry() -> BackendRegistry:
    """Build a registry with the MetaBox RCWA backend."""
    reg = BackendRegistry()
    reg.register(MetaboxBackend())
    return reg


__all__ = [
    "SimulationBackend", "BackendRegistry",
    "MetaboxBackend",
    "default_registry",
]
