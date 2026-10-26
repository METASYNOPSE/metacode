"""Domain types pure data - no I/O no compute"""

from metaopticsai.domain.handles import Artifact, ArtifactKind
from metaopticsai.domain.design import MetalensDesignParams, PillarMaterial
from metaopticsai.domain.incidence import IncidenceSpec
from metaopticsai.domain.results import (
    UnitCellResult, RCWASweepResult, FieldResult, PSFResult, FOMResult,
)
from metaopticsai.domain.materials import MaterialIndex
from metaopticsai.domain.state import DesignState
from metaopticsai.domain.templates import list_templates, get_template, TEMPLATES


__all__ = [
    "Artifact", "ArtifactKind",
    "MetalensDesignParams", "PillarMaterial",
    "IncidenceSpec",
    "UnitCellResult", "RCWASweepResult", "FieldResult", "PSFResult", "FOMResult",
    "MaterialIndex",
    "DesignState",
    "TEMPLATES", "list_templates", "get_template",
]