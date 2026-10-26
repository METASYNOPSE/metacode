"""Design workflow tools — validation, templates, optimization tips, and workflow guide.

These bridge the LLM agent with the canonical MetalensDesignParams lifecycle,
providing the same guarantees as the physics tools: never raise past the boundary,
always return structured results.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.templates import get_template, list_templates
from metaopticsai.tools.base import BaseTool


# ── Validate Design ─────────────────────────────────────────────────────────


class ValidateDesignInput(BaseModel):
    params: dict[str, Any] = Field(
        ...,
        description=(
            "Full MetalensDesignParams dict: wavelength_nm, focal_length_mm, "
            "diameter_mm, pillar_material, pillar_height_nm, min_diameter_nm, "
            "max_diameter_nm, periodicity_nm."
        ),
    )


class ValidateDesignOutput(BaseModel):
    valid: bool
    errors: list[dict[str, str]]
    warnings: list[str]
    derived: dict[str, float]


class ValidateDesignTool(BaseTool[ValidateDesignInput, ValidateDesignOutput]):
    name = "validate_design"
    description = (
        "Validate a candidate design against MetalensDesignParams constraints. "
        "Always call this before run_simulation or run_rcwa_sweep."
    )
    Input = ValidateDesignInput
    Output = ValidateDesignOutput

    def _run(self, p: ValidateDesignInput) -> ValidateDesignOutput:
        warnings: list[str] = []
        try:
            design = MetalensDesignParams.from_dict(p.params)
        except ValidationError as e:
            errs = [
                {"field": ".".join(str(loc) for loc in err["loc"]), "msg": err["msg"]}
                for err in e.errors()
            ]
            return ValidateDesignOutput(
                valid=False, errors=errs, warnings=[], derived={},
            )

        if design.aspect_ratio > 10:
            warnings.append(
                f"Aspect ratio {design.aspect_ratio:.1f} > 10 may be hard to fabricate."
            )
        if design.numerical_aperture > 0.85:
            warnings.append(
                f"NA={design.numerical_aperture:.2f} is very high; phase "
                "quantization will reduce Strehl."
            )
        return ValidateDesignOutput(
            valid=True,
            errors=[],
            warnings=warnings,
            derived={
                "numerical_aperture": design.numerical_aperture,
                "f_number": design.f_number,
                "aspect_ratio": design.aspect_ratio,
                "mean_diameter_nm": design.mean_diameter_nm,
            },
        )


# ── List Templates ───────────────────────────────────────────────────────────


class _Empty(BaseModel):
    pass


class ListTemplatesOutput(BaseModel):
    templates: list[dict[str, Any]]
    count: int


class ListTemplatesTool(BaseTool[_Empty, ListTemplatesOutput]):
    name = "list_templates"
    description = (
        "List all verified starting-point meta-atom templates "
        "(name, wavelength, material, shape). Call this first when "
        "the user requests a new design."
    )
    Input = _Empty
    Output = ListTemplatesOutput

    def _run(self, _p: _Empty) -> ListTemplatesOutput:
        tpl = list_templates()
        return ListTemplatesOutput(templates=tpl, count=len(tpl))


# ── Get Template ─────────────────────────────────────────────────────────────


class GetTemplateInput(BaseModel):
    name: str = Field(..., description="Template name from list_templates.")


class GetTemplateOutput(BaseModel):
    template: dict[str, Any]


class GetTemplateTool(BaseTool[GetTemplateInput, GetTemplateOutput]):
    name = "get_template"
    description = "Return the full parameter dict for one verified design template."
    Input = GetTemplateInput
    Output = GetTemplateOutput

    def _run(self, p: GetTemplateInput) -> GetTemplateOutput:
        try:
            return GetTemplateOutput(template=get_template(p.name))
        except KeyError as e:
            raise ValueError(str(e)) from e


# ── Get Workflow Guide ───────────────────────────────────────────────────────


class GetWorkflowGuideOutput(BaseModel):
    guide: str


class GetWorkflowGuideTool(BaseTool[_Empty, GetWorkflowGuideOutput]):
    name = "get_workflow_guide"
    description = (
        "Return the canonical five-stage meta-optics design workflow. "
        "Consult this whenever unsure which tool to call next."
    )
    Input = _Empty
    Output = GetWorkflowGuideOutput

    def _run(self, _p: _Empty) -> GetWorkflowGuideOutput:
        from metaopticsai.orchestration.workflow import WORKFLOW_GUIDE
        return GetWorkflowGuideOutput(guide=WORKFLOW_GUIDE)


# ── Get Optimization Tips ────────────────────────────────────────────────────


class GetOptimizationTipsInput(BaseModel):
    result: dict[str, Any] = Field(
        ...,
        description=(
            "The result object returned by run_simulation or run_rcwa_sweep. "
            "Must contain phase_coverage and mean_transmission."
        ),
    )


class GetOptimizationTipsOutput(BaseModel):
    tips: list[str]


class GetOptimizationTipsTool(BaseTool[GetOptimizationTipsInput, GetOptimizationTipsOutput]):
    name = "get_optimization_tips"
    description = (
        "Given a simulation result, return concrete suggestions for improving "
        "phase coverage or transmission."
    )
    Input = GetOptimizationTipsInput
    Output = GetOptimizationTipsOutput

    def _run(self, p: GetOptimizationTipsInput) -> GetOptimizationTipsOutput:
        r = p.result
        cov = float(r.get("phase_coverage", r.get("phase_coverage_2pi_fraction", 0.0)))
        trans = float(r.get("mean_transmission", 0.0))
        tips: list[str] = []

        if cov < 0.9:
            tips.append(
                f"Phase coverage {cov:.0%} < 90%: widen the diameter sweep range "
                "(reduce min_diameter_nm and/or increase max_diameter_nm), or "
                "increase pillar_height_nm to accumulate more propagation phase."
            )
        if trans < 0.7:
            tips.append(
                f"Mean transmission {trans:.0%} < 70%: pillars may be too tall "
                "or too close to the unit-cell wall. Reduce pillar_height_nm or "
                "cap max_diameter_nm to about (periodicity_nm - 40)."
            )
        if cov >= 0.95 and trans >= 0.85:
            tips.append(
                "Both metrics are strong; the design is ready to move to phase-"
                "mask synthesis and GDS export."
            )
        if not tips:
            tips.append(
                "Metrics are borderline. Try increasing n_samples to 30-40 for a "
                "smoother phase curve, then re-simulate."
            )
        return GetOptimizationTipsOutput(tips=tips)
