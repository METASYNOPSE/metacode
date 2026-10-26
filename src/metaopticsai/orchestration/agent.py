"""Groq-based LLM agent wired to the canonical MetaOpticsAI tool registry.

Usage (programmatic):
    from metaopticsai.orchestration.agent import run_agent, build_registry_and_store
    registry, store = build_registry_and_store()
    result = run_agent("Design a TiO2 metalens for 532 nm.", registry=registry)

Usage (CLI):  python -m metaopticsai.orchestration "your prompt"
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a careful meta-optics design assistant. You have access to a
toolbox of verified physics functions backed by a metabox3 RCWA solver.
Never invent parameters — start from a template.

The correct procedure for any design request is:

  1. Call `list_templates` to see verified starting points.
  2. Call `get_template` on the most relevant one.
  3. If you need guidance, call `get_workflow_guide`.
  4. Adapt the template's `params` to the user's request.
  5. Call `validate_design(params=...)`. Fix any reported errors, then
     validate again.
  6. Call `run_rcwa_sweep(...)` with the validated params.
  7. If phase_coverage_2pi_fraction < 0.9 OR mean_transmission < 0.7,
     call `get_optimization_tips(result=...)` and adjust, then re-simulate.

Cap yourself at 5 tool calls total. Once metrics are acceptable OR you
have applied one round of tips, stop and report the final design
parameters and metrics in a short, structured summary.
"""


@dataclass
class AgentResult:
    final_text: str
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stopped_on: str = "final_text"


def _extract_tool_calls(msg) -> list[Any]:
    tc = getattr(msg, "tool_calls", None) or []
    return list(tc)


def build_registry_and_store():
    """Construct a default BackendRegistry, InMemoryArtifactStore, and ToolRegistry.

    Returns (registry, store) so callers can inject them.
    """
    from metaopticsai.physics.backends import default_registry
    from metaopticsai.tools import build_tool_registry

    class _InMemoryStore:
        def __init__(self):
            self._data: dict[str, Any] = {}

        def put(self, kind: str, payload: Any, **metadata: Any) -> str:
            from metaopticsai.domain.handles import Artifact
            art = Artifact.new(kind, payload, **metadata)
            self._data[art.handle] = art
            return art.handle

        def get(self, handle: str, *, expected_kind: str | None = None) -> Any:
            art = self._data.get(handle)
            if art is None:
                raise KeyError(f"No artifact with handle {handle!r}")
            if expected_kind and art.kind != expected_kind:
                raise TypeError(
                    f"Expected kind {expected_kind!r}, got {art.kind!r}"
                )
            return art

        def list(self, kind: str | None = None) -> list:
            arts = list(self._data.values())
            if kind:
                arts = [a for a in arts if a.kind == kind]
            return [a.summary() for a in arts]

    backends = default_registry()
    store = _InMemoryStore()
    tool_reg = build_tool_registry(store, backends)
    return tool_reg, store


def run_agent(
    user_prompt: str,
    *,
    registry=None,
    store=None,
    max_iterations: int | None = None,
) -> AgentResult:
    """Drive the Groq tool-calling loop for a single user request.

    If `registry` is None, a default registry is built automatically.
    """
    from metaopticsai.orchestration.groq_client import get_client, get_model_config, load_env
    from metaopticsai.orchestration.tool_adapter import registry_to_openai_tools

    load_env()
    if max_iterations is None:
        max_iterations = int(os.environ.get("AGENT_MAX_ITERATIONS", "5"))
    max_iterations = max(1, min(max_iterations, 5))

    if registry is None:
        registry, store = build_registry_and_store()

    client = get_client()
    cfg = get_model_config()
    tools_schema = registry_to_openai_tools(registry)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    tool_call_log: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        resp = client.chat.completions.create(
            model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            tools=tools_schema,
            tool_choice="auto",
            messages=messages,
        )
        msg = resp.choices[0].message
        calls = _extract_tool_calls(msg)

        assistant_turn: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    },
                }
                for c in calls
            ]
        messages.append(assistant_turn)

        if not calls:
            return AgentResult(
                final_text=msg.content or "",
                iterations=iteration,
                tool_calls=tool_call_log,
                stopped_on="final_text",
            )

        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError as e:
                args = {}
                result_payload = {"ok": False, "error": f"Malformed JSON: {e}"}
            else:
                result_payload = _dispatch(registry, name, args)

            tool_call_log.append({
                "iteration": iteration,
                "name": name,
                "args": args,
                "result": result_payload,
            })

            llm_visible = _strip_large_arrays(result_payload)
            messages.append({
                "role": "tool",
                "tool_call_id": c.id,
                "name": name,
                "content": json.dumps(llm_visible, default=str)[:6000],
            })

    return AgentResult(
        final_text="[Agent hit iteration cap without a final text response.]",
        iterations=max_iterations,
        tool_calls=tool_call_log,
        stopped_on="iteration_cap",
    )


def _dispatch(registry, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Route a tool call through the canonical registry. Never raises."""
    import asyncio
    try:
        tool = registry.get(name)
    except KeyError:
        return {"ok": False, "error": f"Unknown tool {name!r}. Known: {registry.names()}"}
    try:
        out = tool.call_sync(args)
        if isinstance(out, dict):
            return {"ok": True, **out}
        return {"ok": True, "result": out}
    except Exception as e:  # noqa: BLE001
        log.exception("Tool %r raised", name)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _strip_large_arrays(result: dict[str, Any]) -> dict[str, Any]:
    """Remove large array fields before sending to LLM."""
    STRIP_KEYS = {"sweep_values_nm", "amplitudes", "phases_rad", "_full",
                  "sweep_values", "psf_2d", "intensity_2d"}
    if not isinstance(result, dict):
        return result
    return {k: v for k, v in result.items() if k not in STRIP_KEYS}
