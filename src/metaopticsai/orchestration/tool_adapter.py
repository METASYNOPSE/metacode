"""Convert canonical BaseTool instances into JSON-Schema descriptors
for Groq/OpenAI tool-calling APIs.

This is a one-way adapter — the canonical tools know nothing about Groq;
the adapter sits entirely outside the scientific layer.
"""
from __future__ import annotations

import json
from typing import Any

from metaopticsai.tools.base import BaseTool, ToolRegistry


def _pydantic_to_openai_params(model_cls) -> dict[str, Any]:
    """Convert a Pydantic model's JSON schema to the OpenAI parameter shape."""
    try:
        schema = model_cls.model_json_schema()
    except Exception:
        return {"type": "object", "properties": {}, "required": []}

    props: dict[str, Any] = {}
    required: list[str] = schema.get("required", [])

    for field_name, field_schema in schema.get("properties", {}).items():
        # Unwrap $defs references if needed
        if "$ref" in field_schema:
            ref_key = field_schema["$ref"].split("/")[-1]
            defs = schema.get("$defs", {})
            field_schema = defs.get(ref_key, field_schema)

        param: dict[str, Any] = {}
        if "description" in field_schema:
            param["description"] = field_schema["description"]
        if "type" in field_schema:
            param["type"] = field_schema["type"]
        elif "anyOf" in field_schema:
            types = [t.get("type") for t in field_schema["anyOf"] if "type" in t]
            if types:
                param["type"] = types[0]
        if "enum" in field_schema:
            param["enum"] = field_schema["enum"]
        if "default" in field_schema:
            param["default"] = field_schema["default"]
        if "minimum" in field_schema:
            param["minimum"] = field_schema["minimum"]
        if "maximum" in field_schema:
            param["maximum"] = field_schema["maximum"]
        props[field_name] = param

    return {"type": "object", "properties": props, "required": required}


def tool_to_openai_descriptor(tool: BaseTool) -> dict[str, Any]:
    """Convert a BaseTool to an OpenAI/Groq tool descriptor dict."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": _pydantic_to_openai_params(tool.Input),
        },
    }


def registry_to_openai_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Convert all tools in a registry to OpenAI tool descriptors."""
    return [tool_to_openai_descriptor(t) for t in registry]
