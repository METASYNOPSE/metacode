"""Optional AI orchestration — Groq-based tool-calling agent.

Import this subpackage only when LLM functionality is needed.
The scientific pipeline in `metaopticsai.physics`, `metaopticsai.tools`,
etc. works without any LLM credentials.
"""
from metaopticsai.orchestration.agent import run_agent, build_registry_and_store, AgentResult
from metaopticsai.orchestration.groq_client import load_env, get_client, get_model_config
from metaopticsai.orchestration.workflow import get_workflow_guide, WORKFLOW_GUIDE

__all__ = [
    "run_agent",
    "build_registry_and_store",
    "AgentResult",
    "load_env",
    "get_client",
    "get_model_config",
    "get_workflow_guide",
    "WORKFLOW_GUIDE",
]
