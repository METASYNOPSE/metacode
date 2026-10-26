"""Groq LLM client setup.

Reads credentials from config/.env or repo-root .env (never overwrites
already-set environment variables).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "llama-3.3-70b-versatile"


def load_env(env_paths: Iterable[Path] | None = None) -> None:
    """Populate os.environ from a .env file if one exists.

    Order: existing os.environ > config/.env > repo-root/.env.
    Hand-parsed KEY=VALUE so python-dotenv is not required.
    """
    if env_paths is None:
        root = Path(__file__).resolve().parents[4]
        env_paths = [root / "config" / ".env", root / ".env"]
    for p in env_paths:
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def get_client():
    """Return a `groq.Groq` instance. Raises RuntimeError if no key."""
    load_env()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy config/.env.example to "
            "config/.env and fill in your Groq key "
            "(https://console.groq.com/keys)."
        )
    from groq import Groq
    return Groq(api_key=api_key)


def get_model_config() -> dict[str, Any]:
    load_env()
    return {
        "model":       os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.1")),
        "max_tokens":  int(os.environ.get("GROQ_MAX_TOKENS", "1024")),
    }
