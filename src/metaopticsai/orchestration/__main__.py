"""CLI: python -m metaopticsai.orchestration "your prompt"."""
from __future__ import annotations

import argparse
import json
import sys

from metaopticsai.orchestration.agent import run_agent
from metaopticsai.orchestration.groq_client import load_env


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="metaopticsai.orchestration",
        description="Run the MetaOpticsAI design agent for one prompt.",
    )
    ap.add_argument("prompt", nargs="+", help="User request in natural language.")
    ap.add_argument("--max-iterations", type=int, default=None,
                    help="Override AGENT_MAX_ITERATIONS (capped at 5).")
    ap.add_argument("--show-tools", action="store_true",
                    help="Print each tool call and result.")
    args = ap.parse_args()

    load_env()
    prompt = " ".join(args.prompt)
    try:
        r = run_agent(prompt, max_iterations=args.max_iterations)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.show_tools:
        for i, tc in enumerate(r.tool_calls, 1):
            print(f"\n--- tool call {i} (iter {tc['iteration']}) ---")
            print(f"name: {tc['name']}")
            print(f"args: {json.dumps(tc['args'], indent=2)[:400]}")
            print(f"result: {json.dumps(tc['result'], default=str)[:400]}")

    print("\n=== Agent response ===")
    print(r.final_text)
    print(f"\n(iterations: {r.iterations}, stopped_on: {r.stopped_on}, "
          f"tool_calls: {len(r.tool_calls)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
