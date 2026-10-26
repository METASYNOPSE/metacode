"""MetaOpticsAI end-to-end demo.

Exercises the canonical scientific pipeline directly (no LLM required), or
runs the full Groq-backed agent when GROQ_API_KEY is set.

Usage:
    python examples/demo.py --no-llm          # tool dispatch only
    python examples/demo.py                    # live agent run
    python examples/demo.py --prompt "..."     # custom prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from metaopticsai.orchestration.agent import build_registry_and_store
from metaopticsai.orchestration.groq_client import load_env


DEFAULT_PROMPT = (
    "Design a TiO2 metalens for green light (532 nm). Start from a "
    "verified template, validate it, run a 20-diameter RCWA sweep, "
    "and report phase coverage and mean transmission."
)


def dry_run() -> int:
    """Exercise the canonical tool registry without an LLM."""
    print("=== dry-run (no LLM) ===")

    registry, store = build_registry_and_store()

    from metaopticsai.physics.backends import default_registry
    backends = default_registry()
    print(f"backends available: {backends.available()}\n")

    def call(name, args=None):
        tool = registry.get(name)
        return tool.call_sync(args or {})

    # 1. list templates
    r = call("list_templates")
    print("[list_templates]")
    for t in r.get("templates", []):
        print(f"  - {t['name']:32s} {t['material']:5s} @ {t['wavelength_nm']:.0f} nm")

    # 2. get template
    r = call("get_template", {"name": "tio2_metalens_532nm"})
    params = r["template"]["params"]
    print(f"\n[get_template] wavelength_nm={params['wavelength_nm']}, "
          f"material={params['pillar_material']}, D={params['diameter_mm']} mm")

    # 3. validate
    r = call("validate_design", {"params": params})
    print(f"\n[validate_design] valid={r['valid']}, "
          f"NA={r['derived']['numerical_aperture']:.3f}, "
          f"aspect={r['derived']['aspect_ratio']:.2f}")
    for w in r.get("warnings", []):
        print(f"  warn: {w}")

    # 4. simulate
    sweep_input = {
        "wavelength_nm": params["wavelength_nm"],
        "pillar_material": params["pillar_material"],
        "pillar_height_nm": params["pillar_height_nm"],
        "period_nm": params["periodicity_nm"],
        "min_diameter_nm": params["min_diameter_nm"],
        "max_diameter_nm": params["max_diameter_nm"],
        "n_samples": 20,
        "backend": "auto",
    }
    try:
        r = call("run_rcwa_sweep", sweep_input)
        print(f"\n[run_rcwa_sweep] backend={r['backend']}, "
              f"n_samples={r['n_samples']}, "
              f"phase_coverage={r['phase_coverage_2pi_fraction']:.2%}, "
              f"mean_transmission={r['mean_transmission']:.2%}, "
              f"elapsed={r['elapsed_s']:.2f} s")
        sim_result = r
    except Exception as e:
        print(f"\n[run_rcwa_sweep] FAILED: {e}")
        sim_result = {"phase_coverage_2pi_fraction": 0.0, "mean_transmission": 0.0}

    # 5. tips
    r = call("get_optimization_tips", {"result": sim_result})
    print("\n[get_optimization_tips]")
    for tip in r.get("tips", []):
        print(f"  - {tip}")

    print("\n=== dry-run complete ===")
    return 0


def live_run(prompt: str, show_tools: bool = True) -> int:
    from metaopticsai.orchestration.agent import run_agent
    registry, store = build_registry_and_store()
    try:
        result = run_agent(prompt, registry=registry, store=store)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if show_tools:
        print("\n--- tool calls ---")
        for i, tc in enumerate(result.tool_calls, 1):
            print(f"\n[{i}] {tc['name']}  (iter {tc['iteration']})")
            print(f"    args: {json.dumps(tc['args'])[:220]}")
            r = tc["result"]
            if isinstance(r, dict) and r.get("ok") is False:
                print(f"    ERROR: {r.get('error')}")
            elif tc["name"] == "run_rcwa_sweep" and r.get("ok"):
                print(f"    result: backend={r.get('backend')}, "
                      f"cov={r.get('phase_coverage_2pi_fraction', 0):.2%}, "
                      f"trans={r.get('mean_transmission', 0):.2%}")
            else:
                short = json.dumps(r, default=str)
                print(f"    result: {short[:220]}" + ("..." if len(short) > 220 else ""))

    print("\n=== Agent response ===")
    print(result.final_text)
    print(f"\n(iterations={result.iterations}, "
          f"stopped_on={result.stopped_on}, "
          f"tool_calls={len(result.tool_calls)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip the LLM; exercise the tools directly.")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                    help="Prompt for the agent (live mode only).")
    ap.add_argument("--quiet", action="store_true",
                    help="Do not print per-tool trace in live mode.")
    args = ap.parse_args()

    load_env()

    if args.no_llm:
        return dry_run()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set — falling back to --no-llm dry run.")
        print("(Set it in config/.env or the repo-root .env to run live.)\n")
        return dry_run()

    return live_run(args.prompt, show_tools=not args.quiet)


if __name__ == "__main__":
    sys.exit(main())
