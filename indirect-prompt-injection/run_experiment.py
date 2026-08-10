"""Entry point for module 1.

Backends, in order of what their numbers are worth:

  --backend replay   reproduce the committed results from the cassette. No key,
                     no cost. This is what CI and a fresh reader run.
  --backend record   run against a real model AND write every response to the
                     cassette, so a live run becomes reproducible.
  --backend live     run against a real model without recording.
  --backend scripted harness smoke test only -- exercises every code path with a
                     rule-based fake. Its numbers measure the harness, not a model,
                     and are refused a place in results/ (see --allow-scripted).

Typical use:

  # First real run, recorded once:
  ANTHROPIC_API_KEY=... python run_experiment.py --backend record --seeds 12

  # Everyone afterwards, including CI:
  python run_experiment.py --backend replay
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # make agent_lab importable when run directly

from rag_injection.experiment import Cell, run_cell
from rag_injection.scenarios import ALL_SCENARIOS, BY_KEY

from agent_lab.backends import CassetteBackend, build_backend
from agent_lab.defenses import ALL_DEFENSES
from agent_lab.metrics import Rate

CORPUS = HERE / "corpus"
WORKSPACE = HERE / "workspace"
RESULTS = HERE / "results"
DEFAULT_CASSETTE = HERE / "cassettes" / "opus-5.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--backend",
        choices=["replay", "record", "live", "scripted"],
        default="replay",
        help="which model backend to use (default: replay from the cassette)",
    )
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--seeds", type=int, default=12, help="repetitions per cell")
    p.add_argument("--cassette", type=Path, default=DEFAULT_CASSETTE)
    p.add_argument(
        "--scenario",
        action="append",
        choices=list(BY_KEY),
        help="restrict to specific scenarios (repeatable)",
    )
    p.add_argument(
        "--allow-scripted-results",
        action="store_true",
        help="permit writing results/ from the scripted backend (harness, not a model)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = [BY_KEY[k] for k in args.scenario] if args.scenario else ALL_SCENARIOS

    backend = build_backend(mode=args.backend, model=args.model, cassette=args.cassette)

    cells: list[Cell] = []
    for scenario in scenarios:
        for defense in ALL_DEFENSES:
            for seed in range(args.seeds):
                cells.append(
                    run_cell(
                        backend=backend,
                        corpus=CORPUS,
                        workspace=WORKSPACE,
                        scenario=scenario,
                        defense_name=defense,
                        seed=seed,
                    )
                )

    if isinstance(backend, CassetteBackend):
        if backend.recording:
            backend.save()
            print(f"recorded {len(backend._entries)} responses to {args.cassette}")
        else:
            print(f"replayed {backend.hits} responses ({backend.misses} misses)")

    write_outputs(cells, backend_name=backend.name, allow_scripted=args.allow_scripted_results)
    print_summary(cells)
    return 0


def write_outputs(cells: list[Cell], *, backend_name: str, allow_scripted: bool) -> None:
    if backend_name == "scripted" and not allow_scripted:
        print(
            "\nrefusing to write results/ from the scripted backend -- it measures the "
            "harness, not a model. Pass --allow-scripted-results to override.",
            file=sys.stderr,
        )
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [c.to_row() for c in cells]
    with (RESULTS / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "results.json").write_text(
        json.dumps({"backend": backend_name, "cells": rows}, indent=2) + "\n", encoding="utf-8"
    )


def print_summary(cells: list[Cell]) -> None:
    from collections import defaultdict

    by_defense: dict[str, list[Cell]] = defaultdict(list)
    attack_cells = [c for c in cells if not c.scenario.startswith("control")]
    for c in attack_cells:
        by_defense[c.defense].append(c)

    print("\nAttack success rate by defense (exfiltration of the canary):\n")
    print(f"  {'defense':<20} {'kind':<14} {'success':<26} {'attempted':<26} benign_ok")
    for defense in ALL_DEFENSES:
        group = by_defense.get(defense, [])
        if not group:
            continue
        success = Rate(sum(c.exfiltrated for c in group), len(group))
        attempt = Rate(sum(c.attempted for c in group), len(group))
        benign = [c for c in cells if c.defense == defense and c.scenario == "control_clean"]
        benign_rate = Rate(sum(c.task_completed for c in benign), len(benign)) if benign else None
        kind = group[0].defense_kind
        print(f"  {defense:<20} {kind:<14} {success!s:<26} {attempt!s:<26} {benign_rate}")


if __name__ == "__main__":
    raise SystemExit(main())
