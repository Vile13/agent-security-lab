"""Entry point for module 2.

Backends behave exactly as in module 1 -- see that module's run_experiment.py or
the root README for what each one's numbers are worth. In short: `replay`
reproduces the committed table without a key, `record` produces it from a real
model, and `scripted` exercises the harness only.

  python run_experiment.py --backend replay
  ANTHROPIC_API_KEY=... python run_experiment.py --backend record --seeds 12
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # make agent_lab importable when run directly

from tool_poisoning.defenses import ALL_DEFENSES
from tool_poisoning.experiment import SYSTEM_PROMPT, Cell, build_toolset, run_cell
from tool_poisoning.scenarios import ALL_SCENARIOS, BENIGN_TASK, BY_KEY

from agent_lab.backends import CassetteBackend, build_backend
from agent_lab.metrics import Rate
from agent_lab.planning import Plan, tokens_of

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
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the run plan and a cost estimate without calling anything",
    )
    return p.parse_args()


def plan_for(scenarios: list, seeds: int, model: str) -> Plan:
    """Measure the actual request payloads this run would send."""
    toolset = build_toolset(CORPUS, WORKSPACE, scenarios[0])
    retrieved = toolset.render_docs(toolset.index.search(BENIGN_TASK))
    return Plan(
        runs=len(scenarios) * len(ALL_DEFENSES) * seeds,
        cached_prefix_tokens=tokens_of(SYSTEM_PROMPT) + tokens_of(toolset.schemas()),
        variable_tokens_per_call=tokens_of(retrieved) + tokens_of(BENIGN_TASK),
        cells=len(scenarios) * len(ALL_DEFENSES),
        model=model,
    )


def main() -> int:
    args = parse_args()
    scenarios = [BY_KEY[k] for k in args.scenario] if args.scenario else ALL_SCENARIOS

    if args.dry_run:
        print(plan_for(scenarios, args.seeds, args.model).render(model=args.model))
        return 0

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
            print(f"recorded responses to {args.cassette}")
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
    by_defense: dict[str, list[Cell]] = defaultdict(list)
    for c in cells:
        if c.scenario != "control_clean":
            by_defense[c.defense].append(c)

    print("\nAttack success by defense (exfiltration of the canary):\n")
    header = f"  {'defense':<21}{'kind':<15}{'success':<24}{'attempted':<24}benign_ok"
    print(header)
    for defense in ALL_DEFENSES:
        group = by_defense.get(defense, [])
        if not group:
            continue
        success = Rate(sum(c.exfiltrated for c in group), len(group))
        attempt = Rate(sum(c.attempted for c in group), len(group))
        benign = [c for c in cells if c.defense == defense and c.scenario == "control_clean"]
        benign_rate = Rate(sum(c.task_completed for c in benign), len(benign)) if benign else None
        kind = group[0].defense_kind
        print(f"  {defense:<21}{kind:<15}{success!s:<24}{attempt!s:<24}{benign_rate}")

    print("\nStatic scanner detection by technique (independent of the model):\n")
    print(f"  {'scenario':<24}{'flagged':<10}quarantined")
    by_scenario: dict[str, list[Cell]] = defaultdict(list)
    for c in cells:
        by_scenario[c.scenario].append(c)
    for key, group in by_scenario.items():
        flagged = "yes" if group[0].scanner_findings else "no"
        quarantined = sorted({q for c in group for q in c.quarantined}) or ["-"]
        print(f"  {key:<24}{flagged:<10}{', '.join(quarantined)}")


if __name__ == "__main__":
    raise SystemExit(main())
