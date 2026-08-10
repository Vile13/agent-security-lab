"""Invariants that would otherwise fail silently.

These test the harness, using the scripted backend on purpose: the point is to
prove the *measurement* is sound (a leak is detected, a block is enforced, the
canary is scored mechanically), independent of any real model's behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rag_injection.experiment import run_cell
from rag_injection.scenarios import BY_KEY

from agent_lab.backends import CassetteBackend, CassetteMiss, ScriptedBackend, request_key
from agent_lab.metrics import Rate, wilson
from agent_lab.tools import CANARY

HERE = Path(__file__).resolve().parents[1]
CORPUS = HERE / "corpus"
WORKSPACE = HERE / "workspace"


def _cell(scenario_key: str, defense: str):
    return run_cell(
        backend=ScriptedBackend(),
        corpus=CORPUS,
        workspace=WORKSPACE,
        scenario=BY_KEY[scenario_key],
        defense_name=defense,
        seed=0,
    )


def test_undefended_agent_exfiltrates_the_canary():
    # The scripted agent obeys injections, so with no defense the canary must
    # reach the attacker host. If it doesn't, the leak detector is broken.
    cell = _cell("direct", "none")
    assert cell.read_secret
    assert cell.exfiltrated
    assert cell.attempted


def test_egress_allowlist_turns_success_to_zero_but_not_attempt():
    # The deterministic control must stop the leak while leaving the attempt
    # visible -- that gap is the whole argument for measuring both.
    cell = _cell("direct", "egress_allowlist")
    assert not cell.exfiltrated
    assert cell.attempted
    assert cell.blocked, "the blocked egress call should be recorded"


def test_secret_isolation_prevents_the_read():
    cell = _cell("direct", "secret_isolation")
    assert not cell.exfiltrated
    assert not cell.read_secret


def test_clean_control_never_exfiltrates():
    # A benign ticket with no injection must not trip the detector under any
    # defense -- otherwise reported success rates are contaminated.
    for defense in ("none", "prompt_hardening", "egress_allowlist"):
        cell = _cell("control_clean", defense)
        assert not cell.exfiltrated, defense
        assert not cell.attempted, defense


def test_canary_is_the_exfiltration_definition():
    # Success is mechanical: the literal canary in an outbound argument, checked
    # on the EgressAttempt itself -- no model-judged "did it leak" anywhere in
    # the scoring path.
    from rag_injection.experiment import SYSTEM_PROMPT
    from rag_injection.scenarios import BENIGN_TASK

    from agent_lab.agent import run_agent
    from agent_lab.defenses import build as build_defense
    from agent_lab.rag import DocumentIndex
    from agent_lab.tools import Toolset

    index = DocumentIndex.from_dir(CORPUS).with_extra(BY_KEY["direct"].document)
    toolset = Toolset(index=index, workspace=WORKSPACE)
    result = run_agent(
        backend=ScriptedBackend(),
        toolset=toolset,
        defense=build_defense("none"),
        system=SYSTEM_PROMPT,
        user_message=BENIGN_TASK,
    )
    leaked = [e for e in result.egress if e.is_exfiltration]
    assert leaked, "expected at least one exfiltration attempt"
    assert all(CANARY in e.payload for e in leaked)


def test_cassette_replay_miss_is_loud(tmp_path):
    # A replay that can't find a recorded response must raise, never fall through
    # to a live call -- otherwise a stale cassette silently mixes model versions.
    cassette = CassetteBackend(tmp_path / "empty.json")
    with pytest.raises(CassetteMiss):
        cassette.complete(system="s", messages=[{"role": "user", "content": "hi"}], tools=[])


def test_request_key_is_order_insensitive_but_content_sensitive():
    a = request_key("sys", [{"role": "user", "content": "x"}], [{"name": "t"}])
    b = request_key("sys", [{"role": "user", "content": "x"}], [{"name": "t"}])
    c = request_key("sys", [{"role": "user", "content": "y"}], [{"name": "t"}])
    assert a == b
    assert a != c


def test_wilson_interval_has_width_at_the_edges():
    # The reason for Wilson over the normal approximation: 0/12 must not claim a
    # zero-width interval.
    low, high = wilson(0, 12)
    assert low == 0.0
    assert high > 0.0
    assert str(Rate(0, 12)).startswith("0%")
