"""Cross-cutting tests for the shared agent_lab machinery.

Module-specific harness tests live under each module's own tests/. What belongs
here is behavior of the shared layer that no single module owns -- the metrics,
the defense registry, and the record/replay contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_lab.backends import CassetteBackend, request_key
from agent_lab.defenses import ALL_DEFENSES, Composite, build
from agent_lab.metrics import Rate, rate_differs, wilson


def test_every_named_defense_builds():
    for name in ALL_DEFENSES:
        defense = build(name)
        assert defense.name == name or isinstance(defense, Composite)
        assert defense.kind in {"none", "probabilistic", "deterministic", "layered"}


def test_unknown_defense_is_an_error():
    with pytest.raises(ValueError, match="unknown defense"):
        build("nonexistent")


def test_deterministic_defenses_are_labelled_as_such():
    # The two-column argument in the README rests on this label being right: a
    # deterministic defense's success rate is zero by construction, so the label
    # is what tells a reader not to read that zero as evidence.
    assert build("egress_allowlist").kind == "deterministic"
    assert build("secret_isolation").kind == "deterministic"
    assert build("prompt_hardening").kind == "probabilistic"


def test_rate_differs_uses_disjoint_intervals():
    # 0/12 vs 12/12 clearly differ; 6/12 vs 7/12 clearly do not at this n.
    assert rate_differs(Rate(0, 12), Rate(12, 12))
    assert not rate_differs(Rate(6, 12), Rate(7, 12))


def test_wilson_is_symmetric_at_the_extremes():
    low0, high0 = wilson(0, 20)
    low1, high1 = wilson(20, 20)
    assert low0 == 0.0
    assert high1 == 1.0
    # The interval widths at the two extremes mirror each other.
    assert high0 == pytest.approx(1 - low1, abs=1e-9)


def test_record_then_replay_round_trips(tmp_path: Path):
    # A recorded response must come back byte-identical on replay, keyed only by
    # request content -- that is what makes a committed cassette a faithful
    # stand-in for the live run.
    class Fake:
        model = "fake"

        def complete(self, *, system, messages, tools):
            return {"content": [{"type": "text", "text": "recorded answer"}]}

    path = tmp_path / "c.json"
    recorder = CassetteBackend(path, inner=Fake())
    args = {"system": "s", "messages": [{"role": "user", "content": "q"}], "tools": []}
    first = recorder.complete(**args)
    recorder.save()

    replayer = CassetteBackend(path)
    second = replayer.complete(**args)
    assert first == second
    assert replayer.hits == 1
    assert replayer.misses == 0


def test_cost_estimate_models_the_cache_minimum():
    # A cache_control marker below the model's minimum prefix length is ignored
    # silently by the API. If the estimator assumed it worked, it would quote a
    # saving that never lands -- the one way this tool could mislead a spend
    # decision.
    from agent_lab.planning import CACHE_MIN_TOKENS, Plan

    below = Plan(runs=100, cached_prefix_tokens=CACHE_MIN_TOKENS - 1, cells=10,
                 variable_tokens_per_call=100)
    above = Plan(runs=100, cached_prefix_tokens=CACHE_MIN_TOKENS, cells=10,
                 variable_tokens_per_call=100)
    assert not below.caching_engages
    assert above.caching_engages
    # Below the threshold the estimate must equal the no-caching estimate.
    assert below.cost_usd == pytest.approx(below.cost_without_caching_usd)
    assert above.cost_usd < above.cost_without_caching_usd


def test_cost_estimate_scales_with_run_count():
    from agent_lab.planning import Plan

    small = Plan(runs=10, cached_prefix_tokens=600, cells=5, variable_tokens_per_call=100)
    large = Plan(runs=100, cached_prefix_tokens=600, cells=5, variable_tokens_per_call=100)
    assert large.cost_usd > small.cost_usd
    assert 0.0 < large.output_share < 1.0


def test_request_key_ignores_dict_ordering():
    # Cassette keys must not depend on how a dict happened to be constructed, or
    # a replay would miss on a semantically identical request.
    a = request_key("s", [{"role": "user", "content": "x"}], [{"name": "t"}])
    b = request_key("s", [{"content": "x", "role": "user"}], [{"name": "t"}])
    assert a == b
