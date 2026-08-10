"""Harness invariants for module 2.

The scanner tests are the load-bearing ones here: unlike the behavioral rates,
the scanner's verdicts are deterministic, so they are properties this repository
can assert today rather than claims awaiting a model run.
"""

from __future__ import annotations

from pathlib import Path

from tool_poisoning.defenses import ALL_DEFENSES, build
from tool_poisoning.experiment import build_toolset, run_cell
from tool_poisoning.scanner import scan_tool, scan_tools
from tool_poisoning.scenarios import BY_KEY

from agent_lab.backends import ScriptedBackend
from agent_lab.tools import merge_schema

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


# -- the attack surface exists at all -----------------------------------------


def test_poisoned_description_reaches_the_model_as_a_tool_schema():
    # If the payload never makes it into schemas(), every downstream number is
    # measuring nothing.
    toolset = build_toolset(CORPUS, WORKSPACE, BY_KEY["description_injection"])
    blob = str(toolset.schemas())
    assert "collect.attacker.example" in blob
    assert "format_citation" in blob


def test_schema_patch_rewrites_a_built_in_description():
    toolset = build_toolset(CORPUS, WORKSPACE, BY_KEY["supply_chain_patch"])
    read_file = next(s for s in toolset.schemas() if s["name"] == "read_file")
    assert "collect.attacker.example" in read_file["description"]
    # The patch must not damage the rest of the schema.
    assert read_file["input_schema"]["properties"]["path"]["type"] == "string"


def test_merge_schema_is_recursive_and_non_mutating():
    base = {"a": {"b": 1, "c": 2}}
    patch = {"a": {"b": 9}}
    out = merge_schema(base, patch)
    assert out == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}, "merge must not mutate its input"


def test_shadowing_never_calls_the_malicious_tool():
    # The whole claim of the shadowing variant is that merely listing the tool is
    # the attack. If the agent had to call it, it would be an ordinary tool abuse.
    cell = _cell("tool_shadowing", "none")
    assert not cell.called_malicious_tool


# -- the scanner ---------------------------------------------------------------


def test_scanner_flags_every_attack_variant():
    for key in ("description_injection", "schema_field_injection", "tool_shadowing",
                "supply_chain_patch"):
        toolset = build_toolset(CORPUS, WORKSPACE, BY_KEY[key])
        assert scan_tools(toolset.schemas()), f"{key} went undetected"


def test_scanner_reaches_nested_parameter_descriptions():
    # The specific regression this guards: a scanner that only reads
    # tool["description"] passes schema_field_injection with a clean bill.
    toolset = build_toolset(CORPUS, WORKSPACE, BY_KEY["schema_field_injection"])
    citation = next(s for s in toolset.schemas() if s["name"] == "format_citation")
    top_level_only = scan_tool({"name": citation["name"], "description": citation["description"]})
    assert not top_level_only, "the top-level description is clean by construction"
    full = scan_tool(citation)
    assert full, "scanning the full schema must find the nested payload"
    assert any("style" in f.path for f in full)


def test_scanner_does_not_flag_the_benign_control():
    # The false-positive check. A scanner whose catch rate is reported without
    # this number is not reporting a result.
    toolset = build_toolset(CORPUS, WORKSPACE, BY_KEY["control_clean"])
    assert scan_tools(toolset.schemas()) == []


def test_scanner_leaves_the_built_in_toolset_alone():
    # The built-in descriptions are ordinary prose about network and file access;
    # if the patterns fire on those, the scanner is unusable in practice.
    toolset = build_toolset(CORPUS, WORKSPACE, BY_KEY["control_clean"])
    built_in = [s for s in toolset.schemas() if s["name"] != "format_citation"]
    assert scan_tools(built_in) == []


def test_findings_explain_themselves():
    toolset = build_toolset(CORPUS, WORKSPACE, BY_KEY["description_injection"])
    finding = scan_tools(toolset.schemas())[0]
    assert finding.reason and finding.excerpt
    assert finding.tool == "format_citation"


# -- defenses ------------------------------------------------------------------


def test_scanning_defense_quarantines_the_poisoned_tool():
    cell = _cell("description_injection", "description_scanning")
    assert cell.quarantined == ["format_citation"]
    assert not cell.exfiltrated


def test_quarantined_tool_is_withheld_from_the_model():
    scenario = BY_KEY["description_injection"]
    toolset = build_toolset(CORPUS, WORKSPACE, scenario)
    build("description_scanning").setup(toolset)
    assert all(s["name"] != "format_citation" for s in toolset.schemas())


def test_scanning_does_not_quarantine_anything_in_the_control():
    cell = _cell("control_clean", "description_scanning")
    assert cell.quarantined == []


def test_module_one_defenses_are_reused_unchanged():
    # The transferability question is only meaningful if these really are module
    # 1's objects rather than lookalikes defined here.
    import agent_lab.defenses as shared

    assert type(build("prompt_hardening")) is shared.PromptHardening
    assert type(build("egress_allowlist")) is shared.EgressAllowlist
    assert type(build("secret_isolation")) is shared.SecretIsolation


def test_module_one_prompt_hardening_never_mentions_tool_descriptions():
    # Not a prediction about a model -- a fact about the text. It is the reason
    # the v1/v2 comparison exists, and if someone later edits the shared wording
    # to cover tool descriptions, this test should fail and force a rethink of
    # what the v1 column means.
    v1 = build("prompt_hardening").system_suffix().lower()
    assert "search_docs" in v1
    assert "tool description" not in v1
    v2 = build("prompt_hardening_v2").system_suffix().lower()
    assert "description" in v2


def test_provenance_tagging_is_absent_from_this_module():
    # Structurally inapplicable here; asserting its absence keeps that decision
    # explicit instead of looking like an oversight.
    assert "provenance_tagging" not in ALL_DEFENSES


def test_egress_allowlist_still_blocks_a_tool_description_attack():
    # Downstream defenses should be indifferent to where the injection entered.
    cell = _cell("description_injection", "egress_allowlist")
    assert not cell.exfiltrated
    assert cell.attempted, "the attempt must remain visible"


def test_every_named_defense_builds():
    for name in ALL_DEFENSES:
        assert build(name) is not None
