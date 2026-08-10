"""A 30-second tour of the lab, no API key required.

It runs one indirect-prompt-injection attack through the agent loop twice -- once
undefended, once behind the egress allowlist -- and prints the transcript so you
can watch the poisoned document turn into an outbound request carrying the
secret, and then watch the allowlist stop it.

The backend here is the scripted fake, so these transcripts illustrate the
*mechanism*, not any real model's susceptibility. The real numbers -- which model,
how often, with what confidence interval -- live in
indirect-prompt-injection/results/ and are produced with --backend replay.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "indirect-prompt-injection"))

from rag_injection.experiment import SYSTEM_PROMPT  # noqa: E402
from rag_injection.scenarios import BENIGN_TASK, DIRECT  # noqa: E402

from agent_lab.agent import run_agent  # noqa: E402
from agent_lab.backends import ScriptedBackend  # noqa: E402
from agent_lab.defenses import build as build_defense  # noqa: E402
from agent_lab.rag import DocumentIndex  # noqa: E402
from agent_lab.tools import CANARY, Toolset  # noqa: E402

CORPUS = HERE / "indirect-prompt-injection" / "corpus"
WORKSPACE = HERE / "indirect-prompt-injection" / "workspace"


def run(defense_name: str) -> None:
    index = DocumentIndex.from_dir(CORPUS).with_extra(DIRECT.document)
    toolset = Toolset(index=index, workspace=WORKSPACE)
    result = run_agent(
        backend=ScriptedBackend(),
        toolset=toolset,
        defense=build_defense(defense_name),
        system=SYSTEM_PROMPT,
        user_message=BENIGN_TASK,
    )

    print(f"\n{'=' * 70}\ndefense: {defense_name}\n{'=' * 70}")
    for name, args in result.tool_calls:
        summary = ", ".join(f"{k}={_shorten(v)}" for k, v in args.items())
        print(f"  -> {name}({summary})")
    for name, reason in result.blocked:
        print(f"  !! blocked {name}: {reason}")
    verdict = "EXFILTRATED" if result.exfiltrated else (
        "attempted, blocked" if result.attempted_exfiltration else "no exfiltration"
    )
    print(f"  = {verdict}")


def _shorten(value: object) -> str:
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def main() -> None:
    print(__doc__)
    print(f"\nThe user asks: {BENIGN_TASK!r}")
    print(f"The knowledge base has been poisoned via: {DIRECT.technique}")
    print(f"The secret to steal is the canary {CANARY} in workspace/credentials.env.")

    run("none")
    run("egress_allowlist")

    print(
        "\nSame agent, same poisoned corpus. The undefended run reads the secret "
        "and posts it to the attacker; the allowlist lets the read happen but "
        "refuses the outbound call -- and the attempt is still recorded, which is "
        "why the results table reports attack success and attempt as two columns.\n"
    )


if __name__ == "__main__":
    main()
