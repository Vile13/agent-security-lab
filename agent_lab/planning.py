"""Estimate the size and cost of a run before spending money on it.

A full grid is hundreds of agent runs and thousands of model calls, so the
runner can print a plan first. The estimate is deliberately built from the
*actual* serialized request payloads rather than a guessed constant: the system
prompt, the tool schemas, and the corpus are all measurable without an API key,
which is most of the input side.

It is still an estimate, and says so. Token counts are derived from a
characters-per-token ratio rather than the tokenizer, output length is a
stated assumption, and neither the number of turns an agent takes nor how much
it thinks can be known before it runs. Treat the number as an order of
magnitude for deciding `--seeds`, not as a quote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: Rough English-text ratio. The real tokenizer is only reachable through the
#: API (`messages.count_tokens`), which needs the key this estimate exists to
#: help you decide about spending.
CHARS_PER_TOKEN = 3.7

#: claude-opus-5, USD per million tokens.
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 25.00
#: Cached input reads bill at ~0.1x; the first write costs ~1.25x.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25
#: Minimum cacheable prefix on claude-opus-5. Below this a cache_control marker
#: is silently ignored -- no error, just no cache -- so the estimate must model
#: the threshold rather than assume the marker worked.
CACHE_MIN_TOKENS = 512

#: Observed on the scripted harness: retrieve -> read -> exfiltrate -> answer,
#: with shorter paths on blocked and control runs.
ASSUMED_CALLS_PER_RUN = 3.3
#: Adaptive thinking is on by default on claude-opus-5, so output per call is
#: dominated by thinking rather than the short answers this agent writes.
ASSUMED_OUTPUT_TOKENS_PER_CALL = 600


def tokens_of(payload: Any) -> int:
    """Approximate token count of anything JSON-serializable."""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return int(len(text) / CHARS_PER_TOKEN)


@dataclass
class Plan:
    runs: int
    cached_prefix_tokens: int  # system + tools, identical across runs in a cell
    variable_tokens_per_call: int  # transcript that grows turn to turn
    cells: int  # distinct (scenario x defense) prefixes

    @property
    def calls(self) -> int:
        return int(self.runs * ASSUMED_CALLS_PER_RUN)

    @property
    def caching_engages(self) -> bool:
        """Whether the prefix clears the model's minimum cacheable length."""
        return self.cached_prefix_tokens >= CACHE_MIN_TOKENS

    @property
    def cost_usd(self) -> float:
        if self.caching_engages:
            # The prefix is written once per cell and read on every other call.
            prefix = (
                self.cells * self.cached_prefix_tokens * CACHE_WRITE_MULTIPLIER
                + (self.calls - self.cells) * self.cached_prefix_tokens * CACHE_READ_MULTIPLIER
            )
        else:
            prefix = self.calls * self.cached_prefix_tokens
        variable = self.calls * self.variable_tokens_per_call
        output = self.calls * ASSUMED_OUTPUT_TOKENS_PER_CALL
        input_cost = (prefix + variable) / 1_000_000 * INPUT_PER_MTOK
        output_cost = output / 1_000_000 * OUTPUT_PER_MTOK
        return input_cost + output_cost

    @property
    def output_share(self) -> float:
        output = self.calls * ASSUMED_OUTPUT_TOKENS_PER_CALL / 1_000_000 * OUTPUT_PER_MTOK
        return output / self.cost_usd if self.cost_usd else 0.0

    @property
    def cost_without_caching_usd(self) -> float:
        uncached = self.calls * (self.cached_prefix_tokens + self.variable_tokens_per_call)
        output = self.calls * ASSUMED_OUTPUT_TOKENS_PER_CALL
        return uncached / 1_000_000 * INPUT_PER_MTOK + output / 1_000_000 * OUTPUT_PER_MTOK

    def render(self, *, model: str) -> str:
        if self.caching_engages:
            cache_note = (
                f"  prompt cache        engages "
                f"(prefix >= {CACHE_MIN_TOKENS} tok); "
                f"~${self.cost_without_caching_usd:.2f} without it"
            )
        else:
            cache_note = (
                f"  prompt cache        INACTIVE -- prefix is under the "
                f"{CACHE_MIN_TOKENS}-token minimum,\n"
                "                      so the cache_control marker is ignored "
                "silently. Not a\n"
                "                      bug, and not worth padding the prompt to "
                "fix: the prefix\n"
                "                      is a small share of this run's cost either way."
            )
        lines = [
            "",
            f"  planned runs        {self.runs}  ({self.cells} cells x seeds)",
            f"  model calls         ~{self.calls}  (assuming {ASSUMED_CALLS_PER_RUN} per run)",
            f"  stable prefix       ~{self.cached_prefix_tokens} tokens (system + tools)",
            f"  variable context    ~{self.variable_tokens_per_call} tokens per call",
            cache_note,
            "",
            f"  estimated cost      ~${self.cost_usd:.2f} on {model}",
            f"  output share        ~{self.output_share:.0%} of that is output tokens",
            "",
            "  An estimate, not a quote. Token counts come from a chars-per-token",
            "  ratio rather than the tokenizer, and neither the number of turns an",
            "  agent takes nor how much it thinks is knowable in advance -- and at",
            f"  ~{self.output_share:.0%} output, the thinking assumption",
            f"  ({ASSUMED_OUTPUT_TOKENS_PER_CALL} tok/call) is what this number",
            "  mostly rests on. Scale --seeds from it, then read the real usage off",
            "  results.csv, which records input and output tokens per run.",
            "",
        ]
        return "\n".join(lines)
