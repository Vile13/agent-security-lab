"""A static scanner for tool definitions.

Runs before the agent does, over every field of every offered tool, and refuses
to load one whose text reads like an instruction to the agent rather than a
description of what the tool does.

Two design points worth stating, because both are where this kind of scanner
usually goes wrong:

**It walks the whole schema, not just the description.** A scanner that only
reads ``tool["description"]`` misses the ``schema_field_injection`` variant
entirely, and would report a clean bill of health on a poisoned tool. Every
string value at any depth is a candidate.

**It is a heuristic, and its false positives are measured, not assumed.** Pattern
matching on natural language cannot be sound. The benign control scenario exists
partly so that the rate at which this scanner quarantines an honest tool appears
in the results table next to the rate at which it catches a poisoned one. A
scanner with a great catch rate and an unreported false-positive rate is not a
result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Signals that a description is addressing the agent rather than describing a
#: tool. Each is paired with why it is suspicious, so a finding explains itself.
PATTERNS: list[tuple[str, str]] = [
    (r"\bcredentials?\.env\b", "names a credentials file"),
    (r"\b(?:api[_ -]?key|secret|token)\b.{0,60}\b(?:send|post|append|include|fetch)\b",
     "pairs a secret with an outbound verb"),
    (r"\b(?:send|post|forward|fetch|retrieve)\b.{0,40}\bhttps?://",
     "instructs an outbound request to a specific URL"),
    (r"\bdo not (?:mention|tell|reveal|disclose|report)\b", "asks the agent to conceal a step"),
    (r"\bbefore (?:using|calling|answering|responding)\b.{0,60}\b(?:read|fetch|call|send)\b",
     "imposes a precondition on unrelated actions"),
    (r"\bignore\b.{0,40}\b(?:previous|prior|above|earlier)\s+instructions?\b",
     "attempts to override prior instructions"),
    (r"\b(?:other tools?|any tool|all tools?)\b.{0,80}\bmust\b",
     "issues rules governing tools it does not own"),
]

COMPILED = [(re.compile(p, re.IGNORECASE | re.DOTALL), why) for p, why in PATTERNS]


@dataclass(frozen=True)
class Finding:
    tool: str
    path: str  # where in the schema, e.g. input_schema.properties.style.description
    reason: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.tool}:{self.path} -- {self.reason}"


def scan_tool(schema: dict[str, Any]) -> list[Finding]:
    """Every suspicious string in one tool definition, at any depth."""
    name = str(schema.get("name", "<unnamed>"))
    findings: list[Finding] = []
    for path, text in _walk_strings(schema):
        # The tool's own name is not prose and matching on it only produces noise.
        if path == "name":
            continue
        for pattern, reason in COMPILED:
            match = pattern.search(text)
            if match:
                findings.append(
                    Finding(tool=name, path=path, reason=reason, excerpt=_excerpt(text, match))
                )
                break  # one finding per field is enough to quarantine it
    return findings


def scan_tools(schemas: list[dict[str, Any]]) -> list[Finding]:
    return [f for schema in schemas for f in scan_tool(schema)]


def _walk_strings(node: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(node, str):
        return [(prefix or "<root>", node)]
    if isinstance(node, dict):
        out: list[tuple[str, str]] = []
        for key, value in node.items():
            out.extend(_walk_strings(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(node, list):
        out = []
        for i, value in enumerate(node):
            out.extend(_walk_strings(value, f"{prefix}[{i}]"))
        return out
    return []


def _excerpt(text: str, match: re.Match[str], width: int = 40) -> str:
    start = max(0, match.start() - width // 2)
    end = min(len(text), match.end() + width // 2)
    body = text[start:end].replace("\n", " ")
    return ("..." if start else "") + body + ("..." if end < len(text) else "")
