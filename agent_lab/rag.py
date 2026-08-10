"""A deliberately small retrieval layer.

Scoring is keyword overlap, not embeddings. That is a choice, not a shortcut:
this lab measures what an agent does with retrieved text, so the retriever needs
to be *deterministic* -- every run must surface the same passages, or a change in
attack success rate could just be a change in what got retrieved. An embedding
model would add a second source of variance for no gain in what is being
measured. Embedding-space attacks are a separate module on the roadmap, and that
module will need a real retriever.

Every document carries a ``trust`` level, because the mitigation being tested in
module 1 depends on the system knowing which passages an outsider could have
written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Documents an outsider can influence. In the scenario this is the public
#: support portal: anyone with a support account can file a ticket, and ticket
#: text lands in the same knowledge base as the HR handbook.
UNTRUSTED = "user_editable"
TRUSTED = "internal"


@dataclass(frozen=True)
class Document:
    title: str
    source: str
    trust: str
    body: str

    @property
    def untrusted(self) -> bool:
        return self.trust == UNTRUSTED


def parse_document(text: str, fallback_title: str) -> Document:
    """Read a corpus file: an optional ``---`` header block, then the body."""
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        _, header, body = text.split("---", 2)
        for line in header.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
    return Document(
        title=meta.get("title", fallback_title),
        source=meta.get("source", fallback_title),
        trust=meta.get("trust", TRUSTED),
        body=body.strip(),
    )


class DocumentIndex:
    """Keyword-overlap retrieval over a fixed set of documents."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    @classmethod
    def from_dir(cls, path: Path) -> DocumentIndex:
        docs = [
            parse_document(p.read_text(encoding="utf-8"), p.stem)
            for p in sorted(Path(path).glob("*.md"))
        ]
        return cls(docs)

    def with_extra(self, document: Document) -> DocumentIndex:
        """A copy of this index with one more document -- how a corpus is poisoned."""
        return DocumentIndex([*self.documents, document])

    def search(self, query: str, k: int = 3) -> list[Document]:
        terms = _tokens(query)
        scored = [(self._score(doc, terms), i, doc) for i, doc in enumerate(self.documents)]
        # Index is the tiebreak so ranking is total and stable: two documents with
        # equal overlap must not swap places between runs.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for score, _, doc in scored[:k] if score > 0] or [
            doc for _, _, doc in scored[:k]
        ]

    @staticmethod
    def _score(doc: Document, terms: set[str]) -> int:
        haystack = _tokens(f"{doc.title} {doc.body}")
        return len(terms & haystack)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}
