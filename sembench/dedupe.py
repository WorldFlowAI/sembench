"""Build-time source dedupe for real-data corpora.

Real datasets duplicate documents across rows (LongBench multi-field and
multi-hop subsets repeat Wikipedia articles). Two "unrelated" sources sharing
paragraphs make phantom exact-cache hits and break negative controls, so the
builder drops later sources that overlap an earlier kept source.

Overlap is detected on ROLLING token windows, not aligned blocks: transforms
slice and reorder contexts, which shifts block boundaries, so text shared at
any offset can become a full aligned block downstream. A rolling window at
every position catches all alignments the audit could later see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sembench.exact_cache import chunk_hash
from sembench.schema import SourceRecord
from sembench.tokenization import load_tokenizer


@dataclass(frozen=True)
class DedupeReport:
    kept: int
    dropped: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"kept": self.kept, "dropped": list(self.dropped)}


def _window_hashes(token_ids: list[int], window: int) -> set[str]:
    from sembench.collision_audit import is_degenerate_block

    hashes: set[str] = set()
    for start in range(0, max(0, len(token_ids) - window + 1)):
        span = token_ids[start : start + window]
        # Degenerate spans (punctuation runs) would make every pair of code
        # files look like duplicates; they carry no reuse content.
        if not is_degenerate_block(span):
            hashes.add(chunk_hash(span))
    return hashes


def drop_overlapping_sources(
    records: list[SourceRecord],
    *,
    window: int = 16,
    tokenizer_name: str | None = None,
) -> tuple[list[SourceRecord], DedupeReport]:
    """Order-preserving: keep a source only if its context shares no
    ``window``-token span with any earlier kept source."""
    tokenizer = load_tokenizer(tokenizer_name)
    seen: dict[str, str] = {}
    kept: list[SourceRecord] = []
    dropped: list[dict[str, Any]] = []
    for record in records:
        # Hash everything that reaches prompts: code datasets carry shared
        # text (e.g. GPL headers) in the INPUT snippet, not only the context.
        material = record.context + "\n" + record.input
        hashes = _window_hashes(tokenizer.encode(material), window)
        overlap_owner = next((seen[digest] for digest in hashes if digest in seen), None)
        if overlap_owner is not None:
            dropped.append(
                {
                    "source_id": record.source_id,
                    "dataset": record.dataset,
                    "overlaps": overlap_owner,
                }
            )
            continue
        kept.append(record)
        for digest in hashes:
            seen.setdefault(digest, record.source_id)
    return kept, DedupeReport(kept=len(kept), dropped=dropped)


def trim_per_dataset(records: list[SourceRecord], cap: int | None) -> list[SourceRecord]:
    """Keep the first ``cap`` records per dataset (order-preserving)."""
    if cap is None:
        return records
    counts: dict[str, int] = {}
    out: list[SourceRecord] = []
    for record in records:
        if counts.get(record.dataset, 0) < cap:
            counts[record.dataset] = counts.get(record.dataset, 0) + 1
            out.append(record)
    return out
