"""Phantom-match collision audit for workload manifests.

Synthetic and transformed workloads can accidentally share full token blocks
across items that are supposed to be unrelated. Those collisions produce
exact-prefix hits that masquerade as (or contaminate) semantic-reuse
measurements. A frozen corpus must pass this audit before its checksums are
recorded.

Rules:
- Donor blocks are attributed to the source their TEXT came from, not the
  item that carries them: a negative-control donor records the source it was
  borrowed from in ``metadata.negative_source_id``, and its blocks belong to
  that origin (otherwise every deliberate cross-source negative donor makes
  the origin source's own items look like phantoms).
- Blocks shared within one origin source are by design and ignored.
- A recipient sharing blocks with donor text originating from a DIFFERENT
  source is a phantom match.
- A ``negative_control`` recipient sharing any full block with ITS OWN
  donors breaks that control's unrelatedness promise (per-item rule — with
  per-item cache flushing, only the item's own donors are live at replay).
- Widely shared blocks (template boilerplate appearing across ≥
  ``boilerplate_source_threshold`` distinct origin sources) are reported
  separately: they are not one bad pair but a corpus-wide template artifact,
  and they are excluded from pairwise violations so one popular preamble
  doesn't drown the report. A clean corpus should minimize these; the count
  is surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sembench.exact_cache import chunk_hash, iter_full_blocks
from sembench.schema import WorkloadItem
from sembench.tokenization import load_tokenizer

# Blocks with fewer distinct tokens than this are degenerate — punctuation
# runs ("=====", "----") that the fallback regex tokenizer explodes into one
# token per character. They are not meaningful reuse content under any real
# BPE tokenizer (which folds such runs into 1-2 tokens), so they are counted
# separately rather than treated as phantom evidence.
MIN_DISTINCT_TOKENS_PER_BLOCK = 4


def is_degenerate_block(block: list[int]) -> bool:
    return len(set(block)) < MIN_DISTINCT_TOKENS_PER_BLOCK


@dataclass(frozen=True)
class CollisionReport:
    """Outcome of a manifest collision audit."""

    block_size: int
    items_checked: int
    donor_blocks_indexed: int
    boilerplate_blocks: int
    degenerate_blocks: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "items_checked": self.items_checked,
            "donor_blocks_indexed": self.donor_blocks_indexed,
            "boilerplate_blocks": self.boilerplate_blocks,
            "degenerate_blocks": self.degenerate_blocks,
            "violations": list(self.violations),
            "passed": self.passed,
        }


def audit_manifest_items(
    items: list[WorkloadItem],
    *,
    block_size: int = 16,
    tokenizer_name: str | None = None,
    boilerplate_source_threshold: int = 3,
    max_violations: int = 50,
) -> CollisionReport:
    tokenizer = load_tokenizer(tokenizer_name)

    # Index every donor block by the ORIGIN of the donor text:
    # hash -> {(carrying_item_id, origin_source_id)}. Also keep each item's
    # own donor hashes for the negative-control rule.
    donor_owners: dict[str, set[tuple[str, str]]] = {}
    own_donor_hashes: dict[str, set[str]] = {}
    donor_blocks = 0
    degenerate = 0
    for item in items:
        own = own_donor_hashes.setdefault(item.item_id, set())
        for donor in item.donor_prompts:
            origin = str(
                donor.metadata.get("negative_source_id")
                or donor.metadata.get("entity_swap_source_id")
                or item.source_id
            )
            for block in iter_full_blocks(tokenizer.encode(donor.text), block_size):
                donor_blocks += 1
                if is_degenerate_block(block):
                    degenerate += 1
                    continue
                digest = chunk_hash(block)
                donor_owners.setdefault(digest, set()).add((item.item_id, origin))
                own.add(digest)

    # A phantom block is single-origin by nature (one source's content
    # appearing where it shouldn't); template boilerplate is multi-origin.
    # Clamp the threshold to the corpus's distinct origin count (floor 2) so
    # small fixtures classify their shared preamble correctly.
    distinct_sources = len({source for owners in donor_owners.values() for _, source in owners})
    effective_threshold = max(2, min(boilerplate_source_threshold, distinct_sources))
    boilerplate = {
        digest
        for digest, owners in donor_owners.items()
        if len({source for _, source in owners}) >= effective_threshold
    }

    violations: list[dict[str, Any]] = []

    def record(kind: str, item: WorkloadItem, block_idx: int, digest: str, owners) -> None:
        if len(violations) < max_violations:
            violations.append(
                {
                    "kind": kind,
                    "item_id": item.item_id,
                    "source_id": item.source_id,
                    "recipient_block_index": block_idx,
                    "block_hash": digest,
                    "colliding_donor_items": sorted({owner_item for owner_item, _ in owners}),
                }
            )

    for item in items:
        recipient_tokens = tokenizer.encode(item.recipient_prompt)
        own = own_donor_hashes.get(item.item_id, set())
        for block_idx, block in enumerate(iter_full_blocks(recipient_tokens, block_size)):
            if is_degenerate_block(block):
                continue
            digest = chunk_hash(block)
            owners = donor_owners.get(digest)
            if not owners or digest in boilerplate:
                continue
            if item.negative_control and digest in own:
                record("negative_control_overlap", item, block_idx, digest, owners)
                continue
            foreign = [
                (owner_item, origin) for owner_item, origin in owners if origin != item.source_id
            ]
            if foreign:
                record("phantom_cross_source", item, block_idx, digest, foreign)

    return CollisionReport(
        block_size=block_size,
        items_checked=len(items),
        donor_blocks_indexed=donor_blocks,
        boilerplate_blocks=len(boilerplate),
        degenerate_blocks=degenerate,
        violations=violations,
    )
