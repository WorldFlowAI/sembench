"""Exact hash block-cache baseline and block accounting."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExactLookup:
    """Result of an exact block lookup."""

    total_blocks: int
    hit_block_indices: set[int] = field(default_factory=set)

    @property
    def hit_blocks(self) -> int:
        return len(self.hit_block_indices)


class ExactBlockIndex:
    """In-memory exact block index keyed by token chunk hash."""

    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size
        self._hash_to_donors: dict[str, set[str]] = {}

    def add(self, donor_id: str, token_ids: list[int]) -> int:
        """Add donor chunks and return the number of full chunks indexed."""
        count = 0
        for block in iter_full_blocks(token_ids, self.block_size):
            h = chunk_hash(block)
            self._hash_to_donors.setdefault(h, set()).add(donor_id)
            count += 1
        return count

    def lookup(self, token_ids: list[int]) -> ExactLookup:
        hits: set[int] = set()
        total = 0
        for idx, block in enumerate(iter_full_blocks(token_ids, self.block_size)):
            total += 1
            if chunk_hash(block) in self._hash_to_donors:
                hits.add(idx)
        return ExactLookup(total_blocks=total, hit_block_indices=hits)


def iter_full_blocks(token_ids: list[int], block_size: int):
    """Yield full-size token blocks."""
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        if len(block) == block_size:
            yield block


def chunk_hash(tokens: list[int]) -> str:
    """Stable token block hash compatible with SemBlend's packed-token approach."""
    if not tokens:
        return ""
    data = struct.pack(f"<{len(tokens)}I", *[int(t) & 0xFFFFFFFF for t in tokens])
    return hashlib.sha256(data).hexdigest()[:32]


def full_block_count(token_count: int, block_size: int) -> int:
    return token_count // block_size


def full_block_tokens(block_count: int, block_size: int) -> int:
    return block_count * block_size


def count_full_reuse_blocks(
    reused_positions: set[int],
    token_count: int,
    block_size: int,
) -> int:
    """Count full target blocks whose every token is reused."""
    count = 0
    for block_idx in range(full_block_count(token_count, block_size)):
        start = block_idx * block_size
        if all(pos in reused_positions for pos in range(start, start + block_size)):
            count += 1
    return count


def count_backend_eligible_blocks(
    copy_actions: list[dict],
    token_count: int,
    block_size: int,
) -> int:
    """Count reused full blocks that are contiguous from one donor.

    A block is eligible when all target positions in the block are copied, all
    actions name the same donor if donor IDs are available, and donor positions
    are contiguous. This is an offline proxy for materializability, not backend
    confirmation.
    """
    by_target = {
        int(action["targetPos"]): action
        for action in copy_actions
        if action.get("action") == "copy_from_donor"
        and action.get("targetPos") is not None
        and action.get("donorPos") is not None
    }
    eligible = 0
    for block_idx in range(full_block_count(token_count, block_size)):
        start = block_idx * block_size
        actions = [by_target.get(pos) for pos in range(start, start + block_size)]
        if any(action is None for action in actions):
            continue
        donor_ids = {action.get("donorId") for action in actions if action.get("donorId")}
        if len(donor_ids) > 1:
            continue
        donor_positions = [int(action["donorPos"]) for action in actions]
        if donor_positions != list(range(donor_positions[0], donor_positions[0] + block_size)):
            continue
        eligible += 1
    return eligible
