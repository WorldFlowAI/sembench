"""Offline exact-vs-SemBlend hit-rate benchmark runner."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sembench.exact_cache import (
    ExactBlockIndex,
    count_backend_eligible_blocks,
    count_full_reuse_blocks,
    full_block_tokens,
)
from sembench.schema import RequestMetrics, WorkloadItem, read_jsonl
from sembench.tokenization import Tokenizer, load_tokenizer


@dataclass(frozen=True)
class OfflineConfig:
    """Offline runner configuration."""

    manifest: str
    output: str
    block_size: int = 16
    tokenizer: str | None = None
    semblend_path: str | None = None
    semblend_embedder: str = "jaccard"
    semblend_min_similarity: float = 0.0
    semblend_min_reuse_ratio: float = 0.25
    enable_multi_donor: bool = True
    max_items: int | None = None


def run_offline(config: OfflineConfig) -> list[RequestMetrics]:
    """Run offline exact-cache and SemBlend discovery metrics."""
    items = read_jsonl(config.manifest, max_items=config.max_items)
    tokenizer = load_tokenizer(config.tokenizer)
    pipeline_cls = _load_semblend_pipeline(config.semblend_path)

    previous_multi = os.environ.get("SEMBLEND_MULTI_DONOR")
    previous_fast_path = os.environ.get("SEMBLEND_CHUNK_FAST_PATH")
    previous_embedder = os.environ.get("SEMBLEND_EMBEDDER")
    os.environ["SEMBLEND_MULTI_DONOR"] = "1" if config.enable_multi_donor else "0"
    os.environ["SEMBLEND_CHUNK_FAST_PATH"] = "1"
    os.environ["SEMBLEND_EMBEDDER"] = config.semblend_embedder

    try:
        return [
            _run_one_item(
                item=item,
                tokenizer=tokenizer,
                pipeline_cls=pipeline_cls,
                config=config,
            )
            for item in items
        ]
    finally:
        _restore_env("SEMBLEND_MULTI_DONOR", previous_multi)
        _restore_env("SEMBLEND_CHUNK_FAST_PATH", previous_fast_path)
        _restore_env("SEMBLEND_EMBEDDER", previous_embedder)


def _run_one_item(
    *,
    item: WorkloadItem,
    tokenizer: Tokenizer,
    pipeline_cls: Any,
    config: OfflineConfig,
) -> RequestMetrics:
    donor_tokens: dict[str, list[int]] = {
        donor.donor_id: tokenizer.encode(donor.text) for donor in item.donor_prompts
    }
    recipient_tokens = tokenizer.encode(item.recipient_prompt)

    exact = ExactBlockIndex(config.block_size)
    for donor_id, tokens in donor_tokens.items():
        exact.add(donor_id, tokens)
    exact_lookup = exact.lookup(recipient_tokens)

    semantic = _run_semblend(
        pipeline_cls=pipeline_cls,
        item=item,
        donor_tokens=donor_tokens,
        recipient_tokens=recipient_tokens,
        config=config,
    )

    return RequestMetrics(
        item_id=item.item_id,
        dataset=item.dataset,
        transform=item.transform,
        negative_control=item.negative_control,
        donor_count=len(item.donor_prompts),
        prompt_tokens=len(recipient_tokens),
        total_blocks=exact_lookup.total_blocks,
        exact_hit_blocks=exact_lookup.hit_blocks,
        exact_hit_tokens=full_block_tokens(exact_lookup.hit_blocks, config.block_size),
        semantic_candidate_blocks=semantic["candidate_blocks"],
        semantic_candidate_tokens=semantic["candidate_tokens"],
        semantic_eligible_blocks=semantic["eligible_blocks"],
        semantic_eligible_tokens=semantic["eligible_tokens"],
        semblend_found=semantic["found"],
        semblend_similarity=semantic["similarity"],
        semblend_reuse_ratio=semantic["reuse_ratio"],
        semblend_latency_ms=semantic["latency_ms"],
        donor_ids=semantic["donor_ids"],
        rejection_reason=semantic["rejection_reason"],
        error=semantic["error"],
    )


def _run_semblend(
    *,
    pipeline_cls: Any,
    item: WorkloadItem,
    donor_tokens: dict[str, list[int]],
    recipient_tokens: list[int],
    config: OfflineConfig,
) -> dict[str, Any]:
    empty = {
        "found": False,
        "candidate_blocks": 0,
        "candidate_tokens": 0,
        "eligible_blocks": 0,
        "eligible_tokens": 0,
        "similarity": 0.0,
        "reuse_ratio": 0.0,
        "latency_ms": 0.0,
        "donor_ids": [],
        "rejection_reason": None,
        "error": None,
    }
    if pipeline_cls is None:
        return {**empty, "error": "SemBlend import unavailable"}

    try:
        pipeline = pipeline_cls(
            max_donors=max(len(donor_tokens), 1),
            min_similarity=config.semblend_min_similarity,
            min_reuse_ratio=config.semblend_min_reuse_ratio,
            embedder_type=config.semblend_embedder,
            chunk_size=config.block_size,
            enable_pq_segments=False,
        )
        donor_text_by_id = {donor.donor_id: donor.text for donor in item.donor_prompts}
        for donor_id, tokens in donor_tokens.items():
            pipeline.register_donor(
                request_id=donor_id,
                token_ids=tokens,
                prompt_text=donor_text_by_id.get(donor_id, ""),
            )

        result = pipeline.find_donor(
            token_ids=recipient_tokens,
            prompt_text=item.recipient_prompt,
            top_k=min(8, max(len(donor_tokens), 1)),
        )
        if not getattr(result, "found", False):
            return {
                **empty,
                "latency_ms": float(getattr(getattr(result, "timings", None), "total_ms", 0.0)),
                "rejection_reason": getattr(result, "rejection_reason", None),
            }

        copy_actions = [
            action
            for action in getattr(result, "slot_actions", [])
            if action.get("action") == "copy_from_donor"
        ]
        reused_positions = {int(action["targetPos"]) for action in copy_actions}
        candidate_blocks = count_full_reuse_blocks(
            reused_positions,
            len(recipient_tokens),
            config.block_size,
        )
        eligible_blocks = count_backend_eligible_blocks(
            copy_actions,
            len(recipient_tokens),
            config.block_size,
        )
        donor_ids = list(getattr(result, "donor_ids", []) or [])
        if not donor_ids and getattr(result, "donor_id", None):
            donor_ids = [str(result.donor_id)]

        return {
            **empty,
            "found": True,
            "candidate_blocks": candidate_blocks,
            "candidate_tokens": full_block_tokens(candidate_blocks, config.block_size),
            "eligible_blocks": eligible_blocks,
            "eligible_tokens": full_block_tokens(eligible_blocks, config.block_size),
            "similarity": float(getattr(result, "similarity", 0.0) or 0.0),
            "reuse_ratio": float(getattr(result, "reuse_ratio", 0.0) or 0.0),
            "latency_ms": float(getattr(getattr(result, "timings", None), "total_ms", 0.0)),
            "donor_ids": donor_ids,
            "rejection_reason": getattr(result, "rejection_reason", None),
        }
    except Exception as e:
        return {**empty, "error": f"{type(e).__name__}: {e}"}


def _load_semblend_pipeline(semblend_path: str | None):
    if semblend_path:
        path = str(Path(semblend_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from semblend_core.pipeline import SemBlendPipeline

        return SemBlendPipeline
    except Exception:
        return None


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous
