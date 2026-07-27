"""Result aggregation and JSON writing."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

from sembench.schema import RESULT_VERSION, RequestMetrics


def aggregate_metrics(requests: list[RequestMetrics]) -> dict[str, Any]:
    """Aggregate per-request metrics into benchmark-level rates."""
    total_blocks = sum(r.total_blocks for r in requests)
    prompt_tokens = sum(r.prompt_tokens for r in requests)
    exact_blocks = sum(r.exact_hit_blocks for r in requests)
    exact_tokens = sum(r.exact_hit_tokens for r in requests)
    candidate_blocks = sum(r.semantic_candidate_blocks for r in requests)
    candidate_tokens = sum(r.semantic_candidate_tokens for r in requests)
    eligible_blocks = sum(r.semantic_eligible_blocks for r in requests)
    eligible_tokens = sum(r.semantic_eligible_tokens for r in requests)
    has_semantic_plan = any(
        r.semblend_found
        or r.semantic_candidate_blocks
        or r.semantic_eligible_blocks
        or r.semblend_latency_ms > 0
        for r in requests
    )

    confirmed_values = [r.backend_confirmed_blocks for r in requests]
    has_confirmed = any(v is not None for v in confirmed_values)
    confirmed_blocks = sum(v or 0 for v in confirmed_values) if has_confirmed else None
    confirmed_tokens_values = [r.backend_confirmed_tokens for r in requests]
    confirmed_tokens = sum(v or 0 for v in confirmed_tokens_values) if has_confirmed else None

    negative = [r for r in requests if r.negative_control]
    negative_blocks = sum(r.total_blocks for r in negative)
    negative_eligible = sum(r.semantic_eligible_blocks for r in negative)
    negative_semantic_placements = sum(
        1 for r in negative if r.route_outcome == "semantic_placement"
    )
    negative_confirmed = (
        sum((r.backend_confirmed_blocks or 0) for r in negative) if has_confirmed else None
    )
    route_outcomes: dict[str, int] = {}
    for request in requests:
        if request.route_outcome:
            route_outcomes[request.route_outcome] = route_outcomes.get(request.route_outcome, 0) + 1
    quality_values = [r.quality_pass for r in requests if r.quality_pass is not None]

    exact_rate = _rate(exact_blocks, total_blocks)
    candidate_rate = _rate(candidate_blocks, total_blocks)
    eligible_rate = _rate(eligible_blocks, total_blocks)
    confirmed_rate = (
        _rate(confirmed_blocks or 0, total_blocks) if confirmed_blocks is not None else None
    )

    return {
        "request_count": len(requests),
        "prompt_tokens": prompt_tokens,
        "total_blocks": total_blocks,
        "exact_hit_blocks": exact_blocks,
        "exact_hit_tokens": exact_tokens,
        "semantic_candidate_blocks": candidate_blocks,
        "semantic_candidate_tokens": candidate_tokens,
        "semantic_eligible_blocks": eligible_blocks,
        "semantic_eligible_tokens": eligible_tokens,
        "backend_confirmed_blocks": confirmed_blocks,
        "backend_confirmed_tokens": confirmed_tokens,
        "exact_block_hit_rate": exact_rate,
        "semantic_candidate_block_rate": candidate_rate,
        "semantic_eligible_block_rate": eligible_rate,
        "backend_confirmed_block_rate": confirmed_rate,
        "semantic_eligible_lift": eligible_rate - exact_rate if has_semantic_plan else None,
        "backend_confirmed_lift": (
            confirmed_rate - exact_rate if confirmed_rate is not None else None
        ),
        "exact_token_weighted_reuse_rate": _rate(exact_tokens, prompt_tokens),
        "semantic_candidate_token_weighted_reuse_rate": _rate(candidate_tokens, prompt_tokens),
        "semantic_eligible_token_weighted_reuse_rate": _rate(eligible_tokens, prompt_tokens),
        "negative_control_count": len(negative),
        "negative_control_blocks": negative_blocks,
        "negative_control_semantic_eligible_blocks": negative_eligible,
        "negative_control_semantic_eligible_rate": _rate(negative_eligible, negative_blocks),
        "negative_control_semantic_placements": negative_semantic_placements,
        "negative_control_semantic_placement_rate": _rate(
            negative_semantic_placements,
            len(negative),
        ),
        "negative_control_backend_confirmed_blocks": negative_confirmed,
        "negative_control_backend_confirmed_rate": (
            _rate(negative_confirmed or 0, negative_blocks)
            if negative_confirmed is not None
            else None
        ),
        "route_outcomes": dict(sorted(route_outcomes.items())),
        "semantic_placement_rate_by_request": _rate(
            route_outcomes.get("semantic_placement", 0),
            len(requests),
        ),
        "semblend_hit_rate_by_request": _rate(
            sum(1 for r in requests if r.semblend_found),
            len(requests),
        ),
        "mean_semblend_latency_ms": _mean(
            [r.semblend_latency_ms for r in requests if r.semblend_latency_ms > 0]
        ),
        "mean_ttft_ms": _mean([r.ttft_ms for r in requests if r.ttft_ms is not None]),
        "quality_pass_rate": (
            _rate(sum(1 for v in quality_values if v), len(quality_values))
            if quality_values
            else None
        ),
    }


def aggregate_by_transform(requests: list[RequestMetrics]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[RequestMetrics]] = {}
    for request in requests:
        groups.setdefault(request.transform, []).append(request)
    return {name: aggregate_metrics(group) for name, group in sorted(groups.items())}


def write_result(
    path: str | Path,
    *,
    requests: list[RequestMetrics],
    config: dict[str, Any],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_version": RESULT_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": platform.node(),
        },
        "aggregate": aggregate_metrics(requests),
        "by_transform": aggregate_by_transform(requests),
        "requests": [r.to_dict() for r in requests],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
