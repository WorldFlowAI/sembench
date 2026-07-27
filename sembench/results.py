"""Result aggregation and JSON writing."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

from sembench.schema import RESULT_VERSION, RequestMetrics, RunMetadata


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
        "mean_quality_f1": _mean([r.quality_f1 for r in requests if r.quality_f1 is not None]),
        "mean_quality_rouge_l": _mean(
            [r.quality_rouge_l for r in requests if r.quality_rouge_l is not None]
        ),
        "quality_f1_ci": _ci([r.quality_f1 for r in requests if r.quality_f1 is not None]),
        "quality_rouge_l_ci": _ci(
            [r.quality_rouge_l for r in requests if r.quality_rouge_l is not None]
        ),
        "quality_pass_rate": (
            _rate(sum(1 for v in quality_values if v), len(quality_values))
            if quality_values
            else None
        ),
    }


def paired_summary(requests: list[RequestMetrics]) -> dict[str, Any] | None:
    """Per-item cold/warm pairing: TTFT speedup and warm-vs-cold output
    similarity. Pairs with a contaminated cold arm are excluded and counted."""
    cold = {r.item_id: r for r in requests if r.arm == "cold"}
    warm = {r.item_id: r for r in requests if r.arm == "warm"}
    if not cold or not warm:
        return None
    from sembench.quality import rouge_l
    from sembench.stats import bootstrap_mean

    speedups: list[float] = []
    output_rouge: list[float] = []
    contaminated = 0
    errored = 0
    for item_id, cold_row in cold.items():
        warm_row = warm.get(item_id)
        if warm_row is None:
            continue
        if cold_row.flush_contaminated:
            contaminated += 1
            continue
        if cold_row.error or warm_row.error:
            errored += 1
            continue
        if cold_row.ttft_ms and warm_row.ttft_ms:
            speedups.append(cold_row.ttft_ms / warm_row.ttft_ms)
        if cold_row.output_text and warm_row.output_text:
            output_rouge.append(rouge_l(warm_row.output_text, cold_row.output_text))
    speedup_ci = bootstrap_mean(speedups)
    rouge_ci = bootstrap_mean(output_rouge)
    return {
        "pairs_total": len(cold),
        "pairs_used": len(speedups),
        "pairs_contaminated": contaminated,
        "pairs_errored": errored,
        "ttft_speedup_mean": speedup_ci.point if speedup_ci else None,
        "ttft_speedup_ci": speedup_ci.to_dict() if speedup_ci else None,
        "warm_vs_cold_output_rouge_l_mean": rouge_ci.point if rouge_ci else None,
        "warm_vs_cold_output_rouge_l_ci": rouge_ci.to_dict() if rouge_ci else None,
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
    run: RunMetadata | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_version": RESULT_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": run.to_dict() if run is not None else None,
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": platform.node(),
        },
        "aggregate": aggregate_metrics(requests),
        "paired": paired_summary(requests),
        "by_transform": aggregate_by_transform(requests),
        "requests": [r.to_dict() for r in requests],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _ci(values: list[float]) -> dict[str, float] | None:
    from sembench.stats import bootstrap_mean

    result = bootstrap_mean(values)
    return result.to_dict() if result is not None else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
