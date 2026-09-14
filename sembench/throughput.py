"""Throughput aggregation for runs issued under concurrency.

`requests_per_second` is computed over the whole arm: every donor and
recipient request divided by the dispatcher's wall clock. It is reported
twice -- raw, and with the estimated post-donor settle removed -- because a
settle that exists only to let an engine index donors is idle time the server
was never offered, and leaving it in the denominator understates both arms by
different amounts.

TTFT percentiles come from the per-request records, which carry the value
measured at the streamed first token; nothing here re-derives TTFT from
end-to-end latency.
"""

from __future__ import annotations

import statistics
from typing import Any

RequestRecord = dict[str, Any]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[k]


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "mean": statistics.mean(values) if values else None,
    }


def settle_seconds_est(*, items: int, post_donor_delay_ms: int, concurrency: int) -> float:
    """Wall seconds the arm spent in post-donor settles.

    Ported from `worldflow-loop/experiments/amz-throughput.py:86`: each item
    pays the delay once, and `concurrency` items pay it in parallel.
    """
    if post_donor_delay_ms <= 0 or items <= 0:
        return 0.0
    return items * (post_donor_delay_ms / 1000) / max(1, concurrency)


def request_record(item_id: str, role: str, response: dict[str, Any]) -> RequestRecord:
    """One request's contribution to the throughput document."""
    usage = response.get("usage") or {}
    return {
        "item_id": item_id,
        "role": role,
        "ttft_ms": response.get("ttft_ms"),
        "latency_ms": response.get("latency_ms"),
        "output_tokens": usage.get("completion_tokens"),
        "output_text": response.get("output_text"),
        "error": response.get("error"),
    }


def summarize_throughput(
    *,
    donors: list[RequestRecord],
    recipients: list[RequestRecord],
    wall_seconds: float,
    items: int,
    concurrency: int,
    post_donor_delay_ms: int = 0,
    idle_seconds: float | None = None,
) -> dict[str, Any]:
    """Aggregate per-request records into the throughput document (pure).

    `idle_seconds` is measured wall time with nothing in flight. When the
    runner can supply it, it is what the settle-excluded rate divides by,
    because the ported estimate assumes each settle is serialized in its lane
    and over-counts once the dispatcher overlaps them.
    """
    every = donors + recipients
    n_req = len(every)
    out_tokens = sum(int(r.get("output_tokens") or 0) for r in every)
    errors = sum(1 for r in every if r.get("error"))
    settle = settle_seconds_est(
        items=items, post_donor_delay_ms=post_donor_delay_ms, concurrency=concurrency
    )
    excluded = settle if idle_seconds is None else idle_seconds
    served_seconds = max(wall_seconds - excluded, 0.0)
    return {
        "items": items,
        "requests": n_req,
        "errors": errors,
        "concurrency": concurrency,
        "wall_seconds": round(wall_seconds, 3),
        "settle_seconds_est": round(settle, 3),
        "idle_seconds": None if idle_seconds is None else round(idle_seconds, 3),
        "settle_excluded_basis": "estimate" if idle_seconds is None else "measured_idle",
        "requests_per_second": round(n_req / wall_seconds, 4) if wall_seconds > 0 else None,
        "requests_per_second_excluding_settle": (
            round(n_req / served_seconds, 4) if served_seconds > 0 else None
        ),
        "output_tokens_per_second": round(out_tokens / wall_seconds, 2) if wall_seconds > 0 else None,
        "donor_ttft_ms": summarize([r["ttft_ms"] for r in donors if r.get("ttft_ms") is not None]),
        "recipient_ttft_ms": summarize(
            [r["ttft_ms"] for r in recipients if r.get("ttft_ms") is not None]
        ),
        "recipient_latency_ms": summarize(
            [r["latency_ms"] for r in recipients if r.get("latency_ms") is not None]
        ),
    }
