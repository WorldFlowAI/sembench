"""Concurrent-load runner: request throughput and TTFT under load.

Kept for the `run-load` subcommand and its document shape, but reduced to a
wrapper over the gateway runner's bounded dispatcher, which fixes the three
things that made its own numbers unusable: requests were issued through
`ThreadPoolExecutor.map`, destroying the stream order a self-seeding manifest
depends on; the post-donor settle slept inside a worker thread, depressing
both effective concurrency and the reported rate; and per-item tenant and
template were ignored.

New work should use `run-live-gateway --concurrency`, which writes the same
throughput document *and* a full RequestMetrics row per request (quality,
negative control, route headers, run metadata). This document carries only
the seven-key per-request summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sembench.gateway_live import LiveGatewayConfig, run_live_gateway_measured
from sembench.throughput import percentile, summarize, summarize_throughput

__all__ = [
    "LoadConfig",
    "percentile",
    "run_load",
    "summarize",
    "summarize_load",
]


@dataclass(frozen=True)
class LoadConfig:
    manifest: str
    output: str
    gateway_url: str
    model: str
    tenant: str = "tenant-a"
    template: str = "rag-template-v1"
    concurrency: int = 4
    max_items: int | None = None
    donor_max_tokens: int = 1
    recipient_max_tokens: int = 32
    post_donor_delay_ms: int = 1000
    timeout_seconds: float = 1800.0
    run_id: str = "load"
    # Minimum donor -> recipient separation in completed requests, for
    # manifests that name their donor under metadata.donor_item_id.
    min_donor_gap_requests: int = 0


def summarize_load(
    *,
    donors: list[dict[str, Any]],
    recipients: list[dict[str, Any]],
    wall_seconds: float,
    items: int,
    concurrency: int,
    post_donor_delay_ms: int = 0,
) -> dict[str, Any]:
    """Aggregate per-request records into the load document (pure)."""
    return summarize_throughput(
        donors=donors,
        recipients=recipients,
        wall_seconds=wall_seconds,
        items=items,
        concurrency=concurrency,
        post_donor_delay_ms=post_donor_delay_ms,
    )


def gateway_config(config: LoadConfig) -> LiveGatewayConfig:
    """The gateway-runner configuration this load run is really asking for."""
    return LiveGatewayConfig(
        manifest=config.manifest,
        output=config.output,
        gateway_url=config.gateway_url,
        model=config.model,
        tenant=config.tenant,
        template=config.template,
        max_items=config.max_items,
        donor_max_tokens=config.donor_max_tokens,
        recipient_max_tokens=config.recipient_max_tokens,
        timeout_seconds=config.timeout_seconds,
        post_donor_delay_ms=config.post_donor_delay_ms,
        concurrency=config.concurrency,
        min_donor_gap_requests=config.min_donor_gap_requests,
    )


def run_load(config: LoadConfig) -> dict[str, Any]:
    result = run_live_gateway_measured(gateway_config(config))
    return {
        "run_id": config.run_id,
        "mode": "load",
        "config": config.__dict__,
        **result.throughput,
        "donors": list(result.donor_records),
        "recipients": list(result.recipient_records),
    }
