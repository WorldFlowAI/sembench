"""Concurrent-load runner: request throughput and TTFT under load.

Each manifest item is one stream: its donor prompts, a settle delay, then
its recipient prompt. `concurrency` streams run at once against an
OpenAI-compatible chat endpoint (streamed, so TTFT is measured). Run the
same manifest once with the reuse layer off and once with it on; the two
documents give requests per second, output tokens per second, and TTFT
percentiles for donors and recipients under identical load.
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from sembench.gateway_live import _chat_completion
from sembench.schema import WorkloadItem, read_jsonl


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


def summarize_load(
    *,
    donors: list[dict[str, Any]],
    recipients: list[dict[str, Any]],
    wall_seconds: float,
    items: int,
    concurrency: int,
) -> dict[str, Any]:
    """Aggregate per-request records into the load document (pure)."""
    n_req = len(donors) + len(recipients)
    out_tokens = sum(int(r.get("output_tokens") or 0) for r in donors + recipients)
    errors = sum(1 for r in donors + recipients if r.get("error"))
    return {
        "items": items,
        "requests": n_req,
        "errors": errors,
        "concurrency": concurrency,
        "wall_seconds": round(wall_seconds, 3),
        "requests_per_second": round(n_req / wall_seconds, 4) if wall_seconds > 0 else None,
        "output_tokens_per_second": round(out_tokens / wall_seconds, 2) if wall_seconds > 0 else None,
        "donor_ttft_ms": summarize([r["ttft_ms"] for r in donors if r.get("ttft_ms") is not None]),
        "recipient_ttft_ms": summarize(
            [r["ttft_ms"] for r in recipients if r.get("ttft_ms") is not None]
        ),
        "recipient_latency_ms": summarize(
            [r["latency_ms"] for r in recipients if r.get("latency_ms") is not None]
        ),
    }


def _record(item_id: str, role: str, response: dict[str, Any]) -> dict[str, Any]:
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


def _run_item(config: LoadConfig, item: WorkloadItem) -> tuple[list[dict], dict]:
    donors = []
    for donor in item.donor_prompts:
        response = _chat_completion(
            base_url=config.gateway_url.rstrip("/"),
            model=config.model,
            prompt=donor.text,
            max_tokens=config.donor_max_tokens,
            tenant=config.tenant,
            template=config.template,
            timeout_seconds=config.timeout_seconds,
        )
        donors.append(_record(item.item_id, "donor", response))
    if config.post_donor_delay_ms > 0:
        time.sleep(config.post_donor_delay_ms / 1000)
    response = _chat_completion(
        base_url=config.gateway_url.rstrip("/"),
        model=config.model,
        prompt=item.recipient_prompt,
        max_tokens=config.recipient_max_tokens,
        tenant=config.tenant,
        template=config.template,
        timeout_seconds=config.timeout_seconds,
    )
    return donors, _record(item.item_id, "recipient", response)


def run_load(config: LoadConfig) -> dict[str, Any]:
    items = read_jsonl(config.manifest, max_items=config.max_items)
    donors: list[dict[str, Any]] = []
    recipients: list[dict[str, Any]] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
        for item_donors, recipient in pool.map(lambda it: _run_item(config, it), items):
            donors.extend(item_donors)
            recipients.append(recipient)
    wall = time.perf_counter() - start
    doc = {
        "run_id": config.run_id,
        "mode": "load",
        "config": config.__dict__,
        **summarize_load(
            donors=donors,
            recipients=recipients,
            wall_seconds=wall,
            items=len(items),
            concurrency=config.concurrency,
        ),
        "donors": donors,
        "recipients": recipients,
    }
    return doc
