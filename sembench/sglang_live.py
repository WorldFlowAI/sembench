"""Optional live SGLang replay runner."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from sembench.exact_cache import ExactBlockIndex, full_block_tokens
from sembench.quality import quality_score
from sembench.schema import RequestMetrics, WorkloadItem, read_jsonl
from sembench.tokenization import load_tokenizer


@dataclass(frozen=True)
class LiveSglangConfig:
    """Live SGLang runner configuration."""

    manifest: str
    output: str
    base_url: str
    model: str
    block_size: int = 16
    tokenizer: str | None = None
    max_items: int | None = None
    donor_max_new_tokens: int = 1
    recipient_max_new_tokens: int = 16
    post_donor_delay_ms: int = 0
    flush_per_item: bool = True
    timeout_seconds: int = 3600
    quality_threshold: float = 0.60


async def run_live_sglang(config: LiveSglangConfig) -> list[RequestMetrics]:
    """Replay manifest items against a running SGLang /generate endpoint."""
    try:
        import aiohttp
    except ImportError as e:
        raise RuntimeError(
            "live SGLang replay requires aiohttp. Install with: "
            "python -m pip install -e '.[live]'"
        ) from e

    items = read_jsonl(config.manifest, max_items=config.max_items)
    tokenizer = load_tokenizer(config.tokenizer)
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    base_url = config.base_url.rstrip("/")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results: list[RequestMetrics] = []
        for item in items:
            if config.flush_per_item:
                await _post_json(session, f"{base_url}/flush_cache", {})
            for donor in item.donor_prompts:
                await _send_generate(
                    session,
                    f"{base_url}/generate",
                    donor.text,
                    config.donor_max_new_tokens,
                )
            if config.post_donor_delay_ms > 0:
                await asyncio.sleep(config.post_donor_delay_ms / 1000)
            recipient = await _send_generate(
                session,
                f"{base_url}/generate",
                item.recipient_prompt,
                config.recipient_max_new_tokens,
            )
            results.append(
                _metrics_from_live_item(
                    item=item,
                    tokenizer=tokenizer,
                    config=config,
                    response=recipient,
                )
            )
        return results


def _metrics_from_live_item(
    *,
    item: WorkloadItem,
    tokenizer,
    config: LiveSglangConfig,
    response: dict[str, Any],
) -> RequestMetrics:
    donor_tokens = {donor.donor_id: tokenizer.encode(donor.text) for donor in item.donor_prompts}
    recipient_tokens = tokenizer.encode(item.recipient_prompt)
    exact = ExactBlockIndex(config.block_size)
    for donor_id, tokens in donor_tokens.items():
        exact.add(donor_id, tokens)
    exact_lookup = exact.lookup(recipient_tokens)

    cached_tokens = int(response.get("cached_tokens") or 0)
    confirmed_blocks = cached_tokens // config.block_size
    output_text = response.get("output_text") or ""
    answer_score = quality_score(output_text, item.answers)
    quality_pass = (
        answer_score >= config.quality_threshold
        if answer_score is not None
        else None
    )

    return RequestMetrics(
        item_id=item.item_id,
        dataset=item.dataset,
        transform=item.transform,
        negative_control=item.negative_control,
        donor_count=len(item.donor_prompts),
        prompt_tokens=int(response.get("prompt_tokens") or len(recipient_tokens)),
        total_blocks=exact_lookup.total_blocks,
        exact_hit_blocks=exact_lookup.hit_blocks,
        exact_hit_tokens=full_block_tokens(exact_lookup.hit_blocks, config.block_size),
        semantic_candidate_blocks=0,
        semantic_candidate_tokens=0,
        semantic_eligible_blocks=0,
        semantic_eligible_tokens=0,
        backend_confirmed_blocks=confirmed_blocks,
        backend_confirmed_tokens=cached_tokens,
        ttft_ms=response.get("ttft_ms"),
        latency_ms=response.get("latency_ms"),
        output_text=output_text[:2000],
        quality_pass=quality_pass,
        quality_score=answer_score,
        error=response.get("error"),
    )


async def _post_json(session, url: str, payload: dict[str, Any]) -> None:
    try:
        async with session.post(url=url, json=payload) as response:
            await response.read()
    except Exception:
        return


async def _send_generate(
    session,
    url: str,
    text: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    payload = {
        "text": text,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft_ms: float | None = None
    prompt_tokens = 0
    cached_tokens = 0
    output_text = ""
    error: str | None = None
    try:
        async with session.post(url=url, json=payload) as response:
            if response.status != 200:
                return {
                    "error": response.reason or f"HTTP {response.status}",
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
            async for raw in response.content:
                line = raw.strip()
                if not line:
                    continue
                text_line = line.decode("utf-8")
                if text_line.startswith("data: "):
                    text_line = text_line[6:]
                if text_line == "[DONE]":
                    continue
                try:
                    data = json.loads(text_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data.get("text"), str):
                    output_text = data["text"]
                meta = data.get("meta_info") or {}
                if meta:
                    prompt_tokens = int(meta.get("prompt_tokens") or prompt_tokens)
                    cached_tokens = int(meta.get("cached_tokens") or cached_tokens)
                if data.get("output_ids") and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                usage = data.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = int(details.get("cached_tokens") or cached_tokens)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "ttft_ms": ttft_ms,
        "latency_ms": (time.perf_counter() - start) * 1000,
        "output_text": output_text,
        "error": error,
    }


def run_live_sglang_sync(config: LiveSglangConfig) -> list[RequestMetrics]:
    return asyncio.run(run_live_sglang(config))
