"""Optional live SGLang replay runner."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from sembench.exact_cache import ExactBlockIndex, full_block_tokens
from sembench.quality import quality_score, rouge_l_best, token_f1
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
    paired: bool = False
    warmup_requests: int = 0
    cooldown_ms: int = 0
    capture_logprobs: bool = False
    top_logprobs_num: int = 8


async def run_live_sglang(config: LiveSglangConfig) -> list[RequestMetrics]:
    """Replay manifest items against a running SGLang /generate endpoint."""
    try:
        import aiohttp
    except ImportError as e:
        raise RuntimeError(
            "live SGLang replay requires aiohttp. Install with: python -m pip install -e '.[live]'"
        ) from e

    items = read_jsonl(config.manifest, max_items=config.max_items)
    tokenizer = load_tokenizer(config.tokenizer)
    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    base_url = config.base_url.rstrip("/")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        transport = _HttpTransport(
            session=session,
            base_url=base_url,
            capture_logprobs=config.capture_logprobs,
            top_logprobs_num=config.top_logprobs_num,
        )
        return await replay_items(
            items=items, transport=transport, config=config, tokenizer=tokenizer
        )


class _HttpTransport:
    """Real SGLang transport; tests substitute a fake with the same surface."""

    def __init__(
        self, *, session, base_url: str, capture_logprobs: bool = False, top_logprobs_num: int = 8
    ) -> None:
        self._session = session
        self._base_url = base_url
        self._capture_logprobs = capture_logprobs
        self._top_logprobs_num = top_logprobs_num

    async def flush(self) -> None:
        await _post_json(self._session, f"{self._base_url}/flush_cache", {})

    async def generate(self, text: str, max_new_tokens: int) -> dict[str, Any]:
        return await _send_generate(
            self._session,
            f"{self._base_url}/generate",
            text,
            max_new_tokens,
            capture_logprobs=self._capture_logprobs,
            top_logprobs_num=self._top_logprobs_num,
        )


async def replay_items(
    *,
    items: list[WorkloadItem],
    transport,
    config: LiveSglangConfig,
    tokenizer,
) -> list[RequestMetrics]:
    results: list[RequestMetrics] = []
    for _ in range(config.warmup_requests):
        # Warm the serving stack (compilation, allocator, page tables) before
        # any measured request; responses are discarded.
        await transport.generate("warmup request please ignore", 8)
    for item in items:
        if config.paired:
            cold = await _run_arm(transport=transport, item=item, config=config, seed_donors=False)
            results.append(
                _metrics_from_live_item(
                    item=item, tokenizer=tokenizer, config=config, response=cold, arm="cold"
                )
            )
            if config.cooldown_ms > 0:
                await asyncio.sleep(config.cooldown_ms / 1000)
            warm = await _run_arm(transport=transport, item=item, config=config, seed_donors=True)
            results.append(
                _metrics_from_live_item(
                    item=item, tokenizer=tokenizer, config=config, response=warm, arm="warm"
                )
            )
        else:
            response = await _run_arm(
                transport=transport,
                item=item,
                config=config,
                seed_donors=True,
                flush=config.flush_per_item,
            )
            results.append(
                _metrics_from_live_item(
                    item=item, tokenizer=tokenizer, config=config, response=response
                )
            )
    return results


async def _run_arm(
    *,
    transport,
    item: WorkloadItem,
    config: LiveSglangConfig,
    seed_donors: bool,
    flush: bool = True,
) -> dict[str, Any]:
    if flush:
        await transport.flush()
    if seed_donors:
        for donor in item.donor_prompts:
            await transport.generate(donor.text, config.donor_max_new_tokens)
        if config.post_donor_delay_ms > 0:
            await asyncio.sleep(config.post_donor_delay_ms / 1000)
    return await transport.generate(item.recipient_prompt, config.recipient_max_new_tokens)


def _metrics_from_live_item(
    *,
    item: WorkloadItem,
    tokenizer,
    config: LiveSglangConfig,
    response: dict[str, Any],
    arm: str = "single",
) -> RequestMetrics:
    donor_tokens = {donor.donor_id: tokenizer.encode(donor.text) for donor in item.donor_prompts}
    recipient_tokens = tokenizer.encode(item.recipient_prompt)
    exact = ExactBlockIndex(config.block_size)
    for donor_id, tokens in donor_tokens.items():
        exact.add(donor_id, tokens)
    exact_lookup = exact.lookup(recipient_tokens)

    cached_tokens = int(response.get("cached_tokens") or 0)
    confirmed_blocks = cached_tokens // config.block_size
    # A cold arm reporting more than one block of cached tokens means the
    # flush did not take (stale-cache contamination): the arm's numbers are
    # untrustworthy and the pair must be excluded from paired aggregates.
    flush_contaminated = cached_tokens > config.block_size if arm == "cold" else None
    output_text = response.get("output_text") or ""
    answer_score = quality_score(output_text, item.answers)
    quality_pass = answer_score >= config.quality_threshold if answer_score is not None else None
    answer_f1 = token_f1(output_text, item.answers)
    answer_rouge = rouge_l_best(output_text, item.answers)

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
        quality_f1=answer_f1,
        quality_rouge_l=answer_rouge,
        arm=arm,
        flush_contaminated=flush_contaminated,
        output_token_ids=response.get("output_token_ids"),
        output_top_logprobs=response.get("output_top_logprobs"),
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
    capture_logprobs: bool = False,
    top_logprobs_num: int = 8,
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
    if capture_logprobs:
        payload["return_logprob"] = True
        payload["top_logprobs_num"] = top_logprobs_num
    start = time.perf_counter()
    ttft_ms: float | None = None
    prompt_tokens = 0
    cached_tokens = 0
    output_text = ""
    output_token_logprobs: list | None = None
    output_top_logprobs: list | None = None
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
                    if meta.get("output_token_logprobs") is not None:
                        output_token_logprobs = meta["output_token_logprobs"]
                    if meta.get("output_top_logprobs") is not None:
                        output_top_logprobs = meta["output_top_logprobs"]
                if data.get("output_ids") and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                usage = data.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = int(details.get("cached_tokens") or cached_tokens)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    result: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "ttft_ms": ttft_ms,
        "latency_ms": (time.perf_counter() - start) * 1000,
        "output_text": output_text,
        "error": error,
    }
    if output_token_logprobs is not None:
        # sglang entries are [logprob, token_id, text]; keep chosen ids.
        result["output_token_ids"] = [int(entry[1]) for entry in output_token_logprobs]
    if output_top_logprobs is not None:
        # per position: list of [logprob, token_id, text] → (token_id, logprob)
        result["output_top_logprobs"] = [
            [(int(entry[1]), float(entry[0])) for entry in position]
            for position in output_top_logprobs
        ]
    return result


def run_live_sglang_sync(config: LiveSglangConfig) -> list[RequestMetrics]:
    return asyncio.run(run_live_sglang(config))
