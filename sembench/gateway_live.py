"""Live OpenAI-compatible gateway replay runner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sembench.exact_cache import ExactBlockIndex, full_block_tokens
from sembench.quality import quality_score, rouge_l_best, token_f1
from sembench.schema import RequestMetrics, WorkloadItem, read_jsonl
from sembench.tokenization import load_tokenizer


@dataclass(frozen=True)
class LiveGatewayConfig:
    """Gateway runner configuration.

    Donor requests can be sent directly to a backend URL so the benchmark can
    seed a known KV-resident donor. Recipient requests are sent through the
    gateway URL so TTFT, quality, and route headers are measured on the same
    path clients use.
    """

    manifest: str
    output: str
    gateway_url: str
    model: str
    donor_url: str | None = None
    tenant: str = "tenant-a"
    template: str = "rag-template-v1"
    block_size: int = 16
    tokenizer: str | None = None
    max_items: int | None = None
    donor_max_tokens: int = 1
    recipient_max_tokens: int = 24
    timeout_seconds: float = 900.0
    quality_threshold: float = 0.60


def run_live_gateway(config: LiveGatewayConfig) -> list[RequestMetrics]:
    items = read_jsonl(config.manifest, max_items=config.max_items)
    tokenizer = load_tokenizer(config.tokenizer)
    results: list[RequestMetrics] = []
    donor_base = (config.donor_url or config.gateway_url).rstrip("/")
    gateway_base = config.gateway_url.rstrip("/")

    for item in items:
        donor_ids = []
        response: dict[str, Any] = {}
        error: str | None = None
        start = time.perf_counter()
        try:
            for donor in item.donor_prompts:
                donor_ids.append(donor.donor_id)
                _chat_completion(
                    base_url=donor_base,
                    model=config.model,
                    prompt=donor.text,
                    max_tokens=config.donor_max_tokens,
                    tenant=_tenant_for_item(item, config),
                    template=_template_for_item(item, config),
                    timeout_seconds=config.timeout_seconds,
                )
            response = _chat_completion(
                base_url=gateway_base,
                model=config.model,
                prompt=item.recipient_prompt,
                max_tokens=config.recipient_max_tokens,
                tenant=_tenant_for_item(item, config),
                template=_template_for_item(item, config),
                timeout_seconds=config.timeout_seconds,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - start) * 1000
        results.append(
            _metrics_from_item(
                item=item,
                tokenizer=tokenizer,
                config=config,
                donor_ids=donor_ids,
                response=response,
                latency_ms=latency_ms,
                error=error,
            )
        )
    return results


def _chat_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    tenant: str,
    template: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-tenant-id": tenant,
            "x-tokenizer-id": model,
            "x-chat-template-id": template,
        },
        method="POST",
    )
    start = time.perf_counter()
    ttft_ms: float | None = None
    output: list[str] = []
    usage: dict[str, Any] = {}
    try:
        with urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310 - staging benchmark.
            headers = {key.lower(): value for key, value in resp.headers.items()}
            for raw in resp:
                line = raw.strip()
                if not line:
                    continue
                text = line.decode("utf-8", errors="replace")
                if text.startswith("data: "):
                    text = text[6:]
                if text == "[DONE]":
                    continue
                try:
                    chunk = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or choice.get("text") or ""
                    if piece and ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    if piece:
                        output.append(piece)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "error": f"HTTP {exc.code}: {body}",
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
    except URLError as exc:
        return {
            "error": str(exc.reason),
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
    return {
        "output_text": "".join(output),
        "usage": usage,
        "headers": headers,
        "ttft_ms": ttft_ms,
        "latency_ms": (time.perf_counter() - start) * 1000,
    }


def _metrics_from_item(
    *,
    item: WorkloadItem,
    tokenizer,
    config: LiveGatewayConfig,
    donor_ids: list[str],
    response: dict[str, Any],
    latency_ms: float,
    error: str | None,
) -> RequestMetrics:
    donor_tokens = {donor.donor_id: tokenizer.encode(donor.text) for donor in item.donor_prompts}
    recipient_tokens = tokenizer.encode(item.recipient_prompt)
    exact = ExactBlockIndex(config.block_size)
    for donor_id, tokens in donor_tokens.items():
        exact.add(donor_id, tokens)
    exact_lookup = exact.lookup(recipient_tokens)

    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or len(recipient_tokens))
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = details.get("cached_tokens")
    backend_confirmed_tokens = int(cached_tokens) if cached_tokens is not None else None
    backend_confirmed_blocks = (
        backend_confirmed_tokens // config.block_size
        if backend_confirmed_tokens is not None
        else None
    )
    output_text = response.get("output_text") or ""
    answer_score = quality_score(output_text, item.answers)
    quality_pass = answer_score >= config.quality_threshold if answer_score is not None else None
    answer_f1 = token_f1(output_text, item.answers)
    answer_rouge = rouge_l_best(output_text, item.answers)
    headers = response.get("headers") or {}
    route_header = (
        headers.get("x-semantic-route") or headers.get("x-gateway-route") or headers.get("x-route")
    )

    return RequestMetrics(
        item_id=item.item_id,
        dataset=item.dataset,
        transform=item.transform,
        negative_control=item.negative_control,
        donor_count=len(item.donor_prompts),
        prompt_tokens=prompt_tokens,
        total_blocks=exact_lookup.total_blocks,
        exact_hit_blocks=exact_lookup.hit_blocks,
        exact_hit_tokens=full_block_tokens(exact_lookup.hit_blocks, config.block_size),
        semantic_candidate_blocks=0,
        semantic_candidate_tokens=0,
        semantic_eligible_blocks=0,
        semantic_eligible_tokens=0,
        backend_confirmed_blocks=backend_confirmed_blocks,
        backend_confirmed_tokens=backend_confirmed_tokens,
        donor_ids=donor_ids,
        route_endpoint_id=None,
        route_outcome=None,
        route_total_score=None,
        route_semantic_score=None,
        route_reason=None,
        gateway_route_header=route_header,
        ttft_ms=response.get("ttft_ms"),
        latency_ms=latency_ms,
        output_text=output_text[:2000],
        quality_pass=quality_pass,
        quality_score=answer_score,
        quality_f1=answer_f1,
        quality_rouge_l=answer_rouge,
        error=error or response.get("error"),
    )


def _tenant_for_item(item: WorkloadItem, config: LiveGatewayConfig) -> str:
    return str(item.metadata.get("tenant") or item.metadata.get("tenant_id") or config.tenant)


def _template_for_item(item: WorkloadItem, config: LiveGatewayConfig) -> str:
    return str(item.metadata.get("template") or item.metadata.get("template_id") or config.template)
