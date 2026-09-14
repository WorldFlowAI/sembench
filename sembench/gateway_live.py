"""Live OpenAI-compatible gateway replay runner."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sembench.dispatch import Stage, StageOutcome, run_stages
from sembench.exact_cache import ExactBlockIndex, full_block_tokens
from sembench.pairing import COLD_ARM, SINGLE_ARM, ReplayStep, replay_plan
from sembench.quality import quality_score, rouge_l_best, token_f1
from sembench.replay_stages import build_stages
from sembench.schema import RequestMetrics, WorkloadItem, read_jsonl
from sembench.throughput import request_record, summarize_throughput
from sembench.tokenization import load_tokenizer

# Response headers a router may use to report its placement decision, most
# specific vocabulary first. Routers disagree on names, so read all of them.
ROUTE_OUTCOME_HEADERS = (
    "x-synapse-route-outcome",
    "x-synapse-route",
    "x-semblend-routing-path",
    "x-semrouter-path",
    "x-semantic-route",
    "x-gateway-route",
    "x-route",
)
ROUTE_WORKER_HEADERS = (
    "x-synapse-route-worker",
    "x-semblend-routing-worker",
)
ROUTE_SIMILARITY_HEADERS = (
    "x-synapse-route-similarity",
    "x-semblend-routing-similarity",
)
# Short outcome labels folded onto the long-form ones so a run behind either
# router aggregates. The header value is kept verbatim in gateway_route_header.
ROUTE_OUTCOME_ALIASES = {
    "semantic": "semantic_placement",
    "cold": "cold_route",
    "exact": "exact_route",
    "fail_open": "error_fallback",
}


@dataclass(frozen=True)
class LiveGatewayConfig:
    """Gateway runner configuration.

    Donor requests can be sent directly to a backend URL so the benchmark can
    seed a known KV-resident donor. Recipient requests are sent through the
    gateway URL so TTFT, quality, and route headers are measured on the same
    path clients use.

    For a multi-worker fleet, pass every worker endpoint in ``worker_urls``:
    each donor is then seeded directly on one deterministically chosen worker
    while recipients still go through ``gateway_url``, so the router's placement
    decision is the thing under measurement. With ``worker_urls`` empty the
    single-endpoint path is unchanged.
    """

    manifest: str
    output: str
    gateway_url: str
    model: str
    donor_url: str | None = None
    worker_urls: tuple[str, ...] = ()
    tenant: str = "tenant-a"
    template: str = "rag-template-v1"
    block_size: int = 16
    tokenizer: str | None = None
    max_items: int | None = None
    donor_max_tokens: int = 1
    recipient_max_tokens: int = 24
    timeout_seconds: float = 900.0
    quality_threshold: float = 0.60
    # Engines that index donors off the request path need a moment before
    # the recipient can see them; counted in latency_ms, not in ttft_ms.
    post_donor_delay_ms: int = 0
    # Which arm of a paired cold/warm comparison every row belongs to. Stamped
    # onto RequestMetrics.arm, which is what paired_summary joins on.
    arm: str = SINGLE_ARM
    # Run both arms in one process: per item a cold twin (no donors) then a
    # warm twin (donors), adjacent in the stream. Needs reset_urls.
    paired: bool = False
    # Engine cache-reset endpoints POSTed before each arm, e.g. vLLM's
    # /reset_prefix_cache?reset_external=true. One per worker.
    reset_urls: tuple[str, ...] = ()
    # How many request streams are in flight at once. 1 is the serial path,
    # unchanged; above 1 the throughput document is what the run is for.
    concurrency: int = 1
    # Minimum donor -> recipient separation, in *completed requests*. Under
    # concurrency a manifest gap measured in stream positions means nothing:
    # without this the two can be in flight together.
    min_donor_gap_requests: int = 0


@dataclass(frozen=True)
class GatewayRunResult:
    """One arm: the per-request rows plus its throughput document."""

    requests: tuple[RequestMetrics, ...]
    throughput: dict[str, Any]
    donor_records: tuple[dict[str, Any], ...] = ()
    recipient_records: tuple[dict[str, Any], ...] = ()


def run_live_gateway(config: LiveGatewayConfig) -> list[RequestMetrics]:
    """Replay a manifest and return one row per request."""
    return list(run_live_gateway_measured(config).requests)


def run_live_gateway_measured(config: LiveGatewayConfig) -> GatewayRunResult:
    """Replay a manifest, returning the rows and the arm's throughput document.

    At ``concurrency=1`` this is the serial replay: one stream, each request
    issued and awaited before the next, which is what the TTFT and quality
    arms need. Above 1, requests are issued by a bounded dispatcher that still
    submits in stream order and still holds every recipient behind its donor's
    completion (:func:`build_stages`), so the throughput gate and the serial
    arms replay the same sequence.
    """
    items = read_jsonl(config.manifest, max_items=config.max_items)
    tokenizer = load_tokenizer(config.tokenizer)
    plan = replay_plan(items, paired=config.paired, arm=config.arm)
    if config.paired and not config.reset_urls:
        raise ValueError(
            "paired gateway runs require reset_urls: without an engine cache reset "
            "the cold twin is warmed by the arm before it and the pair measures nothing"
        )
    concurrency = max(1, int(config.concurrency))
    if concurrency > 1 and config.reset_urls:
        raise ValueError(
            "concurrency > 1 cannot be combined with reset_urls: the per-step cache reset "
            "would fire while other requests are in flight and flush their KV mid-run. "
            "Run the paired/cold arms serially and the throughput arms without resets"
        )
    if concurrency == 1:
        return _replay_serial(config, plan, tokenizer)
    return _replay_concurrent(config, plan, tokenizer, concurrency=concurrency)


def _replay_serial(
    config: LiveGatewayConfig,
    plan: list[ReplayStep],
    tokenizer,
) -> GatewayRunResult:
    results: list[RequestMetrics] = []
    donor_records: list[dict[str, Any]] = []
    recipient_records: list[dict[str, Any]] = []
    donor_base = (config.donor_url or config.gateway_url).rstrip("/")
    gateway_base = config.gateway_url.rstrip("/")
    worker_urls = parse_worker_urls(config.worker_urls)

    wall_start = time.perf_counter()
    settle_seconds = 0.0
    for step in plan:
        item = step.item
        donor_ids: list[str] = []
        donor_worker_ids: list[str] = []
        response: dict[str, Any] = {}
        error: str | None = None
        cache_reset = (
            reset_engine_caches(config.reset_urls, timeout_seconds=config.timeout_seconds)
            if config.reset_urls
            else None
        )
        start = time.perf_counter()
        try:
            for donor in item.donor_prompts if step.seed_donors else ():
                donor_ids.append(donor.donor_id)
                donor_base_url = (
                    select_worker_url(worker_urls, donor.donor_id) if worker_urls else donor_base
                )
                donor_worker_ids.append(donor_base_url)
                donor_records.append(
                    request_record(
                        item.item_id,
                        "donor",
                        _donor_request(item=item, donor=donor, config=config, base_url=donor_base_url),
                    )
                )
            if step.seed_donors and config.post_donor_delay_ms > 0:
                time.sleep(config.post_donor_delay_ms / 1000)
                settle_seconds += config.post_donor_delay_ms / 1000
            response = _recipient_request(item=item, config=config, base_url=gateway_base)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - start) * 1000
        recipient_records.append(request_record(item.item_id, "recipient", response))
        results.append(
            _metrics_from_item(
                item=item,
                tokenizer=tokenizer,
                config=config,
                donor_ids=donor_ids,
                donor_worker_ids=donor_worker_ids,
                recipient_url=gateway_base,
                worker_urls=worker_urls,
                response=response,
                latency_ms=latency_ms,
                error=error,
                arm=step.arm,
                cache_reset=cache_reset,
            )
        )
    wall_seconds = time.perf_counter() - wall_start

    return GatewayRunResult(
        requests=tuple(results),
        throughput=_throughput_document(
            config=config,
            donor_records=donor_records,
            recipient_records=recipient_records,
            items=len(plan),
            wall_seconds=wall_seconds,
            concurrency=1,
            max_in_flight=1,
            gap_forced=0,
            idle_seconds=settle_seconds,
        ),
        donor_records=tuple(donor_records),
        recipient_records=tuple(recipient_records),
    )


def _replay_concurrent(
    config: LiveGatewayConfig,
    plan: list[ReplayStep],
    tokenizer,
    *,
    concurrency: int,
) -> GatewayRunResult:
    donor_base = (config.donor_url or config.gateway_url).rstrip("/")
    gateway_base = config.gateway_url.rstrip("/")
    worker_urls = parse_worker_urls(config.worker_urls)
    stages = build_stages(
        plan,
        min_donor_gap_requests=config.min_donor_gap_requests,
        settle_seconds=max(0, config.post_donor_delay_ms) / 1000,
    )
    # Written by the donor stage, read by that step's recipient stage, which
    # the dispatcher only releases once the donor stage has completed.
    donor_failures: dict[int, str] = {}

    def execute(stage: Stage) -> dict[str, Any]:
        step = plan[stage.item_index]
        if stage.kind == "donors":
            try:
                return _send_donors(
                    step.item, config=config, donor_base=donor_base, worker_urls=worker_urls
                )
            except Exception as exc:
                donor_failures[stage.item_index] = f"{type(exc).__name__}: {exc}"
                raise
        if stage.item_index in donor_failures:
            # The serial path never sends a recipient whose donors raised.
            return {"response": {}}
        return {
            "response": _recipient_request(
                item=step.item, config=config, base_url=gateway_base
            )
        }

    report = run_stages(stages, execute=execute, concurrency=concurrency)
    results, donor_records, recipient_records = _rows_from_outcomes(
        plan,
        report.outcomes,
        config=config,
        tokenizer=tokenizer,
        gateway_base=gateway_base,
        worker_urls=worker_urls,
    )

    return GatewayRunResult(
        requests=tuple(results),
        throughput=_throughput_document(
            config=config,
            donor_records=donor_records,
            recipient_records=recipient_records,
            items=len(plan),
            wall_seconds=report.wall_seconds,
            concurrency=report.concurrency,
            max_in_flight=report.max_in_flight,
            gap_forced=report.gap_forced,
            idle_seconds=report.idle_seconds,
        ),
        donor_records=tuple(donor_records),
        recipient_records=tuple(recipient_records),
    )



def _rows_from_outcomes(
    plan: list[ReplayStep],
    outcomes: Sequence[StageOutcome],
    *,
    config: LiveGatewayConfig,
    tokenizer,
    gateway_base: str,
    worker_urls: Sequence[str],
) -> tuple[list[RequestMetrics], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fold the completed stages back into one row per item, in stream order.

    Metrics are built after the dispatcher has stopped the clock: tokenizing
    prompts and scoring answers on the request path would be client-side work
    charged to the arm's throughput.
    """
    by_stage: dict[tuple[int, str], StageOutcome] = {
        (outcome.stage.item_index, outcome.stage.kind): outcome for outcome in outcomes
    }
    results: list[RequestMetrics] = []
    donor_records: list[dict[str, Any]] = []
    recipient_records: list[dict[str, Any]] = []
    for position, step in enumerate(plan):
        donors = by_stage.get((position, "donors"))
        recipient = by_stage.get((position, "recipient"))
        donor_value = (donors.value if donors is not None else None) or {}
        donor_responses = list(donor_value.get("responses") or [])
        recipient_value = (recipient.value if recipient is not None else None) or {}
        response = recipient_value.get("response") or {}
        donor_records.extend(
            request_record(step.item.item_id, "donor", one) for one in donor_responses
        )
        recipient_records.append(request_record(step.item.item_id, "recipient", response))
        results.append(
            _metrics_from_item(
                item=step.item,
                tokenizer=tokenizer,
                config=config,
                donor_ids=list(donor_value.get("donor_ids") or []),
                donor_worker_ids=list(donor_value.get("worker_ids") or []),
                recipient_url=gateway_base,
                worker_urls=worker_urls,
                response=response,
                latency_ms=_service_latency_ms(donor_responses, response),
                error=(donors.error if donors is not None else None)
                or (recipient.error if recipient is not None else None),
                arm=step.arm,
                cache_reset=None,
            )
        )
    return results, donor_records, recipient_records


def _send_donors(
    item: WorkloadItem,
    *,
    config: LiveGatewayConfig,
    donor_base: str,
    worker_urls: Sequence[str],
) -> dict[str, Any]:
    donor_ids: list[str] = []
    worker_ids: list[str] = []
    responses: list[dict[str, Any]] = []
    for donor in item.donor_prompts:
        base_url = select_worker_url(worker_urls, donor.donor_id) if worker_urls else donor_base
        donor_ids.append(donor.donor_id)
        worker_ids.append(base_url)
        responses.append(_donor_request(item=item, donor=donor, config=config, base_url=base_url))
    return {"donor_ids": donor_ids, "worker_ids": worker_ids, "responses": responses}


def _donor_request(
    *,
    item: WorkloadItem,
    donor,
    config: LiveGatewayConfig,
    base_url: str,
) -> dict[str, Any]:
    return _chat_completion(
        base_url=base_url,
        model=config.model,
        prompt=donor.text,
        max_tokens=config.donor_max_tokens,
        tenant=_tenant_for_item(item, config),
        template=_template_for_item(item, config),
        timeout_seconds=config.timeout_seconds,
    )


def _recipient_request(
    *,
    item: WorkloadItem,
    config: LiveGatewayConfig,
    base_url: str,
) -> dict[str, Any]:
    return _chat_completion(
        base_url=base_url,
        model=config.model,
        prompt=item.recipient_prompt,
        max_tokens=config.recipient_max_tokens,
        tenant=_tenant_for_item(item, config),
        template=_template_for_item(item, config),
        timeout_seconds=config.timeout_seconds,
    )


def _service_latency_ms(
    donor_responses: Sequence[dict[str, Any]],
    response: dict[str, Any],
) -> float:
    """Server time this item cost, excluding dispatcher queue and gap waits.

    Under concurrency an item's wall span would include time it spent waiting
    for its donor gap, which is the harness's own delay and not latency the
    engine produced.
    """
    served = [*donor_responses, response]
    return sum(float(one.get("latency_ms") or 0.0) for one in served)


def _throughput_document(
    *,
    config: LiveGatewayConfig,
    donor_records: list[dict[str, Any]],
    recipient_records: list[dict[str, Any]],
    items: int,
    wall_seconds: float,
    concurrency: int,
    max_in_flight: int,
    gap_forced: int,
    idle_seconds: float,
) -> dict[str, Any]:
    return {
        **summarize_throughput(
            donors=donor_records,
            recipients=recipient_records,
            wall_seconds=wall_seconds,
            items=items,
            concurrency=concurrency,
            post_donor_delay_ms=config.post_donor_delay_ms,
            idle_seconds=idle_seconds,
        ),
        "arm": config.arm,
        "max_in_flight": max_in_flight,
        "min_donor_gap_requests": config.min_donor_gap_requests,
        # Stages issued before their donor gap was met because nothing was in
        # flight to satisfy it. Non-zero means the manifest's gaps and its
        # stream order disagree; the throughput number stands, the gap does not.
        "gap_forced_stages": gap_forced,
    }


def reset_engine_caches(urls: Sequence[str], *, timeout_seconds: float = 60.0) -> bool:
    """POST every engine cache-reset endpoint; True only if all of them took.

    A failed reset is not an exception: the run continues and the arm is
    marked contaminated, which is far more useful than a dead run with no
    evidence of why the cold twin was warm.
    """
    reset = True
    for url in urls:
        request = Request(
            url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                reset = reset and 200 <= int(response.status) < 300
        except (HTTPError, URLError, OSError):
            reset = False
    return reset


def cold_arm_contaminated(
    *,
    arm: str,
    cache_reset: bool | None,
    confirmed_tokens: int | None,
    block_size: int,
) -> bool | None:
    """Whether a cold twin's cache reset failed to take.

    Only decidable when this runner actually reset the engine before the arm.
    A cross-run cold baseline with prefix caching on is *supposed* to hit its
    own local cache, so flagging that as contamination would exclude exactly
    the pairs the baseline exists to provide.
    """
    if arm != COLD_ARM or cache_reset is None:
        return None
    if not cache_reset:
        return True
    if confirmed_tokens is None:
        return None
    return confirmed_tokens > block_size


def parse_worker_urls(values: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize repeatable --worker-url values into an ordered, deduped tuple.

    Accepts a single string, a comma-separated string, or any iterable of
    either, so `--worker-url a --worker-url b,c` and `worker_urls=("a", "b")`
    both work.
    """
    if values is None:
        return ()
    raw = [values] if isinstance(values, str) else list(values)
    ordered: list[str] = []
    for entry in raw:
        for part in str(entry).split(","):
            url = part.strip().rstrip("/")
            if url and url not in ordered:
                ordered.append(url)
    return tuple(ordered)


def select_worker_url(worker_urls: Sequence[str], key: str) -> str:
    """Deterministically place `key` (a donor id) on one worker.

    Hashed, not a round-robin counter: placement must not move when items are
    reordered, resumed, or issued concurrently, or a re-run stops being a
    replay of the same fleet layout.
    """
    if not worker_urls:
        raise ValueError("select_worker_url requires at least one worker URL")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return worker_urls[int.from_bytes(digest[:8], "big") % len(worker_urls)]


def normalize_route_outcome(value: str | None) -> str | None:
    """Fold a raw route-outcome header onto the long-form label vocabulary."""
    if not value:
        return None
    label = str(value).strip().lower()
    return ROUTE_OUTCOME_ALIASES.get(label, label) or None


def _first_header(headers: dict[str, str], names: Sequence[str]) -> str | None:
    for name in names:
        value = headers.get(name)
        if value:
            return str(value)
    return None


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


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
    donor_worker_ids: list[str] | None = None,
    recipient_url: str | None = None,
    worker_urls: Sequence[str] = (),
    arm: str = SINGLE_ARM,
    cache_reset: bool | None = None,
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
    route_header = _first_header(headers, ROUTE_OUTCOME_HEADERS)
    route_worker = _first_header(headers, ROUTE_WORKER_HEADERS)
    route_similarity = _float_or_none(_first_header(headers, ROUTE_SIMILARITY_HEADERS))
    # A recipient sent straight at a worker identifies its server by URL; behind
    # a gateway only the router can say, so absent the header this stays None.
    direct_worker = recipient_url if recipient_url in tuple(worker_urls) else None

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
        route_endpoint_id=route_worker,
        route_outcome=normalize_route_outcome(route_header),
        route_total_score=None,
        route_semantic_score=route_similarity,
        route_reason=None,
        gateway_route_header=route_header,
        worker_id=route_worker or direct_worker,
        donor_worker_ids=list(donor_worker_ids or []),
        ttft_ms=response.get("ttft_ms"),
        latency_ms=latency_ms,
        output_text=output_text[:2000],
        quality_pass=quality_pass,
        quality_score=answer_score,
        quality_f1=answer_f1,
        quality_rouge_l=answer_rouge,
        arm=arm,
        flush_contaminated=cold_arm_contaminated(
            arm=arm,
            cache_reset=cache_reset,
            confirmed_tokens=backend_confirmed_tokens,
            block_size=config.block_size,
        ),
        error=error or response.get("error"),
    )


def _tenant_for_item(item: WorkloadItem, config: LiveGatewayConfig) -> str:
    return str(item.metadata.get("tenant") or item.metadata.get("tenant_id") or config.tenant)


def _template_for_item(item: WorkloadItem, config: LiveGatewayConfig) -> str:
    return str(item.metadata.get("template") or item.metadata.get("template_id") or config.template)
