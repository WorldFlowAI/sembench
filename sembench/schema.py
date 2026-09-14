"""Shared schemas for local semantic KV benchmark manifests and results."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "sembench.manifest.v1"
RESULT_VERSION = "sembench.result.v2"

# Provenance vocabulary for RequestMetrics.external_confirmed_tokens. The two
# sources are NOT interchangeable and the difference decides what may be said
# about a single request.
#
# CONNECTOR_AUDIT — the per-request connector audit stream, joined on the
# engine request id (B10). A genuine per-request number.
#
# ARM_PROMETHEUS_DELTA — the per-arm delta of ``vllm:external_prefix_cache_hits``
# scraped before/after the arm. That counter is a process-wide total: it says
# how much external KV the arm allocated in aggregate and cannot attribute any
# of it to a particular request. A row stamped with this source carries an
# ARM-LEVEL quantity, so it supports arm-level statements only — never "this
# request reused N tokens".
EXTERNAL_SOURCE_CONNECTOR_AUDIT = "connector_audit"
EXTERNAL_SOURCE_ARM_PROMETHEUS = "arm_prometheus_delta"
EXTERNAL_TOKEN_SOURCES = (EXTERNAL_SOURCE_CONNECTOR_AUDIT, EXTERNAL_SOURCE_ARM_PROMETHEUS)
PER_REQUEST_EXTERNAL_SOURCES = (EXTERNAL_SOURCE_CONNECTOR_AUDIT,)


@dataclass(frozen=True)
class SourceRecord:
    """Normalized source row loaded from a benchmark dataset."""

    source_id: str
    dataset: str
    context: str
    input: str
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DonorPrompt:
    """One donor request that should be seeded before the recipient request."""

    donor_id: str
    text: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadItem:
    """One donor/recipient replay item."""

    item_id: str
    dataset: str
    source_id: str
    transform: str
    donor_prompts: list[DonorPrompt]
    recipient_prompt: str
    input: str = ""
    answers: list[str] = field(default_factory=list)
    negative_control: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    benchmark_version: str = MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WorkloadItem":
        donors = [DonorPrompt(**donor) for donor in raw.get("donor_prompts", [])]
        return cls(
            item_id=raw["item_id"],
            dataset=raw["dataset"],
            source_id=raw["source_id"],
            transform=raw["transform"],
            donor_prompts=donors,
            recipient_prompt=raw["recipient_prompt"],
            input=raw.get("input", ""),
            answers=list(raw.get("answers", [])),
            negative_control=bool(raw.get("negative_control", False)),
            metadata=dict(raw.get("metadata", {})),
            benchmark_version=raw.get("benchmark_version", MANIFEST_VERSION),
        )


@dataclass(frozen=True)
class RequestMetrics:
    """Per-recipient metric record."""

    item_id: str
    dataset: str
    transform: str
    negative_control: bool
    donor_count: int
    prompt_tokens: int
    total_blocks: int
    exact_hit_blocks: int
    exact_hit_tokens: int
    semantic_candidate_blocks: int
    semantic_candidate_tokens: int
    semantic_eligible_blocks: int
    semantic_eligible_tokens: int
    # vLLM's usage.prompt_tokens_details.cached_tokens: LOCAL prefix cache plus
    # external KV transfer, summed by the engine before it reaches the API
    # (vllm/v1/metrics/stats.py:284). With prefix caching on, a repeated
    # document hits the local cache and shows up here, so this field is not
    # semantic-reuse evidence and must never be gated on as if it were.
    backend_confirmed_blocks: int | None = None
    backend_confirmed_tokens: int | None = None
    # Engine-reported fuzzy-admitted mass (cached_tokens_details["fuzzy"]).
    # On sglang's contiguous path this is a subset of backend_confirmed_tokens;
    # on the segments path it is scatter mass NOT counted there. Local prefix
    # hits never land here, so this IS semantic-reuse evidence.
    fuzzy_confirmed_tokens: int = 0
    # Confirmed reuse that came from the EXTERNAL KV connector only, with the
    # local prefix cache excluded — the one number that answers "did semantic
    # reuse happen" on a prefix-caching-on vLLM arm.
    #
    # None means the external split was never measured; it does NOT mean zero.
    # Read it together with external_confirmed_tokens_source, which says
    # whether the value is per-request or an arm-level aggregate.
    external_confirmed_tokens: int | None = None
    external_confirmed_tokens_source: str | None = None
    # The id the runner put in the request's X-Request-Id header, which vLLM
    # adopts as the engine request id (see sembench.connector_audit for the
    # exact vLLM call sites and the prefix/suffix it adds). This is the join
    # key to the connector's audit stream; without it the audit cannot be
    # attributed to a row and every audit-derived field below stays null.
    engine_request_id: str | None = None
    # The id the ENGINE echoed back on its response chunks (vLLM's
    # `chatcmpl-<sent id>`). Kept beside the sent id so a front end that
    # rewrites or drops X-Request-Id is reported as a mismatch count rather
    # than discovered later as rows the audit silently could not join.
    engine_response_id: str | None = None
    # Manifest expectations for this item, stamped by the row constructor from
    # WorkloadItem.metadata (see :func:`manifest_expectations`). They are pure
    # functions of the tokenizer and the connector's own
    # block_align_spans/supply_at_boundary, so they can be precomputed and
    # asserted against what the engine actually did: expected_supplied_tokens
    # > 0 is the statement "a compatible donor existed for this request", and
    # the pair feeds M1's integrity check (does the audited token_count match
    # what the offline model predicted?). None means the manifest made no
    # claim, which is not the same as a claim of zero.
    #
    # They are NOT M1's denominator. That denominator is the manifest's
    # traffic class -- section 4's alignment_given_opportunity is taken over
    # `same_doc_new_instruction` and `revised_doc` items, whatever the offline
    # model predicted for any one of them.
    expected_supplied_tokens: int | None = None
    expected_span_target_start: int | None = None
    # Traffic class from the manifest (no_reuse, same_doc_new_instruction,
    # revised_doc, rope_delta_sweep, exact_repeat, propagation_probe,
    # reworded_doc). Falls back to `transform` when the manifest carries the
    # class there instead.
    traffic_class: str | None = None
    # For a propagation_probe: the item whose request this one repeats
    # verbatim (manifest `parent_item_id`). M7 compares this row's answer
    # against that item's answer in the SAME arm -- the "served" output -- and
    # against this row's own answer in the cold arm, so the link has to be on
    # the row for the comparison to exist at all.
    propagation_parent_item_id: str | None = None
    # Connector-audit join results (sembench.connector_audit.join_requests).
    # audit_joined tri-states on purpose: None means no audit was joined at
    # all, False means the audit was read and held nothing for this request,
    # True means these fields are measurements. A rate computed over rows
    # whose audit_joined is None is a rate over nothing and must be null.
    audit_joined: bool | None = None
    # The boundary the connector actually saw, from the LAST advertise for
    # this request, or from the last boundary-missed event when nothing was
    # ever advertised.
    audit_observed_boundary: int | None = None
    audit_advertised_tokens: int | None = None
    audit_advertised_target_start: int | None = None
    # Did the observed boundary coincide with a snapped span's start? None
    # when no advertise carried spans to compare against.
    audit_boundary_at_span_start: bool | None = None
    audit_load_allocated: bool | None = None
    audit_materialized: bool | None = None
    audit_declined_reasons: list[str] | None = None
    # Did the connector's lookup find a donor for this request at all, and at
    # which boundary? This is M1's denominator for alignment_given_match:
    # section 4 divides the advertises by the LOOKUP HITS, not by the
    # requests, so a request whose provider found nothing is not counted as a
    # misalignment.
    audit_semantic_lookup_hit: bool | None = None
    audit_lookup_hit_boundary: int | None = None
    # The last semantic_span_boundary_missed event for this request: why the
    # boundary landed outside every span, and where it landed. The reason
    # vocabulary is section 4's boundary_miss_breakdown partition
    # (donor_not_captured / donor_too_short / below_min_semantic_span /
    # true_misalignment), plus `unclassified` for a miss event that carried
    # none of the fields the partition reads.
    audit_boundary_miss_reason: str | None = None
    audit_boundary_missed_at: int | None = None
    # Blocks the connector evicted from vLLM's exact prefix cache to stop a
    # semantically filled block being re-served through the local cache
    # without passing a gate. Section 4 (M7): until this counter exists and
    # reads non-zero on a contaminated workload, every lane-2 quality number
    # is unproven -- including a favourable one.
    audit_prefix_blocks_evicted: int | None = None
    # Engine-side TTFT from --enable-per-request-metrics, measured as
    # (first_token_ts - scheduled_ts) and therefore excluding queue wait, plus
    # the queue wait itself. Under concurrency, client-side ttft_ms below is
    # dominated by queueing and is not comparable across arms; these two are.
    engine_ttft_ms: float | None = None
    queue_time_ms: float | None = None
    semblend_found: bool = False
    semblend_similarity: float = 0.0
    semblend_reuse_ratio: float = 0.0
    semblend_latency_ms: float = 0.0
    donor_ids: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    route_endpoint_id: str | None = None
    route_outcome: str | None = None
    route_semantic_score: float | None = None
    route_total_score: float | None = None
    route_reason: str | None = None
    gateway_route_header: str | None = None
    # Fleet placement: which worker served the recipient, and which worker each
    # donor was seeded on. Empty/None on the single-endpoint path.
    worker_id: str | None = None
    donor_worker_ids: list[str] = field(default_factory=list)
    ttft_ms: float | None = None
    latency_ms: float | None = None
    output_text: str | None = None
    quality_pass: bool | None = None
    quality_score: float | None = None
    quality_f1: float | None = None
    quality_rouge_l: float | None = None
    arm: str = "single"
    flush_contaminated: bool | None = None
    output_token_ids: list[int] | None = None
    output_top_logprobs: list[list[tuple[int, float]]] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    """``metadata[key]`` as an int, or None when it says nothing.

    A missing key, an explicit null and an unparseable value all mean "the
    manifest made no claim" and must stay None: a 0 here would be read
    downstream as the manifest predicting that nothing could be supplied.
    """
    value = metadata.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def manifest_expectations(item: WorkloadItem) -> dict[str, Any]:
    """The manifest-side inputs to the connector-audit join, for one item.

    Every live row constructor stamps these, because they are the half of the
    join no engine can supply: what the offline model predicted the connector
    would do (``expected_supplied_tokens`` / ``expected_span_target_start``),
    which class of traffic the item is (``traffic_class`` -- M1's opportunity
    denominator and M7's probe set), and which earlier item a propagation
    probe repeats (``parent_item_id``).

    Absent keys stay None rather than becoming zeros: "the manifest made no
    claim" and "the manifest predicted nothing" are different statements and
    only the first one may be silent.
    """
    metadata = item.metadata or {}
    return {
        "expected_supplied_tokens": _metadata_int(metadata, "expected_supplied_tokens"),
        "expected_span_target_start": _metadata_int(metadata, "expected_span_target_start"),
        "traffic_class": _metadata_str(metadata, "traffic_class"),
        "propagation_parent_item_id": _metadata_str(metadata, "parent_item_id"),
    }


@dataclass(frozen=True)
class RunMetadata:
    """Reproducibility identity for one benchmark run.

    Every result JSON must be traceable to: which manifest (by checksum),
    which engine/backend, which arm of a paired cold/warm comparison, and
    which sembench code produced it.
    """

    run_id: str
    engine: str
    manifest_path: str
    manifest_sha256: str
    arm: str = "single"
    engine_version: str = ""
    backend_id: str = ""
    baseline_id: str = ""
    sembench_version: str = ""
    sembench_git_sha: str = ""
    sembench_git_dirty: bool = False
    semblend_version: str = ""
    timestamp_utc: str = ""
    python_version: str = field(default_factory=platform.python_version)
    run_host: str = field(default_factory=platform.node)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_sha256(path: str | Path) -> str:
    """SHA256 of the manifest file bytes (manifests are canonical JSONL)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return ""


def _git_state(repo_path: Path) -> tuple[str, bool]:
    def run(args: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(repo_path),
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    sha = run(["rev-parse", "--short=12", "HEAD"])
    dirty = bool(run(["status", "--porcelain"]))
    return sha, dirty


def collect_run_metadata(
    *,
    engine: str,
    manifest: str | Path,
    run_id: str | None = None,
    arm: str = "single",
    engine_version: str = "",
    backend_id: str = "",
    baseline_id: str = "",
) -> RunMetadata:
    sha, dirty = _git_state(Path(__file__).resolve().parents[1])
    return RunMetadata(
        run_id=run_id or uuid.uuid4().hex[:12],
        engine=engine,
        manifest_path=str(manifest),
        manifest_sha256=manifest_sha256(manifest),
        arm=arm,
        engine_version=engine_version,
        backend_id=backend_id,
        baseline_id=baseline_id,
        sembench_version=_package_version("sembench"),
        sembench_git_sha=sha,
        sembench_git_dirty=dirty,
        semblend_version=_package_version("semblend"),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def write_jsonl(path: str | Path, items: list[WorkloadItem]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path, max_items: int | None = None) -> list[WorkloadItem]:
    items: list[WorkloadItem] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(WorkloadItem.from_dict(json.loads(line)))
            if max_items is not None and len(items) >= max_items:
                break
    return items
