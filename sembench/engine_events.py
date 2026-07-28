"""Engine log/audit parsers for backend-confirmed semantic KV reuse.

These parsers intentionally consume serving-engine evidence, not SemBlend's
offline planning output. A semantic candidate becomes backend-confirmed only
when a backend log/audit stream reports a hit, load advertisement, or actual
materialization event.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

VLLM_DONOR_RE = re.compile(
    r"SemBlend donor registered request_id=(?P<request_id>\S+) "
    r"namespace=(?P<namespace>\S+) tokens=(?P<tokens>\d+) blocks=(?P<blocks>\d+)"
)
VLLM_HIT_RE = re.compile(
    r"SemBlend semantic lookup hit request_id=(?P<request_id>\S+) "
    r"donor_id=(?P<donor_id>\S+) similarity=(?P<similarity>[0-9.]+) "
    r"kind=(?P<kind>\S+) reusable_tokens=(?P<reusable_tokens>\d+) "
    r"reason=(?P<reason>\S+)"
)
VLLM_ADVERTISED_RE = re.compile(
    r"SemBlend request-local experimental load advertised "
    r"request_id=(?P<request_id>\S+) donor_id=(?P<donor_id>\S+) "
    r"tokens=(?P<tokens>\d+)"
)
VLLM_MATERIALIZING_RE = re.compile(
    r"SemBlend materializing request_id=(?P<request_id>\S+) "
    r"donor_id=(?P<donor_id>\S+) tokens=(?P<tokens>\d+)"
)
VLLM_MATERIALIZED_RE = re.compile(
    r"SemBlend materialized request_id=(?P<request_id>\S+) "
    r"donor_id=(?P<donor_id>\S+) tokens=(?P<tokens>\d+)"
)
SGLANG_DONOR_RE = re.compile(
    r"(?:\[FUZZY\]\s*)?register_donor(?:_async)?:\s*ok "
    r"request_id=(?P<request_id>\S+).*?tokens=(?P<tokens>\d+)",
    re.IGNORECASE,
)
SGLANG_HIT_RE = re.compile(
    r"(?:semantic|fuzzy).*?(?:hit|matched).*?(?:similarity|score)="
    r"(?P<similarity>[0-9.]+).*?(?:tokens|matched_tokens|reused_tokens)="
    r"(?P<tokens>\d+)",
    re.IGNORECASE,
)
SGLANG_RADIX_SUCCESS_RE = re.compile(
    r"\[FUZZY RADIX\]\s*Fuzzy match success:.*?"
    r"cached=(?P<tokens>\d+).*?"
    r"(?:quality_cosine=(?P<similarity>[0-9.]+))?",
    re.IGNORECASE,
)
SGLANG_REALIZED_RE = re.compile(
    r"\[FUZZY\]\s*Realized\s+(?P<tokens>\d+)\s+fuzzy tokens",
    re.IGNORECASE,
)
SGLANG_SEGMENTED_PHASE_RE = re.compile(
    r"\[FUZZY RADIX\]\s*segmented prefill realized donor phase:.*?"
    r"donor_tokens=(?P<tokens>\d+)\s+target=\[(?P<target_start>\d+),(?P<target_end>\d+)\).*?"
    r"direct_paged_kv=(?P<direct_paged_kv>\S+)",
    re.IGNORECASE,
)
# LMCache log families (format drifts across versions — fixtures pin each
# supported version; see tests/fixtures/lmcache/). LMCache is an EXACT-reuse
# baseline: retrieved tokens are engine-confirmed reuse, but semantic_hits
# stays 0 by definition, so `materialized_semantic_kv_reuse` is False for
# LMCache rows — the flag means what it says.
LMCACHE_RETRIEVE_RE = re.compile(
    r"(?:Retrieved|Reusing)\s+(?P<tokens>\d+)\s*(?:/|out of)\s*(?P<total>\d+)",
    re.IGNORECASE,
)
LMCACHE_STORE_RE = re.compile(
    r"Stor(?:ed|ing)\s+(?P<tokens>\d+)\s+(?:new\s+)?tokens",
    re.IGNORECASE,
)
LMCACHE_HIT_RE = re.compile(
    r"(?:cache\s+hit|lookup\s+hit).*?tokens[=:\s]+(?P<tokens>\d+)",
    re.IGNORECASE,
)
RUNTIME_WARNING_MARKERS = (
    "[E:onnxruntime:",
    "Non-zero status code returned",
    "Shape mismatch attempting to re-use buffer",
)


@dataclass(frozen=True)
class EngineReuseSummary:
    """Backend-confirmed semantic KV reuse evidence for one engine run."""

    engine: str
    donor_registrations: int = 0
    semantic_hits: int = 0
    load_advertisements: int = 0
    materialization_events: int = 0
    materialized_tokens: int = 0
    materialized_units: int = 0
    rope_corrections: int = 0
    engine_boundaries: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    runtime_warnings: list[str] = field(default_factory=list)
    events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def materialized_semantic_kv_reuse(self) -> bool:
        return self.semantic_hits > 0 and (
            self.materialization_events > 0
            or self.materialized_tokens > 0
            or self.materialized_units > 0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["materialized_semantic_kv_reuse"] = self.materialized_semantic_kv_reuse
        return payload


def parse_engine_events(engine: str, text: str) -> EngineReuseSummary:
    normalized = engine.strip().lower().replace("_", "-")
    if normalized == "vllm":
        return parse_vllm_logs(text)
    if normalized in {"trtllm", "trt-llm", "tensorrt-llm"}:
        return parse_trtllm_audit_jsonl(text)
    if normalized == "sglang":
        return parse_sglang_logs(text)
    if normalized == "lmcache":
        return parse_lmcache_logs(text)
    raise ValueError(f"unknown engine: {engine}")


def parse_lmcache_logs(logs: str) -> EngineReuseSummary:
    """Engine-confirmed EXACT reuse from LMCache-integrated serving logs."""
    events: dict[str, list[dict[str, Any]]] = {
        "retrieved": [],
        "stored": [],
        "hits": [],
    }
    errors: list[str] = []
    runtime_warnings: list[str] = []
    for line in logs.splitlines():
        for key, pattern in (
            ("retrieved", LMCACHE_RETRIEVE_RE),
            ("stored", LMCACHE_STORE_RE),
            ("hits", LMCACHE_HIT_RE),
        ):
            match = pattern.search(line)
            if match:
                events[key].append(match.groupdict())
        if "ERROR" in line or "Traceback" in line:
            errors.append(line)
        elif _is_runtime_warning(line):
            runtime_warnings.append(line)
    return EngineReuseSummary(
        engine="lmcache",
        donor_registrations=len(events["stored"]),
        semantic_hits=0,  # exact-reuse baseline: never semantic by definition
        materialization_events=len(events["retrieved"]),
        materialized_tokens=sum(_int(event.get("tokens")) for event in events["retrieved"]),
        errors=errors,
        runtime_warnings=runtime_warnings,
        events=events,
    )


def parse_vllm_logs(logs: str) -> EngineReuseSummary:
    events: dict[str, list[dict[str, Any]]] = {
        "donor_registered": [],
        "semantic_hits": [],
        "advertised_loads": [],
        "materializing": [],
        "materialized": [],
    }
    errors: list[str] = []
    runtime_warnings: list[str] = []
    for line in logs.splitlines():
        for key, pattern in (
            ("donor_registered", VLLM_DONOR_RE),
            ("semantic_hits", VLLM_HIT_RE),
            ("advertised_loads", VLLM_ADVERTISED_RE),
            ("materializing", VLLM_MATERIALIZING_RE),
            ("materialized", VLLM_MATERIALIZED_RE),
        ):
            match = pattern.search(line)
            if match:
                events[key].append(match.groupdict())
        if "ERROR" in line or "Traceback" in line:
            errors.append(line)
        elif _is_runtime_warning(line):
            runtime_warnings.append(line)
    return EngineReuseSummary(
        engine="vllm",
        donor_registrations=len(events["donor_registered"]),
        semantic_hits=len(events["semantic_hits"]),
        load_advertisements=len(events["advertised_loads"]),
        materialization_events=len(events["materialized"]),
        materialized_tokens=sum(_int(event.get("tokens")) for event in events["materialized"]),
        errors=errors,
        runtime_warnings=runtime_warnings,
        events=events,
    )


def parse_trtllm_audit_jsonl(text: str) -> EngineReuseSummary:
    events: dict[str, list[dict[str, Any]]] = {}
    rejection_reasons: dict[str, int] = {}
    errors: list[str] = []
    runtime_warnings: list[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            if _is_runtime_warning(raw):
                runtime_warnings.append(raw)
            else:
                errors.append(raw)
            continue
        name = str(event.get("event") or "unknown")
        events.setdefault(name, []).append(event)
        reason = str(event.get("rejection_reason") or "")
        if reason:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    lookup_events = events.get("lookup", [])
    materialized = events.get("materialized", [])
    boundaries = events.get("engine_blend_boundary", [])
    rope_corrections = sum(1 for event in materialized if event.get("requires_rope_correction"))
    return EngineReuseSummary(
        engine="trtllm",
        donor_registrations=len(events.get("donor_registered", [])),
        semantic_hits=sum(1 for event in lookup_events if event.get("found") is True),
        materialization_events=len(materialized),
        materialized_units=sum(_int(event.get("materialized")) for event in materialized),
        rope_corrections=rope_corrections,
        engine_boundaries=len(boundaries),
        rejection_reasons=rejection_reasons,
        errors=errors,
        runtime_warnings=runtime_warnings,
        events=events,
    )


def parse_sglang_logs(logs: str) -> EngineReuseSummary:
    events: dict[str, list[dict[str, Any]]] = {
        "donor_registered": [],
        "semantic_hits": [],
        "materialized": [],
        "segmented_phases": [],
    }
    errors: list[str] = []
    runtime_warnings: list[str] = []
    for line in logs.splitlines():
        donor = SGLANG_DONOR_RE.search(line)
        if donor:
            events["donor_registered"].append(donor.groupdict())
        hit = SGLANG_HIT_RE.search(line)
        if hit:
            events["semantic_hits"].append(hit.groupdict())
        radix_hit = SGLANG_RADIX_SUCCESS_RE.search(line)
        if radix_hit:
            events["semantic_hits"].append(radix_hit.groupdict())
        realized = SGLANG_REALIZED_RE.search(line)
        if realized:
            events["materialized"].append(realized.groupdict())
        segmented_phase = SGLANG_SEGMENTED_PHASE_RE.search(line)
        if segmented_phase:
            events["segmented_phases"].append(segmented_phase.groupdict())
        if "ERROR" in line or "Traceback" in line:
            errors.append(line)
        elif _is_runtime_warning(line):
            runtime_warnings.append(line)
    materialized = events["segmented_phases"] or events["materialized"]
    return EngineReuseSummary(
        engine="sglang",
        donor_registrations=len(events["donor_registered"]),
        semantic_hits=len(events["semantic_hits"]),
        materialization_events=len(materialized),
        materialized_tokens=sum(_int(event.get("tokens")) for event in materialized),
        errors=errors,
        runtime_warnings=runtime_warnings,
        events=events,
    )


def _is_runtime_warning(line: str) -> bool:
    return any(marker in line for marker in RUNTIME_WARNING_MARKERS)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
