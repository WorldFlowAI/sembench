"""Scrape the engine's own Prometheus counters around a benchmark arm.

The OpenAI usage block cannot answer "how much KV came from the external
connector": ``usage.prompt_tokens_details.cached_tokens`` is local prefix
cache plus external transfer summed together, so with prefix caching on it
is not a semantic-reuse measurement at all. vLLM exposes the split on
``/metrics`` and nothing in this repo read it until now.

Two counters matter, and they mean different things:

``vllm:external_prefix_cache_hits`` / ``vllm:external_prefix_cache_queries``
    Recorded after the scheduler has allocated slots, so the hits counter is
    *advertised-and-allocated*: an attempt that never got scheduled does not
    inflate it. This is the right cross-check for connector-reported reuse.

``vllm:prompt_tokens_by_source{source="external_kv_transfer"}``
    Set before allocation, so it is *advertised-and-accepted* only. A worker
    that later declines materialization does not decrement it. Reported
    under its own name so it is never mistaken for the one above.

Neither counter is materialization. Materialization is only observable in
the connector audit stream.

Counters are read as a before/after window per arm so the result carries a
delta, not a process-lifetime total: a server reused across arms has
non-zero totals that belong to the previous arm.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

_TIMEOUT_SECONDS = 15.0

# Metric family names as vLLM registers them. prometheus_client appends
# "_total" to counters in the exposition text, so both spellings are matched.
EXTERNAL_PREFIX_CACHE_QUERIES = "vllm:external_prefix_cache_queries"
EXTERNAL_PREFIX_CACHE_HITS = "vllm:external_prefix_cache_hits"
PROMPT_TOKENS_BY_SOURCE = "vllm:prompt_tokens_by_source"
EXTERNAL_KV_TRANSFER_SOURCE = "external_kv_transfer"

# Keys used inside every snapshot / delta document.
QUERIES_KEY = "external_prefix_cache_queries"
HITS_KEY = "external_prefix_cache_hits"
EXTERNAL_KV_TRANSFER_TOKENS_KEY = "external_kv_transfer_prompt_tokens"

COUNTER_KEYS = (QUERIES_KEY, HITS_KEY, EXTERNAL_KV_TRANSFER_TOKENS_KEY)

# Shipped with every delta so a reader cannot silently upgrade an
# advertised-and-accepted number into a materialization claim.
COUNTER_SEMANTICS = {
    QUERIES_KEY: (
        "vllm:external_prefix_cache_queries — tokens queried against the external "
        "KV connector; the denominator, not a reuse measurement"
    ),
    HITS_KEY: (
        "vllm:external_prefix_cache_hits — advertised-and-allocated tokens "
        "(recorded after the scheduler allocated slots); NOT materialization"
    ),
    EXTERNAL_KV_TRANSFER_TOKENS_KEY: (
        'vllm:prompt_tokens_by_source{source="external_kv_transfer"} — '
        "advertised-and-accepted tokens (set before allocation); a declined "
        "materialization does not decrement it"
    ),
}

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"[ \t]+(?P<value>\S+)"
    r"(?:[ \t]+\S+)?[ \t]*$"
)
_LABEL_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    """One Prometheus exposition sample line."""

    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def label_dict(self) -> dict[str, str]:
        return dict(self.labels)


def _unescape(raw: str) -> str:
    return raw.replace("\\\\", "\\").replace('\\"', '"').replace("\\n", "\n")


def _parse_value(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def parse_exposition(text: str) -> list[Sample]:
    """Parse Prometheus text exposition into samples, skipping HELP/TYPE."""
    samples: list[Sample] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _SAMPLE_RE.match(stripped)
        if match is None:
            continue
        value = _parse_value(match.group("value"))
        if value is None:
            continue
        raw_labels = match.group("labels") or ""
        labels = tuple(
            (pair.group("key"), _unescape(pair.group("value")))
            for pair in _LABEL_RE.finditer(raw_labels)
        )
        samples.append(Sample(name=match.group("name"), labels=labels, value=value))
    return samples


def counter_value(
    samples: list[Sample],
    name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    """Sum a counter family across label sets, or None when it is absent.

    Absent must not read as zero: a stock arm with no connector never
    registers the external counters at all, and reporting 0.0 there would
    claim a measured miss instead of an unmeasured one.
    """
    wanted = {name, f"{name}_total"}
    required = labels or {}
    total: float | None = None
    for sample in samples:
        if sample.name not in wanted:
            continue
        present = sample.label_dict()
        if any(present.get(key) != value for key, value in required.items()):
            continue
        total = sample.value if total is None else total + sample.value
    return total


def _counters_from_samples(samples: list[Sample]) -> dict[str, float | None]:
    return {
        QUERIES_KEY: counter_value(samples, EXTERNAL_PREFIX_CACHE_QUERIES),
        HITS_KEY: counter_value(samples, EXTERNAL_PREFIX_CACHE_HITS),
        EXTERNAL_KV_TRANSFER_TOKENS_KEY: counter_value(
            samples,
            PROMPT_TOKENS_BY_SOURCE,
            {"source": EXTERNAL_KV_TRANSFER_SOURCE},
        ),
    }


@dataclass(frozen=True)
class MetricsSnapshot:
    """Engine counters read from one endpoint at one instant."""

    url: str
    captured_at_utc: str
    counters: dict[str, float | None] = field(default_factory=dict)
    sample_count: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "captured_at_utc": self.captured_at_utc,
            "counters": dict(self.counters),
            "sample_count": self.sample_count,
            "error": self.error,
        }


def metrics_url(base_url: str) -> str:
    """Normalize a worker base URL to its /metrics endpoint."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/metrics"):
        return trimmed
    return f"{trimmed}/metrics"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def snapshot_from_text(url: str, text: str, captured_at_utc: str | None = None) -> MetricsSnapshot:
    """Build a snapshot from already-fetched exposition text."""
    samples = parse_exposition(text)
    return MetricsSnapshot(
        url=url,
        captured_at_utc=captured_at_utc or _utc_now(),
        counters=_counters_from_samples(samples),
        sample_count=len(samples),
    )


def scrape_metrics(base_url: str, timeout: float = _TIMEOUT_SECONDS) -> MetricsSnapshot:
    """GET <base>/metrics and extract the external-KV counters.

    A scrape failure is recorded on the snapshot rather than raised: losing
    the cross-check must not destroy an arm's measured requests, but it must
    be visible in the result.
    """
    url = metrics_url(base_url)
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        return MetricsSnapshot(
            url=url,
            captured_at_utc=_utc_now(),
            counters={key: None for key in COUNTER_KEYS},
            error=f"{type(exc).__name__}: {exc}",
        )
    return snapshot_from_text(url, text)


def scrape_all(base_urls: list[str], timeout: float = _TIMEOUT_SECONDS) -> list[MetricsSnapshot]:
    """Scrape every worker of an arm, deduped by normalized /metrics URL."""
    seen: set[str] = set()
    snapshots: list[MetricsSnapshot] = []
    for base_url in base_urls:
        url = metrics_url(base_url)
        if url in seen:
            continue
        seen.add(url)
        snapshots.append(scrape_metrics(base_url, timeout=timeout))
    return snapshots


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def counter_deltas(
    before: list[MetricsSnapshot],
    after: list[MetricsSnapshot],
) -> dict[str, Any]:
    """Per-endpoint and fleet-summed deltas over one arm.

    A negative delta means the counter restarted mid-arm (the server was
    recycled); it is surfaced instead of being clamped, because a clamped
    zero would read as "the connector was never consulted".
    """
    before_by_url = {snapshot.url: snapshot for snapshot in before}
    after_by_url = {snapshot.url: snapshot for snapshot in after}
    urls = sorted(set(before_by_url) | set(after_by_url))

    per_endpoint: dict[str, dict[str, float | None]] = {}
    totals: dict[str, float | None] = {key: None for key in COUNTER_KEYS}
    reset_detected = False
    paired = 0
    failed = 0

    for url in urls:
        start = before_by_url.get(url)
        end = after_by_url.get(url)
        if start is None or end is None or not start.ok or not end.ok:
            failed += 1
            per_endpoint[url] = {key: None for key in COUNTER_KEYS}
            continue
        paired += 1
        endpoint_deltas: dict[str, float | None] = {}
        for key in COUNTER_KEYS:
            start_value = start.counters.get(key)
            end_value = end.counters.get(key)
            if start_value is None or end_value is None:
                endpoint_deltas[key] = None
                continue
            delta = end_value - start_value
            if delta < 0:
                reset_detected = True
            endpoint_deltas[key] = delta
            totals[key] = delta if totals[key] is None else totals[key] + delta
        per_endpoint[url] = endpoint_deltas

    return {
        f"{QUERIES_KEY}_delta": totals[QUERIES_KEY],
        f"{HITS_KEY}_delta": totals[HITS_KEY],
        f"{EXTERNAL_KV_TRANSFER_TOKENS_KEY}_delta": totals[EXTERNAL_KV_TRANSFER_TOKENS_KEY],
        "external_prefix_cache_hit_ratio": _ratio(totals[HITS_KEY], totals[QUERIES_KEY]),
        "counter_reset_detected": reset_detected,
        "endpoints_paired": paired,
        "endpoints_failed": failed,
        "per_endpoint": per_endpoint,
    }


@dataclass(frozen=True)
class MetricsWindow:
    """Before/after counter reads bracketing one arm."""

    before: tuple[MetricsSnapshot, ...] = ()
    after: tuple[MetricsSnapshot, ...] = ()

    def delta(self) -> dict[str, Any]:
        return counter_deltas(list(self.before), list(self.after))

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoints": [snapshot.url for snapshot in self.before]
            or [snapshot.url for snapshot in self.after],
            "before": [snapshot.to_dict() for snapshot in self.before],
            "after": [snapshot.to_dict() for snapshot in self.after],
            "delta": self.delta(),
            "semantics": dict(COUNTER_SEMANTICS),
        }
