"""Join the SemBlend vLLM connector's audit JSONL onto sembench request rows.

The connector writes one JSONL event per decision (``schema_version`` 2), and
both roles append to the same file. Every event that names a request carries
``connector_id`` / ``request_id`` / ``request_seq`` / ``event_seq``: a
cross-role join (a scheduler advertise against a worker materialization) is on
``request_id`` alone, and ordering *within* a role is on
``(request_seq, event_seq)``.

**The sembench side of the join is the id the runner sends in
``X-Request-Id``.** vLLM's OpenAI server honours that header —
``_base_request_id`` returns it verbatim when present
(``vllm/entrypoints/serve/engine/serving.py:117-126`` in the pinned 0.29
tree) — but the id the *engine* then uses is not the header verbatim:

- completions build ``cmpl-<header>`` at
  ``vllm/entrypoints/openai/completion/serving.py:143`` and then send
  ``f"{request_id}-{i}"`` per prompt at ``:178``, so a single-prompt request
  reaches the connector as ``cmpl-<header>-0``;
- chat completions build ``chatcmpl-<header>`` at
  ``vllm/entrypoints/openai/chat_completion/serving.py:281-283`` and keep it
  unsuffixed for one prompt, gaining ``f"{request_id}_{i}"`` only when one
  HTTP request carries several prompts (``:306-308``).

So this module matches the header id against the audited request id exactly
first, then against the audited id with that prefix and sub-request suffix
removed — and **refuses** (recording the row as ambiguous) rather than
guessing when one normalized key covers two audited requests.

Only ``runtime_materialized`` counts as backend-confirmed KV reuse, per the
connector's own audit contract. An advertise is a promise, an allocation is a
destination; neither is evidence that KV was written.

Four things the fold keeps beyond that, because section 4's metrics are not
computable without them: the ``semantic_lookup_hit`` events and the boundary
each ran at (M1's ``alignment_given_match`` divides by those, not by
requests); the ``semantic_span_boundary_missed`` payload fields that separate
the four causes of a miss (:func:`boundary_miss_reason`); the
``prefix_cache_blocks_evicted`` count, which is what gates every lane-2
quality claim; and, per request, the LAST of each, because
``get_num_new_matched_tokens`` is re-queried on every scheduling attempt and
counting events would weight a contended request above an uncontended one.

**Every field name below was read off the connector's own ``_audit_event``
call, not inferred from the metric that consumes it.** The one that matters
most is the boundary on a lookup hit: ``semantic_lookup_hit`` carries
``already_computed_tokens`` (``connector.py``, the hit branch of
``get_num_new_matched_tokens``) and carries no ``boundary`` key at all —
``boundary`` is the name the *span* events use
(``semantic_span_load_advertised``, ``semantic_span_boundary_missed``,
``semantic_span_declined_unaligned_boundary``,
``semantic_span_declined_below_min_after_clamp``,
``semantic_span_supply_clamped``). Reading ``boundary`` off a hit silently
produced ``None`` for every hit the connector ever wrote, which dropped the
whole denominator of ``alignment_given_match``.

The three span declines are folded for the same reason. A request whose
lookup hit and whose span was then declined — off an unaligned boundary, or
clamped below ``min_semantic_span``, or missed by the boundary entirely — is
a *misalignment*, which is exactly what M1 exists to count. Without those
branches such a request has no boundary anywhere on its row, drops out of
``alignment_given_match``'s denominator, and the rate reports only the
requests that went well.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sembench.schema import EXTERNAL_SOURCE_CONNECTOR_AUDIT, RequestMetrics

AUDIT_SCHEMA_VERSION = 2

EVENT_REQUEST_FIRST_SEEN = "request_first_seen"
EVENT_LOOKUP_HIT = "semantic_lookup_hit"
EVENT_LOAD_ADVERTISED = "semantic_span_load_advertised"
EVENT_BOUNDARY_MISSED = "semantic_span_boundary_missed"
EVENT_DECLINED_UNALIGNED_BOUNDARY = "semantic_span_declined_unaligned_boundary"
EVENT_DECLINED_BELOW_MIN_AFTER_CLAMP = "semantic_span_declined_below_min_after_clamp"
EVENT_SUPPLY_CLAMPED = "semantic_span_supply_clamped"
EVENT_LOAD_ALLOCATED = "load_allocated"
EVENT_MATERIALIZED = "runtime_materialized"
EVENT_MATERIALIZATION_DECLINED = "runtime_materialization_declined"
EVENT_PREFIX_BLOCKS_EVICTED = "prefix_cache_blocks_evicted"
LOOKUP_SKIPPED_PREFIX = "lookup_skipped_"

# The field a `semantic_lookup_hit` carries the engine's boundary in. It is
# NOT called `boundary`: the hit event is written before any span arithmetic
# runs, from the same `num_computed_tokens` the lookup request carried.
LOOKUP_HIT_BOUNDARY_FIELD = "already_computed_tokens"
# The field every span-scoped event carries the boundary in.
SPAN_BOUNDARY_FIELD = "boundary"

# Every way the connector can decline a span AFTER its lookup already hit.
# Each one leaves the request with a donor and no served span, which is the
# definition of a boundary misalignment, so each one has to keep the request
# inside M1's match denominator.
SPAN_DECLINE_EVENTS = (
    EVENT_DECLINED_UNALIGNED_BOUNDARY,
    EVENT_DECLINED_BELOW_MIN_AFTER_CLAMP,
    EVENT_BOUNDARY_MISSED,
)
# Not a decline: the connector clamps the count and carries on, so the request
# can still be advertised at this boundary. Folded only for the boundary.
SPAN_BOUNDARY_EVENTS = (*SPAN_DECLINE_EVENTS, EVENT_SUPPLY_CLAMPED, EVENT_LOAD_ADVERTISED)

# Section 4's boundary_miss_breakdown partition. Four causes land on the same
# `return 0, False` inside the connector and only the miss event's payload can
# separate them, so the partition is computed here, once, per event.
MISS_DONOR_NOT_CAPTURED = "donor_not_captured"
MISS_DONOR_TOO_SHORT = "donor_too_short"
MISS_BELOW_MIN_SEMANTIC_SPAN = "below_min_semantic_span"
MISS_TRUE_MISALIGNMENT = "true_misalignment"
# Not in section 4: a miss event that carried none of the fields the partition
# reads (a pre-B9 connector). Calling that `true_misalignment` would publish a
# diagnosis the payload does not contain.
MISS_UNCLASSIFIED = "unclassified"
BOUNDARY_MISS_REASONS = (
    MISS_DONOR_NOT_CAPTURED,
    MISS_DONOR_TOO_SHORT,
    MISS_BELOW_MIN_SEMANTIC_SPAN,
    MISS_TRUE_MISALIGNMENT,
    MISS_UNCLASSIFIED,
)

# Prefixes vLLM's OpenAI handlers put in front of the X-Request-Id value.
ENGINE_ID_PREFIXES = ("chatcmpl-", "cmpl-")
# The per-prompt sub-request suffix: "-0" (completions, always) or "_0" (chat,
# only for a multi-prompt request).
_SUBREQUEST_SUFFIX = re.compile(r"[-_]\d+$")

MATCH_EXACT = "exact"
MATCH_NORMALIZED = "normalized"
MATCH_AMBIGUOUS = "ambiguous"
MATCH_MISSING = "missing"


class AuditError(ValueError):
    """The audit file could not be read at all."""


@dataclass(frozen=True)
class SpanRecord:
    """One post-snap span as the connector advertised it."""

    target_start: int | None
    target_end: int | None
    donor_start: int | None


@dataclass(frozen=True)
class LookupHit:
    """One ``semantic_lookup_hit``, folded field for field.

    Every name here is the connector's own: ``already_computed_tokens`` (the
    boundary the lookup ran at), ``reusable_tokens``, ``similarity``,
    ``materialization_kind``, ``confidence_tier``, ``donor_id``, ``reason``.
    """

    boundary: int | None
    donor_id: str | None
    reusable_tokens: int | None
    similarity: float | None
    materialization_kind: str | None
    confidence_tier: str | None
    reason: str | None
    attempt: int | None


@dataclass(frozen=True)
class SpanDecline:
    """One span the connector declined after its lookup had already hit.

    ``event`` is the connector's event name, so the three declines stay
    distinguishable downstream; ``reason`` is the miss partition for a
    ``semantic_span_boundary_missed`` and the event name for the other two,
    which carry their cause in their name.
    """

    event: str
    reason: str
    boundary: int | None
    donor_id: str | None
    attempt: int | None


@dataclass(frozen=True)
class Advertisement:
    """The connector's load promise, as of the last advertise for a request."""

    boundary: int | None
    target_start: int | None
    tokens: int
    donor_id: str | None
    attempt: int | None
    spans: tuple[SpanRecord, ...]
    # Where the advertise sat in the audit file. Both connector roles append
    # to one file, so line order is a real observation order across them, and
    # M2 needs it to tell a materialization of THIS promise from one of the
    # promise it superseded.
    line_no: int = -1

    @property
    def boundary_at_span_start(self) -> bool | None:
        """Did the engine's boundary land exactly on a snapped span start?

        None when the advertise carried no spans — unknown, not False.
        """
        if self.boundary is None or not self.spans:
            return None
        return any(span.target_start == self.boundary for span in self.spans)


@dataclass(frozen=True)
class AuditRequest:
    """Everything the audit says about one engine request id."""

    request_id: str
    connector_ids: tuple[str, ...]
    first_seen: bool
    prompt_tokens: int | None
    advertisement: Advertisement | None
    advertise_count: int
    # One entry per semantic_lookup_hit, in role-local order.
    lookup_hits: tuple[LookupHit, ...]
    # Every span the connector declined after a hit, in role-local order.
    span_declines: tuple[SpanDecline, ...]
    boundary_missed_count: int
    last_missed_boundary: int | None
    boundary_miss_reasons: tuple[str, ...]
    # The last boundary ANY span-scoped event was recorded at, whatever the
    # outcome: advertise, clamp, decline or miss.
    last_span_boundary: int | None
    allocated_tokens: int | None
    materialized_tokens: int | None
    # Materialization mass written against a promise the connector has since
    # superseded. Kept out of M2's numerator and published rather than dropped
    # in silence -- see :func:`_fold`.
    superseded_materialized_tokens: int
    declined_reasons: tuple[str, ...]
    lookup_skipped: tuple[str, ...]
    prefix_blocks_evicted: int
    request_id_reused: bool
    event_count: int

    @property
    def lookup_hit_count(self) -> int:
        return len(self.lookup_hits)

    @property
    def observed_boundary(self) -> int | None:
        """The boundary the connector was last asked about.

        The advertise wins when there is one; otherwise the last boundary any
        span event was recorded at — a miss, an unaligned-boundary decline, a
        below-minimum decline or a clamp — so a request that never got a span
        still reports where it stood instead of reporting nothing.
        """
        if self.advertisement is not None:
            return self.advertisement.boundary
        return self.last_span_boundary

    @property
    def lookup_hit_boundary(self) -> int | None:
        """The boundary the LAST lookup hit ran at.

        Read from the hit's own ``already_computed_tokens``. A hit whose
        payload carried neither that field nor a value falls back to
        :attr:`observed_boundary` — the connector is asked about exactly one
        boundary per attempt, so the span event recorded for the same request
        is the same number — rather than dropping out of M1's denominator,
        which would shrink it in silence.
        """
        if not self.lookup_hits:
            return None
        last = self.lookup_hits[-1].boundary
        return self.observed_boundary if last is None else last

    @property
    def last_boundary_miss_reason(self) -> str | None:
        """Why the LAST boundary miss happened, deduped per section 4.

        M1 counts one miss per request id, not one per scheduling attempt:
        ``get_num_new_matched_tokens`` is re-queried while a request stays
        queued, so counting events would weight a contended request higher
        than an uncontended one.
        """
        return self.boundary_miss_reasons[-1] if self.boundary_miss_reasons else None

    @property
    def last_span_decline(self) -> SpanDecline | None:
        """The LAST span decline, deduped per request for the same reason."""
        return self.span_declines[-1] if self.span_declines else None


@dataclass(frozen=True)
class MatchResult:
    """How a row's request id resolved against the audit."""

    record: AuditRequest | None
    kind: str


@dataclass(frozen=True)
class AuditStats:
    """What the file held, including what could not be parsed."""

    path: str
    lines: int
    events: int
    malformed_lines: int
    other_schema_versions: int
    engine_scope_events: int
    requests: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "lines": self.lines,
            "events": self.events,
            "malformed_lines": self.malformed_lines,
            "other_schema_versions": self.other_schema_versions,
            "engine_scope_events": self.engine_scope_events,
            "requests": self.requests,
        }


@dataclass(frozen=True)
class JoinReport:
    """What the join did, so a shrunken denominator is never silent."""

    stats: AuditStats
    rows_total: int
    rows_matched_exact: int
    rows_matched_normalized: int
    rows_unmatched: int
    rows_ambiguous: int
    rows_without_request_id: int
    # Rows whose audited request id was used by more than one request in the
    # same connector instance. vLLM reuses request ids, so such a record mixes
    # two requests' events and its numbers belong to neither.
    rows_reused_request_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit": self.stats.to_dict(),
            "rows_total": self.rows_total,
            "rows_matched_exact": self.rows_matched_exact,
            "rows_matched_normalized": self.rows_matched_normalized,
            "rows_unmatched": self.rows_unmatched,
            "rows_ambiguous": self.rows_ambiguous,
            "rows_without_request_id": self.rows_without_request_id,
            "rows_reused_request_id": self.rows_reused_request_id,
        }


@dataclass(frozen=True)
class _Event:
    """One parsed audit line that names a request."""

    event: str
    request_id: str
    connector_id: str
    request_seq: int | None
    event_seq: int | None
    line_no: int
    fields: dict[str, Any]

    @property
    def order(self) -> tuple[str, int, int, int]:
        """Role-local chronology; an absent sequence sorts first, not last."""
        return (
            self.connector_id,
            -1 if self.request_seq is None else self.request_seq,
            -1 if self.event_seq is None else self.event_seq,
            self.line_no,
        )


def _as_int(value: Any) -> int | None:
    """int(value) when that is meaningful, else None. Never raises."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _span_records(raw: Any) -> tuple[SpanRecord, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(
        SpanRecord(
            target_start=_as_int(span.get("target_start")),
            target_end=_as_int(span.get("target_end")),
            donor_start=_as_int(span.get("donor_start")),
        )
        for span in raw
        if isinstance(span, dict)
    )


def boundary_miss_reason(payload: dict[str, Any]) -> str:
    """Partition one ``semantic_span_boundary_missed`` event by cause.

    Section 4, verbatim::

        boundary_miss_breakdown = boundary_missed events partitioned by reason:
                                  stored_donor_tokens == 0      -> donor_not_captured
                                  stored_donor_tokens < span    -> donor_too_short
                                  n_raw_segments > 0, snapped=0 -> below_min_semantic_span
                                  otherwise                     -> true_misalignment

    The second line is read off the payload the connector actually writes, not
    off the arithmetic the plan describes. **A ``stored_donor_tokens < span``
    test is unreachable.** The connector trims every segment to the captured
    window before it builds ``raw_spans`` —
    ``length = min(seg.token_count, stored_tokens - seg.donor_start)``, and a
    segment whose ``length <= 0`` is dropped and counted in
    ``segments_beyond_capture`` instead — so no entry in ``raw_spans`` can be
    longer than ``stored_donor_tokens``, and the comparison never fires. What
    a capture shortfall actually looks like in the payload is
    ``segments_beyond_capture > 0``: the provider found spans in this donor and
    the stored prefix did not reach them.

    ``n_raw_segments == 0 and n_segments > 0`` is the same finding on a payload
    that predates the per-cause counters: every segment was dropped before
    snapping, and the drop is a capture shortfall unless
    ``segments_wrong_donor`` accounts for it — in which case the planner
    returned another donor's spans and the diagnosis is a misalignment, not a
    short donor.
    """
    stored = _as_int(payload.get("stored_donor_tokens"))
    segments = _as_int(payload.get("n_segments"))
    raw_segments = _as_int(payload.get("n_raw_segments"))
    beyond_capture = _as_int(payload.get("segments_beyond_capture"))
    wrong_donor = _as_int(payload.get("segments_wrong_donor"))
    snapped = payload.get("snapped_spans")
    snapped_count = len(snapped) if isinstance(snapped, list) else None
    if stored is None and raw_segments is None and snapped_count is None and beyond_capture is None:
        return MISS_UNCLASSIFIED
    if stored == 0:
        return MISS_DONOR_NOT_CAPTURED
    if (beyond_capture or 0) > 0:
        return MISS_DONOR_TOO_SHORT
    if (
        beyond_capture is None
        and raw_segments == 0
        and (segments or 0) > 0
        and not (wrong_donor or 0)
    ):
        return MISS_DONOR_TOO_SHORT
    if (raw_segments or 0) > 0 and not snapped_count:
        return MISS_BELOW_MIN_SEMANTIC_SPAN
    return MISS_TRUE_MISALIGNMENT


def parse_events(lines: Iterable[str], *, path: str = "") -> tuple[list[_Event], AuditStats]:
    """Parse audit JSONL into request-scoped events plus what was skipped.

    A truncated or non-JSON line is counted, not fatal: the connector appends
    per event and an engine killed mid-write leaves a partial last line, which
    must not take the whole join down with it.
    """
    events: list[_Event] = []
    line_count = 0
    malformed = 0
    other_schema = 0
    engine_scope = 0
    for line_no, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        line_count += 1
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(record, dict):
            malformed += 1
            continue
        if _as_int(record.get("schema_version")) != AUDIT_SCHEMA_VERSION:
            other_schema += 1
            continue
        request_id = record.get("request_id")
        event = record.get("event")
        if not isinstance(event, str):
            malformed += 1
            continue
        if request_id is None:
            engine_scope += 1
            continue
        events.append(
            _Event(
                event=event,
                request_id=str(request_id),
                connector_id=str(record.get("connector_id") or ""),
                # Absent is None, not zero: event_seq legitimately starts at 0
                # and collapsing the two would reorder a request's first event.
                request_seq=_as_int(record.get("request_seq")),
                event_seq=_as_int(record.get("event_seq")),
                line_no=line_no,
                fields=record,
            )
        )
    stats = AuditStats(
        path=path,
        lines=line_count,
        events=len(events),
        malformed_lines=malformed,
        other_schema_versions=other_schema,
        engine_scope_events=engine_scope,
        requests=len({event.request_id for event in events}),
    )
    return events, stats


def _lookup_hit(payload: dict[str, Any]) -> LookupHit:
    """One ``semantic_lookup_hit`` payload, field for field."""
    similarity = payload.get("similarity")
    return LookupHit(
        boundary=_as_int(payload.get(LOOKUP_HIT_BOUNDARY_FIELD)),
        donor_id=(None if payload.get("donor_id") is None else str(payload["donor_id"])),
        reusable_tokens=_as_int(payload.get("reusable_tokens")),
        similarity=(
            float(similarity)
            if isinstance(similarity, (int, float)) and not isinstance(similarity, bool)
            else None
        ),
        materialization_kind=(
            None
            if payload.get("materialization_kind") is None
            else str(payload["materialization_kind"])
        ),
        confidence_tier=(
            None if payload.get("confidence_tier") is None else str(payload["confidence_tier"])
        ),
        reason=(None if payload.get("reason") is None else str(payload["reason"])),
        attempt=_as_int(payload.get("attempt")),
    )


def _fold(request_id: str, events: Sequence[_Event]) -> AuditRequest:
    """Collapse one request's events into the facts the metrics need.

    Ordering is ``(connector_id, request_seq, event_seq)``: within a role that
    is the connector's own chronology, and across roles the two sequences are
    independent so no cross-role ordering is claimed. A later advertise
    supersedes an earlier one — the connector only re-advertises when the plan
    changed at a new boundary — so the LAST advertise is the promise the
    engine acted on.

    **M2's two sums obey one rule: the last advertise, and the materializations
    that followed it.** The advertised side was already the last advertise;
    the materialized side used to be every ``runtime_materialized`` event on
    the request, which is a different rule on the other half of the same
    ratio — a request re-advertised from 256 to 768 tokens and materialized
    twice contributed 256+768 over 768 and read as 133% of its promise. "Came
    after" is decided by position in the audit file, which is a real total
    order across both roles because both append to one path (the scheduler
    writes the advertise, the worker writes the materialization). Mass written
    against a superseded promise is kept as ``superseded_materialized_tokens``
    rather than dropped: it is evidence of KV that was written, just not of
    this promise being kept.
    """
    ordered = sorted(events, key=lambda event: event.order)
    advertisement: Advertisement | None = None
    advertise_count = 0
    lookup_hits: list[LookupHit] = []
    span_declines: list[SpanDecline] = []
    boundary_missed_count = 0
    last_missed_boundary: int | None = None
    last_span_boundary: int | None = None
    miss_reasons: list[str] = []
    allocated: int | None = None
    materializations: list[tuple[int, int]] = []
    declined: list[str] = []
    skipped: list[str] = []
    evicted = 0
    prompt_tokens: int | None = None
    first_seen = False
    seqs_by_connector: dict[str, set[int]] = {}

    for event in ordered:
        if event.request_seq is not None:
            seqs_by_connector.setdefault(event.connector_id, set()).add(event.request_seq)
        payload = event.fields
        if event.event in SPAN_BOUNDARY_EVENTS:
            boundary = _as_int(payload.get(SPAN_BOUNDARY_FIELD))
            if boundary is not None:
                last_span_boundary = boundary
        if event.event == EVENT_REQUEST_FIRST_SEEN:
            first_seen = True
            prompt_tokens = _as_int(payload.get("prompt_tokens"))
        elif event.event == EVENT_LOOKUP_HIT:
            lookup_hits.append(_lookup_hit(payload))
        elif event.event == EVENT_LOAD_ADVERTISED:
            advertise_count += 1
            advertisement = Advertisement(
                boundary=_as_int(payload.get(SPAN_BOUNDARY_FIELD)),
                target_start=_as_int(payload.get("target_start")),
                tokens=_as_int(payload.get("token_count")) or 0,
                donor_id=(None if payload.get("donor_id") is None else str(payload["donor_id"])),
                attempt=_as_int(payload.get("attempt")),
                spans=_span_records(payload.get("snapped_spans")),
                line_no=event.line_no,
            )
        elif event.event == EVENT_LOAD_ALLOCATED:
            allocated = (allocated or 0) + (_as_int(payload.get("tokens")) or 0)
        elif event.event == EVENT_MATERIALIZED:
            materializations.append((event.line_no, _as_int(payload.get("tokens")) or 0))
        elif event.event == EVENT_MATERIALIZATION_DECLINED:
            declined.append(str(payload.get("declined_reason") or "unspecified"))
        elif event.event == EVENT_PREFIX_BLOCKS_EVICTED:
            evicted += _as_int(payload.get("blocks_evicted")) or 0
        elif event.event.startswith(LOOKUP_SKIPPED_PREFIX):
            skipped.append(event.event[len(LOOKUP_SKIPPED_PREFIX) :])
        if event.event in SPAN_DECLINE_EVENTS:
            reason = (
                boundary_miss_reason(payload)
                if event.event == EVENT_BOUNDARY_MISSED
                else event.event
            )
            span_declines.append(
                SpanDecline(
                    event=event.event,
                    reason=reason,
                    boundary=_as_int(payload.get(SPAN_BOUNDARY_FIELD)),
                    donor_id=(
                        None if payload.get("donor_id") is None else str(payload["donor_id"])
                    ),
                    attempt=_as_int(payload.get("attempt")),
                )
            )
            if event.event == EVENT_BOUNDARY_MISSED:
                boundary_missed_count += 1
                last_missed_boundary = _as_int(payload.get(SPAN_BOUNDARY_FIELD))
                miss_reasons.append(reason)

    advertise_line = -1 if advertisement is None else advertisement.line_no
    kept = [tokens for line_no, tokens in materializations if line_no >= advertise_line]
    superseded = sum(tokens for line_no, tokens in materializations if line_no < advertise_line)

    return AuditRequest(
        request_id=request_id,
        connector_ids=tuple(sorted({event.connector_id for event in ordered})),
        first_seen=first_seen,
        prompt_tokens=prompt_tokens,
        advertisement=advertisement,
        advertise_count=advertise_count,
        lookup_hits=tuple(lookup_hits),
        span_declines=tuple(span_declines),
        boundary_missed_count=boundary_missed_count,
        last_missed_boundary=last_missed_boundary,
        boundary_miss_reasons=tuple(miss_reasons),
        last_span_boundary=last_span_boundary,
        allocated_tokens=allocated,
        # None, not 0, when the worker wrote no materialization event at all:
        # "the audit looked and found none" is decided by `materializations`
        # being empty, not by the mass summing to zero.
        materialized_tokens=(sum(kept) if materializations else None),
        superseded_materialized_tokens=superseded,
        declined_reasons=tuple(declined),
        lookup_skipped=tuple(skipped),
        prefix_blocks_evicted=evicted,
        # vLLM reuses request ids; two arrival numbers from one connector mean
        # one id covered two requests and nothing below can separate them.
        request_id_reused=any(len(seqs) > 1 for seqs in seqs_by_connector.values()),
        event_count=len(ordered),
    )


def normalized_ids(engine_request_id: str) -> tuple[str, ...]:
    """Client-side ids an engine request id could have come from.

    Strips the OpenAI handler's prefix and the per-prompt sub-request suffix
    (both documented at the top of this module), keeping every intermediate
    form so a client id that itself ends in ``-3`` still matches.
    """
    bases = {engine_request_id}
    for prefix in ENGINE_ID_PREFIXES:
        if engine_request_id.startswith(prefix):
            bases.add(engine_request_id[len(prefix) :])
    candidates = set(bases)
    for base in bases:
        trimmed = _SUBREQUEST_SUFFIX.sub("", base)
        if trimmed and trimmed != base:
            candidates.add(trimmed)
    candidates.discard(engine_request_id)
    return tuple(sorted(candidates))


@dataclass(frozen=True)
class AuditIndex:
    """Audited requests, addressable by the id the runner actually sent."""

    records: dict[str, AuditRequest]
    by_normalized_id: dict[str, tuple[str, ...]]
    stats: AuditStats

    def resolve(self, request_id: str | None) -> MatchResult:
        """Exact match, then a unique normalized match, else no match.

        A normalized key covering two audited requests resolves to
        ``MATCH_AMBIGUOUS`` with no record: two engine requests that share a
        client id (a multi-prompt HTTP request) cannot be attributed to one
        row, and picking either would invent a measurement.
        """
        if not request_id:
            return MatchResult(None, MATCH_MISSING)
        exact = self.records.get(request_id)
        if exact is not None:
            return MatchResult(exact, MATCH_EXACT)
        candidates = self.by_normalized_id.get(request_id, ())
        if len(candidates) == 1:
            return MatchResult(self.records[candidates[0]], MATCH_NORMALIZED)
        if len(candidates) > 1:
            return MatchResult(None, MATCH_AMBIGUOUS)
        return MatchResult(None, MATCH_MISSING)


def index_events(events: Sequence[_Event], stats: AuditStats) -> AuditIndex:
    """Group parsed events by request id and build the lookup tables."""
    grouped: dict[str, list[_Event]] = {}
    for event in events:
        grouped.setdefault(event.request_id, []).append(event)
    records = {
        request_id: _fold(request_id, request_events)
        for request_id, request_events in grouped.items()
    }
    normalized: dict[str, list[str]] = {}
    for request_id in records:
        for candidate in normalized_ids(request_id):
            normalized.setdefault(candidate, []).append(request_id)
    return AuditIndex(
        records=records,
        by_normalized_id={key: tuple(sorted(value)) for key, value in normalized.items()},
        stats=stats,
    )


def load_audit(path: str | Path) -> AuditIndex:
    """Read a connector audit JSONL into an :class:`AuditIndex`.

    Raises :class:`AuditError` when the file cannot be read at all — a join
    asked for against a missing audit must fail loudly, because silently
    returning an empty index would publish "no reuse" as a measurement.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise AuditError(
            f"connector audit {source} could not be read ({error}); "
            "without it no per-request external reuse can be attributed"
        ) from error
    events, stats = parse_events(text.splitlines(), path=str(source))
    return index_events(events, stats)


def stamp_row(row: RequestMetrics, record: AuditRequest | None) -> RequestMetrics:
    """A NEW row carrying what the audit says about it.

    ``external_confirmed_tokens`` is set only from ``runtime_materialized``
    mass. A joined request with no materialization event is left at whatever
    it already carried (normally None) rather than being written down as a
    measured zero: the audit confirms materialization affirmatively, and the
    propagation metric reads a null there as "nothing was materialized for
    this request", which a zero would not distinguish from an arm-level value.
    """
    if record is None:
        return replace(row, audit_joined=False)
    advertisement = record.advertisement
    materialized = record.materialized_tokens
    last_hit = record.lookup_hits[-1] if record.lookup_hits else None
    decline = record.last_span_decline
    return replace(
        row,
        external_confirmed_tokens=(
            materialized if materialized is not None else row.external_confirmed_tokens
        ),
        external_confirmed_tokens_source=(
            EXTERNAL_SOURCE_CONNECTOR_AUDIT
            if materialized is not None
            else row.external_confirmed_tokens_source
        ),
        audit_joined=True,
        audit_observed_boundary=record.observed_boundary,
        audit_advertised_tokens=(None if advertisement is None else advertisement.tokens),
        audit_advertised_target_start=(
            None if advertisement is None else advertisement.target_start
        ),
        audit_boundary_at_span_start=(
            None if advertisement is None else advertisement.boundary_at_span_start
        ),
        audit_load_allocated=record.allocated_tokens is not None,
        audit_materialized=materialized is not None,
        audit_declined_reasons=(list(record.declined_reasons) or None),
        audit_semantic_lookup_hit=record.lookup_hit_count > 0,
        audit_lookup_hit_boundary=record.lookup_hit_boundary,
        audit_lookup_reusable_tokens=(None if last_hit is None else last_hit.reusable_tokens),
        audit_boundary_miss_reason=record.last_boundary_miss_reason,
        audit_boundary_missed_at=record.last_missed_boundary,
        audit_span_decline_event=(None if decline is None else decline.event),
        audit_span_decline_reason=(None if decline is None else decline.reason),
        audit_span_declined_at=(None if decline is None else decline.boundary),
        audit_superseded_materialized_tokens=record.superseded_materialized_tokens,
        audit_prefix_blocks_evicted=record.prefix_blocks_evicted,
    )


def join_requests(
    rows: Sequence[RequestMetrics],
    index: AuditIndex,
) -> tuple[list[RequestMetrics], JoinReport]:
    """Stamp every row with its audit facts and report what did not join.

    Every row comes back — an unjoined row is marked ``audit_joined=False``
    and counted, never dropped, because dropping it would shrink the
    denominator of every rate computed afterwards without saying so.
    """
    joined: list[RequestMetrics] = []
    counts = {MATCH_EXACT: 0, MATCH_NORMALIZED: 0, MATCH_AMBIGUOUS: 0, MATCH_MISSING: 0}
    without_id = 0
    reused = 0
    for row in rows:
        if not row.engine_request_id:
            without_id += 1
            joined.append(stamp_row(row, None))
            continue
        match = index.resolve(row.engine_request_id)
        counts[match.kind] += 1
        if match.record is not None and match.record.request_id_reused:
            reused += 1
        joined.append(stamp_row(row, match.record))
    report = JoinReport(
        stats=index.stats,
        rows_total=len(rows),
        rows_matched_exact=counts[MATCH_EXACT],
        rows_matched_normalized=counts[MATCH_NORMALIZED],
        rows_unmatched=counts[MATCH_MISSING],
        rows_ambiguous=counts[MATCH_AMBIGUOUS],
        rows_without_request_id=without_id,
        rows_reused_request_id=reused,
    )
    return joined, report


def join_audit_file(
    rows: Sequence[RequestMetrics],
    path: str | Path,
) -> tuple[list[RequestMetrics], JoinReport]:
    """:func:`load_audit` then :func:`join_requests`, the usual entry point."""
    return join_requests(rows, load_audit(path))


def engine_id_echoes_sent(response_id: str | None, sent_id: str | None) -> bool | None:
    """Did the engine's own id come from the id the runner sent?

    True when ``response_id`` is the sent id or one of the OpenAI handlers'
    forms of it (``chatcmpl-<sent>``, ``cmpl-<sent>-0``). None when either id
    is absent — nothing was observed, which is not a mismatch.
    """
    if not response_id or not sent_id:
        return None
    if response_id == sent_id:
        return True
    return sent_id in normalized_ids(response_id)


def request_id_echo_report(rows: Sequence[RequestMetrics]) -> dict[str, Any]:
    """Whether the engine adopted the ids the runner sent, and where it did not.

    The join to the audit is by construction only as long as the id survives
    the trip: a front end that strips ``X-Request-Id`` (or rewrites it) makes
    vLLM mint its own, and every row then fails to join for a reason that
    looks, downstream, exactly like an arm that materialized nothing. The
    engine echoes its id on every chunk, so the two can be compared per
    request and the count published beside the join report.

    ``examples`` carries at most three mismatching pairs, which is what an
    operator needs to recognize a rewriting proxy without the result growing a
    second copy of every id.
    """
    matched = 0
    mismatched = 0
    missing = 0
    examples: list[dict[str, str | None]] = []
    for row in rows:
        verdict = engine_id_echoes_sent(row.engine_response_id, row.engine_request_id)
        if verdict is None:
            missing += 1
        elif verdict:
            matched += 1
        else:
            mismatched += 1
            if len(examples) < 3:
                examples.append(
                    {"sent": row.engine_request_id, "engine_returned": row.engine_response_id}
                )
    return {
        "rows_checked": matched + mismatched,
        "rows_id_echoed": matched,
        "rows_id_mismatched": mismatched,
        "rows_without_engine_response_id": missing,
        "mismatch_examples": examples,
    }
