"""Result aggregation and JSON writing."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sembench.schema import (
    PER_REQUEST_EXTERNAL_SOURCES,
    RESULT_VERSION,
    RequestMetrics,
    RunMetadata,
)


def aggregate_metrics(requests: list[RequestMetrics]) -> dict[str, Any]:
    """Aggregate per-request metrics into benchmark-level rates."""
    total_blocks = sum(r.total_blocks for r in requests)
    prompt_tokens = sum(r.prompt_tokens for r in requests)
    exact_blocks = sum(r.exact_hit_blocks for r in requests)
    exact_tokens = sum(r.exact_hit_tokens for r in requests)
    candidate_blocks = sum(r.semantic_candidate_blocks for r in requests)
    candidate_tokens = sum(r.semantic_candidate_tokens for r in requests)
    eligible_blocks = sum(r.semantic_eligible_blocks for r in requests)
    eligible_tokens = sum(r.semantic_eligible_tokens for r in requests)
    has_semantic_plan = any(
        r.semblend_found
        or r.semantic_candidate_blocks
        or r.semantic_eligible_blocks
        or r.semblend_latency_ms > 0
        for r in requests
    )

    confirmed_values = [r.backend_confirmed_blocks for r in requests]
    has_confirmed = any(v is not None for v in confirmed_values)
    confirmed_blocks = sum(v or 0 for v in confirmed_values) if has_confirmed else None
    confirmed_tokens_values = [r.backend_confirmed_tokens for r in requests]
    confirmed_tokens = sum(v or 0 for v in confirmed_tokens_values) if has_confirmed else None
    external_values = [r.external_confirmed_tokens for r in requests]
    has_external = any(v is not None for v in external_values)
    external_tokens = sum(v or 0 for v in external_values) if has_external else None

    negative = [r for r in requests if r.negative_control]
    negative_blocks = sum(r.total_blocks for r in negative)
    negative_eligible = sum(r.semantic_eligible_blocks for r in negative)
    negative_semantic_placements = sum(
        1 for r in negative if r.route_outcome == "semantic_placement"
    )
    negative_confirmed = (
        sum((r.backend_confirmed_blocks or 0) for r in negative) if has_confirmed else None
    )
    route_outcomes: dict[str, int] = {}
    for request in requests:
        if request.route_outcome:
            route_outcomes[request.route_outcome] = route_outcomes.get(request.route_outcome, 0) + 1
    quality_values = [r.quality_pass for r in requests if r.quality_pass is not None]

    exact_rate = _rate(exact_blocks, total_blocks)
    candidate_rate = _rate(candidate_blocks, total_blocks)
    eligible_rate = _rate(eligible_blocks, total_blocks)
    confirmed_rate = (
        _rate(confirmed_blocks or 0, total_blocks) if confirmed_blocks is not None else None
    )

    return {
        "request_count": len(requests),
        "prompt_tokens": prompt_tokens,
        "total_blocks": total_blocks,
        "exact_hit_blocks": exact_blocks,
        "exact_hit_tokens": exact_tokens,
        "semantic_candidate_blocks": candidate_blocks,
        "semantic_candidate_tokens": candidate_tokens,
        "semantic_eligible_blocks": eligible_blocks,
        "semantic_eligible_tokens": eligible_tokens,
        "backend_confirmed_blocks": confirmed_blocks,
        "backend_confirmed_tokens": confirmed_tokens,
        # External-connector mass only. backend_confirmed_tokens above is
        # local + external and is not a semantic-reuse measurement.
        "external_confirmed_tokens": external_tokens,
        "external_confirmed_token_weighted_reuse_rate": (
            _rate(external_tokens, prompt_tokens) if external_tokens is not None else None
        ),
        "external_confirmed_token_sources": external_token_sources(requests),
        "exact_block_hit_rate": exact_rate,
        "semantic_candidate_block_rate": candidate_rate,
        "semantic_eligible_block_rate": eligible_rate,
        "backend_confirmed_block_rate": confirmed_rate,
        "semantic_eligible_lift": eligible_rate - exact_rate if has_semantic_plan else None,
        "backend_confirmed_lift": (
            confirmed_rate - exact_rate if confirmed_rate is not None else None
        ),
        "exact_token_weighted_reuse_rate": _rate(exact_tokens, prompt_tokens),
        "semantic_candidate_token_weighted_reuse_rate": _rate(candidate_tokens, prompt_tokens),
        "semantic_eligible_token_weighted_reuse_rate": _rate(eligible_tokens, prompt_tokens),
        "negative_control_count": len(negative),
        "negative_control_blocks": negative_blocks,
        "negative_control_semantic_eligible_blocks": negative_eligible,
        "negative_control_semantic_eligible_rate": _rate(negative_eligible, negative_blocks),
        "negative_control_semantic_placements": negative_semantic_placements,
        "negative_control_semantic_placement_rate": _rate(
            negative_semantic_placements,
            len(negative),
        ),
        "negative_control_backend_confirmed_blocks": negative_confirmed,
        "negative_control_backend_confirmed_rate": (
            _rate(negative_confirmed or 0, negative_blocks)
            if negative_confirmed is not None
            else None
        ),
        "route_outcomes": dict(sorted(route_outcomes.items())),
        "semantic_placement_rate_by_request": _rate(
            route_outcomes.get("semantic_placement", 0),
            len(requests),
        ),
        "semblend_hit_rate_by_request": _rate(
            sum(1 for r in requests if r.semblend_found),
            len(requests),
        ),
        "mean_semblend_latency_ms": _mean(
            [r.semblend_latency_ms for r in requests if r.semblend_latency_ms > 0]
        ),
        "mean_ttft_ms": _mean([r.ttft_ms for r in requests if r.ttft_ms is not None]),
        "mean_quality_f1": _mean([r.quality_f1 for r in requests if r.quality_f1 is not None]),
        "mean_quality_rouge_l": _mean(
            [r.quality_rouge_l for r in requests if r.quality_rouge_l is not None]
        ),
        "quality_f1_ci": _ci([r.quality_f1 for r in requests if r.quality_f1 is not None]),
        "quality_rouge_l_ci": _ci(
            [r.quality_rouge_l for r in requests if r.quality_rouge_l is not None]
        ),
        "quality_pass_rate": (
            _rate(sum(1 for v in quality_values if v), len(quality_values))
            if quality_values
            else None
        ),
        **connector_audit_metrics(requests),
    }


def _pctl(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


# A warm arm counts as a HIT only at >= this many SEMANTIC reuse tokens.
# Single-digit "hits" (e.g. a 3-token shared literal prefix) are numeric
# noise, not reuse.
REUSE_HIT_THRESHOLD_TOKENS = 64

# Mechanisms that are semantic reuse. "exact" is the prefix cache doing its
# ordinary job and is deliberately absent: on a prefix-caching-on arm a
# repeated document hits the local cache, and counting that as a semantic win
# is the single easiest way to publish a speedup the connector did not earn.
SEMANTIC_MECHANISMS = ("scatter", "head", "external")


def semantic_reuse_tokens(row: RequestMetrics) -> int | None:
    """Tokens this row reused SEMANTICALLY, or None when unmeasurable.

    Only two signals qualify, and neither can contain a local prefix hit:

    - ``external_confirmed_tokens`` — external KV connector mass (vLLM);
    - ``fuzzy_confirmed_tokens`` — sglang's fuzzy-admitted mass.

    ``backend_confirmed_tokens`` is explicitly NOT consulted: it is vLLM's
    ``cached_tokens``, which sums local prefix cache and external transfer,
    so with prefix caching on a repeated document reads as reuse there.

    None means no semantic signal was recorded at all — a pre-B12 row, where
    the engine was never asked for the split. None is not zero, and callers
    must not collapse it into one.

    A row the connector audit *did* join and that carries no
    ``runtime_materialized`` is the one case where an absent external count is
    a measurement: the audit looked and nothing was materialized. Those rows
    return 0 (a confirmed miss) rather than None, so they cannot fall through
    to the cached_tokens-based legacy hit.
    """
    external = row.external_confirmed_tokens
    fuzzy = row.fuzzy_confirmed_tokens or 0
    if external is None:
        if row.audit_materialized is False:
            return fuzzy
        return fuzzy or None
    return max(int(external), fuzzy)


def is_reuse_hit(row: RequestMetrics) -> bool | None:
    """Did this row clear the hit threshold on semantic reuse alone?

    None when :func:`semantic_reuse_tokens` is None, so an unmeasured arm
    reads as unmeasured rather than as a clean miss.
    """
    tokens = semantic_reuse_tokens(row)
    return None if tokens is None else tokens >= REUSE_HIT_THRESHOLD_TOKENS


def reuse_mechanism(row: RequestMetrics) -> str:
    """Classify HOW a warm arm reused: 'scatter' (fuzzy mass beyond the
    prefix path), 'head' (fuzzy folded into the prefix), 'external'
    (external-connector mass, the vLLM semantic path), 'exact' (prefix cache
    only — NOT a semantic win), or 'none'. Mechanisms damage quality
    differently (attention-sink positions vs mid-sequence) — never pool them.

    The 'exact' bucket is where a prefix-cache repeat lands, and it is kept
    out of :data:`SEMANTIC_MECHANISMS` on purpose: it is the bucket that
    exists so cached_tokens mass has somewhere honest to go.
    """
    fuzzy = row.fuzzy_confirmed_tokens or 0
    external = row.external_confirmed_tokens or 0
    prefix_path = row.backend_confirmed_tokens or 0  # local + external
    if fuzzy > prefix_path:
        return "scatter"
    if fuzzy > 0:
        return "head"
    if external >= REUSE_HIT_THRESHOLD_TOKENS:
        return "external"
    if prefix_path >= REUSE_HIT_THRESHOLD_TOKENS:
        return "exact"
    return "none"


# The manifest's traffic classes (phase-0 plan §3). The results layer has to
# know them by name because two of the metrics are defined over classes, not
# over requests: M1's opportunity denominator and M7's probe set.
NO_REUSE_CLASS = "no_reuse"
SAME_DOC_NEW_INSTRUCTION_CLASS = "same_doc_new_instruction"
REVISED_DOC_CLASS = "revised_doc"
ROPE_DELTA_SWEEP_CLASS = "rope_delta_sweep"
EXACT_REPEAT_CLASS = "exact_repeat"
# The class whose whole job is to detect propagation: a verbatim repeat of a
# request that was previously served approximate KV, >= 50 requests later.
PROPAGATION_PROBE_CLASS = "propagation_probe"
REWORDED_DOC_CLASS = "reworded_doc"
TRAFFIC_CLASSES = (
    NO_REUSE_CLASS,
    SAME_DOC_NEW_INSTRUCTION_CLASS,
    REVISED_DOC_CLASS,
    ROPE_DELTA_SWEEP_CLASS,
    EXACT_REPEAT_CLASS,
    PROPAGATION_PROBE_CLASS,
    REWORDED_DOC_CLASS,
)

# M1's alignment_given_opportunity denominator, from section 4 verbatim:
# "|{manifest items in same_doc_new_instruction ∪ revised_doc}|". It is the
# classes, not the offline model's per-item prediction: an item whose donor
# should have been reusable and was not is exactly what the metric is for, so
# it cannot be conditioned on the prediction that it would be.
ALIGNMENT_OPPORTUNITY_CLASSES = (SAME_DOC_NEW_INSTRUCTION_CLASS, REVISED_DOC_CLASS)


def traffic_class_of(row: RequestMetrics) -> str:
    """The row's traffic class, falling back to ``transform``.

    Manifests written before the class field existed carry the class in
    ``transform``; reading both means a probe item is found either way, and an
    empty string means the row declares no class at all.
    """
    return row.traffic_class or row.transform or ""


def audit_was_joined(rows: list[RequestMetrics]) -> bool:
    """Did a connector audit get joined onto any of these rows?

    ``audit_joined`` is None until :func:`sembench.connector_audit.join_requests`
    touches a row, so this separates "the audit was read and said nothing about
    these requests" (False on the rows, True here) from "no audit exists"
    (None on the rows, False here). Every audit-derived rate below is null in
    the second case: without the audit, every row's
    ``external_confirmed_tokens`` is null for want of measurement, and a rate
    computed over that reads as a perfect score for a run that measured
    nothing.
    """
    return any(row.audit_joined is not None for row in rows)


def _rate_or_none(numerator: int, denominator: int) -> float | None:
    """A rate, or None when the denominator is empty — never 0/0 as 0.0."""
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


@dataclass(frozen=True)
class AuditableRows:
    """The rows an audit-derived metric may be computed over, and the rest.

    A row is auditable when the connector could have written events about it
    *and* the audit was actually read for it. Two kinds of row cannot be:

    - a **cold-arm** row, which ran against a server with no connector at all.
      A merged cold/warm document holds both arms in one list, and computing
      M1/M2/M7 over that list doubles every denominator with requests no
      connector ever saw — halving each rate for free.
    - a row the audit was never joined to (``audit_joined is None``), which is
      unmeasured rather than negative.
    """

    rows: tuple[RequestMetrics, ...]
    excluded_cold_arm: int
    excluded_not_joined: int


def auditable_rows(rows: list[RequestMetrics]) -> AuditableRows:
    """Split rows into the auditable ones and counts of what was excluded."""
    from sembench.pairing import COLD_ARM

    considered: list[RequestMetrics] = []
    cold = 0
    unjoined = 0
    for row in rows:
        if row.arm == COLD_ARM:
            cold += 1
            continue
        if row.audit_joined is None:
            unjoined += 1
            continue
        considered.append(row)
    return AuditableRows(tuple(considered), cold, unjoined)


def _m1_alignment(considered: list[RequestMetrics], *, joined: bool) -> dict[str, Any]:
    """M1: the three alignment numbers, plus the offline-model integrity check.

    Deduped by request id by construction: one row is one request, and the
    audit fold already collapsed a re-queried request's repeated attempts.
    """
    advertised_at_boundary = [
        row
        for row in considered
        if (row.audit_advertised_tokens or 0) > 0 and (row.audit_observed_boundary or 0) > 0
    ]
    lookup_hits = [
        row
        for row in considered
        if row.audit_semantic_lookup_hit and (row.audit_lookup_hit_boundary or 0) > 0
    ]
    opportunity = [
        row for row in considered if traffic_class_of(row) in ALIGNMENT_OPPORTUNITY_CLASSES
    ]
    aligned = len(advertised_at_boundary)
    # One deliberate deviation. Section 4 shares ONE numerator between the two
    # rates, which holds only if every advertise comes from an item in the
    # opportunity classes -- and it does not, since rope_delta_sweep items
    # carry donors and advertise too. A shared numerator over a two-class
    # denominator can exceed 1.0 and stop being a fraction, so each rate is
    # counted over its own denominator's population. The gap between the two
    # numerators is itself the count of advertises won outside the two classes.
    aligned_opportunity = sum(
        1
        for row in advertised_at_boundary
        if traffic_class_of(row) in ALIGNMENT_OPPORTUNITY_CLASSES
    )
    miss_breakdown: dict[str, int] = {}
    for row in considered:
        if row.audit_boundary_miss_reason and (row.audit_boundary_missed_at or 0) > 0:
            reason = row.audit_boundary_miss_reason
            miss_breakdown[reason] = miss_breakdown.get(reason, 0) + 1
    opportunity_rate = _rate_or_none(aligned_opportunity, len(opportunity)) if joined else None
    return {
        "alignment_given_match": _rate_or_none(aligned, len(lookup_hits)) if joined else None,
        "alignment_given_match_numerator": aligned if joined else None,
        "alignment_given_match_denominator": len(lookup_hits),
        "alignment_given_opportunity": opportunity_rate,
        "alignment_given_opportunity_numerator": aligned_opportunity if joined else None,
        "alignment_given_opportunity_denominator": len(opportunity),
        "boundary_miss_breakdown": dict(sorted(miss_breakdown.items())) if joined else None,
        # The headline alias. The same number as alignment_given_opportunity,
        # kept because it is the name every earlier result document used.
        "boundary_alignment_rate": opportunity_rate,
        **_m1_integrity_check(considered, joined=joined),
    }


def _m1_integrity_check(considered: list[RequestMetrics], *, joined: bool) -> dict[str, Any]:
    """Did the live planner do what the offline model predicted it would?

    Section 4: divergence on the token count "means the offline model and the
    live engine disagree about the planner -- investigate". Only rows carrying
    both a manifest expectation and an advertise can be compared; the rest are
    outside the denominator rather than counted as agreement.
    """
    supplied_claims = [
        row
        for row in considered
        if row.expected_supplied_tokens is not None and row.audit_advertised_tokens is not None
    ]
    supplied_agreed = sum(
        1 for row in supplied_claims if row.audit_advertised_tokens == row.expected_supplied_tokens
    )
    start_claims = [
        row
        for row in considered
        if row.expected_span_target_start is not None
        and row.audit_advertised_target_start is not None
    ]
    start_agreed = sum(
        1
        for row in start_claims
        if row.audit_advertised_target_start == row.expected_span_target_start
    )
    return {
        "expected_supplied_tokens_agreement_rate": (
            _rate_or_none(supplied_agreed, len(supplied_claims)) if joined else None
        ),
        "expected_supplied_tokens_agreement_numerator": supplied_agreed if joined else None,
        "expected_supplied_tokens_agreement_denominator": len(supplied_claims),
        "expected_span_target_start_agreement_rate": (
            _rate_or_none(start_agreed, len(start_claims)) if joined else None
        ),
        "expected_span_target_start_agreement_numerator": start_agreed if joined else None,
        "expected_span_target_start_agreement_denominator": len(start_claims),
    }


def _m2_materialized_reuse(considered: list[RequestMetrics], *, joined: bool) -> dict[str, Any]:
    """M2: the token-weighted headline, and the request-count rate beside it."""
    advertised = [row for row in considered if row.audit_advertised_tokens is not None]
    materialized_rows = [row for row in advertised if row.audit_materialized]
    advertised_tokens = sum(row.audit_advertised_tokens or 0 for row in advertised)
    materialized_tokens = sum(row.external_confirmed_tokens or 0 for row in materialized_rows)
    token_rate = _rate_or_none(materialized_tokens, advertised_tokens) if joined else None
    return {
        # Section 4's formula, and therefore the headline.
        "materialized_reuse_rate": token_rate,
        "materialized_reuse_tokens": materialized_tokens if joined else None,
        "materialized_reuse_advertised_tokens": advertised_tokens if joined else None,
        # Alias of the same token-weighted number, for continuity.
        "materialized_reuse_token_rate": token_rate,
        # The request-count question, under its own name.
        "materialized_reuse_request_rate": (
            _rate_or_none(len(materialized_rows), len(advertised)) if joined else None
        ),
        "materialized_reuse_request_numerator": len(materialized_rows) if joined else None,
        "materialized_reuse_request_denominator": len(advertised),
    }


def _m7_inputs(considered: list[RequestMetrics], *, joined: bool) -> dict[str, Any]:
    """M7's per-arm inputs: the gating counter and one supporting signal.

    Neither is M7 itself, which is the cross-arm answer comparison in
    :func:`paired_summary`.
    """
    probes = [row for row in considered if traffic_class_of(row) == PROPAGATION_PROBE_CLASS]
    cached_without_materialization = sum(
        1
        for row in probes
        if row.external_confirmed_tokens is None and (row.backend_confirmed_tokens or 0) > 0
    )
    evicting_rows = [row for row in considered if row.audit_prefix_blocks_evicted is not None]
    blocks_evicted = sum(row.audit_prefix_blocks_evicted or 0 for row in evicting_rows)
    return {
        "propagation_cached_without_materialization_rate": (
            _rate_or_none(cached_without_materialization, len(probes)) if joined else None
        ),
        "propagation_cached_without_materialization_numerator": (
            cached_without_materialization if joined else None
        ),
        "propagation_cached_without_materialization_denominator": len(probes),
        "prefix_blocks_evicted": blocks_evicted if joined else None,
        "rows_with_prefix_blocks_evicted": (
            sum(1 for row in evicting_rows if (row.audit_prefix_blocks_evicted or 0) > 0)
            if joined
            else None
        ),
    }


def connector_audit_metrics(rows: list[RequestMetrics]) -> dict[str, Any]:
    """M1 boundary alignment and M2 materialized reuse, per section 4.

    Every rate here is computed over :func:`auditable_rows` only — never over
    a cold arm, never over a row the audit was not joined to — and every rate
    publishes its own numerator and denominator, so a 0.0 over three requests
    is never read as a 0.0 over three hundred.

    **M1 — boundary alignment.** Section 4 gives three numbers, all
    conditioned on ``boundary > 0`` and all deduped by ``request_id``::

        alignment_given_match        = |{semantic_span_load_advertised, boundary>0, token_count>0}|
                                     / |{semantic_lookup_hit, boundary>0}|

        alignment_given_opportunity  = same numerator
                                     / |{manifest items in same_doc_new_instruction ∪ revised_doc}|

        boundary_miss_breakdown      = boundary_missed events partitioned by reason

    The two rates differ in what they are conditioned on (and therefore, see
    :func:`_m1_alignment`, in the population their shared numerator is counted
    over): ``alignment_given_match`` asks "when the provider found a donor,
    did the engine's boundary land on a span?", which is the property of the
    *tokenizer and the template* that caveat A of the plan is about;
    ``alignment_given_opportunity`` asks "of the traffic that should have been
    reusable, how much was served?", which is the product number and therefore
    the headline — ``boundary_alignment_rate`` is kept as its alias and
    nothing else.

    Beside them, M1's integrity check: does the connector's advertised
    ``token_count`` match the offline model's ``expected_supplied_tokens``,
    and its ``target_start`` the model's ``expected_span_target_start``?
    Section 4: divergence on the token count "means the offline model and the
    live engine disagree about the planner — investigate".

    **M2 — materialized reuse.** Section 4, verbatim::

        materialized_reuse_rate = Σ runtime_materialized.tokens
                                / Σ semantic_span_load_advertised.token_count

    That token-weighted ratio is what ``materialized_reuse_rate`` carries. The
    request-count form ("how many requests got any of their promise") is a
    different question and has a different name,
    ``materialized_reuse_request_rate``; neither substitutes for the other.

    **M7's inputs, not M7.** ``prefix_blocks_evicted`` is section 4's gating
    counter: until it reads non-zero on a contaminated workload, every lane-2
    quality number is unproven. ``propagation_cached_without_materialization_rate``
    is a per-row supporting signal (probes with no materialization of their
    own but non-zero ``cached_tokens``), NOT M7 — M7 is the cross-arm answer
    comparison in :func:`paired_summary`, which needs both arms.
    """
    auditable = auditable_rows(rows)
    considered = list(auditable.rows)
    # "Was an audit joined to any row these metrics could be computed over" --
    # not "to any row at all". A cold row the join happened to stamp says
    # nothing about the arm that ran the connector.
    joined = audit_was_joined(considered)
    return {
        "connector_audit_present": joined,
        "connector_audit_rows_joined": (
            sum(1 for row in considered if row.audit_joined) if joined else None
        ),
        # What the metrics below were, and were not, computed over.
        "connector_audit_rows_considered": len(considered),
        "connector_audit_rows_excluded_cold_arm": auditable.excluded_cold_arm,
        "connector_audit_rows_excluded_not_joined": auditable.excluded_not_joined,
        **_m1_alignment(considered, joined=joined),
        **_m2_materialized_reuse(considered, joined=joined),
        **_m7_inputs(considered, joined=joined),
    }


def external_token_sources(rows: list[RequestMetrics]) -> list[str]:
    """Distinct provenance labels on ``external_confirmed_tokens``, sorted."""
    return sorted(
        {
            row.external_confirmed_tokens_source
            for row in rows
            if row.external_confirmed_tokens_source
        }
    )


def external_tokens_are_per_request(rows: list[RequestMetrics]) -> bool | None:
    """True only when every external value present is a per-request number.

    False means at least one row carries an ARM-LEVEL value (the per-arm delta
    of ``vllm:external_prefix_cache_hits``), which supports arm-level claims
    only. None means no row declared a source.
    """
    sources = external_token_sources(rows)
    if not sources:
        return None
    return all(source in PER_REQUEST_EXTERNAL_SOURCES for source in sources)


@dataclass(frozen=True)
class _Pair:
    """One item's cold twin and warm twin, both usable."""

    item_id: str
    cold: RequestMetrics
    warm: RequestMetrics


def _clean_pairs(
    cold: dict[str, RequestMetrics],
    warm: dict[str, RequestMetrics],
) -> tuple[list[_Pair], int, int]:
    """Pairs whose cold twin was genuinely cold and whose arms both answered.

    Returns the usable pairs plus the counts it removed, so the summary can
    report what it dropped instead of shrinking a denominator in silence.
    """
    pairs: list[_Pair] = []
    contaminated = 0
    errored = 0
    for item_id, cold_row in cold.items():
        warm_row = warm.get(item_id)
        if warm_row is None:
            continue
        if cold_row.flush_contaminated:
            contaminated += 1
            continue
        if cold_row.error or warm_row.error:
            errored += 1
            continue
        pairs.append(_Pair(item_id, cold_row, warm_row))
    return pairs, contaminated, errored


def _hit_accounting(warm_rows: list[RequestMetrics]) -> dict[str, Any]:
    """Split the warm arm's hits into confirmed-semantic and unverified.

    A pair is *confirmed* when the row carries a semantic signal at all
    (:func:`semantic_reuse_tokens`). A pair is *unverified* when it does not:
    all that exists for it is ``cached_tokens``, which on a prefix-caching-on
    vLLM arm is local + external and cannot distinguish a semantic reuse from
    an ordinary exact-prefix repeat.

    ``hit_rate_external_confirmed`` is computed over confirmed pairs only and
    is None when there are none — so an arm that never measured the split
    fails a None-refusing gate instead of passing it on cached_tokens mass.
    """
    hits: list[bool] = []
    confirmed_hits: list[bool] = []
    unverified_hits: list[bool] = []
    for row in warm_rows:
        confirmed = is_reuse_hit(row)
        if confirmed is None:
            legacy_hit = (row.backend_confirmed_tokens or 0) >= REUSE_HIT_THRESHOLD_TOKENS
            unverified_hits.append(legacy_hit)
            hits.append(legacy_hit)
        else:
            confirmed_hits.append(confirmed)
            hits.append(confirmed)
    return {
        "hits": hits,
        "hit_rate": _rate(sum(hits), len(hits)) if hits else None,
        "hit_rate_external_confirmed": (
            _rate(sum(confirmed_hits), len(confirmed_hits)) if confirmed_hits else None
        ),
        "pairs_external_confirmed": len(confirmed_hits),
        "pairs_external_unconfirmed": len(unverified_hits),
        "hits_unverified_external": sum(unverified_hits),
    }


def _kl_summary(pairs: list[_Pair]) -> dict[str, Any]:
    """Warm-vs-cold top-k KL over pairs that captured logprobs on both sides."""
    from sembench.kl import kl_profile
    from sembench.stats import bootstrap_mean

    kl_means: list[float] = []
    first_divergences: list[int] = []
    for pair in pairs:
        if not pair.cold.output_top_logprobs or not pair.warm.output_top_logprobs:
            continue
        profile = kl_profile(
            pair.cold.output_token_ids or [],
            [[tuple(entry) for entry in pos] for pos in pair.cold.output_top_logprobs],
            pair.warm.output_token_ids or [],
            [[tuple(entry) for entry in pos] for pos in pair.warm.output_top_logprobs],
        )
        kl_means.append(profile.mean_kl_topk)
        if profile.first_token_divergence is not None:
            first_divergences.append(profile.first_token_divergence)
    kl_ci = bootstrap_mean(kl_means)
    return {
        "warm_vs_cold_mean_kl_topk": kl_ci.point if kl_ci else None,
        "warm_vs_cold_mean_kl_topk_ci": kl_ci.to_dict() if kl_ci else None,
        "kl_pairs": len(kl_means),
        "first_token_divergence_median": (
            sorted(first_divergences)[len(first_divergences) // 2] if first_divergences else None
        ),
    }


def _propagation_summary(
    pairs: list[_Pair],
    warm_by_item: dict[str, RequestMetrics],
) -> dict[str, Any]:
    """M7 — contamination / propagation, as section 4 defines it.

    Section 4::

        A4 vs A6 on the 50 propagation_probe items: fraction whose answer in
        A6 matches the *served* output rather than the *cold* (A1) output.

    A propagation probe is a verbatim repeat of an earlier request that was
    served approximate KV (§3), so three answers exist for one prompt and the
    question is which of two the treatment arm reproduces:

    - the **served** answer — the parent item's answer in the *same* arm, i.e.
      the output produced while approximate KV was in play;
    - the **cold** answer — this item's own answer in the baseline arm, which
      is what an uncontaminated engine must return for a verbatim repeat.

    A probe counts as propagated when its treatment answer is strictly closer
    (ROUGE-L) to the served answer than to the cold one. Strictly: a tie is
    not evidence, and contamination is a claim that has to be earned.

    Every probe that cannot be scored is counted and named rather than
    dropped — an unlinked probe, a parent with no answer in this arm, a probe
    missing either answer — because M7's denominator is 50 items by design and
    a silently shrunken one reads as a clean result.
    """
    from sembench.quality import rouge_l

    probes = [pair for pair in pairs if traffic_class_of(pair.warm) == PROPAGATION_PROBE_CLASS]
    scored = 0
    propagated = 0
    unlinked = 0
    without_served_answer = 0
    without_answers = 0
    for pair in probes:
        parent_id = pair.warm.propagation_parent_item_id or pair.cold.propagation_parent_item_id
        if not parent_id:
            unlinked += 1
            continue
        served = warm_by_item.get(parent_id)
        if served is None or not served.output_text:
            without_served_answer += 1
            continue
        if not pair.warm.output_text or not pair.cold.output_text:
            without_answers += 1
            continue
        scored += 1
        to_served = rouge_l(pair.warm.output_text, served.output_text)
        to_cold = rouge_l(pair.warm.output_text, pair.cold.output_text)
        if to_served > to_cold:
            propagated += 1
    return {
        "propagation_definition": (
            "share of propagation_probe items whose treatment-arm answer is closer to the "
            "parent item's answer in the same arm (the served output) than to its own "
            "answer in the baseline arm (the cold output)"
        ),
        "propagation_contamination_rate": _rate_or_none(propagated, scored),
        "propagation_contamination_numerator": propagated,
        "propagation_contamination_denominator": scored,
        "propagation_probe_pairs": len(probes),
        "propagation_probes_unlinked": unlinked,
        "propagation_probes_without_served_answer": without_served_answer,
        "propagation_probes_without_answers": without_answers,
    }


def _engine_ttft_summary(pairs: list[_Pair]) -> dict[str, Any]:
    """Speedup from the engine-side TTFT that excludes queue wait.

    Client-side ``ttft_ms`` under concurrency is mostly queue time, so the
    median-speedup gate is not evaluable from it. This block is reported
    separately rather than folded into the headline: the two numbers answer
    different questions and pooling them would hide which one was measured.
    """
    from sembench.stats import bootstrap_median_ratio

    cold_engine = [
        p.cold.engine_ttft_ms
        for p in pairs
        if not p.cold.negative_control and p.cold.engine_ttft_ms and p.warm.engine_ttft_ms
    ]
    warm_engine = [
        p.warm.engine_ttft_ms
        for p in pairs
        if not p.cold.negative_control and p.cold.engine_ttft_ms and p.warm.engine_ttft_ms
    ]
    engine_ci = bootstrap_median_ratio(cold_engine, warm_engine) if cold_engine else None
    return {
        "engine_ttft_pairs": len(cold_engine),
        "engine_ttft_speedup_median": engine_ci.point if engine_ci else None,
        "engine_ttft_speedup_median_ci": engine_ci.to_dict() if engine_ci else None,
        "queue_time_cold_p50_ms": _pctl(
            [p.cold.queue_time_ms for p in pairs if p.cold.queue_time_ms is not None], 50
        ),
        "queue_time_warm_p50_ms": _pctl(
            [p.warm.queue_time_ms for p in pairs if p.warm.queue_time_ms is not None], 50
        ),
    }


def paired_summary(requests: list[RequestMetrics]) -> dict[str, Any] | None:
    """Per-item cold/warm pairing: TTFT speedups and warm-vs-cold output
    similarity. Pairs with a contaminated cold arm are excluded and counted.

    Speedup semantics (the honesty split):
    - blended: EVERY clean pair contributes; a pair with no confirmed reuse
      in the warm arm contributes its real ratio (typically ~1.0). This is
      the headline number, and its headline statistic is the **median**
      (``blended_ttft_speedup_median``) — ratios are heavy-tailed and the
      mean is kept beside it only as a secondary, tail-sensitive figure.
    - hit-only: pairs whose warm arm reused semantically. Meaningful ONLY
      next to hit_rate — reported together, never alone.
    - negative controls: their speedup must be ~1.0; deviation means the
      cache fired (or slowed things) on unrelated content.

    A "hit" is semantic reuse only (see :func:`semantic_reuse_tokens`); a
    local prefix-cache repeat lands in the 'exact' mechanism bucket and in
    ``hits_unverified_external``, never in ``hit_rate_external_confirmed``.

    This block is also where M7 lives (:func:`_propagation_summary`): it is a
    comparison between the two arms' answers and cannot be computed from one
    arm's rows, which is why the per-arm audit metrics carry only its inputs.
    """
    cold = {r.item_id: r for r in requests if r.arm == "cold"}
    warm = {r.item_id: r for r in requests if r.arm == "warm"}
    if not cold or not warm:
        return None
    from collections import Counter

    from sembench.quality import rouge_l
    from sembench.stats import bootstrap_mean, bootstrap_median, bootstrap_median_ratio

    pairs, contaminated, errored = _clean_pairs(cold, warm)
    timed = [p for p in pairs if p.cold.ttft_ms and p.warm.ttft_ms]
    blended = [p for p in timed if not p.cold.negative_control]
    controls = [p for p in timed if p.cold.negative_control]

    cold_ttfts = [p.cold.ttft_ms for p in timed]
    warm_ttfts = [p.warm.ttft_ms for p in timed]
    speedups = [p.cold.ttft_ms / p.warm.ttft_ms for p in blended]
    negative_speedups = [p.cold.ttft_ms / p.warm.ttft_ms for p in controls]

    accounting = _hit_accounting([p.warm for p in blended])
    hit_pairs = [pair for pair, hit in zip(blended, accounting["hits"]) if hit]
    hit_speedups = [p.cold.ttft_ms / p.warm.ttft_ms for p in hit_pairs]

    mechanisms: Counter[str] = Counter(reuse_mechanism(p.warm) for p in blended)
    output_rouge = [
        rouge_l(p.warm.output_text, p.cold.output_text)
        for p in pairs
        if p.cold.output_text and p.warm.output_text
    ]

    speedup_ci = bootstrap_mean(speedups)
    speedup_median_ci = (
        bootstrap_median_ratio([p.cold.ttft_ms for p in blended], [p.warm.ttft_ms for p in blended])
        if blended
        else None
    )
    hit_ci = bootstrap_mean(hit_speedups)
    hit_median_ci = (
        bootstrap_median_ratio(
            [p.cold.ttft_ms for p in hit_pairs], [p.warm.ttft_ms for p in hit_pairs]
        )
        if hit_pairs
        else None
    )
    negative_ci = bootstrap_mean(negative_speedups)
    negative_median_ci = bootstrap_median(negative_speedups)
    rouge_ci = bootstrap_mean(output_rouge)
    warm_rows = [p.warm for p in pairs]
    return {
        "hit_definition": (
            f">={REUSE_HIT_THRESHOLD_TOKENS} semantic reuse tokens "
            "(external-connector or fuzzy-admitted mass only; cached_tokens, "
            "which is local prefix cache + external, is never counted)"
        ),
        "reuse_mechanisms": dict(mechanisms),
        "semantic_mechanisms": list(SEMANTIC_MECHANISMS),
        "fuzzy_confirmed_tokens_warm": [w.fuzzy_confirmed_tokens or 0 for w in warm.values()],
        "external_confirmed_tokens_warm": [w.external_confirmed_tokens for w in warm.values()],
        "external_confirmed_token_sources": external_token_sources(warm_rows),
        "external_confirmed_is_per_request": external_tokens_are_per_request(warm_rows),
        "pairs_total": len(cold),
        "pairs_used": len(timed),
        "pairs_contaminated": contaminated,
        "pairs_errored": errored,
        "ttft_cold_p50_ms": _pctl(cold_ttfts, 50),
        "ttft_cold_p95_ms": _pctl(cold_ttfts, 95),
        "ttft_warm_p50_ms": _pctl(warm_ttfts, 50),
        "ttft_warm_p95_ms": _pctl(warm_ttfts, 95),
        # The gated statistic. The mean below is secondary.
        "blended_ttft_speedup_median": speedup_median_ci.point if speedup_median_ci else None,
        "blended_ttft_speedup_median_ci": (
            speedup_median_ci.to_dict() if speedup_median_ci else None
        ),
        "blended_ttft_speedup_mean": speedup_ci.point if speedup_ci else None,
        "blended_ttft_speedup_ci": speedup_ci.to_dict() if speedup_ci else None,
        "hit_rate": accounting["hit_rate"],
        "hit_rate_external_confirmed": accounting["hit_rate_external_confirmed"],
        "pairs_external_confirmed": accounting["pairs_external_confirmed"],
        "pairs_external_unconfirmed": accounting["pairs_external_unconfirmed"],
        "hits_unverified_external": accounting["hits_unverified_external"],
        "hit_only_ttft_speedup_median": hit_median_ci.point if hit_median_ci else None,
        "hit_only_ttft_speedup_median_ci": hit_median_ci.to_dict() if hit_median_ci else None,
        "hit_only_ttft_speedup_mean": hit_ci.point if hit_ci else None,
        "hit_only_ttft_speedup_ci": hit_ci.to_dict() if hit_ci else None,
        "negative_control_pairs": len(negative_speedups),
        # The gated statistic for the control, matching the headline: one
        # control pair that happened to hit a slow cold prefill moves a mean
        # far enough to fail (or pass) a deviation gate on its own.
        "negative_control_ttft_speedup_median": (
            negative_median_ci.point if negative_median_ci else None
        ),
        "negative_control_ttft_speedup_median_ci": (
            negative_median_ci.to_dict() if negative_median_ci else None
        ),
        "negative_control_ttft_speedup_mean": negative_ci.point if negative_ci else None,
        "warm_vs_cold_output_rouge_l_mean": rouge_ci.point if rouge_ci else None,
        "warm_vs_cold_output_rouge_l_ci": rouge_ci.to_dict() if rouge_ci else None,
        # Audit-joined metrics are the warm (connector) arm's: the cold arm
        # runs without a connector and has no audit stream to join, and
        # connector_audit_metrics drops any cold row it is handed anyway.
        **connector_audit_metrics(warm_rows),
        # M7 needs both arms and the warm arm's other rows: see
        # _propagation_summary.
        **_propagation_summary(pairs, warm),
        **_kl_summary(pairs),
        **_engine_ttft_summary(pairs),
        # Back-compat alias for the pre-P3 field name.
        "ttft_speedup_mean": speedup_ci.point if speedup_ci else None,
    }


def result_arm(payload: dict[str, Any]) -> str:
    """The arm a result document declares for itself; '' when it declares none.

    Read from ``run.arm``, which :func:`sembench.schema.collect_run_metadata`
    stamps from the flag the arm was actually launched with.
    """
    run = payload.get("run") or {}
    return str(run.get("arm") or "")


def arm_label_conflicts(
    cold_payload: dict[str, Any],
    warm_payload: dict[str, Any],
) -> list[str]:
    """Ways the operator's --cold/--warm assignment contradicts the payloads.

    ``merge-results`` takes the two roles from the command line, so swapping
    the flags silently inverts every speedup in the merged document: a 5x
    reported win is really a 0.2x loss and nothing in the output says so. The
    result documents already know which arm they are, so the claim can be
    checked instead of trusted.

    ``single`` and an absent arm make no claim and are accepted — that is the
    normal shape of a run launched without ``--arm``. A payload declaring the
    *other* role, or declaring ``paired`` (it is already a merged document
    holding both arms), is a conflict.
    """
    from sembench.pairing import SINGLE_ARM

    conflicts: list[str] = []
    for role, payload in (("cold", cold_payload), ("warm", warm_payload)):
        declared = result_arm(payload)
        if declared and declared not in (role, SINGLE_ARM):
            conflicts.append(
                f"--{role} result declares run.arm={declared!r}: "
                f"it was not run as the {role} arm, and merging it as one "
                "inverts or invalidates every paired number downstream"
            )
    return conflicts


def aggregate_by_transform(requests: list[RequestMetrics]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[RequestMetrics]] = {}
    for request in requests:
        groups.setdefault(request.transform, []).append(request)
    return {name: aggregate_metrics(group) for name, group in sorted(groups.items())}


def write_result(
    path: str | Path,
    *,
    requests: list[RequestMetrics],
    config: dict[str, Any],
    run: RunMetadata | None = None,
    engine: dict[str, Any] | None = None,
) -> None:
    """Write one arm's result document.

    ``engine`` carries the arm's serve line, the parsed launch flags, and the
    before/after engine counter window (see ``sembench.engine_config``). The
    key is always present — null when the arm was run without it — so a
    reader can tell "no engine config was recorded" from "these were the
    flags", rather than assuming.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_version": RESULT_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": run.to_dict() if run is not None else None,
        "config": config,
        "engine": engine,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": platform.node(),
        },
        "aggregate": aggregate_metrics(requests),
        "paired": paired_summary(requests),
        "by_transform": aggregate_by_transform(requests),
        "requests": [r.to_dict() for r in requests],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _ci(values: list[float]) -> dict[str, float] | None:
    from sembench.stats import bootstrap_mean

    result = bootstrap_mean(values)
    return result.to_dict() if result is not None else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
