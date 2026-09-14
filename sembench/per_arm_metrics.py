"""Section 4's per-arm metric blocks: the aggregate, M1, M2 and M7's inputs.

Everything here is computable from ONE arm's rows. The cross-arm metrics (M3's
speedup, M4's miss tax, M6's paired quality, M7 itself) need both arms and live
in :mod:`sembench.paired_metrics`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sembench.metric_math import _ci, _mean, _rate, _rate_or_none
from sembench.reuse_signals import (
    audit_was_joined,
    auditable_rows,
    external_token_sources,
)
from sembench.schema import RequestMetrics
from sembench.traffic_classes import (
    ALIGNMENT_OPPORTUNITY_CLASSES,
    DENOMINATOR_FROM_MANIFEST,
    DENOMINATOR_FROM_ROWS_PRESENT,
    PROPAGATION_PROBE_CLASS,
    WRAPPER_STRATA,
    WRAPPER_STRATUM_RULE,
    manifest_class_total,
    traffic_class_of,
    wrapper_head_share,
    wrapper_stratum_of,
)


def aggregate_metrics(
    requests: list[RequestMetrics],
    *,
    manifest_class_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Aggregate per-request metrics into benchmark-level rates.

    ``manifest_class_counts`` is the workload's per-traffic-class item count
    (``sembench.schema.manifest_class_counts``). Section 4's class-scoped
    denominators — M1's opportunity classes, M7's probe set — are counts of
    manifest items, so they are taken from it when the runner carried it into
    the result and from the rows present only when it did not.
    """
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
        # M6: "Report quality per RoPE-delta bucket. A quality result gathered
        # only at |delta| <= 11 does not transfer."
        "quality_by_rope_delta_bucket": quality_by_rope_delta_bucket(requests),
        **connector_audit_metrics(requests, manifest_class_counts=manifest_class_counts),
    }


def quality_by_rope_delta_bucket(requests: list[RequestMetrics]) -> dict[str, Any] | None:
    """Answer quality split by the manifest's RoPE-delta bucket.

    Section 4 (M6): "Report quality per RoPE-delta bucket from the
    rope_delta_sweep class. A quality result gathered only at |delta| <= 11
    does not transfer." Re-rotating donor KV across a large positional delta is
    the specific quality risk lane 2 carries, and a blended mean over a stream
    that is 93% |delta| ~ 0 cannot show it.

    Keys are the bucket as a string, because a result document is JSON. None
    when no row declares a bucket — the split was not measured, which is not
    the same as a workload with no positional deltas.
    """
    groups: dict[int, list[RequestMetrics]] = {}
    for row in requests:
        if row.rope_delta_bucket is None:
            continue
        groups.setdefault(int(row.rope_delta_bucket), []).append(row)
    if not groups:
        return None
    summary: dict[str, Any] = {}
    for bucket, rows in sorted(groups.items()):
        passes = [row.quality_pass for row in rows if row.quality_pass is not None]
        summary[str(bucket)] = {
            "requests": len(rows),
            "mean_quality_f1": _mean([r.quality_f1 for r in rows if r.quality_f1 is not None]),
            "mean_quality_rouge_l": _mean(
                [r.quality_rouge_l for r in rows if r.quality_rouge_l is not None]
            ),
            "quality_pass_rate": (
                _rate(sum(1 for v in passes if v), len(passes)) if passes else None
            ),
            "mean_ttft_ms": _mean([r.ttft_ms for r in rows if r.ttft_ms is not None]),
        }
    return summary


def _m1_breakdowns(
    considered: list[RequestMetrics],
) -> tuple[dict[str, int], dict[str, int]]:
    """M1's two diagnoses: why misses missed, and how spans were declined.

    Together they account for every lookup hit that was not served — a miss
    that reached the boundary-missed branch carries a partition reason, and
    one declined earlier (unaligned boundary, below the minimum after a clamp)
    carries only its event name, which is its cause.
    """
    miss_breakdown: dict[str, int] = {}
    decline_breakdown: dict[str, int] = {}
    for row in considered:
        if row.audit_boundary_miss_reason and (row.audit_boundary_missed_at or 0) > 0:
            reason = row.audit_boundary_miss_reason
            miss_breakdown[reason] = miss_breakdown.get(reason, 0) + 1
        if row.audit_span_decline_event:
            event = row.audit_span_decline_event
            decline_breakdown[event] = decline_breakdown.get(event, 0) + 1
    return (
        dict(sorted(miss_breakdown.items())),
        dict(sorted(decline_breakdown.items())),
    )


def _advertised_at_boundary(rows: Sequence[RequestMetrics]) -> list[RequestMetrics]:
    """M1's shared numerator: advertises that carried tokens at a boundary."""
    return [
        row
        for row in rows
        if (row.audit_advertised_tokens or 0) > 0 and (row.audit_observed_boundary or 0) > 0
    ]


def _lookup_hits_at_boundary(rows: Sequence[RequestMetrics]) -> list[RequestMetrics]:
    """M1's match denominator: lookups that found a donor at a boundary."""
    return [
        row
        for row in rows
        if row.audit_semantic_lookup_hit and (row.audit_lookup_hit_boundary or 0) > 0
    ]


def _opportunity_rows(rows: Sequence[RequestMetrics]) -> list[RequestMetrics]:
    """Rows in section 4's two opportunity classes."""
    return [row for row in rows if traffic_class_of(row) in ALIGNMENT_OPPORTUNITY_CLASSES]


def _m1_stratum_block(rows: list[RequestMetrics]) -> dict[str, Any]:
    """M1's three numbers over one wrapper stratum.

    The opportunity denominator here is always the rows the stratum holds: the
    manifest's per-class counts are not split by wrapper, so there is no
    manifest-side number to divide by and pretending otherwise would mix a
    workload count with a row count.
    """
    advertised = _advertised_at_boundary(rows)
    hits = _lookup_hits_at_boundary(rows)
    opportunity = _opportunity_rows(rows)
    miss_breakdown, decline_breakdown = _m1_breakdowns(rows)
    return {
        "rows_considered": len(rows),
        "alignment_given_match": _rate_or_none(len(advertised), len(hits)),
        "alignment_given_match_numerator": len(advertised),
        "alignment_given_match_denominator": len(hits),
        "alignment_given_opportunity": _rate_or_none(len(advertised), len(opportunity)),
        "alignment_given_opportunity_numerator": len(advertised),
        "alignment_given_opportunity_denominator": len(opportunity),
        "alignment_given_opportunity_denominator_source": DENOMINATOR_FROM_ROWS_PRESENT,
        "boundary_miss_breakdown": miss_breakdown,
        "span_decline_breakdown": decline_breakdown,
    }


def _m1_by_wrapper_stratum(
    considered: list[RequestMetrics],
    *,
    joined: bool,
) -> dict[str, Any] | None:
    """Section 4's split of M1 by wrapper stratum, or None with no audit.

    Every considered row lands in exactly one of the three buckets, so the
    per-stratum numerators sum to the blended ones: the split explains the
    headline and cannot change it.
    """
    if not joined:
        return None
    groups: dict[str, list[RequestMetrics]] = {name: [] for name in WRAPPER_STRATA}
    for row in considered:
        groups[wrapper_stratum_of(row)].append(row)
    return {name: _m1_stratum_block(rows) for name, rows in groups.items()}


def _m1_alignment(
    considered: list[RequestMetrics],
    *,
    joined: bool,
    manifest_class_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """M1: the three alignment numbers, plus the offline-model integrity check.

    Deduped by request id by construction: one row is one request, and the
    audit fold already collapsed a re-queried request's repeated attempts.

    Section 4 shares ONE numerator between the two rates, and that is what is
    published here: ``alignment_given_match_numerator`` and
    ``alignment_given_opportunity_numerator`` are the same count of advertises.
    The shared numerator is not restricted to the two opportunity classes, so
    the opportunity rate CAN exceed 1.0 — a ``rope_delta_sweep`` or
    ``exact_repeat`` item carries a donor and advertises too. Restricting the
    numerator silently would make the rate look like a fraction while
    answering a question section 4 did not ask, so the excess is named instead:
    ``alignment_given_opportunity_numerator_outside_classes`` is exactly how
    many of the advertises came from outside the denominator's population, and
    a rate above 1.0 is read against it.

    Section 4 also forbids publishing the blended rate alone: "Do not report a
    single blended alignment number. Report it separately for the
    shared-wrapper stratum and the ad-hoc stratum." That split is
    ``alignment_by_wrapper_stratum`` (see :data:`WRAPPER_STRATUM_RULE`), and it
    partitions the same rows, so its numerators sum to the blended ones.

    The opportunity denominator is the MANIFEST's per-class item count when the
    result document carries it. A run that errored on half its opportunity
    items, or was cut short by ``--max-items``, otherwise divides by the rows
    that survived and turns its own losses into a better score. The rows-present
    figure is published beside it under
    ``alignment_given_opportunity_rows_present``.
    """
    advertised_at_boundary = _advertised_at_boundary(considered)
    lookup_hits = _lookup_hits_at_boundary(considered)
    opportunity_rows = _opportunity_rows(considered)
    aligned = len(advertised_at_boundary)
    outside_classes = sum(
        1
        for row in advertised_at_boundary
        if traffic_class_of(row) not in ALIGNMENT_OPPORTUNITY_CLASSES
    )
    manifest_denominator = manifest_class_total(
        manifest_class_counts, ALIGNMENT_OPPORTUNITY_CLASSES
    )
    denominator = len(opportunity_rows) if manifest_denominator is None else manifest_denominator
    denominator_source = (
        DENOMINATOR_FROM_ROWS_PRESENT if manifest_denominator is None else DENOMINATOR_FROM_MANIFEST
    )
    miss_breakdown, decline_breakdown = _m1_breakdowns(considered)
    opportunity_rate = _rate_or_none(aligned, denominator) if joined else None
    head_share = wrapper_head_share(considered)
    return {
        "alignment_given_match": _rate_or_none(aligned, len(lookup_hits)) if joined else None,
        "alignment_given_match_numerator": aligned if joined else None,
        "alignment_given_match_denominator": len(lookup_hits),
        "alignment_given_opportunity": opportunity_rate,
        # Section 4's shared numerator: the same count as the match rate's.
        "alignment_given_opportunity_numerator": aligned if joined else None,
        "alignment_given_opportunity_numerator_outside_classes": (
            outside_classes if joined else None
        ),
        "alignment_given_opportunity_denominator": denominator,
        "alignment_given_opportunity_denominator_source": denominator_source,
        # The same numerator over the opportunity rows this document actually
        # holds, so a truncated run is visible as the gap between the two.
        "alignment_given_opportunity_rows_present": (
            _rate_or_none(aligned, len(opportunity_rows)) if joined else None
        ),
        "alignment_given_opportunity_rows_present_denominator": len(opportunity_rows),
        "boundary_miss_breakdown": miss_breakdown if joined else None,
        # The three ways a span is declined AFTER a lookup hit. A misalignment
        # that never reached the boundary-missed branch lands here, and the two
        # breakdowns together account for every hit that was not served.
        "span_decline_breakdown": decline_breakdown if joined else None,
        # The headline alias. The same number as alignment_given_opportunity,
        # kept because it is the name every earlier result document used.
        "boundary_alignment_rate": opportunity_rate,
        # Section 4, line 343: the blended number is the honest headline and
        # this split is the explanation. Null when no audit was joined, for
        # the same reason every rate above is.
        "alignment_by_wrapper_stratum": _m1_by_wrapper_stratum(considered, joined=joined),
        "wrapper_stratum_rule": WRAPPER_STRATUM_RULE,
        # What the PINNED boundary selected on this document, so the rule's
        # own claim can be checked instead of trusted: null when no row
        # declared a wrapper rank, and a false majority flag means the constant
        # no longer describes the manifest it is being applied to.
        "wrapper_stratum_head_share": head_share,
        "wrapper_stratum_head_is_majority": (None if head_share is None else head_share > 0.5),
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
    """M2: the token-weighted headline, and the request-count rate beside it.

    Both sums obey the audit fold's single rule — the last advertise, and the
    materializations that followed it (see :func:`sembench.connector_audit._fold`).
    Mass the worker wrote against a promise the scheduler had already
    superseded is summed separately into
    ``materialized_reuse_superseded_tokens``: counting it in the numerator over
    a denominator that holds only the last promise is how a ratio climbs past
    1.0 without any extra KV being reused.
    """
    advertised = [row for row in considered if row.audit_advertised_tokens is not None]
    materialized_rows = [row for row in advertised if row.audit_materialized]
    advertised_tokens = sum(row.audit_advertised_tokens or 0 for row in advertised)
    materialized_tokens = sum(row.external_confirmed_tokens or 0 for row in materialized_rows)
    superseded_tokens = sum(row.audit_superseded_materialized_tokens or 0 for row in considered)
    token_rate = _rate_or_none(materialized_tokens, advertised_tokens) if joined else None
    return {
        "materialized_reuse_superseded_tokens": superseded_tokens if joined else None,
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


def connector_audit_metrics(
    rows: list[RequestMetrics],
    *,
    manifest_class_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
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

    The two rates share one numerator and differ only in what they are
    conditioned on: ``alignment_given_match`` asks "when the provider found a
    donor, did the engine's boundary land on a span?", which is the property of
    the *tokenizer and the template* that caveat A of the plan is about;
    ``alignment_given_opportunity`` asks "of the traffic that should have been
    reusable, how much was served?", which is the product number and therefore
    the headline — ``boundary_alignment_rate`` is kept as its alias and
    nothing else. Its denominator is the manifest's per-class item count when
    the document carries one (``manifest_class_counts``), because section 4
    counts manifest items, not surviving rows.

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
        # The workload the class-scoped denominators come from, so a reader
        # can check a denominator instead of trusting it. Null when the run
        # did not carry the manifest's counts into its result.
        "manifest_class_counts": dict(manifest_class_counts) if manifest_class_counts else None,
        **_m1_alignment(considered, joined=joined, manifest_class_counts=manifest_class_counts),
        **_m2_materialized_reuse(considered, joined=joined),
        **_m7_inputs(considered, joined=joined),
    }


def aggregate_by_transform(requests: list[RequestMetrics]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[RequestMetrics]] = {}
    for request in requests:
        groups.setdefault(request.transform, []).append(request)
    return {name: aggregate_metrics(group) for name, group in sorted(groups.items())}
