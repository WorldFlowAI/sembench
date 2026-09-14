"""The paired blocks: every metric that needs both arms.

M3's TTFT speedup, M4's miss tax, M6's paired quality, M7's contamination rate
and the warm-vs-cold output comparisons all join a cold twin to a warm twin by
``item_id`` at the same stream position. The per-arm blocks they sit beside are
in :mod:`sembench.per_arm_metrics`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sembench.arm_matrix import M4_CAPTURE_PAIR
from sembench.metric_math import _pctl, _rate
from sembench.pairs import _clean_pairs, _Pair, _pair_traffic_class
from sembench.per_arm_metrics import connector_audit_metrics
from sembench.propagation import _propagation_summary, cold_reference_for
from sembench.reuse_signals import (
    REUSE_HIT_THRESHOLD_TOKENS,
    SEMANTIC_MECHANISMS,
    audit_measured_row,
    audit_was_joined,
    external_token_sources,
    external_tokens_are_per_request,
    is_reuse_hit,
    reuse_mechanism,
)
from sembench.schema import RequestMetrics
from sembench.traffic_classes import (
    NO_REUSE_CLASS,
)


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


# The connector's own lookup counters, named by section 4's M4 decomposition
# ("scheduler-thread lookup: stats_snapshot().lookup_latency_ms_sum /
# lookups_total"). They are read out of the arm's engine counter window when
# that window carries them, and reported as null when it does not — stock
# vLLM's /metrics exposes neither, so on a stock arm the lookup leg of the
# decomposition is unmeasured rather than zero.
LOOKUP_LATENCY_SUM_KEY = "lookup_latency_ms_sum"
LOOKUPS_TOTAL_KEY = "lookups_total"


def _window_counter(delta: dict[str, Any], key: str) -> float | None:
    """One counter out of an engine window delta, by name or ``<name>_delta``."""
    for candidate in (f"{key}_delta", key):
        value = delta.get(candidate)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def lookup_cost_from_engine(engine: dict[str, Any] | None) -> dict[str, Any]:
    """M4's scheduler-thread lookup leg, from the arm's engine counter window.

    Section 4 decomposes the miss tax into a lookup cost, a capture cost and
    an instrumentation cost. Only the first is a per-arm counter ratio, and it
    is only available when the connector exports ``lookup_latency_ms_sum`` and
    ``lookups_total`` onto the endpoint the run scraped. When it does not, the
    three keys are null and ``miss_tax_lookup_cost_source`` says so, because a
    lookup cost of 0.0 would subtract a cost that was never measured.
    """
    window = ((engine or {}).get("prometheus") or {}).get("delta") or {}
    latency_sum = _window_counter(window, LOOKUP_LATENCY_SUM_KEY) if window else None
    lookups = _window_counter(window, LOOKUPS_TOTAL_KEY) if window else None
    measured = latency_sum is not None and lookups is not None and lookups > 0
    return {
        "miss_tax_lookup_latency_ms_sum": latency_sum,
        "miss_tax_lookups_total": lookups,
        "miss_tax_lookup_ms_per_lookup": (latency_sum / lookups) if measured else None,
        "miss_tax_lookup_cost_source": "engine_window" if measured else None,
    }


MISS_TAX_SOURCE_AUDIT = "connector_audit"
MISS_TAX_SOURCE_UNIDENTIFIED = (
    "unidentified: no connector audit was joined to the warm arm, so 'supplied == 0' could "
    "not be read and the missed population could not be identified"
)
MISS_TAX_POPULATION_DEFAULT = (
    "pairs whose warm row the connector audit measured (audit_joined) and that advertised "
    "nothing (audit_advertised_tokens null or 0), negative controls excluded"
)
MISS_TAX_POPULATION_CAPTURE = (
    f"{NO_REUSE_CLASS} pairs whose warm row the connector audit measured (audit_joined) and "
    f"that advertised nothing — section 4's capture leg is the A3 -> A4 delta on "
    f"{NO_REUSE_CLASS} items, negative controls excluded"
)
MISS_TAX_POPULATION_UNIDENTIFIED = (
    "none: without a joined connector audit no pair can be placed in the missed population"
)


def _miss_tax_summary(
    pairs: list[_Pair],
    *,
    engine: dict[str, Any] | None = None,
    joined: bool = False,
    arm_pair: str | None = None,
) -> dict[str, Any]:
    """M4 — the miss tax, as section 4 defines it.

    Section 4, verbatim::

        median ttft_ms(A4 | supplied == 0) - median ttft_ms(A1)
        over the same item_ids

    "Supplied == 0" is read off the audit, not off the outcome: a pair counts
    when its warm row advertised nothing — ``audit_advertised_tokens`` is null
    (the connector said nothing about it) or 0 (it looked and supplied
    nothing). A row that advertised and then failed to materialize is NOT in
    this population: it was served a promise, and its latency carries the cost
    of keeping or breaking that promise rather than the cost of the miss.

    The sign is a tax, not a speedup: positive means the warm arm was slower
    on requests it could not serve, which is the number the connector has to
    pay for out of its wins. Negative controls are excluded for the same
    reason they are excluded from the blended speedup — they are constructed
    not to reuse, and folding them in would price the connector's miss path
    on traffic that was never a candidate.

    Section 4's other two legs are cross-ARM deltas (A3 -> A4 for capture,
    A4 -> A5 for instrumentation) and cannot be computed inside one paired
    document; ``sembench merge-results --pair`` names those pairs, and the
    resulting document's ``miss_tax_ms`` is the leg it measured. The capture
    leg is the one whose population section 4 states separately — "capture
    cost: A3 -> A4 delta on ``no_reuse`` items" — so on an ``m4_capture``
    document the population is additionally restricted to that class. A
    same-document request that merely failed to advertise is a miss, not a
    capture, and pricing the capture path with it charges the connector for
    traffic that had a donor.

    **The population is the pairs the audit MEASURED, row by row.** The filter
    is read off ``audit_advertised_tokens``, and that field is null both when
    the connector looked and advertised nothing and when the audit holds
    nothing about the request at all — every ``MATCH_MISSING`` row, stamped
    ``audit_joined=False``. Collapsing the two prices the connector's miss path
    with requests no connector event ever described, so a pair whose warm row
    the audit did not speak about (:func:`audit_measured_row`) is dropped into
    ``miss_tax_pairs_not_audited_excluded`` before the "advertised nothing"
    filter runs. That is the same rule the per-arm block applies through
    :func:`sembench.reuse_signals.auditable_rows`: one definition of "the audit
    measured this row", shared by M1, M2 and M4.

    The document-level guard sits above it: with no audit joined to the warm
    arm at all, "supplied == 0" was never measurable and every number here is
    null, with ``miss_tax_source`` stating which case the document is in.
    """
    from sembench.stats import bootstrap_median

    capture_leg = str(arm_pair or "") == M4_CAPTURE_PAIR
    candidates = [pair for pair in pairs if not pair.cold.negative_control]
    controls_excluded = len(pairs) - len(candidates)
    audited = [pair for pair in candidates if audit_measured_row(pair.warm)]
    not_audited = len(candidates) - len(audited)
    non_advertising = [pair for pair in audited if (pair.warm.audit_advertised_tokens or 0) == 0]
    advertising = len(audited) - len(non_advertising)
    missed = (
        [pair for pair in non_advertising if _pair_traffic_class(pair) == NO_REUSE_CLASS]
        if capture_leg
        else non_advertising
    )
    outside_class = len(non_advertising) - len(missed)
    timed = [pair for pair in missed if pair.cold.ttft_ms and pair.warm.ttft_ms]
    warm_ttfts = [pair.warm.ttft_ms for pair in timed]
    cold_ttfts = [pair.cold.ttft_ms for pair in timed]
    warm_median = _pctl(warm_ttfts, 50) if joined else None
    cold_median = _pctl(cold_ttfts, 50) if joined else None
    # A per-pair difference CI, because the two medians are over the same
    # item ids and resampling them independently would widen the interval with
    # variance the pairing already removed.
    difference_ci = (
        bootstrap_median([pair.warm.ttft_ms - pair.cold.ttft_ms for pair in timed])
        if joined
        else None
    )
    population = MISS_TAX_POPULATION_CAPTURE if capture_leg else MISS_TAX_POPULATION_DEFAULT
    return {
        "miss_tax_definition": (
            "median warm TTFT minus median cold TTFT over pairs the connector audit measured "
            "and whose warm row advertised nothing (audit_advertised_tokens null or 0); "
            "positive is a tax the connector charges on requests it could not serve"
        ),
        # The rule this document applied, and whether it could be applied at
        # all. Null numbers below mean "unmeasured", never "no tax".
        "miss_tax_population": population if joined else MISS_TAX_POPULATION_UNIDENTIFIED,
        "miss_tax_source": MISS_TAX_SOURCE_AUDIT if joined else MISS_TAX_SOURCE_UNIDENTIFIED,
        "miss_tax_ms": (
            None if warm_median is None or cold_median is None else warm_median - cold_median
        ),
        "miss_tax_ms_median_of_differences": difference_ci.point if difference_ci else None,
        "miss_tax_ms_median_of_differences_ci": (
            difference_ci.to_dict() if difference_ci else None
        ),
        "miss_tax_warm_ttft_p50_ms": warm_median,
        "miss_tax_cold_ttft_p50_ms": cold_median,
        "miss_tax_pairs": len(timed) if joined else None,
        "miss_tax_pairs_without_ttft": len(missed) - len(timed) if joined else None,
        "miss_tax_pairs_advertising_excluded": advertising if joined else None,
        # Pairs whose warm row the audit never spoke about (audit_joined False
        # or absent): unmeasured, not "advertised nothing".
        "miss_tax_pairs_not_audited_excluded": not_audited if joined else None,
        # Non-advertising pairs the capture leg's class filter removed; always
        # 0 on a document that is not the capture leg.
        "miss_tax_pairs_outside_capture_class_excluded": outside_class if joined else None,
        "miss_tax_pairs_negative_control_excluded": controls_excluded,
        **lookup_cost_from_engine(engine),
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


def paired_summary(
    requests: list[RequestMetrics],
    *,
    manifest_class_counts: dict[str, int] | None = None,
    engine: dict[str, Any] | None = None,
    arm_pair: str | None = None,
    baseline_arm: str | None = None,
    cold_reference: Sequence[RequestMetrics] | None = None,
    cold_reference_arm: str | None = None,
) -> dict[str, Any] | None:
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
    M4 (:func:`_miss_tax_summary`) is here for the same reason — it is a
    difference between the arms over the pairs the warm arm could not serve.

    ``manifest_class_counts`` supplies section 4's class-scoped denominators
    (M1's opportunity classes, M7's probe set); ``engine`` is the warm arm's
    engine block, read only for M4's per-lookup cost.

    ``arm_pair`` is the name of section 4's comparison this document is
    (``sembench merge-results --pair``), and two metrics read it: M4's capture
    leg takes its population from it, and M7 uses it to decide whether the
    baseline arm is a legitimate cold reference. ``baseline_arm`` is the same
    question asked of a document nobody labelled — ``run.baseline_id``, stamped
    from the cold arm's ``--backend-id`` — so an unlabelled A4-vs-A6 merge is
    still recognised as one. ``cold_reference`` is a third arm's rows — an A1
    reference run — which M7 uses as the cold answer whatever the baseline arm
    is; see :func:`cold_reference_for`.
    """
    cold = {r.item_id: r for r in requests if r.arm == "cold"}
    warm = {r.item_id: r for r in requests if r.arm == "warm"}
    if not cold or not warm:
        return None
    from collections import Counter

    from sembench.quality import rouge_l
    from sembench.stats import bootstrap_mean, bootstrap_median, bootstrap_median_ratio

    pair_set = _clean_pairs(cold, warm)
    pairs = list(pair_set.pairs)
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
    # The same guard every audit-derived rate in this document obeys: with no
    # audit joined to the warm arm, "supplied == 0" was never measured and M4
    # is unmeasured rather than zero.
    audit_joined = audit_was_joined(warm_rows)
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
        "pairs_contaminated": pair_set.contaminated,
        "pairs_errored": pair_set.errored,
        "pairs_unpaired": pair_set.unpaired,
        # Twins the two arms replayed at different manifest stream positions:
        # they did not replay the same stream, so their ratio is not a
        # measurement of the cache (section 4's M3 pairs at the same position).
        "pairs_stream_position_mismatched": pair_set.stream_position_mismatched,
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
        **connector_audit_metrics(warm_rows, manifest_class_counts=manifest_class_counts),
        # M7 needs both arms and the warm arm's other rows: see
        # _propagation_summary.
        **_propagation_summary(
            pairs,
            warm,
            excluded=pair_set.excluded,
            manifest_class_counts=manifest_class_counts,
            reference=cold_reference_for(
                arm_pair=arm_pair,
                reference_rows=cold_reference,
                reference_arm=cold_reference_arm,
                baseline_arm=baseline_arm,
            ),
        ),
        **_miss_tax_summary(
            pairs,
            engine=engine,
            joined=audit_joined,
            arm_pair=arm_pair,
        ),
        **_kl_summary(pairs),
        **_engine_ttft_summary(pairs),
        # Back-compat alias for the pre-P3 field name.
        "ttft_speedup_mean": speedup_ci.point if speedup_ci else None,
    }
