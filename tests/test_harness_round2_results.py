"""Round-2 harness fixes in the results/schema/stats layer.

Three defects, all of which let a result document overstate a semantic-KV
result without saying anything false-looking:

1. ``cached_tokens`` (``RequestMetrics.backend_confirmed_tokens``) is vLLM's
   local prefix cache **plus** external KV transfer, summed by the engine
   before it reaches the API. With prefix caching on, replaying a document
   hits the local cache, so reading that field as semantic reuse turns an
   ordinary exact-prefix repeat into a connector win. Only
   ``external_confirmed_tokens`` (external-only) and ``fuzzy_confirmed_tokens``
   (sglang fuzzy-admitted mass) are semantic-reuse evidence.

2. The headline speedup was published as a mean over per-pair ratios. Ratios
   are heavy-tailed, so one pair can carry the number; the plan specifies a
   median with a bootstrap CI.

3. ``merge-results`` takes cold/warm from the command line. Swapping the two
   flags inverts every paired number in the merged document and nothing in
   the output says so — but each payload already declares its own arm.
"""

import pytest

from sembench.results import (
    REUSE_HIT_THRESHOLD_TOKENS,
    aggregate_metrics,
    arm_label_conflicts,
    external_tokens_are_per_request,
    is_reuse_hit,
    paired_summary,
    result_arm,
    reuse_mechanism,
    semantic_reuse_tokens,
)
from sembench.schema import (
    EXTERNAL_SOURCE_ARM_PROMETHEUS,
    EXTERNAL_SOURCE_CONNECTOR_AUDIT,
    RequestMetrics,
)
from sembench.stats import bootstrap_median, bootstrap_median_ratio, paired_ratios


def _row(
    arm: str,
    *,
    item_id: str = "i1",
    cached: int | None = None,
    external: int | None = None,
    external_source: str | None = None,
    fuzzy: int = 0,
    ttft: float = 1000.0,
    engine_ttft: float | None = None,
    queue_ms: float | None = None,
    neg: bool = False,
) -> RequestMetrics:
    return RequestMetrics(
        item_id=item_id,
        dataset="d",
        transform="t",
        negative_control=neg,
        donor_count=1,
        prompt_tokens=8192,
        total_blocks=512,
        exact_hit_blocks=0,
        exact_hit_tokens=0,
        semantic_candidate_blocks=0,
        semantic_candidate_tokens=0,
        semantic_eligible_blocks=0,
        semantic_eligible_tokens=0,
        backend_confirmed_blocks=None if cached is None else cached // 16,
        backend_confirmed_tokens=cached,
        fuzzy_confirmed_tokens=fuzzy,
        external_confirmed_tokens=external,
        external_confirmed_tokens_source=external_source,
        ttft_ms=ttft,
        engine_ttft_ms=engine_ttft,
        queue_time_ms=queue_ms,
        latency_ms=ttft * 2,
        output_text="out",
        arm=arm,
    )


# --------------------------------------------------------------------------
# 1. cached_tokens is never semantic reuse
# --------------------------------------------------------------------------


def test_prefix_cache_repeat_is_not_semantic_reuse():
    """The blocker itself: prefix caching on, document replayed, whole prompt
    served from the LOCAL cache, connector contributed nothing."""
    warm = _row("warm", cached=4096, external=0, external_source=EXTERNAL_SOURCE_CONNECTOR_AUDIT)

    assert semantic_reuse_tokens(warm) == 0
    assert is_reuse_hit(warm) is False
    assert reuse_mechanism(warm) == "exact"


def test_external_confirmed_tokens_are_semantic_reuse():
    warm = _row(
        "warm",
        cached=4096,
        external=2048,
        external_source=EXTERNAL_SOURCE_CONNECTOR_AUDIT,
    )

    assert semantic_reuse_tokens(warm) == 2048
    assert is_reuse_hit(warm) is True
    assert reuse_mechanism(warm) == "external"


def test_semantic_reuse_never_reads_cached_tokens():
    """cached_tokens can be arbitrarily large without moving the answer."""
    tiny_external = _row("warm", cached=1_000_000, external=REUSE_HIT_THRESHOLD_TOKENS - 1)

    assert semantic_reuse_tokens(tiny_external) == REUSE_HIT_THRESHOLD_TOKENS - 1
    assert is_reuse_hit(tiny_external) is False


def test_unmeasured_external_split_is_none_not_zero():
    """A pre-B12 row never asked the engine for the split. That is 'unknown',
    and collapsing it to 0 would report a clean miss the run cannot support."""
    legacy = _row("warm", cached=4096, external=None)

    assert semantic_reuse_tokens(legacy) is None
    assert is_reuse_hit(legacy) is None


def test_hit_rate_external_confirmed_excludes_a_prefix_cache_repeat():
    rows = [
        _row("cold", item_id="i1", ttft=1000.0, cached=0, external=0),
        _row("warm", item_id="i1", ttft=200.0, cached=4096, external=0),
    ]
    summary = paired_summary(rows)

    assert summary["hit_rate_external_confirmed"] == 0.0
    assert summary["hit_rate"] == 0.0
    assert summary["reuse_mechanisms"] == {"exact": 1}
    assert summary["pairs_external_confirmed"] == 1
    assert summary["pairs_external_unconfirmed"] == 0


def test_hit_rate_external_confirmed_counts_a_connector_hit():
    rows = [
        _row("cold", item_id="i1", ttft=1000.0, cached=0, external=0),
        _row("warm", item_id="i1", ttft=200.0, cached=4096, external=4096),
    ]
    summary = paired_summary(rows)

    assert summary["hit_rate_external_confirmed"] == 1.0
    assert summary["reuse_mechanisms"] == {"external": 1}


def test_hit_rate_external_confirmed_is_none_when_the_split_was_never_measured():
    """An arm with no external split must fail a None-refusing gate rather
    than pass one on cached_tokens mass."""
    rows = [
        _row("cold", item_id="i1", ttft=1000.0, cached=0),
        _row("warm", item_id="i1", ttft=200.0, cached=4096),
    ]
    summary = paired_summary(rows)

    assert summary["hit_rate_external_confirmed"] is None
    assert summary["pairs_external_confirmed"] == 0
    assert summary["pairs_external_unconfirmed"] == 1
    # The legacy hit is still visible, and it is labelled unverified.
    assert summary["hits_unverified_external"] == 1


def test_hit_definition_names_the_exclusion():
    rows = [_row("cold", ttft=1000.0), _row("warm", ttft=200.0, external=4096)]

    assert "cached_tokens" in paired_summary(rows)["hit_definition"]


def test_exact_is_not_a_semantic_mechanism():
    from sembench.results import SEMANTIC_MECHANISMS

    assert "exact" not in SEMANTIC_MECHANISMS
    assert set(SEMANTIC_MECHANISMS) == {"scatter", "head", "external"}


def test_fuzzy_mass_still_counts_on_the_sglang_path():
    """sglang has no external connector; its fuzzy-admitted mass cannot
    contain a local prefix hit, so it stays admissible evidence."""
    warm = _row("warm", cached=0, fuzzy=REUSE_HIT_THRESHOLD_TOKENS)

    assert semantic_reuse_tokens(warm) == REUSE_HIT_THRESHOLD_TOKENS
    assert is_reuse_hit(warm) is True
    assert reuse_mechanism(warm) == "scatter"


def test_arm_level_external_source_is_flagged_as_not_per_request():
    """The per-arm delta of vllm:external_prefix_cache_hits is a process-wide
    total. Stamped on rows it supports arm-level statements only."""
    rows = [_row("warm", external=512, external_source=EXTERNAL_SOURCE_ARM_PROMETHEUS)]

    assert external_tokens_are_per_request(rows) is False


def test_connector_audit_external_source_is_per_request():
    rows = [_row("warm", external=512, external_source=EXTERNAL_SOURCE_CONNECTOR_AUDIT)]

    assert external_tokens_are_per_request(rows) is True


def test_unlabelled_external_source_is_unknown_provenance():
    assert external_tokens_are_per_request([_row("warm", external=512)]) is None


def test_paired_summary_carries_the_external_provenance():
    rows = [
        _row("cold", item_id="i1", ttft=1000.0, cached=0, external=0),
        _row(
            "warm",
            item_id="i1",
            ttft=200.0,
            cached=4096,
            external=4096,
            external_source=EXTERNAL_SOURCE_ARM_PROMETHEUS,
        ),
    ]
    summary = paired_summary(rows)

    assert summary["external_confirmed_token_sources"] == [EXTERNAL_SOURCE_ARM_PROMETHEUS]
    assert summary["external_confirmed_is_per_request"] is False
    assert summary["external_confirmed_tokens_warm"] == [4096]


def test_aggregate_reports_external_tokens_apart_from_cached():
    rows = [_row("single", cached=4096, external=1024, external_source="connector_audit")]
    aggregate = aggregate_metrics(rows)

    assert aggregate["backend_confirmed_tokens"] == 4096
    assert aggregate["external_confirmed_tokens"] == 1024
    assert aggregate["external_confirmed_token_weighted_reuse_rate"] == pytest.approx(1024 / 8192)
    assert aggregate["external_confirmed_token_sources"] == ["connector_audit"]


def test_aggregate_external_tokens_are_none_when_unmeasured():
    aggregate = aggregate_metrics([_row("single", cached=4096)])

    assert aggregate["external_confirmed_tokens"] is None
    assert aggregate["external_confirmed_token_weighted_reuse_rate"] is None


# --------------------------------------------------------------------------
# 2. the headline speedup is a median with a CI
# --------------------------------------------------------------------------


def test_bootstrap_median_point_and_ci():
    ci = bootstrap_median([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=200)

    assert ci.point == 3.0
    assert ci.lo <= ci.point <= ci.hi


def test_bootstrap_median_is_empty_safe_and_single_safe():
    assert bootstrap_median([]) is None
    single = bootstrap_median([2.5])
    assert (single.point, single.lo, single.hi) == (2.5, 2.5, 2.5)


def test_bootstrap_median_resists_the_tail_that_carries_a_mean():
    from sembench.stats import bootstrap_mean

    ratios = [1.0, 1.0, 1.0, 1.0, 20.0]

    assert bootstrap_mean(ratios).point == pytest.approx(4.8)
    assert bootstrap_median(ratios).point == 1.0


def test_bootstrap_median_ratio_pairs_are_aligned():
    ci = bootstrap_median_ratio([200.0, 210.0, 190.0], [40.0, 42.0, 38.0])

    assert ci.point == pytest.approx(5.0)


def test_paired_ratios_refuses_a_ragged_pairing():
    with pytest.raises(ValueError, match="one denominator per numerator"):
        paired_ratios([1.0, 2.0], [1.0])


def test_paired_ratios_refuses_a_zero_denominator():
    with pytest.raises(ValueError, match="index 1"):
        paired_ratios([1.0, 2.0], [1.0, 0.0])


def test_blended_speedup_median_is_exposed_with_a_ci():
    """The gate key. One pair with a huge ratio must not carry it."""
    rows = []
    for index, warm_ttft in enumerate([1000.0, 1000.0, 1000.0, 1000.0, 50.0]):
        item = f"i{index}"
        rows.append(_row("cold", item_id=item, ttft=1000.0, external=0))
        rows.append(_row("warm", item_id=item, ttft=warm_ttft, external=0))
    summary = paired_summary(rows)

    assert summary["blended_ttft_speedup_median"] == pytest.approx(1.0)
    assert summary["blended_ttft_speedup_mean"] == pytest.approx(4.8)
    ci = summary["blended_ttft_speedup_median_ci"]
    assert set(ci) == {"point", "lo", "hi"}
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_mean_speedup_is_kept_as_a_secondary_field():
    rows = [
        _row("cold", item_id="i1", ttft=200.0, external=0),
        _row("warm", item_id="i1", ttft=40.0, external=4096),
    ]
    summary = paired_summary(rows)

    assert summary["blended_ttft_speedup_mean"] == pytest.approx(5.0)
    assert summary["blended_ttft_speedup_median"] == pytest.approx(5.0)
    assert summary["hit_only_ttft_speedup_median"] == pytest.approx(5.0)
    assert summary["negative_control_ttft_speedup_median"] is None


def test_negative_control_speedup_has_its_own_median():
    rows = [
        _row("cold", item_id="i1", ttft=200.0, external=0),
        _row("warm", item_id="i1", ttft=40.0, external=4096),
        _row("cold", item_id="n1", ttft=200.0, external=0, neg=True),
        _row("warm", item_id="n1", ttft=100.0, external=0, neg=True),
    ]
    summary = paired_summary(rows)

    assert summary["negative_control_pairs"] == 1
    assert summary["negative_control_ttft_speedup_median"] == pytest.approx(2.0)
    # Controls never enter the headline.
    assert summary["blended_ttft_speedup_median"] == pytest.approx(5.0)


def test_engine_side_ttft_speedup_is_reported_separately():
    """Client TTFT under load is mostly queue wait; the engine-side number
    excludes it, and the two are never pooled."""
    rows = [
        _row("cold", item_id="i1", ttft=1000.0, engine_ttft=400.0, queue_ms=600.0, external=0),
        _row("warm", item_id="i1", ttft=500.0, engine_ttft=100.0, queue_ms=400.0, external=4096),
    ]
    summary = paired_summary(rows)

    assert summary["blended_ttft_speedup_median"] == pytest.approx(2.0)
    assert summary["engine_ttft_pairs"] == 1
    assert summary["engine_ttft_speedup_median"] == pytest.approx(4.0)
    assert summary["queue_time_cold_p50_ms"] == 600.0
    assert summary["queue_time_warm_p50_ms"] == 400.0


def test_engine_side_block_is_null_without_per_request_metrics():
    rows = [
        _row("cold", item_id="i1", ttft=1000.0, external=0),
        _row("warm", item_id="i1", ttft=500.0, external=4096),
    ]
    summary = paired_summary(rows)

    assert summary["engine_ttft_pairs"] == 0
    assert summary["engine_ttft_speedup_median"] is None
    assert summary["queue_time_cold_p50_ms"] is None


# --------------------------------------------------------------------------
# 3. merge-results must not trust the operator's cold/warm labels
# --------------------------------------------------------------------------


def _payload(arm: str | None) -> dict:
    run = {"run_id": "r", "engine": "gateway", "manifest_sha256": "a" * 64}
    if arm is not None:
        run = {**run, "arm": arm}
    return {"run": run, "requests": []}


def test_result_arm_reads_the_declared_arm():
    assert result_arm(_payload("warm")) == "warm"


def test_result_arm_is_empty_when_undeclared():
    assert result_arm(_payload(None)) == ""
    assert result_arm({}) == ""


def test_swapped_arm_flags_are_refused():
    conflicts = arm_label_conflicts(_payload("warm"), _payload("cold"))

    assert len(conflicts) == 2
    assert "--cold result declares run.arm='warm'" in conflicts[0]
    assert "--warm result declares run.arm='cold'" in conflicts[1]


def test_one_swapped_arm_flag_is_refused():
    conflicts = arm_label_conflicts(_payload("warm"), _payload("warm"))

    assert len(conflicts) == 1
    assert "--cold" in conflicts[0]


def test_matching_arm_labels_pass():
    assert arm_label_conflicts(_payload("cold"), _payload("warm")) == []


def test_single_and_absent_arms_make_no_claim():
    assert arm_label_conflicts(_payload("single"), _payload("single")) == []
    assert arm_label_conflicts(_payload(None), _payload(None)) == []


def test_an_already_merged_result_is_not_a_single_arm():
    conflicts = arm_label_conflicts(_payload("paired"), _payload("warm"))

    assert len(conflicts) == 1
    assert "paired" in conflicts[0]
