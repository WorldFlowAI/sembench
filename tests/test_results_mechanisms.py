"""Hit-threshold + reuse-mechanism classification tests."""

from sembench.results import REUSE_HIT_THRESHOLD_TOKENS, paired_summary, reuse_mechanism
from sembench.schema import RequestMetrics


def _row(arm, item_id="i1", cached=0, fuzzy=0, ttft=1000.0, neg=False):
    return RequestMetrics(
        item_id=item_id,
        dataset="d",
        transform="t",
        negative_control=neg,
        donor_count=1,
        prompt_tokens=100,
        total_blocks=10,
        exact_hit_blocks=0,
        exact_hit_tokens=0,
        semantic_candidate_blocks=0,
        semantic_candidate_tokens=0,
        semantic_eligible_blocks=0,
        semantic_eligible_tokens=0,
        backend_confirmed_blocks=cached // 16,
        backend_confirmed_tokens=cached,
        fuzzy_confirmed_tokens=fuzzy,
        ttft_ms=ttft,
        latency_ms=ttft * 2,
        output_text="out",
        arm=arm,
    )


def test_mechanism_classification():
    assert reuse_mechanism(_row("warm", fuzzy=18662, cached=3)) == "scatter"
    assert reuse_mechanism(_row("warm", fuzzy=128, cached=128)) == "head"
    assert reuse_mechanism(_row("warm", fuzzy=0, cached=11264)) == "exact"
    assert reuse_mechanism(_row("warm", fuzzy=0, cached=3)) == "none"


def test_tiny_reuse_is_not_a_hit():
    rows = [_row("cold"), _row("warm", cached=3)]
    summary = paired_summary(rows)
    assert summary["hit_rate"] == 0.0
    assert summary["reuse_mechanisms"] == {"none": 1}


def test_scatter_mass_counts_toward_hit():
    rows = [_row("cold"), _row("warm", fuzzy=REUSE_HIT_THRESHOLD_TOKENS)]
    summary = paired_summary(rows)
    assert summary["hit_rate"] == 1.0
    assert summary["reuse_mechanisms"] == {"scatter": 1}
    assert summary["fuzzy_confirmed_tokens_warm"] == [REUSE_HIT_THRESHOLD_TOKENS]
