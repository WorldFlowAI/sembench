"""Traffic-mix ROI model: the miss path must never be free, the hit path
must win above the engagement threshold, and breakeven must move the
right way with overhead and served fraction."""

from __future__ import annotations

from sembench.roi import (
    Bucket,
    CostModel,
    breakeven_hit_rate,
    cold_ttft,
    evaluate,
    hit_ttft,
    miss_ttft,
)


def test_miss_path_costs_more_than_cold_above_threshold():
    c = CostModel()
    assert miss_ttft(4096, c) > cold_ttft(4096, c)
    assert miss_ttft(100, c) == cold_ttft(100, c)  # below min_prompt_tokens: not engaged


def test_hit_path_beats_cold_for_long_prompts():
    c = CostModel()
    assert hit_ttft(8192, c) < cold_ttft(8192, c)


def test_breakeven_rises_with_lookup_overhead():
    low = breakeven_hit_rate(4096, CostModel(lookup_ms=5.0))
    high = breakeven_hit_rate(4096, CostModel(lookup_ms=50.0))
    assert 0.0 < low < high < 1.0


def test_breakeven_falls_with_more_served_fraction():
    less = breakeven_hit_rate(4096, CostModel(served_fraction_on_hit=0.5))
    more = breakeven_hit_rate(4096, CostModel(served_fraction_on_hit=0.9))
    assert more < less


def test_fleet_verdict_flips_with_hit_rate():
    c = CostModel()
    buckets_lose = [Bucket(4096, 1.0, 0.0)]
    buckets_win = [Bucket(4096, 1.0, 0.5)]
    assert evaluate(buckets_lose, c).net_ttft_delta_ms > 0
    assert evaluate(buckets_win, c).net_ttft_delta_ms < 0


def test_short_traffic_is_untouched():
    c = CostModel(min_prompt_tokens=1024)
    report = evaluate([Bucket(256, 1.0, 0.9)], c)
    assert report.net_ttft_delta_ms == 0.0
