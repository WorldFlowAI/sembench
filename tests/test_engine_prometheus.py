"""Engine-side Prometheus counters scraped around each arm (B12).

usage.prompt_tokens_details.cached_tokens is local prefix cache plus
external transfer summed, so it cannot answer "how much came from the
connector". These tests pin the two counters that can, and pin the
distinction the report depends on: an absent counter is not a zero.
"""

from __future__ import annotations

from sembench.prometheus import (
    EXTERNAL_KV_TRANSFER_TOKENS_KEY,
    HITS_KEY,
    QUERIES_KEY,
    MetricsSnapshot,
    MetricsWindow,
    counter_deltas,
    counter_value,
    metrics_url,
    parse_exposition,
    scrape_metrics,
    snapshot_from_text,
)

_EXPOSITION = """
# HELP vllm:external_prefix_cache_queries_total External prefix cache queries.
# TYPE vllm:external_prefix_cache_queries_total counter
vllm:external_prefix_cache_queries_total{model_name="Qwen/Qwen2.5-7B-Instruct",engine="0"} 40960.0
vllm:external_prefix_cache_queries_created{model_name="Qwen/Qwen2.5-7B-Instruct",engine="0"} 1.7e9
# HELP vllm:external_prefix_cache_hits_total External prefix cache hits.
# TYPE vllm:external_prefix_cache_hits_total counter
vllm:external_prefix_cache_hits_total{model_name="Qwen/Qwen2.5-7B-Instruct",engine="0"} 12288.0
# TYPE vllm:prompt_tokens_by_source_total counter
vllm:prompt_tokens_by_source_total{model_name="Qwen/Qwen2.5-7B-Instruct",engine="0",source="new_compute"} 900.0
vllm:prompt_tokens_by_source_total{model_name="Qwen/Qwen2.5-7B-Instruct",engine="0",source="external_kv_transfer"} 15360.0
"""


def _snapshot(url: str, queries, hits, external) -> MetricsSnapshot:
    return MetricsSnapshot(
        url=url,
        captured_at_utc="2026-01-01T00:00:00Z",
        counters={
            QUERIES_KEY: queries,
            HITS_KEY: hits,
            EXTERNAL_KV_TRANSFER_TOKENS_KEY: external,
        },
    )


def test_parse_exposition_reads_counter_total_suffix_and_labels():
    samples = parse_exposition(_EXPOSITION)
    names = {sample.name for sample in samples}
    assert "vllm:external_prefix_cache_queries_total" in names
    hit = next(s for s in samples if s.name == "vllm:external_prefix_cache_hits_total")
    assert hit.value == 12288.0
    assert hit.label_dict()["model_name"] == "Qwen/Qwen2.5-7B-Instruct"


def test_parse_exposition_skips_help_and_type_lines():
    samples = parse_exposition(_EXPOSITION)
    assert all(not sample.name.startswith("#") for sample in samples)
    assert all("HELP" not in sample.name for sample in samples)


def test_counter_value_matches_the_total_suffix_prometheus_adds():
    samples = parse_exposition(_EXPOSITION)
    assert counter_value(samples, "vllm:external_prefix_cache_hits") == 12288.0


def test_counter_value_ignores_the_created_gauge():
    samples = parse_exposition(_EXPOSITION)
    # _created carries a unix timestamp; summing it into the counter would
    # produce a nonsense delta of ~1e9 tokens.
    assert counter_value(samples, "vllm:external_prefix_cache_queries") == 40960.0


def test_counter_value_sums_across_engine_label_sets():
    text = (
        'vllm:external_prefix_cache_hits_total{model_name="m",engine="0"} 10.0\n'
        'vllm:external_prefix_cache_hits_total{model_name="m",engine="1"} 32.0\n'
    )
    assert counter_value(parse_exposition(text), "vllm:external_prefix_cache_hits") == 42.0


def test_counter_value_is_none_when_absent_not_zero():
    """A stock arm never registers the external counters. Reporting 0.0 would
    claim a measured miss instead of an unmeasured one."""
    assert (
        counter_value(
            parse_exposition("vllm:prompt_tokens_total 5.0\n"), "vllm:external_prefix_cache_hits"
        )
        is None
    )


def test_prompt_tokens_by_source_selects_external_kv_transfer_only():
    snapshot = snapshot_from_text("http://w0:8000/metrics", _EXPOSITION)
    assert snapshot.counters[EXTERNAL_KV_TRANSFER_TOKENS_KEY] == 15360.0


def test_snapshot_from_text_captures_the_three_counters():
    snapshot = snapshot_from_text("http://w0:8000/metrics", _EXPOSITION)
    assert snapshot.counters[QUERIES_KEY] == 40960.0
    assert snapshot.counters[HITS_KEY] == 12288.0
    assert snapshot.counters[EXTERNAL_KV_TRANSFER_TOKENS_KEY] == 15360.0
    assert snapshot.ok


def test_counter_deltas_are_per_arm_not_process_lifetime():
    before = [_snapshot("http://w0/metrics", 1000.0, 400.0, 500.0)]
    after = [_snapshot("http://w0/metrics", 41960.0, 12688.0, 15860.0)]
    delta = counter_deltas(before, after)
    assert delta[f"{QUERIES_KEY}_delta"] == 40960.0
    assert delta[f"{HITS_KEY}_delta"] == 12288.0
    assert delta[f"{EXTERNAL_KV_TRANSFER_TOKENS_KEY}_delta"] == 15360.0
    assert delta["external_prefix_cache_hit_ratio"] == 12288.0 / 40960.0
    assert delta["counter_reset_detected"] is False


def test_counter_deltas_sum_across_fleet_workers():
    before = [
        _snapshot("http://w0/metrics", 0.0, 0.0, 0.0),
        _snapshot("http://w1/metrics", 0.0, 0.0, 0.0),
    ]
    after = [
        _snapshot("http://w0/metrics", 100.0, 40.0, 50.0),
        _snapshot("http://w1/metrics", 300.0, 60.0, 70.0),
    ]
    delta = counter_deltas(before, after)
    assert delta[f"{QUERIES_KEY}_delta"] == 400.0
    assert delta[f"{HITS_KEY}_delta"] == 100.0
    assert delta["endpoints_paired"] == 2
    assert delta["per_endpoint"]["http://w1/metrics"][HITS_KEY] == 60.0


def test_counter_deltas_flag_a_counter_reset_instead_of_clamping():
    before = [_snapshot("http://w0/metrics", 5000.0, 900.0, 900.0)]
    after = [_snapshot("http://w0/metrics", 10.0, 2.0, 2.0)]
    delta = counter_deltas(before, after)
    assert delta["counter_reset_detected"] is True
    assert delta[f"{HITS_KEY}_delta"] < 0


def test_counter_deltas_none_when_counter_missing_on_both_ends():
    before = [_snapshot("http://w0/metrics", None, None, None)]
    after = [_snapshot("http://w0/metrics", None, None, None)]
    delta = counter_deltas(before, after)
    assert delta[f"{HITS_KEY}_delta"] is None
    assert delta["external_prefix_cache_hit_ratio"] is None


def test_counter_deltas_count_an_unpaired_endpoint_as_failed():
    before = [_snapshot("http://w0/metrics", 1.0, 1.0, 1.0)]
    after = [_snapshot("http://w1/metrics", 2.0, 2.0, 2.0)]
    delta = counter_deltas(before, after)
    assert delta["endpoints_paired"] == 0
    assert delta["endpoints_failed"] == 2


def test_counter_deltas_skip_a_failed_scrape():
    before = [
        MetricsSnapshot(
            url="http://w0/metrics",
            captured_at_utc="2026-01-01T00:00:00Z",
            counters={
                key: None for key in (QUERIES_KEY, HITS_KEY, EXTERNAL_KV_TRANSFER_TOKENS_KEY)
            },
            error="URLError: refused",
        )
    ]
    after = [_snapshot("http://w0/metrics", 10.0, 5.0, 5.0)]
    delta = counter_deltas(before, after)
    assert delta["endpoints_failed"] == 1
    assert delta[f"{HITS_KEY}_delta"] is None


def test_hit_ratio_is_none_when_no_queries_were_made():
    before = [_snapshot("http://w0/metrics", 10.0, 0.0, 0.0)]
    after = [_snapshot("http://w0/metrics", 10.0, 0.0, 0.0)]
    assert counter_deltas(before, after)["external_prefix_cache_hit_ratio"] is None


def test_metrics_url_normalizes_base_and_metrics_paths():
    assert metrics_url("http://w0:8000") == "http://w0:8000/metrics"
    assert metrics_url("http://w0:8000/") == "http://w0:8000/metrics"
    assert metrics_url("http://w0:8000/metrics") == "http://w0:8000/metrics"


def test_scrape_records_transport_error_instead_of_raising():
    """Losing the cross-check must not destroy an arm's measured requests,
    but it must be visible in the result."""
    snapshot = scrape_metrics("http://127.0.0.1:1", timeout=0.5)
    assert snapshot.error is not None
    assert not snapshot.ok
    assert snapshot.counters[HITS_KEY] is None


def test_window_document_carries_counter_semantics():
    window = MetricsWindow(
        before=(_snapshot("http://w0/metrics", 0.0, 0.0, 0.0),),
        after=(_snapshot("http://w0/metrics", 100.0, 25.0, 30.0),),
    )
    document = window.to_dict()
    assert document["delta"][f"{HITS_KEY}_delta"] == 25.0
    assert "advertised-and-allocated" in document["semantics"][HITS_KEY]
    assert "advertised-and-accepted" in document["semantics"][EXTERNAL_KV_TRANSFER_TOKENS_KEY]
    assert document["endpoints"] == ["http://w0/metrics"]
