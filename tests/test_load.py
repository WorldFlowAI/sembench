"""run-load aggregation: throughput and percentiles from per-request records."""

from __future__ import annotations

from sembench.cli import build_parser
from sembench.load import percentile, summarize_load


def test_percentile_is_nearest_rank_over_sorted_values():
    values = [50.0, 10.0, 30.0, 20.0, 40.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 50.0
    assert percentile([], 50) is None


def test_summarize_load_counts_every_request_and_only_output_tokens():
    donors = [{"ttft_ms": 100.0, "latency_ms": 120.0, "output_tokens": 1} for _ in range(3)]
    recipients = [
        {"ttft_ms": 200.0, "latency_ms": 900.0, "output_tokens": 32},
        {"ttft_ms": None, "latency_ms": 50.0, "output_tokens": None, "error": "HTTP 400"},
        {"ttft_ms": 600.0, "latency_ms": 1300.0, "output_tokens": 32},
        {"ttft_ms": 400.0, "latency_ms": 1100.0, "output_tokens": 32},
    ]
    doc = summarize_load(donors=donors, recipients=recipients, wall_seconds=10.0, items=4, concurrency=2)
    assert doc["requests"] == 7
    assert doc["errors"] == 1
    assert doc["requests_per_second"] == 0.7
    assert doc["output_tokens_per_second"] == 9.9  # (3 + 96) / 10
    assert doc["recipient_ttft_ms"]["p50"] == 400.0  # the errored request has no TTFT
    assert doc["donor_ttft_ms"]["mean"] == 100.0


def test_run_load_cli_parses_concurrency_and_delay():
    args = build_parser().parse_args(
        [
            "run-load",
            "--manifest", "m.jsonl", "--output", "o.json",
            "--gateway-url", "http://127.0.0.1:1", "--model", "m",
            "--concurrency", "8", "--post-donor-delay-ms", "500",
        ]
    )
    assert args.command == "run-load"
    assert args.concurrency == 8 and args.post_donor_delay_ms == 500
