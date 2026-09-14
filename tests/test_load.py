"""run-load aggregation: throughput and percentiles from per-request records."""

from __future__ import annotations

import ast
from pathlib import Path

from sembench import gateway_live
from sembench.cli import build_parser
from sembench.load import LoadConfig, gateway_config, percentile, run_load, summarize_load
from sembench.schema import WorkloadItem, write_jsonl


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


def test_summarize_load_reports_the_settle_it_excluded():
    """A rate that silently includes the harness's own settle is not a rate
    either arm can be compared on, so the exclusion has to be visible."""
    recipients = [{"ttft_ms": 100.0, "latency_ms": 200.0, "output_tokens": 8} for _ in range(4)]
    doc = summarize_load(
        donors=[],
        recipients=recipients,
        wall_seconds=10.0,
        items=4,
        concurrency=2,
        post_donor_delay_ms=1000,
    )
    assert doc["settle_seconds_est"] == 2.0  # 4 items x 1s / 2 lanes
    assert doc["settle_excluded_basis"] == "estimate"
    assert doc["requests_per_second"] == 0.4
    assert doc["requests_per_second_excluding_settle"] == 0.5  # 4 / (10 - 2)


def test_summarize_load_without_a_settle_reports_one_rate_twice():
    recipients = [{"ttft_ms": 100.0, "latency_ms": 200.0, "output_tokens": 8}]
    doc = summarize_load(
        donors=[], recipients=recipients, wall_seconds=10.0, items=1, concurrency=1
    )
    assert doc["settle_seconds_est"] == 0.0
    assert doc["requests_per_second_excluding_settle"] == doc["requests_per_second"]


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


def test_run_load_delegates_to_the_gateway_runner():
    config = LoadConfig(
        manifest="m.jsonl",
        output="o.json",
        gateway_url="http://127.0.0.1:1/",
        model="qwen",
        concurrency=16,
        post_donor_delay_ms=250,
        min_donor_gap_requests=20,
    )
    gateway = gateway_config(config)
    assert gateway.concurrency == 16
    assert gateway.post_donor_delay_ms == 250
    assert gateway.min_donor_gap_requests == 20
    assert gateway.recipient_max_tokens == config.recipient_max_tokens


def test_run_load_document_keeps_its_shape(tmp_path: Path, monkeypatch):
    """`run-load` is now a wrapper, and every key its consumers read has to
    survive that."""
    def fake_chat_completion(*, prompt, **_kwargs):
        return {
            "output_text": "42",
            "usage": {"prompt_tokens": 64, "completion_tokens": 4},
            "headers": {},
            "ttft_ms": 5.0,
            "latency_ms": 20.0,
        }

    monkeypatch.setattr(gateway_live, "_chat_completion", fake_chat_completion)
    manifest = tmp_path / "m.jsonl"
    write_jsonl(
        manifest,
        [
            WorkloadItem(
                item_id=f"i{i}",
                dataset="fixture",
                source_id=f"s{i}",
                transform="instruction_variant",
                donor_prompts=[],
                recipient_prompt=f"recipient-{i}",
                answers=["42"],
            )
            for i in range(3)
        ],
    )

    doc = run_load(
        LoadConfig(
            manifest=str(manifest),
            output=str(tmp_path / "load.json"),
            gateway_url="http://gateway.invalid",
            model="qwen",
            concurrency=3,
            post_donor_delay_ms=0,
            run_id="r1",
        )
    )

    for key in (
        "run_id",
        "mode",
        "config",
        "items",
        "requests",
        "errors",
        "concurrency",
        "wall_seconds",
        "requests_per_second",
        "output_tokens_per_second",
        "donor_ttft_ms",
        "recipient_ttft_ms",
        "recipient_latency_ms",
        "donors",
        "recipients",
    ):
        assert key in doc, key
    assert doc["mode"] == "load"
    assert doc["run_id"] == "r1"
    assert doc["requests"] == 3
    assert [r["item_id"] for r in doc["recipients"]] == ["i0", "i1", "i2"]
    assert set(doc["recipients"][0]) == {
        "item_id",
        "role",
        "ttft_ms",
        "latency_ms",
        "output_tokens",
        "output_text",
        "error",
    }


def test_every_cli_command_is_defined_before_the_main_guard():
    """`python -m sembench.cli run-load` raised NameError because cmd_run_load
    was defined after `if __name__ == "__main__": main()`, which runs first."""
    source = Path("sembench/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    guard_lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.dump(node.test).find("__main__") != -1
    ]
    assert guard_lines, "the __main__ guard moved or vanished"
    defined_late = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.lineno > min(guard_lines)
    ]
    assert defined_late == []
