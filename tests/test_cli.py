import json
from pathlib import Path

import pytest

from sembench.cli import main


def test_cli_build_and_offline_without_semblend(tmp_path: Path):
    manifest = tmp_path / "fixture.jsonl"
    result = tmp_path / "offline.json"

    main(["build", "--profile", "fixture", "--output", str(manifest)])
    main(
        [
            "run-offline",
            "--manifest",
            str(manifest),
            "--output",
            str(result),
            "--block-size",
            "16",
            "--max-items",
            "2",
        ]
    )

    payload = json.loads(result.read_text())
    assert payload["aggregate"]["request_count"] == 2
    assert payload["aggregate"]["total_blocks"] > 0
    assert "exact_block_hit_rate" in payload["aggregate"]


def test_cli_assert_result_gates_with_engine_summary(tmp_path: Path, capsys):
    result = tmp_path / "result.json"
    summary = tmp_path / "engine.json"
    result.write_text(
        json.dumps(
            {
                "aggregate": {
                    "quality_pass_rate": 1.0,
                    "semantic_placement_rate_by_request": 1.0,
                    "negative_control_backend_confirmed_rate": 0.0,
                    "negative_control_semantic_placement_rate": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "materialization_events": 2,
                "materialized_tokens": 512,
                "materialized_units": 0,
                "materialized_semantic_kv_reuse": True,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            "assert-result-gates",
            "--result",
            str(result),
            "--engine-summary",
            str(summary),
            "--min-quality-pass-rate",
            "0.9",
            "--min-semantic-placement-rate",
            "0.9",
            "--min-materialization-events",
            "1",
            "--min-materialized-tokens",
            "1",
            "--require-materialized-reuse",
            "--require-no-engine-errors",
        ]
    )

    assert '"passed": true' in capsys.readouterr().out


def test_cli_assert_result_gates_catches_negative_semantic_placement(tmp_path: Path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "aggregate": {
                    "quality_pass_rate": 1.0,
                    "semantic_placement_rate_by_request": 1.0,
                    "negative_control_semantic_placement_rate": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        main(
            [
                "assert-result-gates",
                "--result",
                str(result),
                "--max-negative-control-semantic-placement-rate",
                "0.0",
            ]
        )


def test_live_gateway_accepts_post_donor_delay():
    """The gateway runner is what long-context campaigns drive; engines that
    index donors asynchronously need the settle delay, and the flag once
    existed only on the non-gateway runner."""
    from sembench.cli import build_parser

    args = build_parser().parse_args(
        [
            "run-live-gateway",
            "--manifest", "m.jsonl", "--output", "o.json",
            "--gateway-url", "http://127.0.0.1:1", "--model", "m",
            "--post-donor-delay-ms", "2000",
        ]
    )
    assert args.post_donor_delay_ms == 2000
