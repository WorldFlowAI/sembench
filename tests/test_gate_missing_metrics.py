"""B11: an unmeasured gate must not read as a pass.

`assert-result-gates` used to skip any gate whose metric was None. Every
paired gate reads `result["paired"]`, which is null on a single-arm result, so
a run that never paired anything satisfied the speedup and contamination gates
by producing no evidence at all. A requested gate with no metric now fails,
and `--allow-missing` is the way to say the skip was intended.
"""

import json
from pathlib import Path

import pytest

from sembench.cli import main


def _result(tmp_path: Path, *, aggregate=None, paired=None, name="result.json") -> str:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "aggregate": aggregate
                if aggregate is not None
                else {
                    "quality_pass_rate": 1.0,
                    "semantic_placement_rate_by_request": 1.0,
                    "backend_confirmed_block_rate": 0.5,
                    "negative_control_backend_confirmed_rate": 0.0,
                    "negative_control_semantic_placement_rate": 0.0,
                },
                "paired": paired,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_single_arm_result_cannot_satisfy_a_speedup_gate(tmp_path: Path, capsys):
    """The exact failure B11 exists to close: `paired` null on every vLLM
    result, so `--min-blended-ttft-speedup` had nothing to evaluate."""
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "assert-result-gates",
                "--result",
                _result(tmp_path, paired=None),
                "--min-blended-ttft-speedup",
                "2.0",
            ]
        )
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "paired_summary" in out
    assert "merge-results" in out


def test_single_arm_result_cannot_satisfy_a_contamination_check(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "assert-result-gates",
                "--result",
                _result(tmp_path, paired=None),
                "--require-contamination-check",
            ]
        )


def test_paired_result_satisfies_the_speedup_gate(tmp_path: Path, capsys):
    result = _result(
        tmp_path,
        paired={"blended_ttft_speedup_mean": 4.8, "pairs_used": 2, "pairs_contaminated": 0},
    )
    main(
        [
            "assert-result-gates",
            "--result",
            result,
            "--min-blended-ttft-speedup",
            "2.0",
            "--require-contamination-check",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["observed"]["paired_present"] is True
    assert payload["observed"]["pairs_used"] == 2


def test_missing_backend_confirmed_rate_fails_when_requested(tmp_path: Path, capsys):
    result = _result(
        tmp_path,
        aggregate={"quality_pass_rate": 1.0, "semantic_placement_rate_by_request": 1.0},
    )
    with pytest.raises(SystemExit):
        main(
            [
                "assert-result-gates",
                "--result",
                result,
                "--min-backend-confirmed-block-rate",
                "0.2",
            ]
        )
    assert "backend_confirmed_block_rate" in capsys.readouterr().out


def test_missing_negative_control_rate_fails_when_requested(tmp_path: Path):
    result = _result(
        tmp_path,
        aggregate={"quality_pass_rate": 1.0, "semantic_placement_rate_by_request": 1.0},
    )
    with pytest.raises(SystemExit):
        main(
            [
                "assert-result-gates",
                "--result",
                result,
                "--max-negative-control-confirmed-rate",
                "0.0",
            ]
        )


def test_allow_missing_reports_what_it_skipped(tmp_path: Path, capsys):
    result = _result(
        tmp_path,
        aggregate={"semantic_placement_rate_by_request": 1.0},
        paired=None,
    )
    main(
        [
            "assert-result-gates",
            "--result",
            result,
            "--min-quality-pass-rate",
            "0.99",
            "--min-blended-ttft-speedup",
            "2.0",
            "--allow-missing",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["allow_missing"] is True
    assert "quality_pass_rate" in payload["missing_metrics"]
    assert "blended_ttft_speedup" in payload["missing_metrics"]
    assert payload["requested_gates"] == ["min_blended_ttft_speedup", "min_quality_pass_rate"]


def test_a_present_metric_is_still_gated_normally(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "assert-result-gates",
                "--result",
                _result(tmp_path, aggregate={"quality_pass_rate": 0.10}),
                "--min-quality-pass-rate",
                "0.80",
            ]
        )


def test_negative_control_speedup_deviation_fails_when_unmeasured(tmp_path: Path):
    """A negative control that produced no number is not a negative control
    that behaved; it used to skip."""
    with pytest.raises(SystemExit):
        main(
            [
                "assert-result-gates",
                "--result",
                _result(tmp_path, paired={"pairs_used": 2}),
                "--max-negative-control-speedup-deviation",
                "0.10",
            ]
        )
