"""Round 2, CLI surface: the headline gate reads the median, and merge-results
checks the arm labels it was handed.

Two ways a paired number lies without anyone noticing:

- The speedup gate read `blended_ttft_speedup_mean`. Speedups are ratios, and
  ratios are heavy-tailed: one pair whose cold arm stalled carries the mean
  over a threshold the typical pair never reached, and the gate goes green on
  an outlier. The gate now reads the median.
- `merge-results` takes cold/warm from the command line. Swapping the flags
  inverts every speedup in the merged document and nothing in the output says
  so. Each arm's own `run.arm` label is now checked against the role it was
  passed as, before anything is joined.
"""

import json
from pathlib import Path

import pytest

from sembench.cli import main

MANIFEST_SHA = "b" * 64


def _result(tmp_path: Path, *, paired=None, name="result.json") -> str:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "aggregate": {
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


def _paired(**kw) -> dict:
    base = {"pairs_used": 8, "pairs_contaminated": 0}
    base.update(kw)
    return base


# --- the gate reads the median -------------------------------------------


def test_the_speedup_gate_reads_the_median_not_the_mean(tmp_path: Path, capsys):
    """A mean carried over the bar by one outlier no longer passes."""
    result = _result(
        tmp_path,
        paired=_paired(blended_ttft_speedup_median=1.1, blended_ttft_speedup_mean=5.0),
    )
    with pytest.raises(SystemExit) as excinfo:
        main(["assert-result-gates", "--result", result, "--min-blended-ttft-speedup", "2.0"])
    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert any("blended_ttft_speedup_median" in failure for failure in payload["failures"])


def test_a_median_above_the_bar_passes_even_when_the_mean_is_below_it(tmp_path: Path, capsys):
    """The mirror image: the gate is not reading the mean in either direction."""
    result = _result(
        tmp_path,
        paired=_paired(blended_ttft_speedup_median=3.0, blended_ttft_speedup_mean=1.2),
    )
    main(["assert-result-gates", "--result", result, "--min-blended-ttft-speedup", "2.0"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True


def test_a_paired_block_with_only_a_mean_cannot_satisfy_the_gate(tmp_path: Path, capsys):
    """A pre-round-2 document has no median, and an absent metric is not a pass."""
    result = _result(tmp_path, paired=_paired(blended_ttft_speedup_mean=9.9))
    with pytest.raises(SystemExit):
        main(["assert-result-gates", "--result", result, "--min-blended-ttft-speedup", "2.0"])
    payload = json.loads(capsys.readouterr().out)
    assert "blended_ttft_speedup_median" in payload["missing_metrics"]
    assert any("--allow-missing" in failure for failure in payload["failures"])


def test_the_gate_report_shows_both_estimators(tmp_path: Path, capsys):
    """The mean stays visible as a secondary number; it is simply not gateable."""
    result = _result(
        tmp_path,
        paired=_paired(blended_ttft_speedup_median=3.0, blended_ttft_speedup_mean=7.5),
    )
    main(["assert-result-gates", "--result", result, "--min-blended-ttft-speedup", "2.0"])
    observed = json.loads(capsys.readouterr().out)["observed"]
    assert observed["blended_ttft_speedup_median"] == pytest.approx(3.0)
    assert observed["blended_ttft_speedup_mean"] == pytest.approx(7.5)


def test_the_median_gate_is_reported_by_the_key_it_reads(tmp_path: Path, capsys):
    result = _result(
        tmp_path,
        paired=_paired(blended_ttft_speedup_median=1.0, blended_ttft_speedup_mean=1.0),
    )
    with pytest.raises(SystemExit):
        main(["assert-result-gates", "--result", result, "--min-blended-ttft-speedup", "2.0"])
    failures = json.loads(capsys.readouterr().out)["failures"]
    assert failures == [
        "blended_ttft_speedup_median: 1.0 < 2.0 (median of paired cold/warm ratios)"
    ]


# --- merge-results checks the labels it was handed ------------------------


def _row(item_id: str, *, ttft: float, cached: int) -> dict:
    return {
        "item_id": item_id,
        "dataset": "t",
        "transform": "instruction_variant",
        "negative_control": False,
        "donor_count": 1,
        "prompt_tokens": 64,
        "total_blocks": 4,
        "exact_hit_blocks": 0,
        "exact_hit_tokens": 0,
        "semantic_candidate_blocks": 0,
        "semantic_candidate_tokens": 0,
        "semantic_eligible_blocks": 0,
        "semantic_eligible_tokens": 0,
        "backend_confirmed_blocks": cached // 16,
        "backend_confirmed_tokens": cached,
        "ttft_ms": ttft,
        "latency_ms": ttft * 1.2,
        "output_text": "the answer is 42 minutes",
        "arm": "single",
    }


def _arm_file(path: Path, rows, *, declared_arm: str, sha: str = MANIFEST_SHA) -> str:
    run = {
        "run_id": path.stem,
        "engine": "gateway",
        "manifest_path": "m.jsonl",
        "manifest_sha256": sha,
    }
    if declared_arm is not None:
        run["arm"] = declared_arm
    path.write_text(
        json.dumps(
            {
                "result_version": "sembench.result.v2",
                "run": run,
                "config": {"mode": "live-gateway"},
                "requests": rows,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _arms(tmp_path: Path, *, cold_arm: str, warm_arm: str, cold_sha: str = MANIFEST_SHA):
    cold = _arm_file(
        tmp_path / "cold.json",
        [_row("i1", ttft=200.0, cached=0), _row("i2", ttft=210.0, cached=0)],
        declared_arm=cold_arm,
        sha=cold_sha,
    )
    warm = _arm_file(
        tmp_path / "warm.json",
        [_row("i1", ttft=40.0, cached=96), _row("i2", ttft=42.0, cached=96)],
        declared_arm=warm_arm,
    )
    return cold, warm


def test_merge_results_refuses_swapped_arm_flags(tmp_path: Path, capsys):
    """The failure this closes: --cold handed the warm run. Every reported
    speedup in the merged document would be the reciprocal of the truth."""
    cold, warm = _arms(tmp_path, cold_arm="warm", warm_arm="cold")
    output = tmp_path / "merged.json"
    with pytest.raises(SystemExit) as excinfo:
        main(["merge-results", "--cold", cold, "--warm", warm, "--output", str(output)])
    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["merged"] is False
    assert len(payload["arm_label_conflicts"]) == 2
    assert any("--cold" in conflict for conflict in payload["arm_label_conflicts"])
    assert any("--warm" in conflict for conflict in payload["arm_label_conflicts"])
    assert not output.exists()


def test_merge_results_refuses_one_swapped_arm_flag(tmp_path: Path, capsys):
    cold, warm = _arms(tmp_path, cold_arm="warm", warm_arm="warm")
    output = tmp_path / "merged.json"
    with pytest.raises(SystemExit):
        main(["merge-results", "--cold", cold, "--warm", warm, "--output", str(output)])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["arm_label_conflicts"]) == 1
    assert not output.exists()


def test_merge_results_accepts_correctly_labelled_arms(tmp_path: Path, capsys):
    cold, warm = _arms(tmp_path, cold_arm="cold", warm_arm="warm")
    output = tmp_path / "merged.json"
    main(["merge-results", "--cold", cold, "--warm", warm, "--output", str(output)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["merged"] is True
    assert output.exists()


def test_merge_results_accepts_arms_that_declare_nothing(tmp_path: Path, capsys):
    """A run launched without --arm makes no claim, which is the normal shape
    of results produced before --arm existed."""
    cold, warm = _arms(tmp_path, cold_arm=None, warm_arm=None)
    output = tmp_path / "merged.json"
    main(["merge-results", "--cold", cold, "--warm", warm, "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["merged"] is True


def test_merge_results_accepts_single_arm_labels(tmp_path: Path, capsys):
    cold, warm = _arms(tmp_path, cold_arm="single", warm_arm="single")
    output = tmp_path / "merged.json"
    main(["merge-results", "--cold", cold, "--warm", warm, "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["merged"] is True


def test_merge_results_refuses_an_already_merged_document(tmp_path: Path, capsys):
    """A paired document holds both arms; feeding it back in as one arm would
    merge a run with itself."""
    cold, warm = _arms(tmp_path, cold_arm="paired", warm_arm="warm")
    output = tmp_path / "merged.json"
    with pytest.raises(SystemExit):
        main(["merge-results", "--cold", cold, "--warm", warm, "--output", str(output)])
    payload = json.loads(capsys.readouterr().out)
    assert "paired" in payload["arm_label_conflicts"][0]
    assert not output.exists()


def test_the_label_check_runs_before_the_join(tmp_path: Path, capsys):
    """Swapped labels are refused on their own terms, not reported as a
    pairing failure that --allow-unpaired could talk past."""
    cold, warm = _arms(tmp_path, cold_arm="warm", warm_arm="cold", cold_sha="c" * 64)
    output = tmp_path / "merged.json"
    with pytest.raises(SystemExit):
        main(
            [
                "merge-results",
                "--cold",
                cold,
                "--warm",
                warm,
                "--output",
                str(output),
                "--allow-unpaired",
            ]
        )
    payload = json.loads(capsys.readouterr().out)
    assert "arm_label_conflicts" in payload
    assert "pairing" not in payload
    assert not output.exists()
