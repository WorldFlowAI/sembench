"""B11: `sembench merge-results` joins two single-arm results into one pair set.

Arms normally run as separate server processes (stock baseline vs connector),
so the cold/warm pairing has to be reconstructed after the fact. It is
reconstructed on item_id and nothing else, and only when both arms provably
replayed the same manifest bytes — the join must never be a heuristic, and a
partial join must never be presented as a paired run.
"""

import json
from pathlib import Path

import pytest

from sembench.cli import main
from sembench.pairing import join_arms, pair_fingerprint, requests_from_result
from sembench.schema import RequestMetrics

MANIFEST_SHA = "a" * 64


def _row(item_id: str, *, ttft: float, cached: int, arm: str = "single", **kw) -> RequestMetrics:
    fields = {
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
        "arm": arm,
    }
    fields.update(kw)
    return RequestMetrics(**fields)


def _write_result(path: Path, rows, *, sha: str = MANIFEST_SHA, run_id: str = "r") -> Path:
    path.write_text(
        json.dumps(
            {
                "result_version": "sembench.result.v2",
                "run": {
                    "run_id": run_id,
                    "engine": "gateway",
                    "manifest_path": "m.jsonl",
                    "manifest_sha256": sha,
                    "arm": "single",
                },
                "config": {"mode": "live-gateway"},
                "requests": [row.to_dict() for row in rows],
            }
        ),
        encoding="utf-8",
    )
    return path


def _arms(tmp_path: Path, *, cold_sha: str = MANIFEST_SHA, warm_sha: str = MANIFEST_SHA):
    cold = _write_result(
        tmp_path / "cold.json",
        [_row("i1", ttft=200.0, cached=0), _row("i2", ttft=210.0, cached=0)],
        sha=cold_sha,
        run_id="cold-run",
    )
    warm = _write_result(
        tmp_path / "warm.json",
        [_row("i1", ttft=40.0, cached=96), _row("i2", ttft=42.0, cached=96)],
        sha=warm_sha,
        run_id="warm-run",
    )
    return cold, warm


def test_merge_produces_a_paired_summary_from_two_single_arm_results(tmp_path: Path):
    cold, warm = _arms(tmp_path)
    output = tmp_path / "merged.json"

    main(["merge-results", "--cold", str(cold), "--warm", str(warm), "--output", str(output)])

    merged = json.loads(output.read_text())
    assert merged["paired"] is not None
    assert merged["paired"]["pairs_used"] == 2
    assert merged["paired"]["blended_ttft_speedup_mean"] == pytest.approx(
        ((200.0 / 40.0) + (210.0 / 42.0)) / 2
    )
    assert merged["paired"]["hit_rate"] == 1.0
    assert [(row["item_id"], row["arm"]) for row in merged["requests"]] == [
        ("i1", "cold"),
        ("i1", "warm"),
        ("i2", "cold"),
        ("i2", "warm"),
    ]


def test_merged_result_records_the_pairing_and_both_arms(tmp_path: Path):
    cold, warm = _arms(tmp_path)
    output = tmp_path / "merged.json"

    main(["merge-results", "--cold", str(cold), "--warm", str(warm), "--output", str(output)])

    merged = json.loads(output.read_text())
    pairing = merged["config"]["pairing"]
    assert pairing["ok"] is True
    assert pairing["pairs"] == 2
    assert pairing["manifest_match"] is True
    assert pairing["problems"] == []
    assert merged["config"]["arms"]["cold"]["run"]["run_id"] == "cold-run"
    assert merged["config"]["arms"]["warm"]["run"]["run_id"] == "warm-run"
    assert merged["run"]["arm"] == "paired"
    assert merged["run"]["run_id"] == "cold-run+warm-run"


def test_merge_refuses_arms_that_replayed_different_manifests(tmp_path: Path):
    cold, warm = _arms(tmp_path, warm_sha="b" * 64)
    output = tmp_path / "merged.json"

    with pytest.raises(SystemExit) as excinfo:
        main(["merge-results", "--cold", str(cold), "--warm", str(warm), "--output", str(output)])
    assert excinfo.value.code == 1
    assert not output.exists()


def test_merge_refuses_when_an_item_has_no_twin(tmp_path: Path):
    cold = _write_result(
        tmp_path / "cold.json",
        [_row("i1", ttft=200.0, cached=0), _row("i2", ttft=210.0, cached=0)],
    )
    warm = _write_result(tmp_path / "warm.json", [_row("i1", ttft=40.0, cached=96)])
    output = tmp_path / "merged.json"

    with pytest.raises(SystemExit):
        main(["merge-results", "--cold", str(cold), "--warm", str(warm), "--output", str(output)])


def test_allow_unpaired_merges_the_clean_subset_and_records_the_rest(tmp_path: Path, capsys):
    cold = _write_result(
        tmp_path / "cold.json",
        [_row("i1", ttft=200.0, cached=0), _row("i2", ttft=210.0, cached=0)],
    )
    warm = _write_result(tmp_path / "warm.json", [_row("i1", ttft=40.0, cached=96)])
    output = tmp_path / "merged.json"

    main(
        [
            "merge-results",
            "--cold",
            str(cold),
            "--warm",
            str(warm),
            "--output",
            str(output),
            "--allow-unpaired",
        ]
    )

    merged = json.loads(output.read_text())
    assert merged["paired"]["pairs_used"] == 1
    assert merged["config"]["pairing"]["cold_only_item_ids"] == ["i2"]
    assert merged["config"]["pairing"]["ok"] is False
    assert "i2" in capsys.readouterr().out


def test_merge_refuses_duplicate_item_ids_rather_than_keeping_the_last(tmp_path: Path):
    cold = _write_result(
        tmp_path / "cold.json",
        [_row("i1", ttft=200.0, cached=0), _row("i1", ttft=900.0, cached=0)],
    )
    warm = _write_result(tmp_path / "warm.json", [_row("i1", ttft=40.0, cached=96)])
    output = tmp_path / "merged.json"

    with pytest.raises(SystemExit):
        main(["merge-results", "--cold", str(cold), "--warm", str(warm), "--output", str(output)])


def test_twins_that_measured_different_requests_are_not_paired(tmp_path: Path):
    """Same item_id, different prompt length: the arms did not replay the same
    stream, so the pair is a fiction even though the join key matches."""
    cold_rows = [_row("i1", ttft=200.0, cached=0)]
    warm_rows = [_row("i1", ttft=40.0, cached=96, total_blocks=40)]

    merged, report = join_arms(
        cold_rows,
        warm_rows,
        cold_manifest_sha256=MANIFEST_SHA,
        warm_manifest_sha256=MANIFEST_SHA,
    )

    assert merged == []
    assert report.fingerprint_mismatch_item_ids == ["i1"]
    assert report.ok is False


def test_pair_fingerprint_is_manifest_derived_not_measurement_derived():
    cold = _row("i1", ttft=200.0, cached=0)
    warm = _row("i1", ttft=40.0, cached=4096, arm="warm")
    # Latency and engine-confirmed reuse are exactly what the arms are meant to
    # differ on; they must not make the twins look like different requests.
    assert pair_fingerprint(cold) == pair_fingerprint(warm)


def test_requests_from_result_tolerates_unknown_and_missing_keys(tmp_path: Path):
    path = tmp_path / "cold.json"
    _write_result(path, [_row("i1", ttft=200.0, cached=0)])
    payload = json.loads(path.read_text())
    payload["requests"][0]["some_future_field"] = 1
    payload["requests"][0].pop("quality_f1", None)

    rows = requests_from_result(payload)
    assert len(rows) == 1
    assert rows[0].item_id == "i1"
    assert rows[0].quality_f1 is None


def test_merge_round_trips_results_written_by_a_real_runner(tmp_path: Path):
    """Against the shape `write_result` actually produces, not a hand-built one."""
    manifest = tmp_path / "m.jsonl"
    main(["build", "--profile", "fixture", "--output", str(manifest)])
    for arm in ("cold", "warm"):
        main(
            [
                "run-offline",
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / f"{arm}.json"),
                "--arm",
                arm,
            ]
        )
    output = tmp_path / "merged.json"
    main(
        [
            "merge-results",
            "--cold",
            str(tmp_path / "cold.json"),
            "--warm",
            str(tmp_path / "warm.json"),
            "--output",
            str(output),
        ]
    )

    merged = json.loads(output.read_text())
    pairing = merged["config"]["pairing"]
    assert pairing["ok"] is True
    assert pairing["pairs"] == pairing["cold_rows"] == pairing["warm_rows"]
    assert merged["paired"]["pairs_total"] == pairing["pairs"]
    assert sorted({row["arm"] for row in merged["requests"]}) == ["cold", "warm"]
