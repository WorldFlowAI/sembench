import json
from pathlib import Path

import pytest

from sembench.cli import main


def _build_fixture(path: Path) -> None:
    main(["build", "--profile", "fixture", "--output", str(path)])


def _offline_result(tmp_path: Path) -> Path:
    manifest = tmp_path / "fixture.jsonl"
    result = tmp_path / "offline.json"
    _build_fixture(manifest)
    main(
        [
            "run-offline",
            "--manifest",
            str(manifest),
            "--output",
            str(result),
            "--block-size",
            "16",
        ]
    )
    return result


def test_build_is_byte_deterministic(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _build_fixture(a)
    _build_fixture(b)
    assert a.read_bytes() == b.read_bytes()


def test_gates_pass_on_clean_offline_result(tmp_path: Path):
    result = _offline_result(tmp_path)
    main(
        [
            "assert-result-gates",
            "--result",
            str(result),
            "--max-negative-control-semantic-placement-rate",
            "0.0",
        ]
    )


def _mutate(result: Path, **aggregate_overrides) -> Path:
    payload = json.loads(result.read_text())
    payload["aggregate"].update(aggregate_overrides)
    mutated = result.with_name("mutated-" + result.name)
    mutated.write_text(json.dumps(payload))
    return mutated


def test_gates_fail_when_negative_control_fires(tmp_path: Path):
    result = _offline_result(tmp_path)
    mutated = _mutate(result, negative_control_semantic_placement_rate=0.5)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "assert-result-gates",
                "--result",
                str(mutated),
                "--max-negative-control-semantic-placement-rate",
                "0.0",
            ]
        )
    assert excinfo.value.code == 1


def test_gates_fail_on_low_quality(tmp_path: Path):
    result = _offline_result(tmp_path)
    mutated = _mutate(result, quality_pass_rate=0.1)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "assert-result-gates",
                "--result",
                str(mutated),
                "--min-quality-pass-rate",
                "0.8",
            ]
        )
    assert excinfo.value.code == 1


def test_gates_fail_without_materialization_evidence(tmp_path: Path):
    result = _offline_result(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "assert-result-gates",
                "--result",
                str(result),
                "--min-materialization-events",
                "1",
            ]
        )
    assert excinfo.value.code == 1


def _strip_quality(tmp_path: Path) -> Path:
    result = _offline_result(tmp_path)
    payload = json.loads(result.read_text())
    payload["aggregate"].pop("quality_pass_rate", None)
    stripped = tmp_path / "stripped.json"
    stripped.write_text(json.dumps(payload))
    return stripped


def test_requested_quality_gate_fails_when_metric_absent(tmp_path: Path):
    """Contract (deliberately tightened): an explicitly requested gate whose
    metric is missing FAILS. Skipping it reads as a pass in CI, which is how a
    gate goes green on a run that never measured the thing it gates."""
    stripped = _strip_quality(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "assert-result-gates",
                "--result",
                str(stripped),
                "--min-quality-pass-rate",
                "0.99",
            ]
        )
    assert excinfo.value.code == 1


def test_allow_missing_restores_the_skip_on_an_absent_metric(tmp_path: Path):
    """The old skip stays available, but only when asked for by name."""
    stripped = _strip_quality(tmp_path)
    main(
        [
            "assert-result-gates",
            "--result",
            str(stripped),
            "--min-quality-pass-rate",
            "0.99",
            "--allow-missing",
        ]
    )


def test_unrequested_gate_still_skips_an_absent_metric(tmp_path: Path):
    """A caller who never asked for the quality gate is not failed by its
    absence — only requested gates are enforced."""
    stripped = _strip_quality(tmp_path)
    main(["assert-result-gates", "--result", str(stripped)])
