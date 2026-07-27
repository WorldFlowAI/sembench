import json
from pathlib import Path

from sembench.cli import main
from sembench.schema import manifest_sha256


def _build_and_run(tmp_path: Path, extra_args: list[str] | None = None) -> dict:
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
            *(extra_args or []),
        ]
    )
    return json.loads(result.read_text())


def test_result_is_v2_with_run_metadata(tmp_path: Path):
    payload = _build_and_run(tmp_path)
    assert payload["result_version"] == "sembench.result.v2"
    run = payload["run"]
    assert run["engine"] == "offline"
    assert run["arm"] == "single"
    assert len(run["run_id"]) > 0
    assert run["manifest_sha256"] == manifest_sha256(tmp_path / "fixture.jsonl")


def test_run_identity_flags_are_recorded(tmp_path: Path):
    payload = _build_and_run(
        tmp_path,
        [
            "--run-id",
            "run-abc",
            "--arm",
            "cold",
            "--backend-id",
            "sglang-fuzzy-pr31057",
            "--baseline-id",
            "exact-prefix",
            "--engine-version",
            "0.5.99",
        ],
    )
    run = payload["run"]
    assert run["run_id"] == "run-abc"
    assert run["arm"] == "cold"
    assert run["backend_id"] == "sglang-fuzzy-pr31057"
    assert run["baseline_id"] == "exact-prefix"
    assert run["engine_version"] == "0.5.99"


def test_checksum_manifest_matches_library_and_is_stable(tmp_path: Path, capsys):
    manifest = tmp_path / "fixture.jsonl"
    main(["build", "--profile", "fixture", "--output", str(manifest)])
    capsys.readouterr()  # drain the build summary

    main(["checksum-manifest", "--manifest", str(manifest)])
    first = json.loads(capsys.readouterr().out)["sha256"]

    main(["checksum-manifest", "--manifest", str(manifest)])
    second = json.loads(capsys.readouterr().out)["sha256"]

    assert first == second == manifest_sha256(manifest)
