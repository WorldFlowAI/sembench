import json
from pathlib import Path

import pytest

from sembench.cli import main


def test_frozen_build_rejects_overrides(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "build",
                "--frozen",
                "fixture-v1",
                "--profile",
                "fixture",
                "--output",
                str(tmp_path / "m.jsonl"),
            ]
        )
    assert "--profile" in str(excinfo.value)


def test_build_requires_profile_or_frozen(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["build", "--output", str(tmp_path / "m.jsonl")])


def test_frozen_build_matches_explicit_fixture_build(tmp_path: Path):
    frozen = tmp_path / "frozen.jsonl"
    explicit = tmp_path / "explicit.jsonl"
    main(["build", "--frozen", "fixture-v1", "--output", str(frozen)])
    main(["build", "--profile", "fixture", "--output", str(explicit)])
    assert frozen.read_bytes() == explicit.read_bytes()


def test_freeze_then_verify_round_trip(tmp_path: Path, capsys):
    main(["freeze", "--spec", "fixture-v1", "--manifests-dir", str(tmp_path)])
    freeze_out = json.loads(capsys.readouterr().out)
    assert (tmp_path / "fixture-v1.jsonl").exists()
    assert (tmp_path / "CHECKSUMS.fixture-v1.json").exists()

    main(["verify-frozen", "--spec", "fixture-v1", "--manifests-dir", str(tmp_path)])
    verify_out = json.loads(capsys.readouterr().out)
    assert verify_out["passed"] is True
    assert verify_out["rebuilt_sha256"] == freeze_out["sha256"]


def test_verify_frozen_fails_on_tamper(tmp_path: Path, capsys):
    main(["freeze", "--spec", "fixture-v1", "--manifests-dir", str(tmp_path)])
    capsys.readouterr()

    checksums = tmp_path / "CHECKSUMS.fixture-v1.json"
    payload = json.loads(checksums.read_text())
    payload["sha256"] = "0" * 64
    checksums.write_text(json.dumps(payload))

    with pytest.raises(SystemExit) as excinfo:
        main(["verify-frozen", "--spec", "fixture-v1", "--manifests-dir", str(tmp_path)])
    assert excinfo.value.code == 1


def test_verify_frozen_without_checksums_is_actionable(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        main(["verify-frozen", "--spec", "fixture-v1", "--manifests-dir", str(tmp_path)])
    assert "freeze" in str(excinfo.value)


def test_unknown_spec_is_actionable(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        main(["freeze", "--spec", "nope", "--manifests-dir", str(tmp_path)])
    assert "fixture-v1" in str(excinfo.value)
