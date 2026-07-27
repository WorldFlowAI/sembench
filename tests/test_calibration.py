import asyncio
import json
from pathlib import Path

import pytest

from sembench.calibration import build_artifact, load_calibration, run_calibration
from sembench.cli import main
from sembench.schema import WorkloadItem


class ScriptedTransport:
    """Returns queued outputs for successive generate calls."""

    def __init__(self, outputs: list[str], cached: int = 0) -> None:
        self._outputs = list(outputs)
        self._cached = cached
        self.flushed = 0

    async def flush(self) -> None:
        self.flushed += 1

    async def generate(self, text: str, max_new_tokens: int):
        out = self._outputs.pop(0) if self._outputs else "empty"
        return {
            "output_text": out,
            "cached_tokens": self._cached,
            "ttft_ms": 10.0,
            "error": None,
        }


def _items(n: int) -> list[WorkloadItem]:
    return [
        WorkloadItem(
            item_id=f"i{k}",
            dataset="t",
            source_id=f"s{k}",
            transform="exact_repeat",
            donor_prompts=[],
            recipient_prompt=f"prompt {k}",
            metadata={"length_band": "8k"},
        )
        for k in range(n)
    ]


def test_calibration_measures_self_agreement():
    # Pair 1 identical outputs (rouge 1.0); pair 2 partly different.
    transport = ScriptedTransport(["a b c d", "a b c d", "a b c d", "a b x y"])
    pairs, contaminated, errored = asyncio.run(
        run_calibration(items=_items(2), transport=transport, max_new_tokens=8)
    )
    assert contaminated == 0 and errored == 0
    scores = [p["rouge_l"] for p in pairs]
    assert scores[0] == pytest.approx(1.0)
    assert 0.0 < scores[1] < 1.0
    assert transport.flushed == 4  # two flushes per pair


def test_cached_tokens_mark_contamination():
    transport = ScriptedTransport(["a", "a"], cached=64)
    pairs, contaminated, _ = asyncio.run(
        run_calibration(items=_items(1), transport=transport, max_new_tokens=8)
    )
    assert not pairs
    assert contaminated == 1


def test_artifact_round_trip(tmp_path: Path):
    pairs = [
        {"item_id": f"i{k}", "length_band": "8k", "rouge_l": 0.8 + 0.01 * (k % 5)}
        for k in range(20)
    ]
    artifact = build_artifact(
        pairs=pairs, contaminated=1, errored=0, model="m", engine="sglang", engine_version="0.5"
    )
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(artifact.to_dict()))
    loaded = load_calibration(path)
    assert loaded.n_pairs == 20
    assert loaded.rouge_l_mean == pytest.approx(artifact.rouge_l_mean)
    assert "8k" in loaded.per_length_band


def _result_with_paired(tmp_path: Path, warm_vs_cold: float) -> Path:
    payload = {
        "aggregate": {},
        "paired": {"warm_vs_cold_output_rouge_l_mean": warm_vs_cold},
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    return path


def _calibration_file(tmp_path: Path, mean: float = 0.85) -> Path:
    pairs = [{"item_id": f"i{k}", "length_band": "", "rouge_l": mean} for k in range(10)]
    artifact = build_artifact(
        pairs=pairs, contaminated=0, errored=0, model="m", engine="sglang", engine_version=""
    )
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(artifact.to_dict()))
    return path


def test_relative_gate_passes_within_drop(tmp_path: Path, capsys):
    result = _result_with_paired(tmp_path, 0.80)
    calib = _calibration_file(tmp_path, 0.85)
    main(
        [
            "assert-result-gates",
            "--result",
            str(result),
            "--noise-floor-calibration",
            str(calib),
            "--max-warm-vs-cold-rouge-drop",
            "0.10",
        ]
    )
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_relative_gate_fails_below_floor(tmp_path: Path):
    result = _result_with_paired(tmp_path, 0.55)
    calib = _calibration_file(tmp_path, 0.85)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "assert-result-gates",
                "--result",
                str(result),
                "--noise-floor-calibration",
                str(calib),
                "--max-warm-vs-cold-rouge-drop",
                "0.10",
            ]
        )
    assert excinfo.value.code == 1


def test_implausibly_high_similarity_is_flagged(tmp_path: Path):
    result = _result_with_paired(tmp_path, 1.0)
    calib = _calibration_file(tmp_path, 0.85)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "assert-result-gates",
                "--result",
                str(result),
                "--noise-floor-calibration",
                str(calib),
                "--max-warm-vs-cold-rouge-drop",
                "0.10",
            ]
        )
    assert excinfo.value.code == 1


def test_relative_gate_requires_calibration(tmp_path: Path):
    result = _result_with_paired(tmp_path, 0.8)
    with pytest.raises(SystemExit):
        main(
            [
                "assert-result-gates",
                "--result",
                str(result),
                "--max-warm-vs-cold-rouge-drop",
                "0.10",
            ]
        )
