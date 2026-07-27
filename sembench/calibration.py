"""Noise-floor calibration: how similar are two COLD runs of the same prompt?

Semantic-reuse quality is judged by warm-vs-cold output similarity, but even
two cold runs of an identical prompt diverge (batching nondeterminism, kernel
scheduling) — ROUGE-L self-agreement is ~0.845 on Qwen2.5-7B, not 1.0. Any
quality gate that ignores this floor either flags healthy reuse as damage or,
worse, celebrates "above-floor" results that signal contamination. Gates are
therefore always calibration-RELATIVE: this module measures the floor per
(model, engine) and emits an artifact the gate command consumes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sembench.quality import rouge_l
from sembench.schema import WorkloadItem
from sembench.stats import bootstrap_mean

CALIBRATION_VERSION = "sembench.calibration.v1"


@dataclass(frozen=True)
class CalibrationArtifact:
    model: str
    engine: str
    engine_version: str
    n_pairs: int
    rouge_l_mean: float
    rouge_l_p05: float
    rouge_l_p50: float
    rouge_l_ci: dict[str, float]
    pairs_contaminated: int
    pairs_errored: int
    created_at_utc: str
    calibration_version: str = CALIBRATION_VERSION
    per_length_band: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_calibration(path: str | Path) -> CalibrationArtifact:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("calibration_version") != CALIBRATION_VERSION:
        raise SystemExit(
            f"calibration artifact version {raw.get('calibration_version')!r} "
            f"!= expected {CALIBRATION_VERSION!r}"
        )
    return CalibrationArtifact(**{k: v for k, v in raw.items()})


async def run_calibration(
    *,
    items: list[WorkloadItem],
    transport,
    max_new_tokens: int,
    cooldown_ms: int = 0,
    warmup_requests: int = 0,
) -> tuple[list[dict[str, Any]], int, int]:
    """Run each item's recipient prompt twice cold; return per-pair records.

    Both runs flush first and seed no donors. A run reporting cached tokens
    (>0 blocks... any nonzero beyond one block) marks the pair contaminated.
    """
    import asyncio

    for _ in range(warmup_requests):
        await transport.generate("warmup request please ignore", 8)
    pairs: list[dict[str, Any]] = []
    contaminated = 0
    errored = 0
    for item in items:
        await transport.flush()
        first = await transport.generate(item.recipient_prompt, max_new_tokens)
        if cooldown_ms > 0:
            await asyncio.sleep(cooldown_ms / 1000)
        await transport.flush()
        second = await transport.generate(item.recipient_prompt, max_new_tokens)

        if first.get("error") or second.get("error"):
            errored += 1
            continue
        if (first.get("cached_tokens") or 0) > 16 or (second.get("cached_tokens") or 0) > 16:
            contaminated += 1
            continue
        pairs.append(
            {
                "item_id": item.item_id,
                "length_band": str(item.metadata.get("length_band", "")),
                "rouge_l": rouge_l(first.get("output_text") or "", second.get("output_text") or ""),
            }
        )
    return pairs, contaminated, errored


def build_artifact(
    *,
    pairs: list[dict[str, Any]],
    contaminated: int,
    errored: int,
    model: str,
    engine: str,
    engine_version: str,
) -> CalibrationArtifact:
    scores = sorted(pair["rouge_l"] for pair in pairs)
    if not scores:
        raise SystemExit("no usable calibration pairs (all contaminated or errored)")
    ci = bootstrap_mean(scores)

    def pct(p: float) -> float:
        rank = (p / 100) * (len(scores) - 1)
        low = int(rank)
        high = min(low + 1, len(scores) - 1)
        frac = rank - low
        return scores[low] * (1 - frac) + scores[high] * frac

    bands: dict[str, dict[str, float]] = {}
    by_band: dict[str, list[float]] = {}
    for pair in pairs:
        if pair["length_band"]:
            by_band.setdefault(pair["length_band"], []).append(pair["rouge_l"])
    for band, values in sorted(by_band.items()):
        bands[band] = {"n": float(len(values)), "mean": sum(values) / len(values)}

    return CalibrationArtifact(
        model=model,
        engine=engine,
        engine_version=engine_version,
        n_pairs=len(scores),
        rouge_l_mean=ci.point,
        rouge_l_p05=pct(5),
        rouge_l_p50=pct(50),
        rouge_l_ci=ci.to_dict(),
        pairs_contaminated=contaminated,
        pairs_errored=errored,
        created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        per_length_band=bands,
    )
