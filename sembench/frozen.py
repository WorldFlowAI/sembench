"""Frozen dataset specs (builder-first freezing).

A frozen corpus is a recipe plus checksums, not redistributed data: the spec
pins every input that affects the bytes of the built manifest (source profile,
HF dataset revision, dataset list, transforms, segmentation parameters), and
``manifests/CHECKSUMS.<spec>.json`` pins the expected output. ``sembench
freeze`` builds and records; ``sembench verify-frozen`` rebuilds and compares.
See docs/DATASET.md for the design.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sembench.longbench import DEFAULT_LONGBENCH_V1_DATASETS
from sembench.transforms import DEFAULT_TRANSFORMS


@dataclass(frozen=True)
class FrozenSpec:
    """Everything that determines the bytes of a frozen manifest."""

    name: str
    profile: str
    datasets: tuple[str, ...]
    transforms: tuple[str, ...]
    max_items_per_dataset: int | None
    max_segments: int
    min_segment_chars: int
    hf_revision: str | None
    tokenizer: str

    def manifest_filename(self) -> str:
        return f"{self.name}.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Machinery-validation spec: no network, deterministic, tiny. The committed
# CHECKSUMS file for this spec is CI's proof that freeze/verify round-trips.
FROZEN_FIXTURE_V1 = FrozenSpec(
    name="fixture-v1",
    profile="fixture",
    datasets=("fixture",),
    transforms=tuple(DEFAULT_TRANSFORMS),
    max_items_per_dataset=None,
    max_segments=4,
    min_segment_chars=400,
    hf_revision=None,
    tokenizer="Qwen/Qwen2.5-7B-Instruct",
)

# The real corpus (docs/DATASET.md). hf_revision intentionally unset until the
# actual freeze moment: freezing against "main" would be a silent unpin.
FROZEN_V1 = FrozenSpec(
    name="v1",
    profile="longbench-v1",
    datasets=tuple(DEFAULT_LONGBENCH_V1_DATASETS),
    transforms=tuple(DEFAULT_TRANSFORMS),
    max_items_per_dataset=15,
    max_segments=4,
    min_segment_chars=400,
    hf_revision=None,  # set at freeze time to the exact THUDM/LongBench commit
    tokenizer="Qwen/Qwen2.5-7B-Instruct",
)

# Original synthetic corpus (redistributable); small default freeze uses a
# reduced source count until the full corpus freeze is approved.
FROZEN_SYNTHETIC_V1 = FrozenSpec(
    name="synthetic-v1",
    profile="synthetic-v1",
    datasets=("synthetic-enterprise-v1",),
    transforms=tuple(DEFAULT_TRANSFORMS),
    max_items_per_dataset=80,
    max_segments=4,
    min_segment_chars=400,
    hf_revision=None,
    tokenizer="Qwen/Qwen2.5-7B-Instruct",
)

FROZEN_SPECS: dict[str, FrozenSpec] = {
    spec.name: spec for spec in (FROZEN_FIXTURE_V1, FROZEN_V1, FROZEN_SYNTHETIC_V1)
}


def get_frozen_spec(name: str) -> FrozenSpec:
    try:
        return FROZEN_SPECS[name]
    except KeyError:
        known = ", ".join(sorted(FROZEN_SPECS))
        raise SystemExit(f"unknown frozen spec {name!r} (known: {known})")


def checksums_path(manifests_dir: str | Path, spec: FrozenSpec) -> Path:
    return Path(manifests_dir) / f"CHECKSUMS.{spec.name}.json"


def read_checksums(manifests_dir: str | Path, spec: FrozenSpec) -> dict[str, Any]:
    path = checksums_path(manifests_dir, spec)
    if not path.exists():
        raise SystemExit(
            f"no committed checksums at {path} — run 'sembench freeze --spec {spec.name}' first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_checksums(
    manifests_dir: str | Path,
    spec: FrozenSpec,
    *,
    manifest_sha256: str,
    workload_items: int,
) -> Path:
    path = checksums_path(manifests_dir, spec)
    payload = {
        "spec": spec.to_dict(),
        "manifest": spec.manifest_filename(),
        "sha256": manifest_sha256,
        "workload_items": workload_items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
