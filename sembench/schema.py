"""Shared schemas for local semantic KV benchmark manifests and results."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "sembench.manifest.v1"
RESULT_VERSION = "sembench.result.v2"


@dataclass(frozen=True)
class SourceRecord:
    """Normalized source row loaded from a benchmark dataset."""

    source_id: str
    dataset: str
    context: str
    input: str
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DonorPrompt:
    """One donor request that should be seeded before the recipient request."""

    donor_id: str
    text: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadItem:
    """One donor/recipient replay item."""

    item_id: str
    dataset: str
    source_id: str
    transform: str
    donor_prompts: list[DonorPrompt]
    recipient_prompt: str
    input: str = ""
    answers: list[str] = field(default_factory=list)
    negative_control: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    benchmark_version: str = MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WorkloadItem":
        donors = [DonorPrompt(**donor) for donor in raw.get("donor_prompts", [])]
        return cls(
            item_id=raw["item_id"],
            dataset=raw["dataset"],
            source_id=raw["source_id"],
            transform=raw["transform"],
            donor_prompts=donors,
            recipient_prompt=raw["recipient_prompt"],
            input=raw.get("input", ""),
            answers=list(raw.get("answers", [])),
            negative_control=bool(raw.get("negative_control", False)),
            metadata=dict(raw.get("metadata", {})),
            benchmark_version=raw.get("benchmark_version", MANIFEST_VERSION),
        )


@dataclass(frozen=True)
class RequestMetrics:
    """Per-recipient metric record."""

    item_id: str
    dataset: str
    transform: str
    negative_control: bool
    donor_count: int
    prompt_tokens: int
    total_blocks: int
    exact_hit_blocks: int
    exact_hit_tokens: int
    semantic_candidate_blocks: int
    semantic_candidate_tokens: int
    semantic_eligible_blocks: int
    semantic_eligible_tokens: int
    backend_confirmed_blocks: int | None = None
    backend_confirmed_tokens: int | None = None
    semblend_found: bool = False
    semblend_similarity: float = 0.0
    semblend_reuse_ratio: float = 0.0
    semblend_latency_ms: float = 0.0
    donor_ids: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    route_endpoint_id: str | None = None
    route_outcome: str | None = None
    route_semantic_score: float | None = None
    route_total_score: float | None = None
    route_reason: str | None = None
    gateway_route_header: str | None = None
    ttft_ms: float | None = None
    latency_ms: float | None = None
    output_text: str | None = None
    quality_pass: bool | None = None
    quality_score: float | None = None
    quality_f1: float | None = None
    quality_rouge_l: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunMetadata:
    """Reproducibility identity for one benchmark run.

    Every result JSON must be traceable to: which manifest (by checksum),
    which engine/backend, which arm of a paired cold/warm comparison, and
    which sembench code produced it.
    """

    run_id: str
    engine: str
    manifest_path: str
    manifest_sha256: str
    arm: str = "single"
    engine_version: str = ""
    backend_id: str = ""
    baseline_id: str = ""
    sembench_version: str = ""
    sembench_git_sha: str = ""
    sembench_git_dirty: bool = False
    semblend_version: str = ""
    timestamp_utc: str = ""
    python_version: str = field(default_factory=platform.python_version)
    run_host: str = field(default_factory=platform.node)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_sha256(path: str | Path) -> str:
    """SHA256 of the manifest file bytes (manifests are canonical JSONL)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return ""


def _git_state(repo_path: Path) -> tuple[str, bool]:
    def run(args: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(repo_path),
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    sha = run(["rev-parse", "--short=12", "HEAD"])
    dirty = bool(run(["status", "--porcelain"]))
    return sha, dirty


def collect_run_metadata(
    *,
    engine: str,
    manifest: str | Path,
    run_id: str | None = None,
    arm: str = "single",
    engine_version: str = "",
    backend_id: str = "",
    baseline_id: str = "",
) -> RunMetadata:
    sha, dirty = _git_state(Path(__file__).resolve().parents[1])
    return RunMetadata(
        run_id=run_id or uuid.uuid4().hex[:12],
        engine=engine,
        manifest_path=str(manifest),
        manifest_sha256=manifest_sha256(manifest),
        arm=arm,
        engine_version=engine_version,
        backend_id=backend_id,
        baseline_id=baseline_id,
        sembench_version=_package_version("sembench"),
        sembench_git_sha=sha,
        sembench_git_dirty=dirty,
        semblend_version=_package_version("semblend"),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def write_jsonl(path: str | Path, items: list[WorkloadItem]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path, max_items: int | None = None) -> list[WorkloadItem]:
    items: list[WorkloadItem] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(WorkloadItem.from_dict(json.loads(line)))
            if max_items is not None and len(items) >= max_items:
                break
    return items
