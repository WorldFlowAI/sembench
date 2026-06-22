"""Shared schemas for local semantic KV benchmark manifests and results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "sembench.manifest.v1"
RESULT_VERSION = "sembench.result.v1"


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
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
