"""Deterministic enterprise-style replay transforms."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from sembench.schema import DonorPrompt, SourceRecord, WorkloadItem

DEFAULT_TRANSFORMS = (
    "exact_repeat",
    "instruction_variant",
    "same_evidence_new_task",
    "leading_evidence_new_task",
    "rag_reorder",
    "multi_donor_composite",
    "fuzzy_edit",
    "negative_control",
)


@dataclass(frozen=True)
class TransformConfig:
    """Workload generation controls."""

    transforms: tuple[str, ...] = DEFAULT_TRANSFORMS
    max_segments: int = 4
    min_segment_chars: int = 400


def build_workload(
    records: list[SourceRecord],
    config: TransformConfig | None = None,
) -> list[WorkloadItem]:
    """Build deterministic donor/recipient pairs from source records."""
    cfg = config or TransformConfig()
    items: list[WorkloadItem] = []
    for idx, record in enumerate(records):
        negative = records[(idx + 1) % len(records)] if len(records) > 1 else record
        for transform in cfg.transforms:
            if transform == "negative_control" and negative.source_id == record.source_id:
                continue
            items.append(_build_item(record, transform, cfg, negative))
    return items


def fixture_records() -> list[SourceRecord]:
    """Small local records used by smoke tests and dependency-free demos."""
    context_a = (
        "Incident report A. The payment service saw elevated latency after a "
        "database failover. The team restored normal routing, verified error "
        "budgets, and added a migration precheck. Customer impact was limited "
        "to delayed checkout sessions in the west region. "
    ) * 18
    context_b = (
        "Contract review B. The supplier agreement requires quarterly security "
        "attestations, sixty day termination notice, and encryption for stored "
        "customer records. The finance team requested a liability cap review. "
    ) * 18
    return [
        SourceRecord(
            source_id="fixture-incident-a",
            dataset="fixture",
            context=context_a,
            input="What was the customer impact and remediation?",
            answers=["Delayed checkout sessions; routing restored and precheck added."],
        ),
        SourceRecord(
            source_id="fixture-contract-b",
            dataset="fixture",
            context=context_b,
            input="Which obligations require legal follow-up?",
            answers=["Termination notice, attestations, encryption, and liability cap."],
        ),
    ]


def _build_item(
    record: SourceRecord,
    transform: str,
    cfg: TransformConfig,
    negative_record: SourceRecord,
) -> WorkloadItem:
    base = _base_prompt(record.context, record.input)
    item_id = stable_id(record.dataset, record.source_id, transform)
    donor_id = stable_id(item_id, "donor")

    if transform == "exact_repeat":
        donors = [DonorPrompt(donor_id=donor_id, text=base, label="base_prompt")]
        recipient = base
        negative = False
        metadata = {"expected_exact": True}
    elif transform == "instruction_variant":
        donors = [DonorPrompt(donor_id=donor_id, text=base, label="base_prompt")]
        recipient = _instruction_variant_prompt(record.context, record.input)
        negative = False
        metadata = {"expected_exact": False, "enterprise_shape": "analyst_qa"}
    elif transform == "same_evidence_new_task":
        donors = [DonorPrompt(donor_id=donor_id, text=base, label="base_prompt")]
        recipient = _same_evidence_task_prompt(record.context, record.input)
        negative = False
        metadata = {"expected_exact": False, "enterprise_shape": "briefing_extraction"}
    elif transform == "leading_evidence_new_task":
        donors = [
            DonorPrompt(
                donor_id=donor_id,
                text=_leading_evidence_donor_prompt(record.context),
                label="leading_evidence_donor",
            )
        ]
        recipient = _leading_evidence_task_prompt(record.context, record.input)
        negative = False
        metadata = {
            "expected_exact": False,
            "enterprise_shape": "leading_evidence_prefix",
            "materialization_shape": "target_prefix_boundary",
        }
    elif transform == "rag_reorder":
        donors = [DonorPrompt(donor_id=donor_id, text=base, label="base_prompt")]
        segments = split_context(record.context, cfg.max_segments, cfg.min_segment_chars)
        recipient = _rag_reorder_prompt(segments, record.input)
        negative = False
        metadata = {"segment_count": len(segments), "enterprise_shape": "retrieval_bundle"}
    elif transform == "multi_donor_composite":
        segments = split_context(record.context, cfg.max_segments, cfg.min_segment_chars)
        donors = [
            DonorPrompt(
                donor_id=stable_id(item_id, "segment", str(i)),
                text=_segment_donor_prompt(segment, i),
                label="segment_donor",
                metadata={"segment_index": i},
            )
            for i, segment in enumerate(segments)
        ]
        recipient = _multi_donor_prompt(segments, record.input)
        negative = False
        metadata = {"segment_count": len(segments), "enterprise_shape": "multi_doc_rag"}
    elif transform == "fuzzy_edit":
        donors = [DonorPrompt(donor_id=donor_id, text=base, label="base_prompt")]
        recipient = _fuzzy_edit_prompt(record.context, record.input)
        negative = False
        metadata = {"expected_exact": False, "enterprise_shape": "metadata_edit"}
    elif transform == "negative_control":
        donor_text = _base_prompt(negative_record.context, negative_record.input)
        donors = [
            DonorPrompt(
                donor_id=donor_id,
                text=donor_text,
                label="unrelated_base_prompt",
                metadata={"negative_source_id": negative_record.source_id},
            )
        ]
        recipient = _instruction_variant_prompt(record.context, record.input)
        negative = True
        metadata = {
            "expected_semantic": False,
            "negative_source_id": negative_record.source_id,
        }
    else:
        raise ValueError(f"unknown transform: {transform}")

    return WorkloadItem(
        item_id=item_id,
        dataset=record.dataset,
        source_id=record.source_id,
        transform=transform,
        donor_prompts=donors,
        recipient_prompt=recipient,
        input=record.input,
        answers=record.answers,
        negative_control=negative,
        metadata={**record.metadata, **metadata},
    )


def _base_prompt(context: str, request: str) -> str:
    return (
        "Workspace evidence review\n\n"
        "Use the source material below to answer the user request. Keep the "
        "answer grounded in the source.\n\n"
        "Source material:\n"
        f"{context.strip()}\n\n"
        "User request:\n"
        f"{request.strip()}\n\n"
        "Answer:"
    )


def _instruction_variant_prompt(context: str, request: str) -> str:
    return (
        "You are an enterprise analyst preparing a concise internal response.\n\n"
        "Evidence packet:\n"
        f"{context.strip()}\n\n"
        "Analyst task:\n"
        f"{request.strip()}\n\n"
        "Return only facts supported by the evidence.\n"
        "Response:"
    )


def _same_evidence_task_prompt(context: str, request: str) -> str:
    return (
        "Prepare an operations brief from the following evidence. Identify the "
        "decision-relevant facts, risks, and follow-up actions.\n\n"
        "Original request:\n"
        f"{request.strip()}\n\n"
        "Evidence:\n"
        f"{context.strip()}\n\n"
        "Brief:"
    )


def _leading_evidence_donor_prompt(context: str) -> str:
    return (
        "Reusable enterprise evidence cache entry\n\n"
        f"{context.strip()}\n\n"
        "Cached evidence summary:"
    )


def _leading_evidence_task_prompt(context: str, request: str) -> str:
    return (
        f"{context.strip()}\n\n"
        "Using only the evidence above, answer the enterprise request.\n"
        f"Request: {request.strip()}\n"
        "Answer:"
    )


def _rag_reorder_prompt(segments: list[str], request: str) -> str:
    rotated = list(reversed(segments))
    parts = ["Retrieved enterprise knowledge bundle"]
    for idx, segment in enumerate(rotated):
        parts.append(
            f"\n[retrieved_record id=R{idx + 1:02d} source=knowledge_base]\n"
            f"{segment.strip()}"
        )
    parts.append(
        "\nUsing the retrieved records, answer the request with citations to "
        "record IDs when possible.\n"
        f"Request: {request.strip()}\n"
        "Answer:"
    )
    return "\n".join(parts)


def _segment_donor_prompt(segment: str, idx: int) -> str:
    return (
        f"Cached enterprise evidence segment {idx + 1}\n\n"
        f"{segment.strip()}\n\n"
        "Segment summary:"
    )


def _multi_donor_prompt(segments: list[str], request: str) -> str:
    parts = ["Cross-document enterprise analysis task"]
    for idx, segment in enumerate(segments):
        parts.append(f"\nDocument {idx + 1}:\n{segment.strip()}")
    parts.append(
        "\nSynthesize across all documents. Use only the provided documents.\n"
        f"Request: {request.strip()}\n"
        "Answer:"
    )
    return "\n".join(parts)


def _fuzzy_edit_prompt(context: str, request: str) -> str:
    edited = re.sub(r"\s+", " ", context.strip())
    edited = edited.replace(". ", ".\n")
    return (
        "Workspace evidence review\n"
        "[metadata source=enterprise_search freshness=latest access=internal]\n\n"
        "Use the following normalized evidence export to answer the request.\n\n"
        f"{edited}\n\n"
        f"Request: {request.strip()}\n"
        "Answer:"
    )


def split_context(
    context: str,
    max_segments: int = 4,
    min_segment_chars: int = 400,
) -> list[str]:
    """Split context into deterministic document-like segments."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", context) if p.strip()]
    if len(paragraphs) >= max_segments:
        return paragraphs[:max_segments]

    text = context.strip()
    if not text:
        return [""]
    target = max(min_segment_chars, len(text) // max(max_segments, 1))
    segments: list[str] = []
    cursor = 0
    while cursor < len(text) and len(segments) < max_segments:
        end = min(len(text), cursor + target)
        if end < len(text):
            boundary = text.rfind(". ", cursor, min(len(text), end + 200))
            if boundary > cursor + min_segment_chars // 2:
                end = boundary + 1
        segment = text[cursor:end].strip()
        if segment:
            segments.append(segment)
        cursor = end
    return segments or [text]


def stable_id(*parts: str) -> str:
    data = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]
