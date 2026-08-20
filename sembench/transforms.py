"""Deterministic enterprise-style replay transforms."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from sembench.paraphrase import (
    block_overlap_ratio,
    derived_int,
    inject_sentence,
    rewrite_preserving_facts,
)
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

# v2 adds the highest-severity quality-risk workload: a SAME-domain donor that
# is a near-duplicate modulo entities. Reusing its KV without corruption of
# the recipient's own facts is the hardest honest test for semantic caches.
TRANSFORMS_V2 = DEFAULT_TRANSFORMS + ("entity_swap_control",)

# v3 adds the PRODUCTION semantic-hit classes: the same evidence and the same
# semantic question re-asked under a different session wrapper (verbatim or
# reworded). Exact-prefix caching misses these (the prompts diverge at
# position ~0); a semantic cache should capture them near-losslessly — they
# are the workload where high-mass reuse must be safe, and the contrast class
# for the adversarial transforms above.
TRANSFORMS_V3 = TRANSFORMS_V2 + ("repeat_shifted", "question_paraphrase")

# v4 adds the doc-v2 class: sparse real content edits at a parameterized
# per-sentence rate with whitespace PRESERVED (fuzzy_edit, by contrast,
# reformats whitespace globally and therefore retokenizes every chunk).
TRANSFORMS_V4 = TRANSFORMS_V3 + ("sparse_edit",)

# v5 adds the verified-paraphrase family: the DOCUMENT BODY is rewritten
# (token-different, facts preserved), so token-identity verification cannot
# apply and acceptance requires meaning-level verification. Each accept
# class ships with its matched leak control (same construction, one digit
# changed on the donor side) so quality scoring can catch a wrongly served
# donor by the answer alone. The two restatement classes state a fact in a
# value-equivalent but digit-disjoint form; verifiers are expected to
# reject them today, and the benchmark records the accept rate so future
# meaning tiers show up as measured progress rather than a claim.
TRANSFORMS_V5 = TRANSFORMS_V4 + (
    "fact_paraphrase",
    "fact_divergent_paraphrase",
    "arithmetic_restatement",
    "complement_restatement",
)

_FACT_DONOR = (
    "The oversight committee's final tally recorded exactly {n} approvals during this period."
)
_FACT_RECIPIENT = (
    "Across this period, precisely {n} approvals appeared in the final "
    "tally kept by the oversight committee."
)
_FACT_QUESTION = (
    "According to the passage, exactly how many approvals did the oversight "
    "committee record? Answer with the number only."
)
_ARITH_DONOR = (
    "The oversight committee logged {a} approvals before the midpoint "
    "review and {b} further approvals afterward."
)
_COMPLEMENT_DONOR = "Of the {total} audit review items, exactly {passed} items passed inspection."
_COMPLEMENT_RECIPIENT = (
    "Of the {total} audit review items, exactly {failed} items failed inspection."
)
_COMPLEMENT_QUESTION = (
    "According to the passage, exactly how many audit review items failed "
    "inspection? Answer with the number only."
)

# Deterministic, answer-preserving rewordings for the three synthetic domains'
# question templates. Applied longest-pattern-first; unmatched questions pass
# through wrapped (still a valid paraphrase-class item via the wrapper).
_PARAPHRASE_REWRITES = (
    (
        "what was the root cause and how long did mitigation take?",
        "how long did mitigation take, and what root cause was identified?",
    ),
    ("and what period does it specify?", "and what time period applies?"),
    ("Which clause of agreement", "Identify the clause of agreement"),
    ("What anomalous error code did", "Which anomalous error code did"),
    ("and how many occurrences were counted?", "and what was the occurrence count?"),
    (
        "What was the customer impact and remediation?",
        "Describe the remediation and the customer impact.",
    ),
    (
        "Which obligations require legal follow-up?",
        "List the obligations that need legal follow-up.",
    ),
)


# Residual verbatim word-block overlap allowed between a source and its
# rewrite. Above this the identity-verified paths could serve the span and
# the paraphrase-class premise is void, so building the item fails loudly.
_MAX_RESIDUAL_OVERLAP = 0.35

_PROBE_SALTS = 64


def _fact_probe_numbers(source_id: str, context: str) -> tuple[int, int]:
    """Probe number plus its divergent twin. The twin always starts with a
    different leading digit so first-token answer scoring can never collide,
    and both values are rejection-sampled against the source text: a probe
    that already occurs anywhere in the context blinds the leak control."""
    for salt in range(_PROBE_SALTS):
        n = derived_int(f"fact-n:{source_id}:{salt}", 23, 89)
        wrong = n + 10 + derived_int(f"fact-wrong:{source_id}:{salt}", 0, 9)
        while str(wrong)[0] == str(n)[0]:
            wrong += 10
        if str(n) not in context and str(wrong) not in context:
            return n, wrong
    raise ValueError(f"no collision-free fact probe for {source_id}")


def _restated_parts(source_id: str, context: str, n: int) -> tuple[int, int]:
    """Two distinct addends of n (requires n >= 22), neither present in the
    source text."""
    for salt in range(_PROBE_SALTS):
        part = derived_int(f"fact-part:{source_id}:{salt}", 11, n - 11)
        other = n - part
        if part != other and str(part) not in context and str(other) not in context:
            return part, other
    raise ValueError(f"no collision-free restated parts for {source_id}")


def _complement_numbers(source_id: str, context: str) -> tuple[int, int]:
    """(total, failed) with passed = total - failed. Same discipline as the
    fact probes: failed and passed always start with different leading
    digits, and none of the three values occur in the source text."""
    for salt in range(_PROBE_SALTS):
        total = derived_int(f"fact-total:{source_id}:{salt}", 60, 95)
        failed = derived_int(f"fact-failed:{source_id}:{salt}", 21, 39)
        passed = total - failed
        if (
            failed != passed
            and str(failed)[0] != str(passed)[0]
            and str(total) not in context
            and str(failed) not in context
            and str(passed) not in context
        ):
            return total, failed
    raise ValueError(f"no collision-free complement numbers for {source_id}")


def _v5_probe_pair(
    record: SourceRecord,
    donor_id: str,
    donor_fact: str,
    recipient_fact: str,
    question: str,
    donor_label: str,
) -> tuple[list[DonorPrompt], str]:
    """Shared assembly for the v5 classes: donor carries the source text
    plus its fact sentence; the recipient carries the fact-preserving
    rewrite plus the restated fact, under the same wrapper so the shared
    header is the only verbatim anchor. A rewrite that stays too close to
    the source voids the class premise and fails the build loudly."""
    rewritten = rewrite_preserving_facts(record.context, seed_key=record.source_id)
    if block_overlap_ratio(record.context, rewritten) > _MAX_RESIDUAL_OVERLAP:
        raise ValueError(f"{record.source_id}: rewrite too shallow for a paraphrase-class item")
    donors = [
        DonorPrompt(
            donor_id=donor_id,
            text=_base_prompt(inject_sentence(record.context, donor_fact), record.input),
            label=donor_label,
        )
    ]
    recipient = _base_prompt(inject_sentence(rewritten, recipient_fact), question)
    return donors, recipient


def _build_v5_probe_item(
    *,
    record: SourceRecord,
    transform: str,
    donor_id: str,
) -> tuple[list[DonorPrompt], str, str, list[str], dict]:
    # Collision haystack covers everything that lands in either prompt:
    # a probe value occurring in the question would blind the leak control
    # exactly like one in the context.
    context = record.context + "\x1f" + record.input
    if transform == "fact_paraphrase":
        n, _ = _fact_probe_numbers(record.source_id, context)
        donors, recipient = _v5_probe_pair(
            record,
            donor_id,
            _FACT_DONOR.format(n=n),
            _FACT_RECIPIENT.format(n=n),
            _FACT_QUESTION,
            "fact_probe_donor",
        )
        return (
            donors,
            recipient,
            _FACT_QUESTION,
            [str(n)],
            {
                "expected_exact": False,
                "hit_class": "fact_paraphrase",
                "expected_verifier_accept": True,
                "fact_probe": n,
            },
        )
    if transform == "fact_divergent_paraphrase":
        n, wrong = _fact_probe_numbers(record.source_id, context)
        donors, recipient = _v5_probe_pair(
            record,
            donor_id,
            _FACT_DONOR.format(n=wrong),
            _FACT_RECIPIENT.format(n=n),
            _FACT_QUESTION,
            "fact_divergent_donor",
        )
        return (
            donors,
            recipient,
            _FACT_QUESTION,
            [str(n)],
            {
                "expected_exact": False,
                "hit_class": "fact_divergent_paraphrase",
                "expected_verifier_accept": False,
                "expected_no_fact_leak": True,
                "fact_probe": n,
                "divergent_donor_value": wrong,
            },
        )
    if transform == "arithmetic_restatement":
        n, _ = _fact_probe_numbers(record.source_id, context)
        part, other = _restated_parts(record.source_id, context, n)
        donors, recipient = _v5_probe_pair(
            record,
            donor_id,
            _ARITH_DONOR.format(a=part, b=other),
            _FACT_RECIPIENT.format(n=n),
            _FACT_QUESTION,
            "arithmetic_restatement_donor",
        )
        return (
            donors,
            recipient,
            _FACT_QUESTION,
            [str(n)],
            {
                "expected_exact": False,
                "hit_class": "arithmetic_restatement",
                "expected_verifier_accept": False,
                "fact_equivalent": True,
                "fact_probe": n,
                "restated_parts": [part, other],
            },
        )
    if transform == "complement_restatement":
        total, failed = _complement_numbers(record.source_id, context)
        donors, recipient = _v5_probe_pair(
            record,
            donor_id,
            _COMPLEMENT_DONOR.format(total=total, passed=total - failed),
            _COMPLEMENT_RECIPIENT.format(total=total, failed=failed),
            _COMPLEMENT_QUESTION,
            "complement_restatement_donor",
        )
        return (
            donors,
            recipient,
            _COMPLEMENT_QUESTION,
            [str(failed)],
            {
                "expected_exact": False,
                "hit_class": "complement_restatement",
                "expected_verifier_accept": False,
                "fact_equivalent": True,
                "fact_probe": failed,
                "restated_total": total,
            },
        )
    raise ValueError(f"unknown v5 transform: {transform}")


def _paraphrase_request(request: str) -> str:
    reworded = request
    for old_pat, new_pat in _PARAPHRASE_REWRITES:
        reworded = reworded.replace(old_pat, new_pat)
    return reworded


@dataclass(frozen=True)
class TransformConfig:
    """Workload generation controls."""

    transforms: tuple[str, ...] = DEFAULT_TRANSFORMS
    max_segments: int = 4
    min_segment_chars: int = 400
    negative_selection: str = "cross_domain"  # v1 frozen specs pin "adjacent"


def _record_domain(record: SourceRecord) -> str:
    return str(record.metadata.get("domain") or record.dataset)


def _pick_negative(records: list[SourceRecord], idx: int, selection: str) -> SourceRecord:
    """Choose the unrelated donor for negative controls.

    "adjacent" (legacy, pinned by v1 frozen specs) takes the next record —
    which in a domain-ordered corpus is usually the SAME domain, making the
    "unrelated" donor a near-duplicate modulo entities (74-79% fuzzy-alignable
    in practice: an entity-swap pair, not an unrelated one). "cross_domain"
    takes the first record of a DIFFERENT domain in cyclic order.
    """
    n = len(records)
    if n <= 1:
        return records[idx]
    if selection == "cross_domain":
        home = _record_domain(records[idx])
        for offset in range(1, n):
            candidate = records[(idx + offset) % n]
            if _record_domain(candidate) != home:
                return candidate
    return records[(idx + 1) % n]


def build_workload(
    records: list[SourceRecord],
    config: TransformConfig | None = None,
) -> list[WorkloadItem]:
    """Build deterministic donor/recipient pairs from source records."""
    cfg = config or TransformConfig()
    items: list[WorkloadItem] = []
    for idx, record in enumerate(records):
        negative = _pick_negative(records, idx, cfg.negative_selection)
        entity_swap = records[(idx + 1) % len(records)] if len(records) > 1 else record
        for transform in cfg.transforms:
            if transform == "negative_control" and negative.source_id == record.source_id:
                continue
            if transform == "entity_swap_control" and (
                entity_swap.source_id == record.source_id
                or _record_domain(entity_swap) != _record_domain(record)
            ):
                continue
            items.append(
                _build_item(
                    record,
                    transform,
                    cfg,
                    negative if transform != "entity_swap_control" else entity_swap,
                )
            )
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
    question = record.input
    answers = record.answers

    if transform == "repeat_shifted":
        donors = [DonorPrompt(donor_id=donor_id, text=base, label="base_prompt")]
        recipient = _shifted_session_prompt(record.context, record.input)
        negative = False
        metadata = {"expected_exact": False, "hit_class": "true_repeat"}
    elif transform == "question_paraphrase":
        donors = [DonorPrompt(donor_id=donor_id, text=base, label="base_prompt")]
        recipient = _shifted_session_prompt(record.context, _paraphrase_request(record.input))
        negative = False
        metadata = {"expected_exact": False, "hit_class": "paraphrase"}
    elif transform == "exact_repeat":
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
    elif transform == "sparse_edit":
        donors = [DonorPrompt(donor_id=donor_id, text=base, label="base_prompt")]
        recipient = _sparse_edit_prompt(record.context, record.input, record.source_id)
        negative = False
        metadata = {
            "expected_exact": False,
            "enterprise_shape": "doc_v2_sparse_edit",
        }
    elif transform == "entity_swap_control":
        donor_text = _base_prompt(negative_record.context, negative_record.input)
        donors = [
            DonorPrompt(
                donor_id=donor_id,
                text=donor_text,
                label="entity_swap_donor",
                metadata={"entity_swap_source_id": negative_record.source_id},
            )
        ]
        recipient = base
        negative = False
        metadata = {
            "entity_swap": True,
            "entity_swap_source_id": negative_record.source_id,
            "expected_no_fact_leak": True,
        }
    elif transform in (
        "fact_paraphrase",
        "fact_divergent_paraphrase",
        "arithmetic_restatement",
        "complement_restatement",
    ):
        donors, recipient, question, answers, metadata = _build_v5_probe_item(
            record=record, transform=transform, donor_id=donor_id
        )
        negative = False
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
        input=question,
        answers=answers,
        negative_control=negative,
        metadata={**record.metadata, **metadata},
    )


def _shifted_session_prompt(context: str, request: str) -> str:
    """Same evidence + request as _base_prompt behind a different session
    wrapper: prompts diverge at position ~0 (exact-prefix misses) while the
    semantic content is unchanged — the canonical semantic-cache hit."""
    return (
        "Support session follow-up (ref: knowledge-workspace)\n\n"
        "A teammate previously reviewed this material; answer the request "
        "using the same source, grounded and concise.\n\n"
        "Source material:\n"
        f"{context.strip()}\n\n"
        "User request:\n"
        f"{request.strip()}\n\n"
        "Answer:"
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
        f"Reusable enterprise evidence cache entry\n\n{context.strip()}\n\nCached evidence summary:"
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
            f"\n[retrieved_record id=R{idx + 1:02d} source=knowledge_base]\n{segment.strip()}"
        )
    parts.append(
        "\nUsing the retrieved records, answer the request with citations to "
        "record IDs when possible.\n"
        f"Request: {request.strip()}\n"
        "Answer:"
    )
    return "\n".join(parts)


def _segment_donor_prompt(segment: str, idx: int) -> str:
    return f"Cached enterprise evidence segment {idx + 1}\n\n{segment.strip()}\n\nSegment summary:"


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


def _sparse_edit_prompt(
    context: str,
    request: str,
    source_id: str,
    edit_rate: float = 0.04,
) -> str:
    """Doc-v2 class: edit ~edit_rate of sentences IN PLACE, preserving all
    other bytes (incl. whitespace), under a shifted session wrapper.

    Deterministic per source (string-keyed seed). Edited sentences get a
    clearly-marked revision suffix so answers stay derivable from the
    unedited evidence.
    """
    import hashlib as _hashlib

    sentences = re.split(r"(?<=\.) ", context)
    seed = int(_hashlib.sha256(f"sparse-edit:{source_id}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    edited = []
    for idx, sent in enumerate(sentences):
        if sent and rng.random() < edit_rate:
            edited.append(sent.rstrip(".") + " (rev-2 annotation r" + str(idx) + ").")
        else:
            edited.append(sent)
    body = " ".join(edited)
    return (
        "SESSION[doc-v2 review] channel=editorial pass=2\n"
        "Compare-ready evidence copy follows.\n\n"
        f"{body}\n\n"
        f"Request: {request.strip()}\n"
        "Answer:"
    )


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
