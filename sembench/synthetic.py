"""Synthetic enterprise corpus (fully original, redistributable).

Three document domains modeled on enterprise LLM workloads — support-ticket
RAG, contract review, and log triage — generated deterministically so the
corpus can be frozen and rebuilt byte-identically anywhere.

Design constraints (docs/DATASET.md):
- String-keyed per-source seeds (never arithmetic seed offsets, which alias
  streams and create phantom overlap between "independent" sources).
- Every source gets its own entity vocabulary (names carry a per-source tag),
  so unrelated sources cannot share full token blocks: the corpus passes the
  collision audit by construction rather than by luck.
- Context lengths cover 2k-16k tokens with the 8k+ band oversampled — the
  fuzzy-match benefit threshold sits around 8k, and the benchmark must
  resolve the benefit-vs-context curve rather than average over it.
- Each document embeds extractable facts; the paired question asks for them,
  so quality scoring has a ground truth.
"""

from __future__ import annotations

import random

from sembench.schema import SourceRecord

SYNTHETIC_DATASET = "synthetic-enterprise-v1"
_SEED_NAMESPACE = "sembench-synthetic-v1"

# (band_label, target_word_count, weight). Words approximate tokens closely
# enough for banding; the 8k+ bands carry most of the weight.
_LENGTH_BANDS = (
    ("2k", 1_500, 1),
    ("4k", 3_000, 2),
    ("8k", 6_000, 3),
    ("12k", 9_000, 3),
    ("16k", 12_000, 3),
)

_DOMAINS = ("support_ticket", "contract_review", "log_triage")

_SERVICES = ("checkout", "billing", "auth", "search", "inventory", "routing")
_REGIONS = ("west", "east", "central", "north")
_SEVERITIES = ("sev1", "sev2", "sev3")
_CLAUSE_TOPICS = (
    "data retention",
    "security attestation",
    "termination notice",
    "liability cap",
    "service credits",
    "audit rights",
)
_COMPONENTS = ("gateway", "scheduler", "worker", "cache", "replicator", "indexer")


def synthetic_records(
    sources_per_domain: int = 80,
    domains: tuple[str, ...] = _DOMAINS,
) -> list[SourceRecord]:
    records = []
    for domain in domains:
        for index in range(sources_per_domain):
            records.append(_build_source(domain=domain, index=index))
    return records


def _build_source(*, domain: str, index: int) -> SourceRecord:
    rng = random.Random(f"{_SEED_NAMESPACE}-{domain}-{index}")
    tag = f"{rng.randrange(16**4):04x}"
    band_label, target_words, _ = _pick_band(rng)

    if domain == "support_ticket":
        context, question, answer = _support_ticket(rng, tag, target_words)
    elif domain == "contract_review":
        context, question, answer = _contract_review(rng, tag, target_words)
    else:
        context, question, answer = _log_triage(rng, tag, target_words)

    return SourceRecord(
        source_id=f"syn-{domain}-{index:03d}-{tag}",
        dataset=SYNTHETIC_DATASET,
        context=context,
        input=question,
        answers=[answer],
        metadata={"domain": domain, "length_band": band_label},
    )


def _pick_band(rng: random.Random):
    total = sum(weight for _, _, weight in _LENGTH_BANDS)
    roll = rng.randrange(total)
    for band in _LENGTH_BANDS:
        roll -= band[2]
        if roll < 0:
            return band
    return _LENGTH_BANDS[-1]


def _fill(rng: random.Random, sentences: list[str], target_words: int) -> str:
    """Repeat-shuffle sentences until the document reaches the target size."""
    doc: list[str] = []
    words = 0
    while words < target_words:
        batch = sentences[:]
        rng.shuffle(batch)
        for sentence in batch:
            doc.append(sentence)
            words += len(sentence.split())
            if words >= target_words:
                break
    return " ".join(doc)


def _support_ticket(rng: random.Random, tag: str, target_words: int):
    service = f"{rng.choice(_SERVICES)}-{tag}"
    region = rng.choice(_REGIONS)
    severity = rng.choice(_SEVERITIES)
    minutes = rng.randrange(18, 240)
    root_cause = rng.choice(
        (
            f"a stale config push to {service}",
            f"connection pool exhaustion in {service}",
            f"a failed database failover behind {service}",
            f"certificate expiry on the {service} edge",
        )
    )
    # Every template span carries a per-source entity within ~10 words: any
    # tag-free run of 16+ tokens is identical across sources and becomes a
    # phantom block (found empirically by the collision audit).
    sentences = [
        f"Ticket {tag}-{i}: {severity} incident affecting {service} in the {region} region. "
        f"Symptom set {i} on {service} includes elevated latency and retry storms, with "
        f"queue depth alerts for {service} observed by the {region} on-call rotation."
        for i in range(1, 9)
    ]
    sentences.append(
        f"Investigation for {service} concluded the root cause was {root_cause}, "
        f"with full mitigation completed in {minutes} minutes."
    )
    context = _fill(rng, sentences, target_words)
    question = (
        f"For the {severity} incident on {service}, what was the root cause "
        "and how long did mitigation take?"
    )
    answer = f"{root_cause}; mitigated in {minutes} minutes"
    return context, question, answer


def _contract_review(rng: random.Random, tag: str, target_words: int):
    supplier = f"supplier-{tag}"
    topic = rng.choice(_CLAUSE_TOPICS)
    days = rng.randrange(30, 121)
    clause_no = rng.randrange(4, 19)
    sentences = [
        f"Agreement {tag} clause {i}: {supplier} shall maintain documented controls "
        f"for obligation area {i} under schedule {tag}, with counsel review for "
        f"{supplier} renewed annually."
        for i in range(1, 10)
    ]
    # The fact values (topic, days) birthday-collide across 80 sources, so the
    # fact sentence itself must carry the per-source tag mid-span.
    sentences.append(
        f"Clause {clause_no} of agreement {tag} sets the {topic} requirement "
        f"at {days} days under schedule {tag}, flagged for legal follow-up."
    )
    context = _fill(rng, sentences, target_words)
    question = f"Which clause of agreement {tag} covers {topic}, and what period does it specify?"
    answer = f"clause {clause_no}; {days} days"
    return context, question, answer


def _log_triage(rng: random.Random, tag: str, target_words: int):
    component = f"{rng.choice(_COMPONENTS)}-{tag}"
    error_code = f"E{rng.randrange(1000, 9999)}"
    count = rng.randrange(40, 900)
    sentences = [
        f"log[{tag}.{i}] {component} emitted status batch {i} with heartbeat and "
        f"compaction counters for {component} within normal {tag} thresholds."
        for i in range(1, 11)
    ]
    sentences.append(
        f"Triage summary: {component} logged {count} occurrences of {error_code}, "
        f"the only anomalous signature in the {tag} window."
    )
    context = _fill(rng, sentences, target_words)
    question = (
        f"What anomalous error code did {component} log, and how many occurrences were counted?"
    )
    answer = f"{error_code}; {count} occurrences"
    return context, question, answer
