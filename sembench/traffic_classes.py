"""The manifest's vocabulary: traffic classes, denominators, wrapper strata.

Two of section 4's metrics are defined over manifest CLASSES rather than over
requests (M1's opportunity denominator, M7's probe set) and one over the
manifest's instruction WRAPPERS (M1's two strata), so the names, the stratum
boundary and the rule for reading either off a row live in one place that both
the per-arm and the paired blocks import.
"""

from __future__ import annotations

from collections.abc import Sequence

from sembench.schema import RequestMetrics

# The manifest's traffic classes (phase-0 plan §3). The results layer has to
# know them by name because two of the metrics are defined over classes, not
# over requests: M1's opportunity denominator and M7's probe set.
NO_REUSE_CLASS = "no_reuse"
SAME_DOC_NEW_INSTRUCTION_CLASS = "same_doc_new_instruction"
REVISED_DOC_CLASS = "revised_doc"
ROPE_DELTA_SWEEP_CLASS = "rope_delta_sweep"
EXACT_REPEAT_CLASS = "exact_repeat"
# The class whose whole job is to detect propagation: a verbatim repeat of a
# request that was previously served approximate KV, >= 50 requests later.
PROPAGATION_PROBE_CLASS = "propagation_probe"
REWORDED_DOC_CLASS = "reworded_doc"
TRAFFIC_CLASSES = (
    NO_REUSE_CLASS,
    SAME_DOC_NEW_INSTRUCTION_CLASS,
    REVISED_DOC_CLASS,
    ROPE_DELTA_SWEEP_CLASS,
    EXACT_REPEAT_CLASS,
    PROPAGATION_PROBE_CLASS,
    REWORDED_DOC_CLASS,
)

# M1's alignment_given_opportunity denominator, from section 4 verbatim:
# "|{manifest items in same_doc_new_instruction ∪ revised_doc}|". It is the
# classes, not the offline model's per-item prediction: an item whose donor
# should have been reusable and was not is exactly what the metric is for, so
# it cannot be conditioned on the prediction that it would be.
ALIGNMENT_OPPORTUNITY_CLASSES = (SAME_DOC_NEW_INSTRUCTION_CLASS, REVISED_DOC_CLASS)

# M1's two strata (section 4, line 343): "Do not report a single blended
# alignment number. Report it separately for the shared-wrapper stratum and the
# ad-hoc stratum."
#
# The plan defines NEITHER term, so this boundary is an interpretation and is
# published as one (see docs/METRICS.md). The rejected reading is per-pair
# wrapper identity — "shared" = the donor and the recipient were written with
# the same wrapper — which is degenerate here: M1's opportunity population is
# same_doc_new_instruction union revised_doc, and same_doc_new_instruction is
# wrapper-MISMATCHED by construction, so that stratum would be near-empty and
# would explain nothing. The reading taken instead is popularity, which is what
# line 128's mechanism ("the shared wrapper never enters the prefix cache") is
# about: the wrapper the bulk of the stream sits on.
#
# The boundary is the manifest's own popularity order. Stream B draws its eight
# wrappers from Zipf(s=1.1) (`phase0-build-manifest.py`: WRAPPERS,
# zipf_weights(len(WRAPPERS), 1.1)), which puts 39.8% of the stream on rank 0,
# 18.6% on rank 1 and under 9% on every rank below — so ranks 0-1 are the
# smallest prefix of the order carrying a MAJORITY of the traffic (58.4%
# modelled; 56.1% of the 1,250 real rows in phase0-stream-b.jsonl). That is the
# rule: a wrapper is "shared" when it is in the head that the bulk of the
# stream sits on, so the RECIPIENT's own wrapper is routinely already resident
# in the prefix cache and its boundary is non-zero — lane 2's precondition,
# and what line 128's mechanism is about. (It is NOT a claim that the donor was
# written with the same wrapper: in M1's opportunity population the donor's
# wrapper differs by construction, which is exactly why the identity reading
# was rejected above.) Everything below the head is ad-hoc traffic whose
# recipient wrapper is usually cold, so the shared token-level tail that
# alignment is a property of (caveat A of the plan) is a different one. The two
# strata therefore answer different questions, and a blended rate over them is
# dominated by whichever wrapper happened to be popular.
SHARED_WRAPPER_MAX_RANK = 1
SHARED_WRAPPER_STRATUM = "shared_wrapper"
AD_HOC_WRAPPER_STRATUM = "ad_hoc"
# Rows whose manifest named no wrapper (a pre-round-6 manifest, or a synthetic
# stream). They are neither stratum and are published as their own bucket
# rather than folded into one, so the per-stratum numerators still sum to the
# blended numerator and nobody has to guess where a row went.
UNSTRATIFIED_WRAPPER = "unstratified"
WRAPPER_STRATA = (SHARED_WRAPPER_STRATUM, AD_HOC_WRAPPER_STRATUM, UNSTRATIFIED_WRAPPER)
WRAPPER_STRATUM_RULE = (
    f"shared_wrapper = wrapper_rank <= {SHARED_WRAPPER_MAX_RANK} (the Zipf(s=1.1) head the "
    "majority of the stream sits on, so the recipient's OWN wrapper is routinely resident "
    "and its boundary is non-zero; not a claim about the donor's wrapper, which differs by "
    "construction); ad_hoc = any lower rank; unstratified = the manifest "
    "named no wrapper. Each stratum's opportunity denominator is the rows it holds, because "
    "the manifest's class counts are not split by wrapper. The plan defines neither term: "
    "this is the popularity reading, chosen over donor/recipient wrapper identity, which is "
    "degenerate for M1's opportunity population. The constant is pinned, so "
    "wrapper_stratum_head_share reports what it actually selected on THIS document and "
    "wrapper_stratum_head_is_majority is false when it no longer selects a majority."
)


def wrapper_stratum_of(row: RequestMetrics) -> str:
    """Which of M1's strata this row belongs to.

    Null rank means the manifest named no wrapper, which is not rank 0: a row
    that declares nothing cannot be counted as the most popular wrapper.
    """
    if row.wrapper_rank is None:
        return UNSTRATIFIED_WRAPPER
    return (
        SHARED_WRAPPER_STRATUM
        if int(row.wrapper_rank) <= SHARED_WRAPPER_MAX_RANK
        else AD_HOC_WRAPPER_STRATUM
    )


def wrapper_head_share(rows: Sequence[RequestMetrics]) -> float | None:
    """Share of the RANKED rows the pinned head constant actually selected.

    :data:`SHARED_WRAPPER_MAX_RANK` is pinned to the manifest it was derived
    from, so on a manifest with a different wrapper count or a different Zipf
    exponent the published rule ("the head the majority of the stream sits on")
    would silently become false. This is the document's own measurement of that
    claim; unstratified rows are outside it, because a row that named no
    wrapper says nothing about the popularity order.
    """
    ranked = [row for row in rows if row.wrapper_rank is not None]
    if not ranked:
        return None
    head = sum(1 for row in ranked if wrapper_stratum_of(row) == SHARED_WRAPPER_STRATUM)
    return head / len(ranked)


# Where a class-scoped denominator came from. Section 4's denominators are
# counts of MANIFEST ITEMS, so a run that lost rows (errors, a stopped run, a
# --max-items cut) must still divide by what it was asked to serve. When the
# manifest counts did not reach the result document the rows present are used
# instead, and the substitution is named rather than assumed.
DENOMINATOR_FROM_MANIFEST = "manifest"
DENOMINATOR_FROM_ROWS_PRESENT = "rows_present"


def manifest_class_total(counts: dict[str, int] | None, classes: Sequence[str]) -> int | None:
    """How many manifest items fall in ``classes``, or None with no counts.

    None means "the manifest's own count never reached this document", which
    is the only honest reason to fall back to the rows that happen to be
    present.
    """
    if not counts:
        return None
    return sum(int(counts.get(name, 0)) for name in classes)


def traffic_class_of(row: RequestMetrics) -> str:
    """The row's traffic class, falling back to ``transform``.

    Manifests written before the class field existed carry the class in
    ``transform``; reading both means a probe item is found either way, and an
    empty string means the row declares no class at all.
    """
    return row.traffic_class or row.transform or ""
