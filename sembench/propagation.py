"""M7 — contamination / propagation, and the cold reference it is read against.

Section 4 compares the treatment arm's answer for a ``propagation_probe``
against the **cold (A1)** output, not against whatever a merged document calls
its baseline arm: the ``m7_propagation`` merge's baseline is A4, the product
arm, which can be contaminated on the same probe. Resolving that reference, and
scoring the probes against it, is the whole of this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sembench.arm_matrix import PHASE0_ARM_PAIRS, arm_id_of
from sembench.metric_math import _rate_or_none
from sembench.pairs import _Pair
from sembench.schema import RequestMetrics
from sembench.traffic_classes import (
    DENOMINATOR_FROM_MANIFEST,
    DENOMINATOR_FROM_ROWS_PRESENT,
    PROPAGATION_PROBE_CLASS,
    manifest_class_total,
    traffic_class_of,
)

# Section 4's cold reference for M7: "fraction whose answer in A6 matches the
# served output rather than the cold (A1) output". A1 is named, and it is not
# always the arm a merged document calls "cold" — the m7_propagation pair's
# baseline is A4, the product arm, which can be contaminated on the same probe.
PROPAGATION_COLD_REFERENCE_ARM = "A1"
# Where the cold answer came from: a separate A1 reference run supplied to the
# merge, or the document's own baseline arm (legitimate only when that arm IS
# section 4's cold reference).
COLD_REFERENCE_FROM_REFERENCE_ARM = "reference_arm"
# A reference run whose own document declares no phase-0 arm. It still answers
# the probes, but nothing checked WHICH arm it is, so the source says so rather
# than asserting the operator's --cold-reference-arm as a fact.
COLD_REFERENCE_FROM_REFERENCE_ARM_UNDECLARED = "reference_arm_undeclared"
COLD_REFERENCE_FROM_BASELINE_ARM = "baseline_arm"
# Arm pairs whose baseline arm is section 4's cold reference. Every other
# labelled pair needs a reference run of its own before M7 can be published.
COLD_REFERENCE_BASELINE_ARMS = (PROPAGATION_COLD_REFERENCE_ARM,)
# A document that names no arm anywhere: no --pair, and a run whose baseline id
# matches no phase-0 arm. Nothing on it says the cold rows came from section
# 4's cold reference, so M7 is not published — but the state is named rather
# than left as a null a reader would take for A1, because it is the state an
# operator fixes with --backend-id, --pair or --cold-reference.
COLD_REFERENCE_ARM_UNDECLARED = "undeclared"
# A document whose baseline is a KNOWN arm that is not A1 (an m7_propagation
# merge, labelled or not) and that was given no reference run. The field says
# which arm M7 still needs, and cannot be misread as the arm that answered:
# no arm answered as cold here at all.
COLD_REFERENCE_ARM_REQUIRED = f"required: {PROPAGATION_COLD_REFERENCE_ARM}"


@dataclass(frozen=True)
class ColdReference:
    """Which arm supplies M7's cold answer, and whether it is available.

    ``rows_by_item`` None with ``missing`` false means "use the pair's own cold
    twin", which is correct exactly when the document's baseline arm IS section
    4's cold reference. ``missing`` means no arm on this document answered as
    cold — either its baseline is a known non-A1 arm, or it declares no arm at
    all — and section 4's metric cannot be computed from it.

    ``arm`` is never null and is never provenance in the ``missing`` case: it
    reads :data:`COLD_REFERENCE_ARM_UNDECLARED` when nothing named an arm and
    :data:`COLD_REFERENCE_ARM_REQUIRED` when a known non-cold arm was the
    baseline, so a reader is handed neither a null that looks like A1 nor an
    "A1" that no A1 run stands behind.
    """

    rows_by_item: dict[str, RequestMetrics] | None
    arm: str | None
    source: str | None
    missing: bool


def _reference_rows_by_item(rows: Sequence[RequestMetrics]) -> dict[str, RequestMetrics]:
    """Index a reference run's rows by item, first row per item wins.

    A reference document is normally a single-arm A1 result. If a merged
    document is handed over instead, its warm rows are the treatment arm and
    are skipped: the cold reference is the baseline half of it.
    """
    from sembench.pairing import WARM_ARM

    by_item: dict[str, RequestMetrics] = {}
    for row in rows:
        if row.arm == WARM_ARM:
            continue
        by_item.setdefault(row.item_id, row)
    return by_item


def cold_reference_for(
    *,
    arm_pair: str | None,
    reference_rows: Sequence[RequestMetrics] | None,
    reference_arm: str | None = None,
    baseline_arm: str | None = None,
) -> ColdReference:
    """Resolve M7's cold reference for one document.

    Publishing M7 takes an **affirmative** signal that some run answered as
    section 4's cold arm. There are two, and everything else suppresses the
    metric:

    - a reference run was supplied — it is the cold reference, whatever the
      document's own baseline arm is. ``reference_arm`` is the arm that
      document declares for itself (the CLI checks it), and None means it
      declared none, which the source records instead of asserting A1;
    - the baseline arm IS section 4's cold reference — a pair whose baseline is
      A1, or an unlabelled join whose ``run.baseline_id`` names A1 — so the
      pair's own cold twin answers.

    Both remaining shapes are ``missing``, because on neither does anything
    say an A1 run took part:

    - the baseline is some OTHER known arm (``m7_propagation`` merges A4
      against A6, labelled or not) — the two arms can drift towards the served
      answer together, so the rate would under-report;
    - the document names no arm anywhere: no ``--pair``, and a baseline id that
      matches no phase-0 arm. This is the DEFAULT shape — ``--backend-id``
      defaults to the empty string and a plain ``merge-results`` stamps
      ``run.baseline_id`` from the cold run_id — so it was the same
      0.0-on-a-contaminated-workload hole one door over: an A4-vs-A6 merge
      that labels nothing is indistinguishable from an A1-vs-A4 merge that
      labels nothing, and "indistinguishable" is not permission to publish.
      The arm reads ``undeclared``, which names the remedy: ``--backend-id``,
      ``--pair``, or ``--cold-reference``.

    ``baseline_arm`` is the merged document's own ``run.baseline_id``, stamped
    from the cold arm's ``--backend-id``. Reading it is what lets an ordinary
    A1-vs-A4 merge keep its number without an extra reference run, and it is
    the only thing that can: a document that declares nothing is not an A1
    document that forgot to say so.
    """
    if reference_rows:
        declared = str(reference_arm or "").strip()
        return ColdReference(
            rows_by_item=_reference_rows_by_item(reference_rows),
            arm=declared or COLD_REFERENCE_ARM_UNDECLARED,
            source=(
                COLD_REFERENCE_FROM_REFERENCE_ARM
                if declared
                else COLD_REFERENCE_FROM_REFERENCE_ARM_UNDECLARED
            ),
            missing=False,
        )
    pair = PHASE0_ARM_PAIRS.get(str(arm_pair or ""))
    baseline = pair.baseline if pair is not None else arm_id_of(baseline_arm or "")
    if baseline in COLD_REFERENCE_BASELINE_ARMS:
        return ColdReference(None, baseline, COLD_REFERENCE_FROM_BASELINE_ARM, False)
    if not baseline:
        return ColdReference(None, COLD_REFERENCE_ARM_UNDECLARED, None, True)
    return ColdReference(None, COLD_REFERENCE_ARM_REQUIRED, None, True)


def cold_reference_conflicts(
    reference_rows: Sequence[RequestMetrics],
    arm_rows: Sequence[RequestMetrics],
    *,
    reference_path: str,
) -> list[str]:
    """Ways a ``--cold-reference`` document cannot answer this merge's items.

    A reference from another manifest, or one whose rows are all stamped
    ``arm='warm'`` (a merged document handed over by mistake), holds no cold
    answer for any item here. Scoring it produces zero scored probes, which is
    refused at the CLI rather than published: see
    :func:`_propagation_summary`, which nulls the rate in the same case for
    documents assembled any other way.
    """
    usable = set(_reference_rows_by_item(reference_rows))
    if usable & {row.item_id for row in arm_rows}:
        return []
    return [
        f"--cold-reference {reference_path!r} holds no cold-arm row for any item in this "
        "merge: it replayed another manifest, or every row in it is stamped arm='warm'. "
        "Merging it would publish M7 against a reference that answered nothing"
    ]


@dataclass(frozen=True)
class _PropagationCounts:
    """One pass over the probes against one cold reference."""

    propagated: int
    scored: int
    unlinked: int
    without_served_answer: int
    without_answers: int
    without_cold_reference: int


def _reference_answers(
    probes: Sequence[_Pair],
    rows_by_item: dict[str, RequestMetrics],
) -> tuple[dict[str, str], int]:
    """Cold answers from the reference run, and positions that did not match.

    The reference run has to have replayed the same stream: a row that sat at
    a different manifest position answered a different sequence of requests,
    and its output is not this probe's cold answer. A row that declares no
    position makes no claim and is accepted, exactly as the pairing does.
    """
    answers: dict[str, str] = {}
    mismatched = 0
    for pair in probes:
        reference = rows_by_item.get(pair.item_id)
        if reference is None:
            continue
        position = pair.warm.stream_position
        if (
            position is not None
            and reference.stream_position is not None
            and reference.stream_position != position
        ):
            mismatched += 1
            continue
        if reference.output_text:
            answers[pair.item_id] = reference.output_text
    return answers, mismatched


def _count_propagation(
    probes: Sequence[_Pair],
    warm_by_item: dict[str, RequestMetrics],
    cold_answers: dict[str, str] | None,
) -> _PropagationCounts:
    """Score the probes against one cold reference.

    ``cold_answers`` None means "the pair's own cold twin"; a dict means the
    reference run's answers, and a probe absent from it has no cold answer at
    all rather than a cold answer of "".
    """
    from sembench.quality import rouge_l

    propagated = 0
    scored = 0
    unlinked = 0
    without_served_answer = 0
    without_answers = 0
    without_cold_reference = 0
    for pair in probes:
        parent_id = pair.warm.propagation_parent_item_id or pair.cold.propagation_parent_item_id
        if not parent_id:
            unlinked += 1
            continue
        served = warm_by_item.get(parent_id)
        if served is None or not served.output_text:
            without_served_answer += 1
            continue
        if not pair.warm.output_text:
            without_answers += 1
            continue
        if cold_answers is None:
            cold_text = pair.cold.output_text
            if not cold_text:
                without_answers += 1
                continue
        else:
            cold_text = cold_answers.get(pair.item_id)
            if not cold_text:
                without_cold_reference += 1
                continue
        scored += 1
        if rouge_l(pair.warm.output_text, served.output_text) > rouge_l(
            pair.warm.output_text, cold_text
        ):
            propagated += 1
    return _PropagationCounts(
        propagated=propagated,
        scored=scored,
        unlinked=unlinked,
        without_served_answer=without_served_answer,
        without_answers=without_answers,
        without_cold_reference=without_cold_reference,
    )


def _reference_scored_nothing(reference: ColdReference, counts: _PropagationCounts) -> bool:
    """Was a reference run supplied that scored no probe at all?

    ``cold_reference_for`` marks a reference present as soon as it holds rows,
    but a reference from another manifest, or one whose rows are all the warm
    arm, yields no cold answer for any probe: every probe lands in
    ``propagation_probes_without_cold_reference`` and the rate would be 0 over
    the whole probe set. That is the failure the explicit reference exists to
    prevent, one door over, so it suppresses the rate exactly as an absent
    reference does.
    """
    return (
        reference.source
        in (COLD_REFERENCE_FROM_REFERENCE_ARM, COLD_REFERENCE_FROM_REFERENCE_ARM_UNDECLARED)
        and counts.scored == 0
    )


def _propagation_summary(
    pairs: list[_Pair],
    warm_by_item: dict[str, RequestMetrics],
    *,
    excluded: Sequence[RequestMetrics] = (),
    manifest_class_counts: dict[str, int] | None = None,
    reference: ColdReference | None = None,
) -> dict[str, Any]:
    """M7 — contamination / propagation, as section 4 defines it.

    Section 4::

        A4 vs A6 on the 50 propagation_probe items: fraction whose answer in
        A6 matches the *served* output rather than the *cold* (A1) output.

    A propagation probe is a verbatim repeat of an earlier request that was
    served approximate KV (§3), so three answers exist for one prompt and the
    question is which of two the treatment arm reproduces:

    - the **served** answer — the parent item's answer in the *same* arm, i.e.
      the output produced while approximate KV was in play;
    - the **cold** answer — this item's answer in **A1**, which is what an
      uncontaminated engine must return for a verbatim repeat.

    A probe counts as propagated when its treatment answer is strictly closer
    (ROUGE-L) to the served answer than to the cold one. Strictly: a tie is
    not evidence, and contamination is a claim that has to be earned.

    **The cold answer is A1's, not the merge baseline's.** Section 4 names A1,
    and the ``m7_propagation`` merge's baseline is A4 — the product arm, which
    can be contaminated on the same probe. When both sides drift towards the
    served answer they drift together, the strict comparison finds no
    difference, and the rate under-reports (a fully contaminated workload can
    read 0.0). A document that declares no arm anywhere cannot be told from
    that one, so it is suppressed the same way: publishing takes an
    affirmative A1 signal, never the absence of a contradicting one. The
    reference is therefore explicit: ``cold_reference`` carries an A1
    reference run's rows, ``propagation_cold_reference_arm`` and
    ``propagation_cold_reference_source`` say what answered, and on a document
    whose baseline is not A1 and that was given no reference run the rate is
    **null** with ``propagation_cold_reference_missing`` true. A reference that
    was supplied and scored no probe at all (another manifest, or rows stamped
    as the warm arm) is the same hole one door over and is nulled the same way,
    under ``propagation_cold_reference_unusable``. The
    baseline-referenced number is still computed — it is a useful arm-vs-arm
    diagnostic — but only under
    ``propagation_contamination_rate_vs_baseline_arm``, where nobody can read
    it as section 4's metric.

    **The denominator is the probe set, not the probes that could be scored.**
    Section 4 says "A4 vs A6 on the 50 ``propagation_probe`` items", and a
    probe that could not be scored is not evidence of no contamination — it is
    a probe that was not read. Dividing by the scored subset turns every
    failure to score into a better contamination number, which is the exact
    direction a contamination metric must not drift. So the headline rate is
    ``propagated / |manifest propagation_probe items|`` and every exclusion is
    published beside it: probes whose pair was not clean
    (``propagation_probes_excluded_unclean_pair`` — no twin, a contaminated
    cold arm, an error, a stream-position mismatch), unlinked probes, probes
    whose parent never answered in this arm, and probes missing an answer.
    The scored-only rate is kept under
    ``propagation_contamination_rate_scored_only``, which is the number to
    read when the exclusions are large, and never the headline.
    """
    probes = [pair for pair in pairs if traffic_class_of(pair.warm) == PROPAGATION_PROBE_CLASS]
    excluded_probes = sum(1 for row in excluded if traffic_class_of(row) == PROPAGATION_PROBE_CLASS)
    manifest_probes = manifest_class_total(manifest_class_counts, (PROPAGATION_PROBE_CLASS,))
    probe_set = len(probes) + excluded_probes if manifest_probes is None else manifest_probes
    probe_set_source = (
        DENOMINATOR_FROM_ROWS_PRESENT if manifest_probes is None else DENOMINATOR_FROM_MANIFEST
    )
    if reference is None:
        reference = cold_reference_for(arm_pair=None, reference_rows=None)
    # Always computed, never the headline: the treatment arm against whatever
    # this document calls its baseline.
    baseline = _count_propagation(probes, warm_by_item, None)
    if reference.missing:
        # No cold answer exists on this document at all: every probe that got
        # as far as needing one is counted, and no rate is published.
        answers: dict[str, str] | None = {}
        mismatched = 0
    elif reference.rows_by_item is not None:
        answers, mismatched = _reference_answers(probes, reference.rows_by_item)
    else:
        answers, mismatched = None, 0
    counts = baseline if answers is None else _count_propagation(probes, warm_by_item, answers)
    # A reference was supplied and scored NOTHING: wrong manifest, wrong arm
    # stamps, or every probe at another stream position. Publishing 0 / probe
    # set there is the same 0.0-on-a-contaminated-workload the explicit
    # reference exists to prevent, so it is nulled and named.
    unusable = _reference_scored_nothing(reference, counts)
    published = None if reference.missing or unusable else counts.propagated
    return {
        "propagation_definition": (
            "share of the propagation_probe SET whose treatment-arm answer is closer to the "
            "parent item's answer in the same arm (the served output) than to its own "
            "answer in the cold reference arm (section 4's A1); probes that could not be "
            "scored stay in the denominator and are published as the exclusion counters "
            "beside it"
        ),
        # Which arm answered as "cold", and whether one did at all. Never a
        # bare arm name when nothing answered: "undeclared" when the document
        # named no arm, "required: A1" when its baseline is a known arm that
        # is not section 4's cold reference.
        "propagation_cold_reference_arm": reference.arm,
        "propagation_cold_reference_source": reference.source,
        "propagation_cold_reference_missing": reference.missing,
        # A reference document WAS supplied and answered none of the probes.
        # Distinct from _missing (none was supplied at all) and published so a
        # null rate cannot be read as "the reference was fine".
        "propagation_cold_reference_unusable": unusable,
        "propagation_contamination_rate": (
            None if published is None else _rate_or_none(published, probe_set)
        ),
        "propagation_contamination_numerator": published,
        "propagation_contamination_denominator": probe_set,
        "propagation_probe_set_source": probe_set_source,
        # The same numerator over the probes that could actually be read. Use
        # it to judge the headline, never in place of it.
        "propagation_contamination_rate_scored_only": (
            None if published is None else _rate_or_none(published, counts.scored)
        ),
        "propagation_contamination_scored_denominator": (
            None if published is None else counts.scored
        ),
        # The treatment arm against this document's own baseline. On an
        # m7_propagation merge that baseline is A4, which can be contaminated
        # on the same probe, so this is an arm-vs-arm diagnostic and NOT
        # section 4's metric.
        "propagation_contamination_rate_vs_baseline_arm": _rate_or_none(
            baseline.propagated, probe_set
        ),
        "propagation_contamination_numerator_vs_baseline_arm": baseline.propagated,
        "propagation_probe_pairs": len(probes),
        "propagation_probes_excluded_unclean_pair": excluded_probes,
        "propagation_probes_unlinked": counts.unlinked,
        "propagation_probes_without_served_answer": counts.without_served_answer,
        "propagation_probes_without_answers": counts.without_answers,
        # Probes for which the cold reference arm holds no usable answer: it
        # never ran them, or it ran them at another stream position.
        "propagation_probes_without_cold_reference": counts.without_cold_reference,
        "propagation_probes_reference_position_mismatched": mismatched,
        # Probes the manifest declares that this document holds no row for at
        # all: a run stopped early, or a --max-items cut. Non-zero means the
        # probe set was never fully replayed.
        "propagation_probes_absent_from_run": max(
            0,
            probe_set - len(probes) - excluded_probes,
        ),
    }
