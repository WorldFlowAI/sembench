"""The phase-0 arm matrix, and what a result document says about its arm.

Section 3 names eight arms and section 4 names five cold/warm comparisons over
them. A merged document that cannot say WHICH comparison it is cannot be read,
so the matrix, the checks that refuse a mislabelled merge, and the readers that
pull an arm's identity back out of a result document live together here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def result_arm(payload: dict[str, Any]) -> str:
    """The arm a result document declares for itself; '' when it declares none.

    Read from ``run.arm``, which :func:`sembench.schema.collect_run_metadata`
    stamps from the flag the arm was actually launched with.
    """
    run = payload.get("run") or {}
    return str(run.get("arm") or "")


@dataclass(frozen=True)
class Arm:
    """One arm of the phase-0 matrix (section 3's arm table)."""

    arm_id: str
    label: str
    purpose: str

    def names(self) -> tuple[str, ...]:
        return (self.arm_id, self.label)


PHASE0_ARMS = {
    arm.arm_id: arm
    for arm in (
        Arm(
            "A0",
            "stock_nopc",
            "prefix caching off, no connector — the cold floor, never a baseline",
        ),
        Arm("A1", "stock_pc", "prefix caching on, no connector — THE baseline"),
        Arm("A2", "stock_pc_rerun", "identical repeat of A1 — the cold-vs-cold noise floor"),
        Arm("A3", "conn_discovery", "mode=discovery_only — lookup cost without capture"),
        Arm("A4", "conn_span", "mode=semantic_span_experimental — the product arm"),
        Arm(
            "A5",
            "conn_span_noaudit",
            "A4 with audit and log_decisions off — prices instrumentation",
        ),
        Arm("A6", "conn_span_nomitigation", "A4 with contamination eviction off"),
        Arm("A7", "fleet_3worker", "A4 x 3 replicas behind the llm-d scorer"),
    )
}


@dataclass(frozen=True)
class ArmPair:
    """A comparison section 4 names, and which metric it is the input to."""

    name: str
    baseline: str
    treatment: str
    metric: str
    measures: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.name,
            "metric": self.metric,
            "measures": self.measures,
            "baseline_arm": self.baseline,
            "baseline_arm_label": PHASE0_ARMS[self.baseline].label,
            "treatment_arm": self.treatment,
            "treatment_arm_label": PHASE0_ARMS[self.treatment].label,
        }


# Every cold/warm join section 4 asks for by name. `merge-results --pair`
# takes one of these, records it on the merged document, and checks the two
# source arms against it — so a merged A3-vs-A5 document cannot be published
# as if it were the M3 headline.
PHASE0_ARM_PAIRS = {
    pair.name: pair
    for pair in (
        ArmPair(
            "m3_ttft",
            "A1",
            "A4",
            "M3",
            "TTFT speedup and M6 answer quality: the product arm against THE baseline",
        ),
        ArmPair(
            "m6_noise_floor",
            "A1",
            "A2",
            "M6",
            "cold-vs-cold noise floor; the non-inferiority margin M6 is judged against",
        ),
        ArmPair(
            "m4_capture",
            "A3",
            "A4",
            "M4",
            "capture leg of the miss tax: discovery-only against the product arm",
        ),
        ArmPair(
            "m4_instrumentation",
            "A5",
            "A4",
            "M4",
            "instrumentation leg: the audit-off arm against the product arm; "
            "subtract it from any published tax",
        ),
        ArmPair(
            "m7_propagation",
            "A4",
            "A6",
            "M7",
            "contamination: the product arm against the same arm with eviction off",
        ),
    )
}


# The two pairs whose NAME changes how a metric is computed, not just how the
# document is labelled: m4_capture restricts the miss tax to no_reuse items
# (section 4's capture leg), and m7_propagation is the one comparison whose
# baseline arm is not section 4's cold reference.
M4_CAPTURE_PAIR = "m4_capture"
M7_PROPAGATION_PAIR = "m7_propagation"


def arm_pair(name: str) -> ArmPair:
    """One of section 4's named arm pairs, or a ValueError listing them all."""
    pair = PHASE0_ARM_PAIRS.get(str(name))
    if pair is None:
        known = ", ".join(sorted(PHASE0_ARM_PAIRS))
        raise ValueError(f"unknown arm pair {name!r}; section 4 names: {known}")
    return pair


def _declared_arm_id(payload: dict[str, Any], role: str) -> str:
    """What a result document calls the arm it ran, '' when it says nothing."""
    run = payload.get("run") or {}
    for key in ("backend_id", "baseline_id") if role == "cold" else ("backend_id",):
        value = str(run.get(key) or "").strip()
        if value:
            return value
    return ""


def _declaration_names_arm(declared: str, name: str) -> bool:
    """Does a backend id name this arm? Word-boundary match, either spelling."""
    return bool(re.search(rf"(?<![a-z0-9]){name.lower()}(?![a-z0-9])", declared.lower()))


def arm_id_of(declared: str) -> str:
    """Which phase-0 arm a backend/baseline id names, '' when it names none.

    The id is free text an operator wrote (``--backend-id "A4 conn_span"``),
    and both spellings are accepted — the arm id and the plan's config name.
    The LONGEST matching name wins, so ``conn_span_noaudit`` resolves to A5 and
    not to A4, whose label is a prefix of it.

    An id that matches nothing is not an error: it is a run that never said
    which arm it was, and callers treat that as undeclared rather than as a
    contradiction.
    """
    haystack = str(declared or "").strip()
    if not haystack:
        return ""
    best_id = ""
    best_length = 0
    for arm in PHASE0_ARMS.values():
        for name in arm.names():
            if len(name) > best_length and _declaration_names_arm(haystack, name):
                best_id, best_length = arm.arm_id, len(name)
    return best_id


def result_backend_arm(payload: dict[str, Any]) -> str:
    """Which phase-0 arm a result document declares it ran, '' when none."""
    return arm_id_of(_declared_arm_id(payload, "cold"))


def cold_reference_arm_conflicts(payload: dict[str, Any], asserted_arm: str | None) -> list[str]:
    """Ways a ``--cold-reference`` document contradicts ``--cold-reference-arm``.

    The flag is an operator assertion and it defaults to A1, so without this
    check any result file handed in as the reference stamps
    ``propagation_cold_reference_arm: "A1"`` on a document whose entire point
    is reference provenance. A reference that declares no arm is accepted —
    refusing it would only teach people to stop labelling runs — and is
    recorded as undeclared rather than as an asserted A1.
    """
    declared = result_backend_arm(payload)
    asserted = arm_id_of(asserted_arm or "")
    if not declared or not asserted or declared == asserted:
        return []
    return [
        f"--cold-reference-arm {asserted_arm!r} but that result declares "
        f"{_declared_arm_id(payload, 'cold')!r} ({declared}): M7 would name an arm that did "
        "not answer, in the one block whose purpose is reference provenance"
    ]


def arm_pair_conflicts(
    pair: ArmPair,
    cold_payload: dict[str, Any],
    warm_payload: dict[str, Any],
) -> list[str]:
    """Ways the two result documents contradict the pair they are merged as.

    The check is on ``run.backend_id`` (``baseline_id`` for the cold side),
    which is the only place an operator writes down *which* phase-0 arm a run
    was. An arm that labelled itself is checked; one that did not is left
    alone, because refusing an unlabelled run would just teach people to pass
    ``--pair`` less. Matching is on either spelling — the arm id (``A4``) or
    the plan's config name (``conn_span``).
    """
    conflicts: list[str] = []
    for role, payload, expected in (
        ("cold", cold_payload, pair.baseline),
        ("warm", warm_payload, pair.treatment),
    ):
        declared = _declared_arm_id(payload, role)
        if not declared:
            continue
        names = PHASE0_ARMS[expected].names()
        if not any(_declaration_names_arm(declared, name) for name in names):
            conflicts.append(
                f"--pair {pair.name} expects the {role} arm to be {expected} "
                f"({PHASE0_ARMS[expected].label}), but that result declares "
                f"{declared!r}: merging it under this pair would publish "
                f"{pair.metric} against an arm it was not measured on"
            )
    return conflicts


def arm_label_conflicts(
    cold_payload: dict[str, Any],
    warm_payload: dict[str, Any],
) -> list[str]:
    """Ways the operator's --cold/--warm assignment contradicts the payloads.

    ``merge-results`` takes the two roles from the command line, so swapping
    the flags silently inverts every speedup in the merged document: a 5x
    reported win is really a 0.2x loss and nothing in the output says so. The
    result documents already know which arm they are, so the claim can be
    checked instead of trusted.

    ``single`` and an absent arm make no claim and are accepted — that is the
    normal shape of a run launched without ``--arm``. A payload declaring the
    *other* role, or declaring ``paired`` (it is already a merged document
    holding both arms), is a conflict.
    """
    from sembench.pairing import SINGLE_ARM

    conflicts: list[str] = []
    for role, payload in (("cold", cold_payload), ("warm", warm_payload)):
        declared = result_arm(payload)
        if declared and declared not in (role, SINGLE_ARM):
            conflicts.append(
                f"--{role} result declares run.arm={declared!r}: "
                f"it was not run as the {role} arm, and merging it as one "
                "inverts or invalidates every paired number downstream"
            )
    return conflicts


def result_manifest_class_counts(payload: dict[str, Any]) -> dict[str, int] | None:
    """The manifest's per-class item counts a result document carries.

    Written by the runner into ``config.manifest_class_counts`` because the
    runner is the only stage that reads the manifest. ``merge-results`` reads
    it back out so the merged document's class-scoped denominators (M1's
    opportunity classes, M7's probe set) are still the manifest's counts and
    not the rows the two arms happened to produce.
    """
    config = payload.get("config") or {}
    counts = config.get("manifest_class_counts")
    if not isinstance(counts, dict):
        return None
    clean = {
        str(name): int(value)
        for name, value in counts.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return clean or None


def result_arm_pair_name(config: dict[str, Any] | None) -> str | None:
    """Which of section 4's comparisons a document's config declares.

    ``merge-results --pair`` writes the whole :class:`ArmPair` into
    ``config.arm_pair``; the name is the part the metrics read (M4's capture
    population, M7's cold reference). Null means an unlabelled join.
    """
    pair = (config or {}).get("arm_pair")
    if not isinstance(pair, dict):
        return None
    name = str(pair.get("pair") or "").strip()
    return name or None
