"""Deterministic cold/warm arm pairing.

A paired measurement is only trustworthy when every warm request has exactly
one cold twin that replayed the same prompt, the same donor gap and the same
stream position. That has to be a *checked* property, not an assumption: a
paired summary silently computed over a partial or mismatched join reads the
same as a real one.

So the join is by construction, never by heuristic:

- the replay order is generated from the manifest (:func:`replay_plan`), so a
  paired run emits exactly one ``cold`` and one ``warm`` row per ``item_id``
  at adjacent stream positions;
- two arms run as separate processes (the usual stock-vs-connector A/B) are
  joined on ``item_id`` only when both replayed the same manifest bytes, and
  every pair is checked against a manifest-derived fingerprint;
- anything that does not join cleanly is reported and counted, never dropped.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from sembench.schema import RequestMetrics, RunMetadata, WorkloadItem

COLD_ARM = "cold"
WARM_ARM = "warm"
SINGLE_ARM = "single"
PAIRED_ARMS = (COLD_ARM, WARM_ARM)
MERGED_ARM = "paired"
VALID_ARMS = (COLD_ARM, WARM_ARM, SINGLE_ARM)


@dataclass(frozen=True)
class ReplayStep:
    """One request to issue: which item, which arm, where in the stream."""

    item: WorkloadItem
    arm: str
    stream_position: int
    seed_donors: bool


def duplicate_item_ids(items: list[WorkloadItem]) -> list[str]:
    """Item ids appearing more than once, in first-seen order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in items:
        if item.item_id in seen and item.item_id not in duplicates:
            duplicates.append(item.item_id)
        seen.add(item.item_id)
    return duplicates


def replay_plan(
    items: list[WorkloadItem],
    *,
    paired: bool = False,
    arm: str = SINGLE_ARM,
) -> list[ReplayStep]:
    """The exact ordered sequence of requests a run will issue.

    In paired mode each item yields its cold twin first and its warm twin
    immediately after, so both arms see the same prompt at the same position
    in the stream and the warm arm's only warmth is its own donors. In
    single-arm mode (including a whole-run ``cold``/``warm`` arm that will be
    joined later by :func:`join_arms`) the plan is the manifest order and
    donors are seeded for every item, so the two runs replay byte-identical
    streams and the donor gap is the same on both sides.
    """
    if arm not in VALID_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {VALID_ARMS}")
    if paired and arm != SINGLE_ARM:
        raise ValueError(
            f"paired mode stamps cold/warm per item: do not also pin the run to arm={arm!r}"
        )
    if paired or arm in PAIRED_ARMS:
        duplicates = duplicate_item_ids(items)
        if duplicates:
            raise ValueError(
                "paired arms join on item_id, so item ids must be unique; "
                f"manifest repeats: {', '.join(duplicates[:10])}"
            )

    steps: list[ReplayStep] = []
    for item in items:
        if paired:
            steps.append(ReplayStep(item, COLD_ARM, len(steps), False))
            steps.append(ReplayStep(item, WARM_ARM, len(steps), True))
        else:
            steps.append(ReplayStep(item, arm, len(steps), True))
    return steps


def pair_fingerprint(row: RequestMetrics) -> tuple[str, str, bool, int, int]:
    """Manifest-derived identity of the request a row measured.

    Every component is a pure function of the replayed item (its dataset,
    transform, control flag, donor gap and tokenized prompt length), so two
    rows sharing an ``item_id`` but not a fingerprint did not replay the same
    stream and must not be paired.
    """
    return (
        row.dataset,
        row.transform,
        bool(row.negative_control),
        int(row.donor_count),
        int(row.total_blocks),
    )


def index_by_item(rows: list[RequestMetrics]) -> tuple[dict[str, RequestMetrics], list[str]]:
    """Map rows by ``item_id``; report ids that occur more than once.

    A dict comprehension would silently keep the last row per id, which is how
    a half-joined pair set still produces a confident-looking summary.
    """
    index: dict[str, RequestMetrics] = {}
    duplicates: list[str] = []
    for row in rows:
        if row.item_id in index:
            if row.item_id not in duplicates:
                duplicates.append(row.item_id)
            continue
        index[row.item_id] = row
    return index, duplicates


@dataclass(frozen=True)
class PairingReport:
    """What the join actually produced, including everything it could not pair."""

    pairs: int
    cold_rows: int
    warm_rows: int
    manifest_sha256_cold: str = ""
    manifest_sha256_warm: str = ""
    manifest_match: bool = False
    duplicate_cold_item_ids: list[str] = field(default_factory=list)
    duplicate_warm_item_ids: list[str] = field(default_factory=list)
    cold_only_item_ids: list[str] = field(default_factory=list)
    warm_only_item_ids: list[str] = field(default_factory=list)
    fingerprint_mismatch_item_ids: list[str] = field(default_factory=list)

    @property
    def problems(self) -> list[str]:
        """Reasons this join is not a clean one-to-one pairing."""
        issues: list[str] = []
        if not self.manifest_match:
            issues.append(
                "arms replayed different manifests "
                f"(cold={self.manifest_sha256_cold or 'unknown'}, "
                f"warm={self.manifest_sha256_warm or 'unknown'}): "
                "prompts, donor gaps and stream positions are not comparable"
            )
        if self.duplicate_cold_item_ids:
            issues.append(f"cold arm repeats item ids: {_preview(self.duplicate_cold_item_ids)}")
        if self.duplicate_warm_item_ids:
            issues.append(f"warm arm repeats item ids: {_preview(self.duplicate_warm_item_ids)}")
        if self.cold_only_item_ids:
            issues.append(f"cold rows with no warm twin: {_preview(self.cold_only_item_ids)}")
        if self.warm_only_item_ids:
            issues.append(f"warm rows with no cold twin: {_preview(self.warm_only_item_ids)}")
        if self.fingerprint_mismatch_item_ids:
            issues.append(
                f"twins measured different requests: {_preview(self.fingerprint_mismatch_item_ids)}"
            )
        if not self.pairs:
            issues.append("no item paired across the two arms")
        return issues

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok, "problems": self.problems}


def _preview(values: list[str], limit: int = 10) -> str:
    head = ", ".join(values[:limit])
    return head if len(values) <= limit else f"{head} (+{len(values) - limit} more)"


def join_arms(
    cold_rows: list[RequestMetrics],
    warm_rows: list[RequestMetrics],
    *,
    cold_manifest_sha256: str = "",
    warm_manifest_sha256: str = "",
) -> tuple[list[RequestMetrics], PairingReport]:
    """Join two single-arm runs into one paired row list.

    Rows are restamped ``cold``/``warm`` from the role they were supplied as,
    and emitted cold-then-warm in the cold run's order so the merged result
    replays in stream order. Only items present exactly once on both sides
    with matching fingerprints are paired; the rest are reported.
    """
    cold_index, cold_duplicates = index_by_item(cold_rows)
    warm_index, warm_duplicates = index_by_item(warm_rows)

    merged: list[RequestMetrics] = []
    mismatched: list[str] = []
    for item_id, cold_row in cold_index.items():
        warm_row = warm_index.get(item_id)
        if warm_row is None:
            continue
        if pair_fingerprint(cold_row) != pair_fingerprint(warm_row):
            mismatched.append(item_id)
            continue
        merged.append(replace(cold_row, arm=COLD_ARM))
        merged.append(replace(warm_row, arm=WARM_ARM))

    paired_ids = {row.item_id for row in merged}
    report = PairingReport(
        pairs=len(merged) // 2,
        cold_rows=len(cold_rows),
        warm_rows=len(warm_rows),
        manifest_sha256_cold=cold_manifest_sha256,
        manifest_sha256_warm=warm_manifest_sha256,
        manifest_match=bool(cold_manifest_sha256) and cold_manifest_sha256 == warm_manifest_sha256,
        duplicate_cold_item_ids=cold_duplicates,
        duplicate_warm_item_ids=warm_duplicates,
        cold_only_item_ids=sorted(set(cold_index) - paired_ids - set(mismatched)),
        warm_only_item_ids=sorted(set(warm_index) - paired_ids - set(mismatched)),
        fingerprint_mismatch_item_ids=sorted(mismatched),
    )
    return merged, report


def requests_from_result(payload: dict[str, Any]) -> list[RequestMetrics]:
    """Rebuild ``RequestMetrics`` rows from a written result document.

    Unknown keys are dropped so a result written by an older or newer sembench
    still loads; missing keys fall back to the dataclass defaults.
    """
    known = {f.name for f in fields(RequestMetrics)}
    rows: list[RequestMetrics] = []
    for raw in payload.get("requests") or []:
        rows.append(RequestMetrics(**{key: value for key, value in raw.items() if key in known}))
    return rows


def result_manifest_sha256(payload: dict[str, Any]) -> str:
    """The manifest checksum a result run was produced against ('' if absent)."""
    run = payload.get("run") or {}
    return str(run.get("manifest_sha256") or "")


def merged_run_metadata(
    cold_payload: dict[str, Any],
    warm_payload: dict[str, Any],
    *,
    run_id: str | None = None,
) -> RunMetadata:
    """Run identity for a merged result: warm arm's engine, cold arm as baseline.

    ``baseline_id`` is run IDENTITY and falls back to the cold run's id when
    the cold arm declared no backend/baseline label, so that a merged document
    always says which run was the baseline. ``baseline_arm_declared`` carries
    only what an operator actually declared and has no such fallback: a run id
    is free text ("phase0-g5-a1-rack-cold") that can name an arm by accident,
    and M7 reads the declaration, never the identity.
    """
    cold_run = cold_payload.get("run") or {}
    warm_run = warm_payload.get("run") or {}
    cold_id = str(cold_run.get("run_id") or "cold")
    warm_id = str(warm_run.get("run_id") or "warm")
    declared_baseline = str(cold_run.get("backend_id") or cold_run.get("baseline_id") or "")
    return RunMetadata(
        run_id=run_id or f"{cold_id}+{warm_id}",
        engine=str(warm_run.get("engine") or cold_run.get("engine") or ""),
        manifest_path=str(warm_run.get("manifest_path") or cold_run.get("manifest_path") or ""),
        manifest_sha256=result_manifest_sha256(warm_payload)
        or result_manifest_sha256(cold_payload),
        arm=MERGED_ARM,
        engine_version=str(warm_run.get("engine_version") or ""),
        backend_id=str(warm_run.get("backend_id") or ""),
        baseline_id=declared_baseline or cold_id,
        baseline_arm_declared=declared_baseline,
        sembench_version=str(warm_run.get("sembench_version") or ""),
        sembench_git_sha=str(warm_run.get("sembench_git_sha") or ""),
        sembench_git_dirty=bool(warm_run.get("sembench_git_dirty") or False),
        semblend_version=str(warm_run.get("semblend_version") or ""),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
