"""Turn a replay plan into dispatchable stages carrying their gaps.

The donor -> recipient gap is the thing a concurrent run most easily
destroys: a manifest that separates a donor and its recipient by 200 stream
positions separates them by nothing at all if both are in flight together. So
the gap travels with the work as a dependency measured in *completed
requests*, and the dispatcher, not the worker thread, is what waits.
"""

from __future__ import annotations

from sembench.dispatch import Dependency, Stage
from sembench.pairing import ReplayStep
from sembench.schema import WorkloadItem


def donor_item_refs(item: WorkloadItem) -> tuple[str, ...]:
    """Item ids earlier in the stream whose request seeds this one.

    A self-seeding manifest carries no ``donor_prompts``: the donor is an
    earlier request in the same stream, named under ``metadata`` as
    ``donor_item_id`` (or ``donor_item_ids``). Without this the runner cannot
    tell which earlier request a recipient must wait behind.
    """
    raw = item.metadata.get("donor_item_ids")
    if raw is None:
        raw = item.metadata.get("donor_item_id")
    if raw is None:
        return ()
    values = [raw] if isinstance(raw, str) else list(raw)
    return tuple(str(value) for value in values if str(value))


def build_stages(
    plan: list[ReplayStep],
    *,
    min_donor_gap_requests: int = 0,
    settle_seconds: float = 0.0,
) -> tuple[Stage, ...]:
    """Expand a replay plan into dispatchable stages carrying their gaps.

    A step becomes a donor stage (when it seeds donors) and a recipient stage
    that depends on it, so the settle an engine needs to index donors is
    waited for by the dispatcher instead of inside a worker slot. In-manifest
    donor references become a dependency with the configured gap in completed
    requests; a reference that is unknown or points forward in the stream is
    dropped, because no completion could ever satisfy it.
    """
    stages: list[Stage] = []
    recipient_keys: dict[str, str] = {}
    for position, step in enumerate(plan):
        item = step.item
        depends_on: list[Dependency] = []
        if step.seed_donors and item.donor_prompts:
            donor_key = f"{position}:{item.item_id}:donors"
            stages.append(
                Stage(
                    key=donor_key,
                    kind="donors",
                    item_index=position,
                    requests=len(item.donor_prompts),
                )
            )
            depends_on.append(Dependency(key=donor_key, min_seconds=max(0.0, settle_seconds)))
        gap = int(item.metadata.get("donor_gap_requests") or min_donor_gap_requests or 0)
        for ref in donor_item_refs(item):
            ref_key = recipient_keys.get(ref)
            if ref_key is None:
                continue
            depends_on.append(Dependency(key=ref_key, min_gap_requests=max(0, gap)))
        recipient_key = f"{position}:{item.item_id}:recipient"
        stages.append(
            Stage(
                key=recipient_key,
                kind="recipient",
                item_index=position,
                depends_on=tuple(depends_on),
            )
        )
        recipient_keys[item.item_id] = recipient_key
    return tuple(stages)
