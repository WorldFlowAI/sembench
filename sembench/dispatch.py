"""Bounded, order-preserving dispatch for concurrent replay runs.

Stream order is part of the experiment. A recipient may only be issued once
the request that seeds it has *completed*, and that donor -> recipient gap is
counted in completed requests, never in wall time, so raising concurrency
cannot quietly turn a 200-request gap into an adjacent pair.

Submission follows manifest order among the stages that are *ready*: a stage
still waiting on its own dependency steps aside rather than holding the queue
behind it, because a head-of-line wait would collapse the run back to one
stream at a time and the measured rate would be the harness's. At
``concurrency=1`` there is no stepping aside -- every stage is issued and
awaited one at a time, in exactly manifest order.

Nothing here knows about HTTP. The caller supplies ``execute`` and gets back
one outcome per stage, in the order the stages were given, plus the wall clock
the whole run took -- the denominator of requests-per-second for the arm.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Dependency:
    """A stage that must finish first, plus the gap that must follow it.

    ``min_gap_requests`` is the donor -> recipient separation the manifest
    asks for, measured in completed requests. ``min_seconds`` is the settle an
    engine that indexes donors off the request path needs; it is waited for by
    the dispatcher, not inside a worker slot, so it does not depress
    concurrency or the reported throughput.
    """

    key: str
    min_gap_requests: int = 0
    min_seconds: float = 0.0


@dataclass(frozen=True)
class Stage:
    """One dispatchable unit of work in the stream."""

    key: str
    kind: str
    item_index: int = 0
    requests: int = 1
    depends_on: tuple[Dependency, ...] = ()


@dataclass(frozen=True)
class StageOutcome:
    """What one stage produced. Returned in the order the stages were given."""

    stage: Stage
    value: Any = None
    error: str | None = None
    gap_forced: bool = False


@dataclass(frozen=True)
class DispatchReport:
    """Outcomes plus the arm-level facts a throughput number is built from."""

    outcomes: tuple[StageOutcome, ...]
    wall_seconds: float
    concurrency: int
    max_in_flight: int
    gap_forced: int
    # Wall seconds with nothing in flight, i.e. the arm waiting on a settle
    # rather than on the engine. Measured, so a throughput number can exclude
    # it without guessing at how well the settles overlapped.
    idle_seconds: float = 0.0


@dataclass(frozen=True)
class Progress:
    """Completion bookkeeping the readiness predicates read.

    Immutable: each completion produces a new Progress, so a predicate can
    never observe a half-updated view.
    """

    completed_requests: int = 0
    ranks: Mapping[str, int] = field(default_factory=dict)
    finished_at: Mapping[str, float] = field(default_factory=dict)

    def with_completion(self, stage: Stage, *, now: float) -> "Progress":
        completed = self.completed_requests + max(1, stage.requests)
        return Progress(
            completed_requests=completed,
            ranks={**self.ranks, stage.key: completed},
            finished_at={**self.finished_at, stage.key: now},
        )


def gap_shortfall(stage: Stage, *, progress: Progress) -> int:
    """Completed requests still owed before `stage` may be issued.

    Zero means the gap is satisfied. A dependency that has not completed at
    all owes at least one request, whatever its configured gap.
    """
    shortfall = 0
    for dep in stage.depends_on:
        rank = progress.ranks.get(dep.key)
        if rank is None:
            shortfall = max(shortfall, max(1, dep.min_gap_requests))
            continue
        shortfall = max(shortfall, dep.min_gap_requests - (progress.completed_requests - rank))
    return max(0, shortfall)


def settle_remaining(stage: Stage, *, progress: Progress, now: float) -> float:
    """Seconds of post-donor settle still owed before `stage` may be issued."""
    remaining = 0.0
    for dep in stage.depends_on:
        if dep.min_seconds <= 0:
            continue
        finished = progress.finished_at.get(dep.key)
        if finished is None:
            remaining = max(remaining, dep.min_seconds)
            continue
        remaining = max(remaining, dep.min_seconds - (now - finished))
    return max(0.0, remaining)


def run_stages(
    stages: Sequence[Stage],
    *,
    execute: Callable[[Stage], Any],
    concurrency: int = 1,
) -> DispatchReport:
    """Issue `stages` with at most `concurrency` requests in flight.

    Outcomes come back in `stages` order whatever order they complete in.

    When nothing is in flight and no remaining stage is ready, waiting can
    produce no further completions, so the queue head is unblocked: a stage
    owed only a settle is slept for, and a stage short of its completed-request
    gap is issued anyway and counted in `gap_forced`. A manifest whose gaps
    agree with its stream order never reaches that second branch.
    """
    width = max(1, int(concurrency))
    # At width 1 the queue is strictly head-of-line: order is the guarantee,
    # and there is no second lane for a look-ahead to fill anyway.
    lookahead = 1 if width == 1 else max(64, 4 * width)
    ordered: list[StageOutcome | None] = [None] * len(stages)
    pending: dict[Future, tuple[int, Stage]] = {}
    forced_positions: set[int] = set()
    queue: list[int] = list(range(len(stages)))
    progress = Progress()
    max_in_flight = 0
    idle_seconds = 0.0
    start = time.perf_counter()
    # Only stretches where the arm drained count as idle; the few microseconds
    # before the first submit are not a settle.
    empty_since: float | None = None

    def ready(position: int) -> bool:
        stage = stages[position]
        return (
            gap_shortfall(stage, progress=progress) == 0
            and settle_remaining(stage, progress=progress, now=time.perf_counter()) <= 0
        )

    def submit(index: int) -> None:
        nonlocal idle_seconds, empty_since
        if not pending and empty_since is not None:
            idle_seconds += time.perf_counter() - empty_since
            empty_since = None
        position = queue.pop(index)
        stage = stages[position]
        pending[pool.submit(_guarded, execute, stage)] = (position, stage)

    with ThreadPoolExecutor(max_workers=width) as pool:
        while queue or pending:
            index = 0
            while index < min(len(queue), lookahead) and len(pending) < width:
                if ready(queue[index]):
                    submit(index)
                    continue
                index += 1

            if not pending and queue:
                window = min(len(queue), lookahead)
                unblocked = next((i for i in range(window) if ready(queue[i])), None)
                if unblocked is not None:
                    submit(unblocked)
                else:
                    head = stages[queue[0]]
                    settle = settle_remaining(head, progress=progress, now=time.perf_counter())
                    if settle > 0:
                        time.sleep(settle)
                        continue
                    forced_positions.add(queue[0])
                    submit(0)

            max_in_flight = max(max_in_flight, len(pending))
            # With every lane busy only a completion can change anything, so
            # there is nothing to wake up early for.
            timeout = (
                None
                if len(pending) >= width
                else _next_settle(stages, queue, progress=progress, lookahead=lookahead)
            )
            done, _ = wait(list(pending), timeout=timeout, return_when=FIRST_COMPLETED)
            for future in done:
                position, stage = pending.pop(future)
                value, error = future.result()
                progress = progress.with_completion(stage, now=time.perf_counter())
                ordered[position] = StageOutcome(
                    stage=stage,
                    value=value,
                    error=error,
                    gap_forced=position in forced_positions,
                )
            if not pending and empty_since is None:
                empty_since = time.perf_counter()
    wall_seconds = time.perf_counter() - start

    return DispatchReport(
        outcomes=tuple(outcome for outcome in ordered if outcome is not None),
        wall_seconds=wall_seconds,
        concurrency=width,
        max_in_flight=max_in_flight,
        gap_forced=len(forced_positions),
        idle_seconds=idle_seconds,
    )


def _next_settle(
    stages: Sequence[Stage],
    queue: Sequence[int],
    *,
    progress: Progress,
    lookahead: int,
) -> float | None:
    """When to wake even if nothing completes, because a settle will expire."""
    now = time.perf_counter()
    waiting = [
        settle_remaining(stages[position], progress=progress, now=now)
        for position in queue[:lookahead]
    ]
    pending_settles = [value for value in waiting if value > 0]
    return max(min(pending_settles), 0.001) if pending_settles else None


def _guarded(execute: Callable[[Stage], Any], stage: Stage) -> tuple[Any, str | None]:
    try:
        return execute(stage), None
    except Exception as exc:  # one dead endpoint must not abort the arm
        return None, f"{type(exc).__name__}: {exc}"
