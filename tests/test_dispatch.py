"""Bounded ordered dispatch: order, in-flight ceiling, and the donor gap.

The gap between a donor and its recipient is the property a concurrent run
destroys first, and it destroys it silently: the arm still completes, the
numbers still look like numbers. So the gap is asserted here in the units the
manifest states it in -- completed requests -- not in wall time.
"""

from __future__ import annotations

import threading
import time

from sembench.dispatch import (
    Dependency,
    Progress,
    Stage,
    gap_shortfall,
    run_stages,
    settle_remaining,
)


def _stage(key: str, *, depends_on=(), requests: int = 1, item_index: int = 0) -> Stage:
    return Stage(
        key=key,
        kind="recipient",
        item_index=item_index,
        requests=requests,
        depends_on=tuple(depends_on),
    )


def test_gap_shortfall_is_zero_only_once_the_donor_completed_and_the_gap_elapsed():
    stage = _stage("r", depends_on=[Dependency(key="d", min_gap_requests=3)])

    not_started = Progress(completed_requests=10, ranks={})
    assert gap_shortfall(stage, progress=not_started) == 3

    just_done = Progress(completed_requests=10, ranks={"d": 10})
    assert gap_shortfall(stage, progress=just_done) == 3

    two_since = Progress(completed_requests=12, ranks={"d": 10})
    assert gap_shortfall(stage, progress=two_since) == 1

    three_since = Progress(completed_requests=13, ranks={"d": 10})
    assert gap_shortfall(stage, progress=three_since) == 0


def test_an_uncompleted_dependency_blocks_even_with_a_zero_gap():
    """A zero-gap dependency still means "after it finished", not "whenever"."""
    stage = _stage("r", depends_on=[Dependency(key="d")])
    assert gap_shortfall(stage, progress=Progress(completed_requests=5, ranks={})) == 1
    assert gap_shortfall(stage, progress=Progress(completed_requests=5, ranks={"d": 5})) == 0


def test_gap_shortfall_takes_the_worst_dependency():
    stage = _stage(
        "r",
        depends_on=[
            Dependency(key="a", min_gap_requests=1),
            Dependency(key="b", min_gap_requests=8),
        ],
    )
    progress = Progress(completed_requests=10, ranks={"a": 9, "b": 9})
    assert gap_shortfall(stage, progress=progress) == 7


def test_settle_remaining_counts_down_from_the_dependency_finishing():
    stage = _stage("r", depends_on=[Dependency(key="d", min_seconds=2.0)])
    unfinished = Progress(completed_requests=1, ranks={}, finished_at={})
    assert settle_remaining(stage, progress=unfinished, now=100.0) == 2.0

    finished = Progress(completed_requests=1, ranks={"d": 1}, finished_at={"d": 100.0})
    assert settle_remaining(stage, progress=finished, now=100.5) == 1.5
    assert settle_remaining(stage, progress=finished, now=103.0) == 0.0


def test_progress_with_completion_does_not_mutate_the_previous_view():
    before = Progress()
    after = before.with_completion(_stage("d", requests=3), now=1.0)
    assert before.completed_requests == 0 and dict(before.ranks) == {}
    assert after.completed_requests == 3 and after.ranks["d"] == 3


def test_run_stages_returns_outcomes_in_submission_order_not_completion_order():
    stages = [_stage(f"s{i}", item_index=i) for i in range(6)]

    def execute(stage: Stage) -> str:
        # Later stages finish first; submission order must survive that.
        time.sleep(0.02 * (len(stages) - stage.item_index))
        return stage.key

    report = run_stages(stages, execute=execute, concurrency=6)
    assert [outcome.stage.key for outcome in report.outcomes] == [s.key for s in stages]
    assert [outcome.value for outcome in report.outcomes] == [s.key for s in stages]


def test_run_stages_holds_the_in_flight_count_at_the_concurrency_ceiling():
    stages = [_stage(f"s{i}", item_index=i) for i in range(12)]
    lock = threading.Lock()
    live = {"now": 0, "peak": 0}

    def execute(stage: Stage) -> None:
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.01)
        with lock:
            live["now"] -= 1

    report = run_stages(stages, execute=execute, concurrency=4)
    assert live["peak"] <= 4
    assert live["peak"] > 1  # it really did run them together
    assert report.max_in_flight <= 4
    assert len(report.outcomes) == 12


def test_concurrency_one_issues_one_at_a_time_in_order():
    stages = [_stage(f"s{i}", item_index=i) for i in range(5)]
    lock = threading.Lock()
    live = {"now": 0, "peak": 0}
    order: list[str] = []

    def execute(stage: Stage) -> None:
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            order.append(stage.key)
        time.sleep(0.005)
        with lock:
            live["now"] -= 1

    report = run_stages(stages, execute=execute, concurrency=1)
    assert live["peak"] == 1
    assert order == [s.key for s in stages]
    assert report.gap_forced == 0


def test_a_recipient_waits_for_its_donor_gap_in_completed_requests():
    """The point of the whole module: at concurrency 8 a recipient whose donor
    is 4 requests back must still see 4 completions before it is issued."""
    donor = _stage("donor", item_index=0)
    fillers = [_stage(f"filler{i}", item_index=i + 1) for i in range(10)]
    recipient = _stage(
        "recipient",
        item_index=11,
        depends_on=[Dependency(key="donor", min_gap_requests=4)],
    )
    stages = [donor, *fillers, recipient]

    lock = threading.Lock()
    completed: list[str] = []
    started_after = {"count": None}

    def execute(stage: Stage) -> None:
        if stage.key == "recipient":
            with lock:
                started_after["count"] = len(completed)
        time.sleep(0.001 if stage.key == "donor" else 0.02)
        with lock:
            completed.append(stage.key)

    report = run_stages(stages, execute=execute, concurrency=4)
    donor_rank = completed.index("donor") + 1
    assert started_after["count"] - donor_rank >= 4
    assert report.gap_forced == 0


def test_a_blocked_stage_steps_aside_instead_of_holding_the_queue():
    """Head-of-line waiting would collapse the arm back to one stream while a
    settle elapses, and the measured rate would be the harness's, not the
    engine's."""
    stages = [
        _stage("donor", item_index=0),
        _stage("blocked", item_index=1, depends_on=[Dependency(key="donor", min_seconds=0.2)]),
        _stage("behind", item_index=2),
    ]
    order: list[str] = []
    lock = threading.Lock()

    def execute(stage: Stage) -> None:
        with lock:
            order.append(stage.key)

    run_stages(stages, execute=execute, concurrency=4)
    assert order.index("behind") < order.index("blocked")


def test_at_concurrency_one_a_blocked_stage_does_hold_the_queue():
    """One lane means order is the only guarantee left; nothing overtakes."""
    stages = [
        _stage("donor", item_index=0),
        _stage("blocked", item_index=1, depends_on=[Dependency(key="donor", min_seconds=0.05)]),
        _stage("behind", item_index=2),
    ]
    order: list[str] = []

    run_stages(stages, execute=lambda stage: order.append(stage.key), concurrency=1)
    assert order == ["donor", "blocked", "behind"]


def test_an_unsatisfiable_gap_is_forced_rather_than_deadlocking():
    """Nothing in flight can never produce another completion, so waiting for
    one would hang the arm. The stage is issued and the shortfall is counted."""
    stages = [
        _stage("donor", item_index=0),
        _stage(
            "recipient",
            item_index=1,
            depends_on=[Dependency(key="donor", min_gap_requests=50)],
        ),
    ]
    report = run_stages(stages, execute=lambda stage: stage.key, concurrency=1)
    assert report.gap_forced == 1
    assert [outcome.stage.key for outcome in report.outcomes] == ["donor", "recipient"]
    assert [outcome.gap_forced for outcome in report.outcomes] == [False, True]


def test_a_settle_is_waited_out_by_the_dispatcher_not_inside_a_worker():
    stages = [
        _stage("donor", item_index=0),
        _stage(
            "recipient",
            item_index=1,
            depends_on=[Dependency(key="donor", min_seconds=0.15)],
        ),
    ]
    started: dict[str, float] = {}
    finished: dict[str, float] = {}

    def execute(stage: Stage) -> None:
        started[stage.key] = time.perf_counter()
        finished[stage.key] = time.perf_counter()

    run_stages(stages, execute=execute, concurrency=4)
    assert started["recipient"] - finished["donor"] >= 0.15


def test_a_failing_stage_is_recorded_and_the_rest_of_the_arm_continues():
    stages = [_stage(f"s{i}", item_index=i) for i in range(3)]

    def execute(stage: Stage) -> str:
        if stage.key == "s1":
            raise RuntimeError("endpoint refused")
        return "ok"

    report = run_stages(stages, execute=execute, concurrency=2)
    assert [outcome.value for outcome in report.outcomes] == ["ok", None, "ok"]
    assert report.outcomes[1].error == "RuntimeError: endpoint refused"


def test_wall_seconds_covers_the_whole_arm():
    stages = [_stage(f"s{i}", item_index=i) for i in range(4)]
    report = run_stages(stages, execute=lambda stage: time.sleep(0.02), concurrency=2)
    assert report.wall_seconds >= 0.04  # two batches of two
    assert report.concurrency == 2
