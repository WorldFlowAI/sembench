"""Which cold twin goes with which warm twin, and which pairs are usable.

Every paired metric starts here. A paired measurement is only trustworthy when
the twins replayed the same item at the same stream position and both arms
answered, so the join is checked rather than assumed and everything it removes
is counted instead of dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from sembench.schema import RequestMetrics
from sembench.traffic_classes import traffic_class_of


@dataclass(frozen=True)
class _Pair:
    """One item's cold twin and warm twin, both usable."""

    item_id: str
    cold: RequestMetrics
    warm: RequestMetrics


def _pair_traffic_class(pair: _Pair) -> str:
    """The pair's traffic class, from either twin.

    Both twins replay the same manifest item, so they declare the same class;
    reading the warm row first and falling back to the cold one means a class
    stamped on only one side is still found.
    """
    return traffic_class_of(pair.warm) or traffic_class_of(pair.cold)


@dataclass(frozen=True)
class _PairSet:
    """The usable pairs, and every candidate that did not become one.

    ``excluded`` holds the cold row of each dropped candidate so a class-scoped
    denominator (M7's probe set) can say how many of ITS items were lost, not
    just how many were lost overall.
    """

    pairs: tuple[_Pair, ...]
    contaminated: int
    errored: int
    unpaired: int
    stream_position_mismatched: int
    excluded: tuple[RequestMetrics, ...]


def _clean_pairs(
    cold: dict[str, RequestMetrics],
    warm: dict[str, RequestMetrics],
) -> _PairSet:
    """Pairs whose cold twin was genuinely cold and whose arms both answered.

    Returns the usable pairs plus everything it removed, so the summary can
    report what it dropped instead of shrinking a denominator in silence.

    One check is new in round 5. Section 4 states M3 as "per ``item_id``, at
    the same stream position", and an item's manifest stream position is
    stamped on both twins. Two twins at different positions did not replay the
    same stream — a re-ordered or edited manifest between the arms, which the
    ``manifest_sha256`` check in ``merge-results`` catches only when both arms
    recorded one — and their TTFT ratio measures the reorder, not the cache.
    A row that declares no position makes no claim and is not excluded by it.
    """
    pairs: list[_Pair] = []
    excluded: list[RequestMetrics] = []
    contaminated = 0
    errored = 0
    unpaired = 0
    position_mismatched = 0
    for item_id, cold_row in cold.items():
        warm_row = warm.get(item_id)
        if warm_row is None:
            unpaired += 1
            excluded.append(cold_row)
            continue
        if cold_row.flush_contaminated:
            contaminated += 1
            excluded.append(cold_row)
            continue
        if cold_row.error or warm_row.error:
            errored += 1
            excluded.append(cold_row)
            continue
        if (
            cold_row.stream_position is not None
            and warm_row.stream_position is not None
            and cold_row.stream_position != warm_row.stream_position
        ):
            position_mismatched += 1
            excluded.append(cold_row)
            continue
        pairs.append(_Pair(item_id, cold_row, warm_row))
    return _PairSet(
        pairs=tuple(pairs),
        contaminated=contaminated,
        errored=errored,
        unpaired=unpaired,
        stream_position_mismatched=position_mismatched,
        excluded=tuple(excluded),
    )
