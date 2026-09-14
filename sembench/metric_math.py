"""Small shared arithmetic for the metric blocks.

Four rules live here rather than in each block, because the difference between
them is a reporting decision, not a calculation:

- :func:`_rate` returns 0.0 on an empty denominator (the legacy block rates);
- :func:`_rate_or_none` returns None instead, so 0/0 is never published as a
  clean 0.0 — every section-4 rate uses this one;
- :func:`_pctl` is a linear-interpolating percentile over a list;
- :func:`_mean` / :func:`_ci` are the mean and its bootstrap interval.
"""

from __future__ import annotations


def _pctl(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _rate_or_none(numerator: int, denominator: int) -> float | None:
    """A rate, or None when the denominator is empty — never 0/0 as 0.0."""
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _ci(values: list[float]) -> dict[str, float] | None:
    from sembench.stats import bootstrap_mean

    result = bootstrap_mean(values)
    return result.to_dict() if result is not None else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
