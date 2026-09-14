"""Bootstrap confidence intervals for aggregate metrics.

Dependency-free port of the house bootstrap methodology (95% percentile CI,
seeded resampling so reports are reproducible). Every headline number in a
published result should carry one of these.

Mean and median are both provided and they are not interchangeable. TTFT
speedup ratios are heavy-tailed — one pair that went from a 4 s cold prefill
to a 60 ms warm one drags a mean far above what a typical request sees — so
the headline speedup gate is specified on the **median**
(:func:`bootstrap_median_ratio`) and the mean is reported beside it as a
secondary, tail-sensitive figure.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

N_BOOTSTRAP = 2000
CI_ALPHA = 0.05
DEFAULT_SEED = "sembench-ci"


@dataclass(frozen=True)
class CIResult:
    point: float
    lo: float
    hi: float

    def to_dict(self) -> dict[str, float]:
        return {"point": self.point, "lo": self.lo, "hi": self.hi}


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    """Median with the usual even-length midpoint average."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    n_boot: int,
    seed: str,
) -> CIResult | None:
    """Seeded percentile bootstrap of ``statistic`` over ``values``."""
    if not values:
        return None
    point = statistic(values)
    if len(values) == 1:
        return CIResult(point, point, point)
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = _percentile(stats, 100 * CI_ALPHA / 2)
    hi = _percentile(stats, 100 * (1 - CI_ALPHA / 2))
    return CIResult(point, lo, hi)


def bootstrap_mean(
    values: list[float],
    n_boot: int = N_BOOTSTRAP,
    seed: str = DEFAULT_SEED,
) -> CIResult | None:
    """95% percentile-bootstrap CI for the mean; None on empty input."""
    return _bootstrap_ci(values, _mean, n_boot, seed)


def bootstrap_median(
    values: list[float],
    n_boot: int = N_BOOTSTRAP,
    seed: str = DEFAULT_SEED,
) -> CIResult | None:
    """95% percentile-bootstrap CI for the median; None on empty input.

    The median is what the speedup gate is written against: it answers "what
    did a typical pair do", which a mean over ratios does not.
    """
    return _bootstrap_ci(values, _median, n_boot, seed)


def paired_ratios(numerators: Sequence[float], denominators: Sequence[float]) -> list[float]:
    """Elementwise ``numerator / denominator`` for an aligned pair of arms.

    Raises rather than skipping: a dropped pair is an invisible change to the
    denominator of every rate computed beside this one, and a zero denominator
    means the caller handed over a measurement it should have filtered.
    """
    if len(numerators) != len(denominators):
        raise ValueError(
            "paired ratio needs one denominator per numerator; "
            f"got {len(numerators)} and {len(denominators)}"
        )
    for index, denominator in enumerate(denominators):
        if denominator <= 0:
            raise ValueError(f"paired ratio denominator at index {index} is {denominator!r}")
    return [float(n) / float(d) for n, d in zip(numerators, denominators)]


def bootstrap_median_ratio(
    numerators: list[float],
    denominators: list[float],
    n_boot: int = N_BOOTSTRAP,
    seed: str = DEFAULT_SEED,
) -> CIResult | None:
    """95% CI for the median of the **paired** ratios ``numerator/denominator``.

    Pairs are resampled as units — ``(cold[i], warm[i])`` travels together —
    because the two arms are a within-item comparison. Resampling the arms
    independently would break that pairing and produce a CI for a quantity
    nobody measured.

    This is the estimator behind ``blended_ttft_speedup_median``: cold TTFT
    over warm TTFT, per item, median across items.
    """
    return bootstrap_median(paired_ratios(numerators, denominators), n_boot=n_boot, seed=seed)
