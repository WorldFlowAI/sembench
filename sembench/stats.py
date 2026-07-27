"""Bootstrap confidence intervals for aggregate metrics.

Dependency-free port of the house bootstrap methodology (95% percentile CI,
seeded resampling so reports are reproducible). Every headline number in a
published result should carry one of these.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

N_BOOTSTRAP = 2000
CI_ALPHA = 0.05


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


def bootstrap_mean(
    values: list[float],
    n_boot: int = N_BOOTSTRAP,
    seed: str = "sembench-ci",
) -> CIResult | None:
    """95% percentile-bootstrap CI for the mean; None on empty input."""
    if not values:
        return None
    point = sum(values) / len(values)
    if len(values) == 1:
        return CIResult(point, point, point)
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(sample) / n)
    stats.sort()
    lo = _percentile(stats, 100 * CI_ALPHA / 2)
    hi = _percentile(stats, 100 * (1 - CI_ALPHA / 2))
    return CIResult(point, lo, hi)
