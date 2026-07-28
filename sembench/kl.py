"""Per-token distributional divergence between warm and cold runs (H5).

ROUGE on decoded text tells you THAT outputs diverged; token logprob streams
tell you WHERE and HOW MUCH the sampling distribution shifted before the
decoded tokens visibly differ. Under the sequential deterministic protocol
(see docs/CALIBRATION.md) cold logprobs are reproducible, so any
distributional shift in the warm arm is mechanism-caused.

What SGLang exposes per generated position with `return_logprob` +
`top_logprobs_num=k` is the top-k (logprob, token_id) list — a truncated
distribution. We therefore compute a truncated-support KL:

    KL_t(cold_t || warm_t) over the UNION of top-k token ids at position t,
    with each side renormalized over that union; token ids missing from one
    side get that side's floor MINUS a fixed margin (1 nat below its k-th
    logprob) — an absent token is strictly less likely than any present one,
    and without the margin a k=1 comparison of disjoint singletons would
    renormalize to identical distributions and read KL=0. Still understates
    true KL; stated in the field name (kl_topk).

Alignment: positions compare 1:1 from the first generated token. Once the
two streams have emitted different token ids the comparison after that point
mixes "different prefix" effects; the first-divergence index is reported so
consumers can split pre/post divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

TopK = list[tuple[int, float]]  # (token_id, logprob), descending by logprob


@dataclass(frozen=True)
class KlProfile:
    """Positional divergence profile for one warm/cold pair."""

    positions_compared: int
    first_token_divergence: int | None  # index of first differing chosen token
    mean_kl_topk: float
    max_kl_topk: float
    kl_at_positions: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions_compared": self.positions_compared,
            "first_token_divergence": self.first_token_divergence,
            "mean_kl_topk": self.mean_kl_topk,
            "max_kl_topk": self.max_kl_topk,
        }


def kl_topk_at_position(cold: TopK, warm: TopK) -> float:
    """Truncated-support KL(cold || warm) over the union of token ids."""
    if not cold or not warm:
        return 0.0
    cold_map = dict(cold)
    warm_map = dict(warm)
    support = set(cold_map) | set(warm_map)
    absent_margin = 1.0  # nats below the k-th logprob for ids not in the list
    cold_floor = min(cold_map.values()) - absent_margin
    warm_floor = min(warm_map.values()) - absent_margin

    def renorm(logprobs: dict[int, float], floor: float) -> dict[int, float]:
        raw = {tid: math.exp(logprobs.get(tid, floor)) for tid in support}
        total = sum(raw.values())
        return {tid: value / total for tid, value in raw.items()}

    p = renorm(cold_map, cold_floor)
    q = renorm(warm_map, warm_floor)
    return sum(p[tid] * math.log(p[tid] / q[tid]) for tid in support if p[tid] > 0)


def kl_profile(
    cold_chosen: list[int],
    cold_topk: list[TopK],
    warm_chosen: list[int],
    warm_topk: list[TopK],
) -> KlProfile:
    """Positional KL profile for a warm/cold logprob stream pair."""
    n = min(len(cold_topk), len(warm_topk))
    per_position = [kl_topk_at_position(cold_topk[t], warm_topk[t]) for t in range(n)]

    first_divergence: int | None = None
    for t in range(min(len(cold_chosen), len(warm_chosen))):
        if cold_chosen[t] != warm_chosen[t]:
            first_divergence = t
            break

    return KlProfile(
        positions_compared=n,
        first_token_divergence=first_divergence,
        mean_kl_topk=(sum(per_position) / n) if n else 0.0,
        max_kl_topk=max(per_position) if per_position else 0.0,
        kl_at_positions=per_position,
    )
