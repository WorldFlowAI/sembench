import pytest

from sembench.kl import kl_profile, kl_topk_at_position


def _topk(*pairs):
    return [(tid, lp) for tid, lp in pairs]


def test_identical_distributions_zero_kl():
    dist = _topk((1, -0.1), (2, -2.5), (3, -3.0))
    assert kl_topk_at_position(dist, dist) == pytest.approx(0.0, abs=1e-12)


def test_shifted_distribution_positive_kl():
    cold = _topk((1, -0.1), (2, -2.5))
    warm = _topk((2, -0.1), (1, -2.5))  # swapped preference
    kl = kl_topk_at_position(cold, warm)
    assert kl > 0.1


def test_disjoint_support_uses_floors():
    cold = _topk((1, -0.1), (2, -2.0))
    warm = _topk((3, -0.1), (4, -2.0))
    kl = kl_topk_at_position(cold, warm)
    assert kl > 0.5  # strongly divergent, finite (no inf/nan)
    assert kl == kl  # not NaN


def test_empty_side_is_zero():
    assert kl_topk_at_position([], _topk((1, -0.1))) == 0.0


def test_profile_alignment_and_divergence_index():
    cold_topk = [_topk((1, -0.1)), _topk((2, -0.1)), _topk((3, -0.1))]
    warm_topk = [_topk((1, -0.1)), _topk((9, -0.1))]  # shorter + diverges at pos 1
    profile = kl_profile([1, 2, 3], cold_topk, [1, 9], warm_topk)
    assert profile.positions_compared == 2  # min length
    assert profile.first_token_divergence == 1
    assert profile.kl_at_positions[0] == pytest.approx(0.0, abs=1e-12)
    # Disjoint singletons with a 1-nat absent-margin give ~0.46 nats.
    assert profile.kl_at_positions[1] > 0.3
    assert profile.max_kl_topk == max(profile.kl_at_positions)


def test_profile_no_divergence():
    topk = [_topk((1, -0.1), (2, -2.0))] * 3
    profile = kl_profile([1, 1, 1], topk, [1, 1, 1], topk)
    assert profile.first_token_divergence is None
    assert profile.mean_kl_topk == pytest.approx(0.0, abs=1e-12)
