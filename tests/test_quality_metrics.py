import pytest

from sembench.quality import quality_score, rouge_l, rouge_l_best, token_f1
from sembench.stats import bootstrap_mean


def test_token_f1_golden_values():
    # identical → 1.0
    assert token_f1("clause 7; 90 days", ["clause 7; 90 days"]) == pytest.approx(1.0)
    # half overlap: prediction {a,b}, reference {a,c} → p=r=f1=0.5
    assert token_f1("alpha beta", ["alpha gamma"]) == pytest.approx(0.5)
    # disjoint → 0.0
    assert token_f1("alpha beta", ["gamma delta"]) == pytest.approx(0.0)
    # no references → None
    assert token_f1("anything", []) is None
    # best-of over references
    assert token_f1("alpha beta", ["gamma", "alpha beta"]) == pytest.approx(1.0)


def test_rouge_l_golden_values():
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)
    # hyp "a b c d", ref "a c b d": LCS = 3 ("a b d" or "a c d") → f1 = 3/4
    assert rouge_l("a b c d", "a c b d") == pytest.approx(0.75)
    assert rouge_l("x y z", "p q r") == pytest.approx(0.0)
    assert rouge_l("", "ref") == pytest.approx(0.0)
    # order sensitivity: same bag of words, reversed order → LCS 1 of 3
    bag = rouge_l("a b c", "c b a")
    assert bag < 0.5


def test_rouge_l_best_over_references():
    assert rouge_l_best("a b c", ["z", "a b c"]) == pytest.approx(1.0)
    assert rouge_l_best("a b c", []) is None


def test_corruption_degrades_all_three_metrics():
    reference = "the incident was mitigated in 42 minutes by rolling back the config push"
    clean = reference
    corrupted = "the incident was mitigated in 97 hours by restarting the database cluster"
    for metric in (
        lambda out: quality_score(out, [reference]),
        lambda out: token_f1(out, [reference]),
        lambda out: rouge_l_best(out, [reference]),
    ):
        assert metric(clean) > metric(corrupted)


def test_normalization_is_case_and_punct_insensitive():
    assert token_f1("Clause 7, 90 DAYS!", ["clause 7; 90 days"]) == pytest.approx(1.0)
    assert rouge_l("The Cat SAT.", "the cat sat") == pytest.approx(1.0)


def test_bootstrap_mean_properties():
    assert bootstrap_mean([]) is None
    single = bootstrap_mean([0.5])
    assert single.point == single.lo == single.hi == 0.5
    ci = bootstrap_mean([0.4, 0.5, 0.6, 0.45, 0.55] * 20, n_boot=500)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.hi - ci.lo < 0.1  # tight for n=100 low-variance data
    # deterministic across calls (seeded)
    ci2 = bootstrap_mean([0.4, 0.5, 0.6, 0.45, 0.55] * 20, n_boot=500)
    assert (ci.lo, ci.hi) == (ci2.lo, ci2.hi)
