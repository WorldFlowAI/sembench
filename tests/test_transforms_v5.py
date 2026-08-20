"""v5 transform family: verified-paraphrase and restatement classes.

Guarded failure modes: fact preservation under the deterministic rewriter
(digits/entities must survive), the leak-control construction (donor digit
must differ from the recipient digit with a different leading digit so
first-token scoring can never collide), and value-equivalence with digit
disjointness for the restatement classes (the verifier-reject contract).
"""

import json
import re

from sembench.paraphrase import block_overlap_ratio, rewrite_preserving_facts
from sembench.transforms import (
    TRANSFORMS_V5,
    TransformConfig,
    build_workload,
    fixture_records,
)

V5_ONLY = (
    "fact_paraphrase",
    "fact_divergent_paraphrase",
    "arithmetic_restatement",
    "complement_restatement",
)


def _v5_items():
    return build_workload(fixture_records(), TransformConfig(transforms=TRANSFORMS_V5))


def _item(items, transform):
    return next(i for i in items if i.transform == transform)


def _digits(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def test_v5_tuple_extends_v4_and_workload_is_deterministic():
    for name in V5_ONLY:
        assert name in TRANSFORMS_V5

    first = _v5_items()
    second = _v5_items()

    def serialize(items):
        return [json.dumps(i.to_dict(), sort_keys=True) for i in items]

    assert serialize(first) == serialize(second)
    assert {i.transform for i in first} >= set(V5_ONLY)


def test_rewriter_preserves_digits_and_mid_sentence_entities():
    text = (
        "The Meridian review found 47 approvals across 12 million records. "
        "A report from the finance team said the Q3 window was large. "
        "Many teams need new tooling before the audit."
    )
    out = rewrite_preserving_facts(text, seed_key="unit-test")

    assert out != text
    assert sorted(_digits(out)) == sorted(_digits(text))
    assert "Meridian" in out
    assert "Q3" in out
    assert rewrite_preserving_facts(text, seed_key="unit-test") == out


def test_fact_paraphrase_pair_shares_the_probe_number():
    item = _item(_v5_items(), "fact_paraphrase")
    n = str(item.metadata["fact_probe"])

    assert item.answers == [n]
    assert n in item.donor_prompts[0].text
    assert n in item.recipient_prompt
    assert item.metadata["expected_verifier_accept"] is True
    assert item.recipient_prompt != item.donor_prompts[0].text
    assert item.recipient_prompt.startswith("Workspace evidence review")
    assert "number only" in item.input


def test_fact_divergent_donor_number_differs_with_new_leading_digit():
    item = _item(_v5_items(), "fact_divergent_paraphrase")
    n = str(item.metadata["fact_probe"])
    wrong = str(item.metadata["divergent_donor_value"])

    assert item.answers == [n]
    assert wrong != n
    assert wrong[0] != n[0]
    assert wrong in item.donor_prompts[0].text
    assert n not in item.donor_prompts[0].text
    assert n in item.recipient_prompt
    assert item.metadata["expected_verifier_accept"] is False
    assert item.metadata["expected_no_fact_leak"] is True
    assert item.negative_control is False


def test_arithmetic_restatement_is_value_equivalent_but_digit_disjoint():
    item = _item(_v5_items(), "arithmetic_restatement")
    n = item.metadata["fact_probe"]
    a = item.metadata["restated_parts"][0]
    b = item.metadata["restated_parts"][1]

    assert a + b == n
    assert str(n) not in (str(a), str(b))
    assert str(a) in item.donor_prompts[0].text
    assert str(b) in item.donor_prompts[0].text
    assert str(n) in item.recipient_prompt
    assert item.answers == [str(n)]
    assert item.metadata["expected_verifier_accept"] is False
    assert item.metadata["fact_equivalent"] is True


def test_complement_restatement_complements_within_the_stated_total():
    item = _item(_v5_items(), "complement_restatement")
    total = item.metadata["restated_total"]
    failed = item.metadata["fact_probe"]
    passed = total - failed

    assert 0 < failed < total
    assert str(passed) in item.donor_prompts[0].text
    assert str(failed) not in _digits(item.donor_prompts[0].text)
    assert str(failed) in item.recipient_prompt
    assert str(total) in item.donor_prompts[0].text
    assert str(total) in item.recipient_prompt
    assert item.answers == [str(failed)]
    assert item.metadata["expected_verifier_accept"] is False
    assert item.metadata["fact_equivalent"] is True


def _context_body(prompt: str) -> str:
    body = prompt.split("Source material:\n", 1)[1]
    return body.split("\n\nUser request:", 1)[0]


def test_v5_recipients_are_token_divergent_from_donor_bodies():
    """The paraphrase classes must not degrade into near-identical text the
    identity-verified paths would silently serve (the contamination failure
    the corpus guard exists for)."""
    records = fixture_records()
    items = build_workload(records, TransformConfig(transforms=V5_ONLY))
    for item in items:
        donor_sents = set(_context_body(item.donor_prompts[0].text).split(". "))
        recip_sents = set(_context_body(item.recipient_prompt).split(". "))
        shared = donor_sents & recip_sents
        assert len(shared) <= max(1, len(donor_sents) // 4), item.transform


def test_merge_never_decapitalizes_entities():
    text = (
        "The audit passed. IT operations restored the service. "
        "The vendor was chosen. One Medical signed the agreement. "
        "The review closed. There Inc. reported the findings."
    )
    out = rewrite_preserving_facts(text, seed_key="k")

    assert "IT operations" in out
    assert "One Medical" in out
    assert "There Inc." in out


def test_abbreviation_periods_survive_merges():
    text = "Approved by the U.S. The team checked the records daily."
    out = rewrite_preserving_facts(text, seed_key="k")

    assert "U.S." in out


def test_leading_punctuation_stays_in_place():
    text = "Costs grew here (the baseline) as they were checked over."
    out = rewrite_preserving_facts(text, seed_key="k")

    assert "(" not in out or re.search(r"\(\w", out)


def test_v5_invariants_hold_on_the_synthetic_corpus():
    """The fixture corpus is digit-free and repetitive; the real corpora
    are not. Every v5 invariant (probe absent from the opposite side,
    rewrite genuinely divergent) must hold on number-dense enterprise
    prose, not just on the degenerate fixture."""
    from sembench.synthetic import synthetic_records

    records = synthetic_records(sources_per_domain=6)
    items = build_workload(records, TransformConfig(transforms=V5_ONLY))
    assert len(items) == len(records) * len(V5_ONLY)
    for item in items:
        donor_body = _context_body(item.donor_prompts[0].text)
        recip_body = _context_body(item.recipient_prompt)
        assert block_overlap_ratio(donor_body, recip_body) <= 0.4, item.transform
        if item.transform == "fact_divergent_paraphrase":
            n = str(item.metadata["fact_probe"])
            assert n not in item.donor_prompts[0].text
            assert str(item.metadata["divergent_donor_value"]) not in recip_body
        if item.transform == "complement_restatement":
            failed = str(item.metadata["fact_probe"])
            assert failed not in re.findall(r"\d+", donor_body)
        if item.transform == "arithmetic_restatement":
            for part in item.metadata["restated_parts"]:
                assert str(part) not in recip_body
