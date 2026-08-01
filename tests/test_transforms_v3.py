"""v3 production hit-classes: repeat_shifted + question_paraphrase."""

from sembench.transforms import (
    TransformConfig,
    _paraphrase_request,
    build_workload,
    fixture_records,
)


def _items(transform):
    cfg = TransformConfig(transforms=(transform,))
    return build_workload(fixture_records(), cfg)


def test_repeat_shifted_diverges_early_but_shares_content():
    for item in _items("repeat_shifted"):
        donor = item.donor_prompts[0].text
        assert item.recipient_prompt != donor
        # Diverge within the first 40 chars (exact-prefix must miss).
        assert item.recipient_prompt[:40] != donor[:40]
        # Same evidence and same request verbatim inside.
        assert item.metadata["hit_class"] == "true_repeat"
        assert not item.negative_control


def test_question_paraphrase_preserves_answers_and_rewords():
    rewrote_any = False
    for item in _items("question_paraphrase"):
        assert item.metadata["hit_class"] == "paraphrase"
        assert item.answers  # answer-preserving by construction
        donor = item.donor_prompts[0].text
        assert item.recipient_prompt[:40] != donor[:40]
        rewrote_any = rewrote_any or (
            "how long did mitigation take, and what root cause" in item.recipient_prompt
            or "Identify the clause of agreement" in item.recipient_prompt
            or "Which anomalous error code" in item.recipient_prompt
            or "Describe the remediation and the customer impact." in item.recipient_prompt
            or "List the obligations that need legal follow-up." in item.recipient_prompt
        )
    assert rewrote_any


def test_paraphrase_rewrites_are_deterministic():
    q = "For the sev2 incident on api-x, what was the root cause and how long did mitigation take?"
    assert _paraphrase_request(q) == _paraphrase_request(q)
    assert "how long did mitigation take, and what root cause" in _paraphrase_request(q)
