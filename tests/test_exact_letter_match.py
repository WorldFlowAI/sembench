"""M6's letter-match leg: LongBench-v2 multiple-choice rows score by letter."""

from __future__ import annotations

import pytest

from sembench.quality import answer_letter, exact_letter_match, quality_score
from sembench.traffic_classes import LONGBENCH_V2_MC_CLASS, TRAFFIC_CLASSES


@pytest.mark.parametrize(
    ("reply", "letter"),
    [
        ("B", "B"),
        ("b", "B"),
        ("(C)", "C"),
        ("D.", "D"),
        ("Answer: A", "A"),
        ("Final answer: c", "C"),
        ("The correct choice is B because the narrator says so.", "B"),
        ("I think it is either A or D; going with D.", "A"),
        ("None of these.", None),
        ("", None),
    ],
)
def test_the_letter_is_read_the_way_the_prompt_asked_for_it(reply, letter):
    assert answer_letter(reply) == letter


def test_exact_letter_match_is_binary_and_never_null_for_a_reply():
    assert exact_letter_match("B", ["B"]) == 1.0
    assert exact_letter_match("Answer: B", ["B"]) == 1.0
    assert exact_letter_match("C", ["B"]) == 0.0
    assert exact_letter_match("no idea", ["B"]) == 0.0
    assert exact_letter_match("B", []) is None


def test_token_recall_would_have_misread_a_letter_inside_a_sentence():
    """Why the leg is not scored by the default recall scorer."""
    reply = "The answer is D, since option B is contradicted by the text."
    assert exact_letter_match(reply, ["B"]) == 0.0
    assert quality_score(reply, ["B"]) == 1.0


def test_the_class_is_registered():
    assert LONGBENCH_V2_MC_CLASS in TRAFFIC_CLASSES


def test_a_multiple_choice_row_is_scored_by_letter_end_to_end(tmp_path, monkeypatch):
    """Through the runner: the row carries the letter verdict and no F1/ROUGE."""
    import json

    from sembench import gateway_live
    from sembench.cli import main
    from sembench.schema import WorkloadItem, write_jsonl

    def reply(*, base_url, model, prompt, max_tokens, tenant, template, timeout_seconds):
        text = "Answer: B" if "mc" in prompt else "42"
        return {
            "output_text": text,
            "usage": {"prompt_tokens": 64, "completion_tokens": 4},
            "headers": {},
            "ttft_ms": 5.0,
            "latency_ms": 10.0,
        }

    monkeypatch.setattr(gateway_live, "_chat_completion", reply)
    monkeypatch.setattr("sembench.cli.scrape_all", lambda urls, **kw: [])
    items = [
        WorkloadItem(
            item_id="mc-0",
            dataset="longbench_v2",
            source_id="s-mc-0",
            transform="instruction_variant",
            donor_prompts=[],
            recipient_prompt="recipient-mc Answer with a single letter.",
            answers=["B"],
            metadata={"traffic_class": LONGBENCH_V2_MC_CLASS, "expected_answer_letter": "B"},
        ),
        WorkloadItem(
            item_id="qa-0",
            dataset="fixture",
            source_id="s-qa-0",
            transform="instruction_variant",
            donor_prompts=[],
            recipient_prompt="recipient-qa",
            answers=["42"],
            metadata={},
        ),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, items)
    out = tmp_path / "result.json"
    main(
        [
            "run-live-gateway",
            "--manifest",
            str(manifest),
            "--output",
            str(out),
            "--gateway-url",
            "http://gateway:8000",
            "--model",
            "qwen",
            "--skip-verify",
        ]
    )
    rows = {r["item_id"]: r for r in json.loads(out.read_text())["requests"]}
    assert rows["mc-0"]["quality_score"] == 1.0
    assert rows["mc-0"]["quality_pass"] is True
    assert rows["mc-0"]["quality_f1"] is None
    assert rows["mc-0"]["quality_rouge_l"] is None
    assert rows["qa-0"]["quality_f1"] is not None
