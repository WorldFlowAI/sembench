import json
from pathlib import Path

import pytest

from sembench.cli import main
from sembench.collision_audit import audit_manifest_items
from sembench.schema import DonorPrompt, WorkloadItem, read_jsonl, write_jsonl

FIXTURE_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "fixture-v1.jsonl"


def _item(
    item_id: str,
    source_id: str,
    donor_text: str,
    recipient: str,
    *,
    negative: bool = False,
    donor_meta: dict | None = None,
) -> WorkloadItem:
    return WorkloadItem(
        item_id=item_id,
        dataset="t",
        source_id=source_id,
        transform="negative_control" if negative else "exact_repeat",
        donor_prompts=[
            DonorPrompt(
                donor_id=f"{item_id}-d0",
                text=donor_text,
                label="base_prompt",
                metadata=donor_meta or {},
            )
        ],
        recipient_prompt=recipient,
        negative_control=negative,
    )


WORDS_A = " ".join(f"alpha{i}" for i in range(64))
WORDS_B = " ".join(f"beta{i}" for i in range(64))
WORDS_C = " ".join(f"gamma{i}" for i in range(64))


def test_clean_manifest_passes():
    items = [
        _item("a1", "src-a", WORDS_A, WORDS_A + " question one"),
        _item("b1", "src-b", WORDS_B, WORDS_B + " question two"),
    ]
    report = audit_manifest_items(items, block_size=16)
    assert report.passed
    assert report.boilerplate_blocks == 0


def test_planted_cross_source_phantom_fails():
    # src-b's recipient embeds src-a's document verbatim: single-origin
    # collision, must be flagged regardless of corpus size.
    items = [
        _item("a1", "src-a", WORDS_A, WORDS_A + " question one"),
        _item("b1", "src-b", WORDS_B, WORDS_A + " unrelated question"),
    ]
    report = audit_manifest_items(items, block_size=16)
    assert not report.passed
    assert report.violations[0]["kind"] == "phantom_cross_source"
    assert report.violations[0]["item_id"] == "b1"


def test_negative_control_donor_origin_is_not_a_phantom():
    # Negative control for src-b borrows src-a's doc as its donor and says so
    # in metadata; src-a's own item sharing blocks with that donor is fine.
    items = [
        _item("a1", "src-a", WORDS_A, WORDS_A + " question one"),
        _item(
            "b-neg",
            "src-b",
            WORDS_A,
            WORDS_B + " unrelated request",
            negative=True,
            donor_meta={"negative_source_id": "src-a"},
        ),
    ]
    report = audit_manifest_items(items, block_size=16)
    assert report.passed


def test_negative_control_overlapping_its_own_donor_fails():
    items = [
        _item(
            "b-neg",
            "src-b",
            WORDS_A,
            WORDS_A + " supposedly unrelated",
            negative=True,
            donor_meta={"negative_source_id": "src-a"},
        ),
    ]
    report = audit_manifest_items(items, block_size=16)
    assert not report.passed
    assert report.violations[0]["kind"] == "negative_control_overlap"


def test_multi_origin_boilerplate_is_classified_not_flagged():
    preamble = " ".join(f"tmpl{i}" for i in range(32))
    items = [
        _item("a1", "src-a", preamble + " " + WORDS_A, preamble + " " + WORDS_A + " q"),
        _item("b1", "src-b", preamble + " " + WORDS_B, preamble + " " + WORDS_B + " q"),
        _item("c1", "src-c", preamble + " " + WORDS_C, preamble + " " + WORDS_C + " q"),
    ]
    report = audit_manifest_items(items, block_size=16)
    assert report.passed
    assert report.boilerplate_blocks > 0


def test_frozen_fixture_manifest_is_clean():
    report = audit_manifest_items(read_jsonl(FIXTURE_MANIFEST), block_size=16)
    assert report.passed
    assert report.boilerplate_blocks == 1  # the shared workspace preamble


def test_cli_audit_exit_codes(tmp_path: Path, capsys):
    clean = tmp_path / "clean.jsonl"
    write_jsonl(
        clean,
        [
            _item("a1", "src-a", WORDS_A, WORDS_A + " q"),
            _item("b1", "src-b", WORDS_B, WORDS_B + " q"),
        ],
    )
    main(["audit-manifest", "--manifest", str(clean)])
    assert json.loads(capsys.readouterr().out)["passed"] is True

    dirty = tmp_path / "dirty.jsonl"
    write_jsonl(
        dirty,
        [
            _item("a1", "src-a", WORDS_A, WORDS_A + " q"),
            _item("b1", "src-b", WORDS_B, WORDS_A + " q"),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main(["audit-manifest", "--manifest", str(dirty)])
    assert excinfo.value.code == 1
