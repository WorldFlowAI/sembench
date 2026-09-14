"""Round 6, second pass: the holes the round-6 verifier found in round 6.

Each of the four fixes round 6 shipped closed the case it was filed against
and left a door open one inch away. Every test here walks through that inch:

- **M4's audit guard was document-level.** ``_miss_tax_summary`` took one
  boolean for the whole document and then still read "advertised nothing" off
  rows the audit had never spoken about, so on a MIXED-join document 19 of 20
  unaudited pairs priced the miss path while ``miss_tax_source`` asserted the
  population came from the connector audit.
- **An unusable cold reference published 0.0.** A reference from another
  manifest, or one whose rows are all the warm arm, scores no probe at all —
  and the document published ``propagation_contamination_rate: 0.0`` beside
  ``propagation_cold_reference_missing: false``, which is the exact
  0.0-on-a-contaminated-workload the explicit reference exists to prevent.
- **The unlabelled merge still read its own contaminated baseline.** A4 vs A6
  merged without ``--pair`` published the pre-round-6 number, even though
  ``run.baseline_id`` on the document already said the baseline was A4.
- **``--cold-reference-arm`` was recorded as fact.** It defaults to A1, so any
  result file handed in stamped ``propagation_cold_reference_arm: "A1"`` with
  nothing checked, in the one block whose purpose is reference provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sembench.results import (
    COLD_REFERENCE_ARM_UNDECLARED,
    COLD_REFERENCE_FROM_BASELINE_ARM,
    COLD_REFERENCE_FROM_REFERENCE_ARM,
    COLD_REFERENCE_FROM_REFERENCE_ARM_UNDECLARED,
    PROPAGATION_PROBE_CLASS,
    arm_id_of,
    connector_audit_metrics,
    paired_summary,
)
from sembench.schema import RequestMetrics

SAME_DOC = "same_doc_new_instruction"
SERVED = "Donaghadee, County Down"
COLD_ANSWER = "Northern Ireland"


def _row(item_id: str, **kw) -> RequestMetrics:
    fields: dict[str, object] = {
        "item_id": item_id,
        "dataset": "longbench",
        "transform": SAME_DOC,
        "negative_control": False,
        "donor_count": 1,
        "prompt_tokens": 3000,
        "total_blocks": 187,
        "exact_hit_blocks": 0,
        "exact_hit_tokens": 0,
        "semantic_candidate_blocks": 0,
        "semantic_candidate_tokens": 0,
        "semantic_eligible_blocks": 0,
        "semantic_eligible_tokens": 0,
        "arm": "warm",
        "audit_joined": True,
        "traffic_class": SAME_DOC,
        "ttft_ms": 50.0,
    }
    fields.update(kw)
    return RequestMetrics(**fields)


# --------------------------------------------------------------------------
# 1 — M4's audit guard is row-level
# --------------------------------------------------------------------------


def _mixed_join_rows() -> list[RequestMetrics]:
    """One audited pair, nineteen the audit never saw.

    The audit WAS read for this document — every warm row carries
    ``audit_joined``, False on the nineteen the join found no connector events
    for (``MATCH_MISSING``, which is what ``engine_response_id`` and the join
    report exist for). Their ``audit_advertised_tokens`` is null for want of a
    measurement, not because the connector advertised nothing.
    """
    rows = [
        _row("audited", arm="cold", ttft_ms=110.0, audit_joined=None),
        _row("audited", arm="warm", ttft_ms=100.0, audit_joined=True, audit_advertised_tokens=0),
    ]
    for index in range(19):
        item = f"unjoined{index}"
        rows.append(_row(item, arm="cold", ttft_ms=100.0, audit_joined=None))
        rows.append(_row(item, arm="warm", ttft_ms=900.0, audit_joined=False))
    return rows


def test_the_miss_tax_reads_only_the_pairs_the_audit_measured():
    """The document-level guard passes; nineteen of twenty rows are unmeasured.

    Before the row-level filter this published ``miss_tax_ms = 800.0`` over
    ``miss_tax_pairs = 20`` with ``miss_tax_source = 'connector_audit'`` — an
    80x-wrong tax whose source string claimed the population came from an
    audit that had never seen 19 of those requests.
    """
    paired = paired_summary(_mixed_join_rows())

    assert paired is not None
    assert paired["connector_audit_present"] is True
    assert paired["miss_tax_pairs"] == 1
    assert paired["miss_tax_ms"] == -10.0
    assert paired["miss_tax_pairs_not_audited_excluded"] == 19
    assert paired["miss_tax_source"] == "connector_audit"
    assert "audit_joined" in paired["miss_tax_population"]


def test_the_miss_tax_is_empty_when_the_audit_joined_no_warm_row():
    """The audit was read and matched nothing: zero pairs, not a 300 ms tax."""
    rows = []
    for index in range(3):
        item = f"i{index}"
        rows.append(_row(item, arm="cold", ttft_ms=100.0, audit_joined=None))
        rows.append(_row(item, arm="warm", ttft_ms=400.0, audit_joined=False))

    paired = paired_summary(rows)

    assert paired is not None
    assert paired["connector_audit_present"] is True
    assert paired["miss_tax_pairs"] == 0
    assert paired["miss_tax_ms"] is None
    assert paired["miss_tax_pairs_not_audited_excluded"] == 3


def test_the_capture_leg_also_drops_the_pairs_the_audit_never_saw():
    """The class filter runs after the audit filter, not instead of it."""
    rows = [
        _row("nr1", arm="cold", ttft_ms=100.0, traffic_class="no_reuse", audit_joined=None),
        _row(
            "nr1",
            arm="warm",
            ttft_ms=130.0,
            traffic_class="no_reuse",
            audit_advertised_tokens=0,
        ),
        _row("nr2", arm="cold", ttft_ms=100.0, traffic_class="no_reuse", audit_joined=None),
        _row("nr2", arm="warm", ttft_ms=800.0, traffic_class="no_reuse", audit_joined=False),
    ]

    capture = paired_summary(rows, arm_pair="m4_capture")

    assert capture is not None
    assert capture["miss_tax_pairs"] == 1
    assert capture["miss_tax_ms"] == 30.0
    assert capture["miss_tax_pairs_not_audited_excluded"] == 1
    assert capture["miss_tax_pairs_outside_capture_class_excluded"] == 0


# --------------------------------------------------------------------------
# 2 — a supplied reference that scores nothing is not a 0.0
# --------------------------------------------------------------------------


def _contaminated_probe_rows() -> list[RequestMetrics]:
    return [
        _row("parent", arm="cold", ttft_ms=200.0, output_text=SERVED, audit_joined=None),
        _row("parent", arm="warm", ttft_ms=100.0, output_text=SERVED),
        _row(
            "probe",
            arm="cold",
            ttft_ms=200.0,
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id="parent",
            output_text=SERVED,
            audit_joined=None,
        ),
        _row(
            "probe",
            arm="warm",
            ttft_ms=100.0,
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id="parent",
            output_text=SERVED,
        ),
    ]


def _reference_rows(*, item_prefix: str = "", arm: str = "single") -> list[RequestMetrics]:
    return [
        _row(f"{item_prefix}parent", arm=arm, output_text=COLD_ANSWER, audit_joined=None),
        _row(
            f"{item_prefix}probe",
            arm=arm,
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id=f"{item_prefix}parent",
            output_text=COLD_ANSWER,
            audit_joined=None,
        ),
    ]


@pytest.mark.parametrize(
    ("label", "reference"),
    [
        ("another manifest", _reference_rows(item_prefix="other-")),
        ("every row stamped warm", _reference_rows(arm="warm")),
    ],
)
def test_a_reference_that_scores_no_probe_publishes_no_rate(label, reference):
    """A fully contaminated probe set, and a reference that answered none of it.

    Both shapes score zero probes: the wrong-manifest reference holds no row
    for this item_id, and the all-warm reference is skipped by
    ``_reference_rows_by_item`` (a merged document's warm rows are the
    treatment arm, not a cold answer). Before the guard the document published
    0.0 with ``propagation_cold_reference_missing: false`` beside it.
    """
    paired = paired_summary(
        _contaminated_probe_rows(),
        arm_pair="m7_propagation",
        cold_reference=reference,
        cold_reference_arm="A1",
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None, label
    assert paired["propagation_cold_reference_unusable"] is True
    assert paired["propagation_cold_reference_missing"] is False
    assert paired["propagation_contamination_rate"] is None
    assert paired["propagation_contamination_numerator"] is None
    assert paired["propagation_contamination_rate_scored_only"] is None
    assert paired["propagation_probes_without_cold_reference"] == 1


def test_a_reference_that_scores_a_probe_is_not_flagged_unusable():
    paired = paired_summary(
        _contaminated_probe_rows(),
        arm_pair="m7_propagation",
        cold_reference=_reference_rows(),
        cold_reference_arm="A1",
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None
    assert paired["propagation_cold_reference_unusable"] is False
    assert paired["propagation_contamination_numerator"] == 1


def test_a_reference_that_declares_no_arm_is_accepted_and_named_as_such():
    paired = paired_summary(
        _contaminated_probe_rows(),
        arm_pair="m7_propagation",
        cold_reference=_reference_rows(),
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None
    assert (
        paired["propagation_cold_reference_source"] == COLD_REFERENCE_FROM_REFERENCE_ARM_UNDECLARED
    )
    assert paired["propagation_cold_reference_arm"] == COLD_REFERENCE_ARM_UNDECLARED
    assert paired["propagation_contamination_numerator"] == 1


# --------------------------------------------------------------------------
# 3 — an unlabelled merge reads run.baseline_id, not "assume A1"
# --------------------------------------------------------------------------


def test_an_unlabelled_merge_of_a_non_cold_baseline_has_no_cold_reference():
    """No ``--pair``, but the document says its baseline arm was A4.

    ``--pair`` is optional and unlabelled is the default, so an operator who
    merges A4 against A6 and forgets ``--pair m7_propagation`` got the
    pre-round-6 number back: 0.0 on a workload contaminated end to end.
    """
    paired = paired_summary(
        _contaminated_probe_rows(),
        baseline_arm="A4 conn_span",
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None
    assert paired["propagation_cold_reference_missing"] is True
    assert paired["propagation_contamination_rate"] is None
    assert paired["propagation_contamination_numerator"] is None
    # The arm-vs-arm diagnostic is still there, under the name that cannot be
    # read as section 4's metric.
    assert paired["propagation_contamination_rate_vs_baseline_arm"] == 0.0


def test_an_unlabelled_merge_of_the_a1_baseline_names_the_arm_it_read():
    paired = paired_summary(
        _contaminated_probe_rows(),
        baseline_arm="A1 stock_pc",
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None
    assert paired["propagation_cold_reference_arm"] == "A1"
    assert paired["propagation_cold_reference_source"] == COLD_REFERENCE_FROM_BASELINE_ARM
    assert paired["propagation_cold_reference_missing"] is False


def test_a_backend_id_names_the_arm_it_spells_longest():
    """``conn_span_noaudit`` is A5, even though A4's label is a prefix of it."""
    assert arm_id_of("A4 conn_span") == "A4"
    assert arm_id_of("conn_span_noaudit") == "A5"
    assert arm_id_of("stock_pc") == "A1"
    assert arm_id_of("cold-run-3") == ""
    assert arm_id_of("") == ""


# --------------------------------------------------------------------------
# 4 — merge-results checks the reference document
# --------------------------------------------------------------------------


def _arm_document(tmp_path: Path, name: str, *, arm: str, rows, backend_id: str = "") -> str:
    payload = {
        "run": {
            "run_id": name,
            "arm": arm,
            "manifest_sha256": "deadbeef",
            "backend_id": backend_id,
            "baseline_id": backend_id,
        },
        "config": {"manifest_class_counts": {PROPAGATION_PROBE_CLASS: 50}},
        "engine": None,
        "requests": [row.to_dict() for row in rows],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _m7_arms(tmp_path: Path) -> tuple[str, str]:
    rows = _contaminated_probe_rows()
    cold = _arm_document(
        tmp_path,
        "cold",
        arm="cold",
        backend_id="A4 conn_span",
        rows=[row for row in rows if row.arm == "cold"],
    )
    warm = _arm_document(
        tmp_path,
        "warm",
        arm="warm",
        backend_id="A6 conn_span_nomitigation",
        rows=[row for row in rows if row.arm == "warm"],
    )
    return cold, warm


def _merge(argv: list[str]) -> None:
    from sembench.cli import main

    main(argv)


def test_merge_results_refuses_a_reference_that_contradicts_its_arm(tmp_path, capsys):
    cold, warm = _m7_arms(tmp_path)
    reference = _arm_document(
        tmp_path, "reference", arm="single", backend_id="A4 conn_span", rows=_reference_rows()
    )

    with pytest.raises(SystemExit) as excinfo:
        _merge(
            [
                "merge-results",
                "--cold",
                cold,
                "--warm",
                warm,
                "--output",
                str(tmp_path / "merged.json"),
                "--pair",
                "m7_propagation",
                "--cold-reference",
                reference,
            ]
        )

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["merged"] is False
    assert any("--cold-reference-arm" in problem for problem in payload["arm_label_conflicts"])


def test_merge_results_refuses_a_reference_from_another_manifest(tmp_path, capsys):
    cold, warm = _m7_arms(tmp_path)
    reference = _arm_document(
        tmp_path,
        "reference",
        arm="single",
        backend_id="A1 stock_pc",
        rows=_reference_rows(item_prefix="other-"),
    )

    with pytest.raises(SystemExit) as excinfo:
        _merge(
            [
                "merge-results",
                "--cold",
                cold,
                "--warm",
                warm,
                "--output",
                str(tmp_path / "merged.json"),
                "--pair",
                "m7_propagation",
                "--cold-reference",
                reference,
            ]
        )

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert any("--cold-reference" in problem for problem in payload["arm_label_conflicts"])


def test_merge_results_records_the_arm_the_reference_declares(tmp_path, capsys):
    """The flag is an assertion; the document's own backend id is evidence."""
    cold, warm = _m7_arms(tmp_path)
    reference = _arm_document(tmp_path, "reference", arm="single", rows=_reference_rows())
    output = tmp_path / "merged.json"

    _merge(
        [
            "merge-results",
            "--cold",
            cold,
            "--warm",
            warm,
            "--output",
            str(output),
            "--pair",
            "m7_propagation",
            "--cold-reference",
            reference,
        ]
    )
    capsys.readouterr()

    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["config"]["cold_reference_arm"] is None
    assert (
        merged["paired"]["propagation_cold_reference_source"]
        == COLD_REFERENCE_FROM_REFERENCE_ARM_UNDECLARED
    )
    assert merged["paired"]["propagation_contamination_rate"] == pytest.approx(1 / 50)


def test_an_unlabelled_a4_vs_a6_merge_publishes_no_contamination_rate(tmp_path, capsys):
    """End to end: no ``--pair``, and ``run.baseline_id`` carries the A4."""
    cold, warm = _m7_arms(tmp_path)
    output = tmp_path / "merged.json"

    _merge(["merge-results", "--cold", cold, "--warm", warm, "--output", str(output)])
    capsys.readouterr()

    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["run"]["baseline_id"] == "A4 conn_span"
    assert merged["paired"]["propagation_cold_reference_missing"] is True
    assert merged["paired"]["propagation_contamination_rate"] is None


def test_a_labelled_reference_survives_the_check(tmp_path, capsys):
    cold, warm = _m7_arms(tmp_path)
    reference = _arm_document(
        tmp_path, "reference", arm="single", backend_id="A1 stock_pc", rows=_reference_rows()
    )
    output = tmp_path / "merged.json"

    _merge(
        [
            "merge-results",
            "--cold",
            cold,
            "--warm",
            warm,
            "--output",
            str(output),
            "--pair",
            "m7_propagation",
            "--cold-reference",
            reference,
        ]
    )
    capsys.readouterr()

    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["config"]["cold_reference_arm"] == "A1"
    assert (
        merged["paired"]["propagation_cold_reference_source"] == COLD_REFERENCE_FROM_REFERENCE_ARM
    )
    assert merged["paired"]["propagation_cold_reference_unusable"] is False


# --------------------------------------------------------------------------
# 5 — the pinned stratum boundary reports what it selected
# --------------------------------------------------------------------------


def _wrapper_row(item_id: str, *, rank: int | None) -> RequestMetrics:
    return _row(
        item_id,
        wrapper_id=None if rank is None else f"w{rank + 1}-wrapper",
        wrapper_rank=rank,
        audit_semantic_lookup_hit=True,
        audit_lookup_hit_boundary=1024,
        audit_advertised_tokens=0,
    )


def test_the_head_share_is_measured_on_the_document_not_assumed():
    """``SHARED_WRAPPER_MAX_RANK`` is pinned to one manifest; this is the check."""
    majority = connector_audit_metrics(
        [
            _wrapper_row("a", rank=0),
            _wrapper_row("b", rank=1),
            _wrapper_row("c", rank=4),
            _wrapper_row("legacy", rank=None),
        ]
    )

    assert majority["wrapper_stratum_head_share"] == pytest.approx(2 / 3)
    assert majority["wrapper_stratum_head_is_majority"] is True


def test_the_head_share_flags_a_manifest_the_constant_no_longer_describes():
    minority = connector_audit_metrics(
        [_wrapper_row("a", rank=0), _wrapper_row("c", rank=4), _wrapper_row("d", rank=6)]
    )

    assert minority["wrapper_stratum_head_share"] == pytest.approx(1 / 3)
    assert minority["wrapper_stratum_head_is_majority"] is False
    assert "wrapper_stratum_head_share" in minority["wrapper_stratum_rule"]


def test_the_head_share_is_null_when_no_row_declared_a_wrapper():
    unranked = connector_audit_metrics([_wrapper_row("legacy", rank=None)])

    assert unranked["wrapper_stratum_head_share"] is None
    assert unranked["wrapper_stratum_head_is_majority"] is None


# --------------------------------------------------------------------------
# The docs are part of the contract
# --------------------------------------------------------------------------

# The keys THIS pass added, named. A hardcoded list can only assert what the
# author remembered, so it is no longer the doc contract: that is
# ``test_metrics_round6_third_pass.test_every_key_a_written_document_emits_is_documented``,
# which derives the key set from a document ``write_result`` actually wrote.
# This stays as the named check for the five keys the second pass introduced.
SECOND_PASS_KEYS = (
    "miss_tax_pairs_not_audited_excluded",
    "propagation_cold_reference_unusable",
    "reference_arm_undeclared",
    "wrapper_stratum_head_share",
    "wrapper_stratum_head_is_majority",
)

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("key", SECOND_PASS_KEYS)
def test_every_key_this_pass_adds_is_documented(key):
    assert key in (_ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8"), key
