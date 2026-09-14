"""Round 6: the four defects round 5's adversarial verifier found.

Every one of them publishes a number that looks like a measurement:

- **M4 without an audit join.** ``_miss_tax_summary`` had no joined guard, and
  its population filter collapsed "no audit row at all" (None) with "the
  connector looked and supplied nothing" (0). With the audit missing, every
  pair became the missed population and the whole cold-warm TTFT delta was
  republished as a tax, while every sibling metric on the same document said
  the audit was absent.
- **M7 against the merge baseline.** Section 4 compares the treatment answer
  against the *cold* (A1) output. The ``m7_propagation`` pair's baseline is
  A4, which can be contaminated on the same probe; when both sides drift
  together the rate reads 0.0 and the contamination disappears.
- **M1 blended only.** Section 4: "Do not report a single blended alignment
  number. Report it separately for the shared-wrapper stratum and the ad-hoc
  stratum." The manifest carries ``wrapper_id`` / ``wrapper_rank`` and nothing
  read them.
- **The capture leg's population.** Section 4 defines M4's capture cost as the
  A3 -> A4 delta **on no_reuse items**; the leg was taken over every
  non-advertising pair, so same-document traffic that merely failed to
  advertise priced the capture path.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sembench.results import (
    AD_HOC_WRAPPER_STRATUM,
    COLD_REFERENCE_ARM_UNDECLARED,
    COLD_REFERENCE_FROM_BASELINE_ARM,
    COLD_REFERENCE_FROM_REFERENCE_ARM,
    NO_REUSE_CLASS,
    PROPAGATION_PROBE_CLASS,
    SHARED_WRAPPER_MAX_RANK,
    SHARED_WRAPPER_STRATUM,
    UNSTRATIFIED_WRAPPER,
    connector_audit_metrics,
    paired_summary,
    wrapper_stratum_of,
)
from sembench.schema import RequestMetrics, WorkloadItem, manifest_expectations

SAME_DOC = "same_doc_new_instruction"


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
# 1 — M4 is null without an audit join
# --------------------------------------------------------------------------


def test_the_miss_tax_is_null_when_no_audit_was_joined():
    """Twenty pairs, cold 400 ms, warm 100 ms, and no audit anywhere.

    Without the guard every pair passes the "advertised nothing" filter
    (``audit_advertised_tokens`` is null because nobody looked), the whole
    -300 ms delta is published as ``miss_tax_ms``, and the sign says the
    connector's miss path is three times FASTER than the baseline.
    """
    rows: list[RequestMetrics] = []
    for index in range(20):
        item = f"i{index}"
        rows.append(_row(item, arm="cold", ttft_ms=400.0, audit_joined=None))
        rows.append(_row(item, arm="warm", ttft_ms=100.0, audit_joined=None))

    paired = paired_summary(rows)

    assert paired is not None
    assert paired["connector_audit_present"] is False
    assert paired["miss_tax_ms"] is None
    assert paired["miss_tax_ms_median_of_differences"] is None
    assert paired["miss_tax_ms_median_of_differences_ci"] is None
    assert paired["miss_tax_warm_ttft_p50_ms"] is None
    assert paired["miss_tax_cold_ttft_p50_ms"] is None
    assert paired["miss_tax_pairs"] is None
    assert "could not be identified" in paired["miss_tax_source"]


def test_the_miss_tax_still_reads_the_gap_when_the_audit_was_joined():
    """The guard nulls an unmeasured tax, not a measured one."""
    rows = [
        _row("i1", arm="cold", ttft_ms=100.0, audit_joined=None),
        _row("i1", arm="warm", ttft_ms=130.0, audit_advertised_tokens=0),
    ]

    paired = paired_summary(rows)

    assert paired is not None
    assert paired["miss_tax_ms"] == 30.0
    assert paired["miss_tax_pairs"] == 1
    assert paired["miss_tax_source"] == "connector_audit"


# --------------------------------------------------------------------------
# 2 — M7's cold reference is A1, not the merge baseline
# --------------------------------------------------------------------------

SERVED = "Donaghadee, County Down"
COLD_ANSWER = "Northern Ireland"


def _contaminated_probe_rows() -> list[RequestMetrics]:
    """A4 (the merge baseline) and A6 (the treatment) both drifted.

    The probe repeats ``parent`` verbatim. In the treatment arm the parent was
    served approximate KV and answered ``SERVED``; the probe answers ``SERVED``
    too. The baseline arm here is A4, which is contaminated on the same probe
    and answers ``SERVED`` as well — so a comparison against it sees no
    difference at all.
    """
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


def _a1_reference_rows() -> list[RequestMetrics]:
    return [
        _row("parent", arm="single", output_text=COLD_ANSWER, audit_joined=None),
        _row(
            "probe",
            arm="single",
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id="parent",
            output_text=COLD_ANSWER,
            audit_joined=None,
        ),
    ]


def test_m7_reads_the_cold_reference_arm_not_the_merge_baseline():
    """Both merged arms drift to the served answer; A1 did not.

    Against the merge baseline the probe's answer is exactly as close to the
    served output as to the "cold" one, the strict comparison declines to call
    it propagation, and the rate reads 0.0 on a workload that is contaminated
    end to end.
    """
    paired = paired_summary(
        _contaminated_probe_rows(),
        arm_pair="m7_propagation",
        cold_reference=_a1_reference_rows(),
        cold_reference_arm="A1",
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None
    assert paired["propagation_cold_reference_arm"] == "A1"
    assert paired["propagation_cold_reference_source"] == COLD_REFERENCE_FROM_REFERENCE_ARM
    assert paired["propagation_cold_reference_missing"] is False
    assert paired["propagation_contamination_rate"] != 0.0
    assert paired["propagation_contamination_rate"] == pytest.approx(1 / 50)
    assert paired["propagation_contamination_numerator"] == 1
    # The number the defect published, kept under a name nobody can mistake
    # for section 4's metric.
    assert paired["propagation_contamination_rate_vs_baseline_arm"] == 0.0


def test_m7_is_null_on_a_propagation_document_with_no_cold_reference():
    paired = paired_summary(
        _contaminated_probe_rows(),
        arm_pair="m7_propagation",
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None
    assert paired["propagation_cold_reference_missing"] is True
    assert paired["propagation_cold_reference_source"] is None
    assert paired["propagation_contamination_rate"] is None
    assert paired["propagation_contamination_numerator"] is None
    assert paired["propagation_contamination_rate_scored_only"] is None
    assert paired["propagation_probes_without_cold_reference"] == 1
    assert paired["propagation_contamination_rate_vs_baseline_arm"] == 0.0


def test_an_unlabelled_join_that_declares_no_arm_publishes_no_rate():
    """A join that names no arm ANYWHERE publishes no contamination rate.

    Round 6, second pass (finding 1): this branch used to read the document's
    own cold twin, which is right for an A1-vs-A4 merge and wrong for exactly
    the A4-vs-A6 merge it cannot be told apart from — and it is the DEFAULT
    branch, because ``--backend-id`` defaults to empty. Publishing takes an
    affirmative A1 signal now; the arm reads ``undeclared``, which names the
    remedy without asserting that A1 was consulted.
    """
    paired = paired_summary(_contaminated_probe_rows())

    assert paired is not None
    assert paired["propagation_cold_reference_arm"] == COLD_REFERENCE_ARM_UNDECLARED
    assert paired["propagation_cold_reference_source"] is None
    assert paired["propagation_cold_reference_missing"] is True
    assert paired["propagation_contamination_rate"] is None
    assert paired["propagation_contamination_numerator"] is None
    # The arm-vs-arm diagnostic stays, under the name that is not the metric.
    assert paired["propagation_contamination_rate_vs_baseline_arm"] == 0.0


def test_an_unlabelled_join_whose_baseline_id_names_a1_still_reads_its_cold_arm():
    """The ordinary A1-vs-A4 merge keeps its number without a reference run —
    it just has to say so, which ``--backend-id A1 stock_pc`` does."""
    paired = paired_summary(_contaminated_probe_rows(), baseline_arm="A1 stock_pc")

    assert paired is not None
    assert paired["propagation_cold_reference_source"] == COLD_REFERENCE_FROM_BASELINE_ARM
    assert paired["propagation_cold_reference_arm"] == "A1"
    assert paired["propagation_cold_reference_missing"] is False
    assert paired["propagation_contamination_rate"] == 0.0


def test_a_reference_row_at_another_stream_position_is_not_a_reference():
    """The reference run must have replayed the same stream: a row at a
    different position answered a different prompt sequence."""
    reference = [
        _row("parent", arm="single", output_text=COLD_ANSWER, audit_joined=None),
        _row(
            "probe",
            arm="single",
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id="parent",
            output_text=COLD_ANSWER,
            stream_position=9,
            audit_joined=None,
        ),
    ]
    rows = [
        replace(row, stream_position=3) if row.item_id == "probe" else row
        for row in _contaminated_probe_rows()
    ]

    paired = paired_summary(
        rows,
        arm_pair="m7_propagation",
        cold_reference=reference,
        cold_reference_arm="A1",
    )

    assert paired is not None
    assert paired["propagation_probes_reference_position_mismatched"] == 1
    assert paired["propagation_probes_without_cold_reference"] == 1
    # Round 6, second pass: the reference answered nothing, so the rate is
    # null rather than a 0 over the probe set (see the unusable-reference
    # tests below).
    assert paired["propagation_contamination_numerator"] is None
    assert paired["propagation_cold_reference_unusable"] is True


# --------------------------------------------------------------------------
# 3 — M1 is stratified by wrapper popularity
# --------------------------------------------------------------------------


def _wrapper_row(item_id: str, *, rank: int | None, advertised: int, **kw) -> RequestMetrics:
    wrapper = None if rank is None else f"w{rank + 1}-wrapper"
    return _row(
        item_id,
        wrapper_id=wrapper,
        wrapper_rank=rank,
        audit_semantic_lookup_hit=True,
        audit_lookup_hit_boundary=1024,
        audit_advertised_tokens=advertised,
        audit_observed_boundary=1024 if advertised else None,
        **kw,
    )


def test_the_per_stratum_alignment_numerators_sum_to_the_blended_numerator():
    """The split explains the blended number; it cannot change it."""
    rows = [
        _wrapper_row("w1a", rank=0, advertised=3776),
        _wrapper_row("w1b", rank=0, advertised=0),
        _wrapper_row("w2a", rank=1, advertised=3776),
        _wrapper_row("w6a", rank=5, advertised=3776),
        _wrapper_row("w8a", rank=7, advertised=0),
        _wrapper_row("legacy", rank=None, advertised=3776),
    ]

    metrics = connector_audit_metrics(rows, manifest_class_counts={SAME_DOC: 6})
    strata = metrics["alignment_by_wrapper_stratum"]

    assert set(strata) == {SHARED_WRAPPER_STRATUM, AD_HOC_WRAPPER_STRATUM, UNSTRATIFIED_WRAPPER}
    for key in ("alignment_given_match_numerator", "alignment_given_opportunity_numerator"):
        assert sum(stratum[key] for stratum in strata.values()) == metrics[key]
    for key in ("alignment_given_match_denominator",):
        assert sum(stratum[key] for stratum in strata.values()) == metrics[key]
    # And the split is a real split: two head wrappers, two tail ones.
    assert strata[SHARED_WRAPPER_STRATUM]["rows_considered"] == 3
    assert strata[AD_HOC_WRAPPER_STRATUM]["rows_considered"] == 2
    assert strata[UNSTRATIFIED_WRAPPER]["rows_considered"] == 1
    assert strata[SHARED_WRAPPER_STRATUM]["alignment_given_match"] == pytest.approx(2 / 3)
    assert strata[AD_HOC_WRAPPER_STRATUM]["alignment_given_match"] == 0.5


def test_the_stratum_boundary_is_the_zipf_head():
    assert SHARED_WRAPPER_MAX_RANK == 1
    assert wrapper_stratum_of(_wrapper_row("a", rank=0, advertised=0)) == SHARED_WRAPPER_STRATUM
    assert wrapper_stratum_of(_wrapper_row("b", rank=1, advertised=0)) == SHARED_WRAPPER_STRATUM
    assert wrapper_stratum_of(_wrapper_row("c", rank=2, advertised=0)) == AD_HOC_WRAPPER_STRATUM
    assert wrapper_stratum_of(_wrapper_row("d", rank=None, advertised=0)) == UNSTRATIFIED_WRAPPER


def test_the_wrapper_identity_is_stamped_from_the_manifest():
    item = WorkloadItem(
        item_id="sd-086-recip",
        dataset="longbench-hotpotqa",
        source_id="doc-1",
        transform=SAME_DOC,
        donor_prompts=[],
        recipient_prompt="recipient",
        metadata={"traffic_class": SAME_DOC, "wrapper_id": "w2-retrieval", "wrapper_rank": 1},
    )

    stamped = manifest_expectations(item)

    assert stamped["wrapper_id"] == "w2-retrieval"
    assert stamped["wrapper_rank"] == 1
    assert RequestMetrics(**{**_row("x").to_dict(), **stamped}).wrapper_rank == 1


def test_the_stratum_split_is_null_without_an_audit_join():
    rows = [_row("i1", audit_joined=None, wrapper_rank=0)]

    assert connector_audit_metrics(rows)["alignment_by_wrapper_stratum"] is None


# --------------------------------------------------------------------------
# 4 — the capture leg is the no_reuse population
# --------------------------------------------------------------------------


def _miss_pair(item_id: str, *, traffic_class: str, cold: float, warm: float):
    return [
        _row(item_id, arm="cold", ttft_ms=cold, traffic_class=traffic_class, audit_joined=None),
        _row(
            item_id,
            arm="warm",
            ttft_ms=warm,
            traffic_class=traffic_class,
            audit_advertised_tokens=0,
        ),
    ]


def test_the_capture_leg_reads_only_no_reuse_pairs():
    """Section 4: "capture cost: A3 -> A4 delta on no_reuse items".

    A same-document request that failed to advertise is a miss, not a capture:
    folding it in prices the capture path with traffic that had a donor.
    """
    rows = [
        *_miss_pair("nr1", traffic_class=NO_REUSE_CLASS, cold=100.0, warm=130.0),
        *_miss_pair("sd1", traffic_class=SAME_DOC, cold=100.0, warm=500.0),
    ]

    capture = paired_summary(rows, arm_pair="m4_capture")
    unlabelled = paired_summary(rows)

    assert capture is not None and unlabelled is not None
    assert capture["miss_tax_pairs"] == 1
    assert capture["miss_tax_ms"] == 30.0
    assert capture["miss_tax_pairs_outside_capture_class_excluded"] == 1
    assert NO_REUSE_CLASS in capture["miss_tax_population"]
    # The unlabelled join keeps section 4's plain "supplied == 0" population.
    assert unlabelled["miss_tax_pairs"] == 2
    assert unlabelled["miss_tax_pairs_outside_capture_class_excluded"] == 0


# --------------------------------------------------------------------------
# merge-results carries the reference arm
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


def test_merge_results_takes_the_cold_reference_arm(tmp_path, capsys):
    from sembench.cli import main

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
    reference = _arm_document(
        tmp_path, "reference", arm="single", backend_id="A1 stock_pc", rows=_a1_reference_rows()
    )
    output = tmp_path / "merged.json"

    main(
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
    assert merged["config"]["cold_reference_result"] == reference
    assert merged["config"]["cold_reference_arm"] == "A1"
    assert (
        merged["paired"]["propagation_cold_reference_source"] == COLD_REFERENCE_FROM_REFERENCE_ARM
    )
    assert merged["paired"]["propagation_contamination_rate"] == pytest.approx(1 / 50)


# --------------------------------------------------------------------------
# The docs are part of the contract
# --------------------------------------------------------------------------

ROUND6_KEYS = (
    "miss_tax_source",
    "miss_tax_population",
    "miss_tax_pairs_outside_capture_class_excluded",
    "propagation_cold_reference_arm",
    "propagation_cold_reference_source",
    "propagation_cold_reference_missing",
    "propagation_contamination_rate_vs_baseline_arm",
    "propagation_probes_without_cold_reference",
    "propagation_probes_reference_position_mismatched",
    "alignment_by_wrapper_stratum",
    "wrapper_stratum_rule",
    "wrapper_id",
    "wrapper_rank",
)

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("key", ROUND6_KEYS)
def test_every_key_this_round_adds_is_documented(key):
    assert key in (_ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8"), key


def test_the_readme_documents_the_new_merge_flag():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "--cold-reference" in readme
    assert "--cold-reference-arm" in readme
