"""The metrics exactly as the phase-0 execution plan's section 4 defines them.

Section 4 is the authority for M1, M2 and M7, and every formula it states is
quoted beside the test that pins it. The failures this file exists to prevent
are the ones that look like measurements:

- a manifest-side input nothing fills, so the join has only half its key and
  every rate computed from it is null or wrong (blocker 1);
- a denominator that silently includes an arm the connector never ran in, so
  every rate is halved for free (blocker 2);
- one blended alignment number where section 4 names three, with no
  numerator, no denominator and no miss diagnosis (blocker 3);
- M2's headline carrying the request-count rate under the token-weighted
  formula's name (blocker 4);
- M7 reported as a per-row cached-tokens heuristic rather than the cross-arm
  answer comparison section 4 defines (blocker 5);
- a control gate on the mean, which one outlier pair can flip (blocker 6).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sembench import gateway_live
from sembench.cli import main
from sembench.connector_audit import (
    MISS_BELOW_MIN_SEMANTIC_SPAN,
    MISS_DONOR_NOT_CAPTURED,
    MISS_DONOR_TOO_SHORT,
    MISS_TRUE_MISALIGNMENT,
    MISS_UNCLASSIFIED,
    engine_id_echoes_sent,
    join_audit_file,
    request_id_echo_report,
)
from sembench.gateway_live import LiveGatewayConfig, run_live_gateway
from sembench.results import (
    ALIGNMENT_OPPORTUNITY_CLASSES,
    PROPAGATION_PROBE_CLASS,
    TRAFFIC_CLASSES,
    aggregate_metrics,
    auditable_rows,
    connector_audit_metrics,
    paired_summary,
)
from sembench.schema import RequestMetrics, WorkloadItem, manifest_expectations, write_jsonl

GATEWAY = "http://gateway:8000"


# --------------------------------------------------------------------------
# Fixtures: manifest items shaped like the real phase-0 stream, and an audit
# stream shaped like the connector's.
# --------------------------------------------------------------------------


def _item(
    item_id: str,
    *,
    traffic_class: str,
    supplied: int | None = None,
    target_start: int | None = None,
    parent: str | None = None,
) -> WorkloadItem:
    """One manifest row with the metadata phase0-stream-b.jsonl carries.

    The key names are the real ones: `expected_supplied_tokens`,
    `expected_span_target_start`, `traffic_class` and `parent_item_id` all sit
    under `metadata`, which is what `sembench.manifest.v1` requires so that
    `WorkloadItem.from_dict` loads a new manifest unchanged.
    """
    return WorkloadItem(
        item_id=item_id,
        dataset="longbench-hotpotqa",
        source_id=f"doc-{item_id}",
        transform=traffic_class,
        donor_prompts=[],
        recipient_prompt=f"recipient-{item_id}",
        answers=["Northern Ireland"],
        metadata={
            "traffic_class": traffic_class,
            "expected_supplied_tokens": supplied,
            "expected_span_target_start": target_start,
            "parent_item_id": parent,
            "stream": "phase0-stream-b",
            "block_size": 16,
        },
    )


def _audit_event(event: str, *, request_id: str, **fields) -> str:
    record: dict[str, object] = {
        "schema_version": 2,
        "event": event,
        "connector_id": fields.pop("connector_id", "sched-1"),
        "request_id": request_id,
        "request_seq": fields.pop("request_seq", 1),
        "event_seq": fields.pop("event_seq", 0),
        "time_unix_s": 1_700_000_000.0,
    }
    record.update(fields)
    return json.dumps(record, sort_keys=True)


def _row(item_id: str, **kw) -> RequestMetrics:
    fields: dict[str, object] = {
        "item_id": item_id,
        "dataset": "longbench",
        "transform": "same_doc_new_instruction",
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
        "ttft_ms": 50.0,
    }
    fields.update(kw)
    return RequestMetrics(**fields)


class _Gateway:
    """Chat endpoint double: one answer per prompt, plus an engine id."""

    def __init__(self, answers: dict[str, str] | None = None, *, echo_id: bool = True) -> None:
        self.answers = answers or {}
        self.echo_id = echo_id

    def __call__(self, *, base_url, model, prompt, max_tokens, tenant, template, timeout_seconds):
        from sembench.request_ids import current_request_id

        sent = current_request_id()
        return {
            "output_text": self.answers.get(prompt, "42"),
            "usage": {"prompt_tokens": 3000},
            "headers": {},
            "ttft_ms": 5.0,
            "latency_ms": 10.0,
            "request_id": sent,
            "response_id": (f"chatcmpl-{sent}" if self.echo_id else "chatcmpl-front-end-minted"),
        }


def _replay(tmp_path: Path, items: list[WorkloadItem], **overrides) -> list[RequestMetrics]:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, items)
    return run_live_gateway(
        LiveGatewayConfig(
            manifest=str(manifest),
            output=str(tmp_path / "result.json"),
            gateway_url=GATEWAY,
            model="qwen",
            run_id="run-1",
            **overrides,
        )
    )


# --------------------------------------------------------------------------
# Blocker 1 — the join's manifest-side inputs
# --------------------------------------------------------------------------


def test_manifest_expectations_reads_the_real_metadata_keys():
    item = _item(
        "sd-086-recip",
        traffic_class="same_doc_new_instruction",
        supplied=3776,
        target_start=32,
        parent="sd-086-seed",
    )

    assert manifest_expectations(item) == {
        "expected_supplied_tokens": 3776,
        "expected_span_target_start": 32,
        "traffic_class": "same_doc_new_instruction",
        "propagation_parent_item_id": "sd-086-seed",
        # Round 5: M6's quality split and M3's same-position pairing are both
        # manifest facts, so they are stamped by the same function.
        "rope_delta_bucket": None,
        "stream_position": None,
        # Round 6: M1's shared-wrapper vs ad-hoc strata are a manifest fact too.
        "wrapper_id": None,
        "wrapper_rank": None,
    }


def test_a_manifest_that_claims_nothing_stamps_none_not_zero():
    """`expected_supplied_tokens: null` is "the manifest made no claim". A 0
    there would be read downstream as a prediction that nothing could be
    supplied, which is a different statement about the same item."""
    item = _item("nr-001", traffic_class="no_reuse")

    assert manifest_expectations(item)["expected_supplied_tokens"] is None
    assert manifest_expectations(item)["propagation_parent_item_id"] is None


def test_the_row_constructor_stamps_the_manifest_inputs(tmp_path, monkeypatch):
    """Declared on the row and filled by nothing was the whole bug: the audit
    side of the join arrives from the connector, the manifest side has to be
    stamped here or M1 has no denominator and M7 has no parent link."""
    monkeypatch.setattr(gateway_live, "_chat_completion", _Gateway())

    rows = _replay(
        tmp_path,
        [
            _item(
                "sd-001-recip",
                traffic_class="same_doc_new_instruction",
                supplied=3776,
                target_start=32,
                parent="sd-001-seed",
            ),
            _item("pp-001", traffic_class=PROPAGATION_PROBE_CLASS, parent="sd-001-recip"),
        ],
    )

    served, probe = rows
    assert served.expected_supplied_tokens == 3776
    assert served.expected_span_target_start == 32
    assert served.traffic_class == "same_doc_new_instruction"
    assert served.propagation_parent_item_id == "sd-001-seed"
    assert probe.traffic_class == PROPAGATION_PROBE_CLASS
    assert probe.propagation_parent_item_id == "sd-001-recip"


def test_the_results_layer_knows_every_manifest_traffic_class():
    """Including propagation_probe, which is the class M7 is defined over."""
    for traffic_class in (
        "no_reuse",
        "same_doc_new_instruction",
        "revised_doc",
        "rope_delta_sweep",
        "exact_repeat",
        "propagation_probe",
        "reworded_doc",
    ):
        assert traffic_class in TRAFFIC_CLASSES
    assert PROPAGATION_PROBE_CLASS in TRAFFIC_CLASSES
    # Section 4: alignment_given_opportunity is over these two classes.
    assert ALIGNMENT_OPPORTUNITY_CLASSES == ("same_doc_new_instruction", "revised_doc")


# --------------------------------------------------------------------------
# Blocker 2 — M1/M2 must not be computed over cold + unjoined rows
# --------------------------------------------------------------------------


def test_cold_and_unjoined_rows_are_excluded_and_counted():
    rows = [
        _row("i1", traffic_class="same_doc_new_instruction"),
        _row("i1", traffic_class="same_doc_new_instruction", arm="cold", audit_joined=None),
        _row("i2", traffic_class="same_doc_new_instruction", audit_joined=None),
    ]

    auditable = auditable_rows(rows)

    assert [row.item_id for row in auditable.rows] == ["i1"]
    assert auditable.excluded_cold_arm == 1
    assert auditable.excluded_not_joined == 1


def test_a_merged_document_reports_why_rows_left_the_denominator():
    warm = [
        _row(
            "i1",
            traffic_class="same_doc_new_instruction",
            audit_advertised_tokens=512,
            audit_observed_boundary=32,
            audit_materialized=True,
            external_confirmed_tokens=512,
        ),
        _row("i2", traffic_class="same_doc_new_instruction", audit_joined=False),
    ]
    cold = [replace(row, arm="cold", audit_joined=None, ttft_ms=200.0) for row in warm]

    metrics = aggregate_metrics(cold + warm)

    assert metrics["connector_audit_rows_considered"] == 2
    assert metrics["connector_audit_rows_excluded_cold_arm"] == 2
    assert metrics["connector_audit_rows_excluded_not_joined"] == 0
    # 1 of 2 opportunity rows, not 1 of 4.
    assert metrics["alignment_given_opportunity_denominator"] == 2
    assert metrics["alignment_given_opportunity"] == 0.5


# --------------------------------------------------------------------------
# Blocker 3 — M1 is three numbers, all conditioned on boundary > 0
# --------------------------------------------------------------------------


def test_m1_emits_all_three_numbers_with_explicit_numerators():
    """Section 4::

    alignment_given_match        = |{semantic_span_load_advertised, boundary>0, token_count>0}|
                                 / |{semantic_lookup_hit, boundary>0}|

    alignment_given_opportunity  = same numerator
                                 / |{manifest items in same_doc_new_instruction ∪ revised_doc}|
    """
    rows = [
        # Lookup hit at a non-zero boundary, and a span was advertised there.
        _row(
            "served",
            traffic_class="same_doc_new_instruction",
            audit_semantic_lookup_hit=True,
            audit_lookup_hit_boundary=32,
            audit_advertised_tokens=3776,
            audit_observed_boundary=32,
        ),
        # Lookup hit at a non-zero boundary, no span: a real misalignment.
        _row(
            "missed",
            traffic_class="revised_doc",
            audit_semantic_lookup_hit=True,
            audit_lookup_hit_boundary=1008,
            audit_boundary_miss_reason=MISS_TRUE_MISALIGNMENT,
            audit_boundary_missed_at=1008,
        ),
        # No lookup hit at all: outside the match denominator, inside the
        # opportunity one, because the opportunity is a property of the item.
        _row("no-donor", traffic_class="revised_doc", audit_semantic_lookup_hit=False),
    ]

    metrics = connector_audit_metrics(rows)

    assert metrics["alignment_given_match_numerator"] == 1
    assert metrics["alignment_given_match_denominator"] == 2
    assert metrics["alignment_given_match"] == 0.5
    assert metrics["alignment_given_opportunity_numerator"] == 1
    assert metrics["alignment_given_opportunity_denominator"] == 3
    assert metrics["alignment_given_opportunity"] == pytest.approx(1 / 3)
    assert metrics["boundary_miss_breakdown"] == {MISS_TRUE_MISALIGNMENT: 1}
    # The old name survives only as the headline's alias.
    assert metrics["boundary_alignment_rate"] == metrics["alignment_given_opportunity"]


def test_m1_is_conditioned_on_a_non_zero_boundary():
    """A boundary-0 request is lane 1, not lane 2: the connector serving it
    would prove nothing about spans at a non-zero boundary, and counting it
    would make the metric a tautology."""
    rows = [
        _row(
            "boundary-zero",
            traffic_class="same_doc_new_instruction",
            audit_semantic_lookup_hit=True,
            audit_lookup_hit_boundary=0,
            audit_advertised_tokens=2048,
            audit_observed_boundary=0,
        ),
    ]

    metrics = connector_audit_metrics(rows)

    assert metrics["alignment_given_match_numerator"] == 0
    assert metrics["alignment_given_match_denominator"] == 0
    assert metrics["alignment_given_match"] is None
    assert metrics["alignment_given_opportunity"] == 0.0


def test_an_advertise_of_zero_tokens_is_not_an_alignment():
    rows = [
        _row(
            "empty-span",
            traffic_class="same_doc_new_instruction",
            audit_semantic_lookup_hit=True,
            audit_lookup_hit_boundary=32,
            audit_advertised_tokens=0,
            audit_observed_boundary=32,
        )
    ]

    metrics = connector_audit_metrics(rows)

    assert metrics["alignment_given_match"] == 0.0


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"stored_donor_tokens": 0, "n_raw_segments": 0}, MISS_DONOR_NOT_CAPTURED),
        (
            # A captured donor the provider's spans reach past. The connector
            # drops each such segment before it builds raw_spans and counts it
            # in segments_beyond_capture -- which is why a
            # `stored_donor_tokens < longest raw span` test never fires: every
            # raw span was already trimmed to the captured window.
            {
                "stored_donor_tokens": 512,
                "n_segments": 2,
                "n_raw_segments": 0,
                "segments_wrong_donor": 0,
                "segments_beyond_capture": 2,
                "raw_spans": [],
                "snapped_spans": [],
            },
            MISS_DONOR_TOO_SHORT,
        ),
        (
            {
                "stored_donor_tokens": 4096,
                "n_raw_segments": 3,
                "raw_spans": [{"target_start": 32, "length": 55, "donor_start": 16}],
                "snapped_spans": [],
            },
            MISS_BELOW_MIN_SEMANTIC_SPAN,
        ),
        (
            {
                "stored_donor_tokens": 4096,
                "n_raw_segments": 1,
                "raw_spans": [{"target_start": 32, "length": 1024, "donor_start": 16}],
                "snapped_spans": [{"target_start": 48, "target_end": 1024, "donor_start": 32}],
            },
            MISS_TRUE_MISALIGNMENT,
        ),
        ({}, MISS_UNCLASSIFIED),
    ],
)
def test_the_miss_breakdown_separates_the_four_causes(tmp_path, payload, expected):
    """Section 4 partitions boundary_missed events by reason::

    stored_donor_tokens == 0      -> donor_not_captured
    stored_donor_tokens < span    -> donor_too_short
    n_raw_segments > 0, snapped=0 -> below_min_semantic_span
    otherwise                     -> true_misalignment

    A payload carrying none of those fields is `unclassified` rather than a
    fourth-bucket diagnosis the event does not support.
    """
    path = tmp_path / "audit.jsonl"
    path.write_text(
        _audit_event(
            "semantic_span_boundary_missed",
            request_id="chatcmpl-sb-i1",
            boundary=1008,
            **payload,
        )
        + "\n",
        encoding="utf-8",
    )

    rows, _ = join_audit_file(
        [_row("i1", engine_request_id="sb-i1", traffic_class="same_doc_new_instruction")], path
    )

    assert rows[0].audit_boundary_miss_reason == expected
    assert rows[0].audit_boundary_missed_at == 1008
    assert connector_audit_metrics(rows)["boundary_miss_breakdown"] == {expected: 1}


def test_a_lookup_hit_event_reaches_the_row(tmp_path):
    """M1's match denominator is the lookup hits, so the fold has to keep
    them; without the event the rate is null, not 1.0."""
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "\n".join(
            [
                _audit_event(
                    "semantic_lookup_hit",
                    request_id="chatcmpl-sb-i1",
                    event_seq=0,
                    boundary=32,
                    donor_id="d1",
                    similarity=0.91,
                ),
                _audit_event(
                    "semantic_span_load_advertised",
                    request_id="chatcmpl-sb-i1",
                    event_seq=1,
                    boundary=32,
                    target_start=32,
                    token_count=3776,
                    donor_id="d1",
                    snapped_spans=[{"target_start": 32, "target_end": 3808, "donor_start": 13}],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, _ = join_audit_file(
        [
            _row(
                "i1",
                engine_request_id="sb-i1",
                traffic_class="same_doc_new_instruction",
                expected_supplied_tokens=3776,
                expected_span_target_start=32,
            )
        ],
        path,
    )

    assert rows[0].audit_semantic_lookup_hit is True
    assert rows[0].audit_lookup_hit_boundary == 32
    metrics = connector_audit_metrics(rows)
    assert metrics["alignment_given_match"] == 1.0
    # M1's integrity check: the live planner agreed with the offline model.
    assert metrics["expected_supplied_tokens_agreement_rate"] == 1.0
    assert metrics["expected_span_target_start_agreement_rate"] == 1.0


def test_a_planner_disagreeing_with_the_offline_model_is_visible():
    """Section 4: "Divergence on token_count means the offline model and the
    live engine disagree about the planner — investigate"."""
    rows = [
        _row(
            "i1",
            traffic_class="same_doc_new_instruction",
            expected_supplied_tokens=3776,
            audit_advertised_tokens=2048,
            audit_observed_boundary=32,
        )
    ]

    metrics = connector_audit_metrics(rows)

    assert metrics["expected_supplied_tokens_agreement_denominator"] == 1
    assert metrics["expected_supplied_tokens_agreement_numerator"] == 0
    assert metrics["expected_supplied_tokens_agreement_rate"] == 0.0


# --------------------------------------------------------------------------
# Blocker 4 — M2's headline is the token-weighted formula
# --------------------------------------------------------------------------


def test_m2_headline_is_tokens_over_tokens_not_requests_over_requests():
    """Section 4::

    materialized_reuse_rate = Σ runtime_materialized.tokens
                            / Σ semantic_span_load_advertised.token_count

    The two forms disagree whenever the served requests are not the large
    ones, which is exactly when the difference matters.
    """
    rows = [
        _row("big", audit_advertised_tokens=4096, audit_materialized=False),
        _row(
            "small",
            audit_advertised_tokens=512,
            audit_materialized=True,
            external_confirmed_tokens=512,
        ),
    ]

    metrics = connector_audit_metrics(rows)

    assert metrics["materialized_reuse_rate"] == pytest.approx(512 / 4608)
    assert metrics["materialized_reuse_tokens"] == 512
    assert metrics["materialized_reuse_advertised_tokens"] == 4608
    assert metrics["materialized_reuse_token_rate"] == metrics["materialized_reuse_rate"]
    # The request-count rate is a different question under a different name:
    # half the requests were served, but only 11% of the promised mass.
    assert metrics["materialized_reuse_request_rate"] == 0.5
    assert metrics["materialized_reuse_rate"] != metrics["materialized_reuse_request_rate"]


# --------------------------------------------------------------------------
# Blocker 5 — M7 is a cross-arm answer comparison
# --------------------------------------------------------------------------


def _probe_arms(*, probe_answer: str) -> list[RequestMetrics]:
    """One served parent and one probe repeating it, in both arms.

    The parent was served approximate KV in the warm arm and answered
    "Donaghadee"; the cold arm answers "Northern Ireland" for both requests.
    """
    return [
        _row(
            "parent",
            arm="cold",
            ttft_ms=200.0,
            traffic_class="same_doc_new_instruction",
            output_text="Northern Ireland",
        ),
        _row(
            "parent",
            arm="warm",
            ttft_ms=100.0,
            traffic_class="same_doc_new_instruction",
            output_text="Donaghadee",
            external_confirmed_tokens=3776,
        ),
        _row(
            "probe",
            arm="cold",
            ttft_ms=200.0,
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id="parent",
            output_text="Northern Ireland",
        ),
        _row(
            "probe",
            arm="warm",
            ttft_ms=100.0,
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id="parent",
            output_text=probe_answer,
        ),
    ]


def test_m7_counts_a_probe_answering_like_the_served_output():
    """Section 4: "fraction whose answer in A6 matches the *served* output
    rather than the *cold* (A1) output".

    ``baseline_arm`` names A1 because M7 is published only when some run is
    identified as section 4's cold arm (round 6, second pass: a document that
    identifies no arm is indistinguishable from an A4-vs-A6 merge that
    labelled nothing). These tests are about the scoring, so they declare it.
    """
    paired = paired_summary(_probe_arms(probe_answer="Donaghadee"), baseline_arm="A1 stock_pc")

    assert paired is not None
    assert paired["propagation_contamination_denominator"] == 1
    assert paired["propagation_contamination_numerator"] == 1
    assert paired["propagation_contamination_rate"] == 1.0


def test_m7_does_not_count_a_probe_answering_like_the_cold_arm():
    paired = paired_summary(
        _probe_arms(probe_answer="Northern Ireland"), baseline_arm="A1 stock_pc"
    )

    assert paired is not None
    assert paired["propagation_contamination_numerator"] == 0
    assert paired["propagation_contamination_rate"] == 0.0


def test_m7_reports_probes_it_could_not_score_instead_of_dropping_them():
    """A shrunken denominator on a 50-item probe set reads as a clean result,
    so every excluded probe is counted and named."""
    rows = [row for row in _probe_arms(probe_answer="Donaghadee") if row.item_id != "parent"]
    unlinked = [
        replace(row, propagation_parent_item_id=None) if row.item_id == "probe" else row
        for row in rows
    ]

    no_parent = paired_summary(rows, baseline_arm="A1 stock_pc")
    no_link = paired_summary(unlinked, baseline_arm="A1 stock_pc")

    assert no_parent is not None and no_link is not None
    # The parent is not in this arm at all, so there is no served answer.
    assert no_parent["propagation_probes_without_served_answer"] == 1
    # The probe stays in the denominator: a probe that could not be scored is
    # not evidence of no contamination, so the headline rate divides by the
    # probe SET and the scored-only rate keeps its own name.
    assert no_parent["propagation_contamination_denominator"] == 1
    assert no_parent["propagation_contamination_rate"] == 0.0
    assert no_parent["propagation_contamination_scored_denominator"] == 0
    assert no_parent["propagation_contamination_rate_scored_only"] is None
    assert no_link["propagation_probes_unlinked"] == 1


def test_the_per_row_cached_token_figure_is_kept_under_its_own_name():
    """The old M7 was this: probes with no materialization of their own but
    non-zero cached_tokens. It is a supporting signal, not the metric."""
    rows = [
        _row(
            "p1",
            traffic_class=PROPAGATION_PROBE_CLASS,
            external_confirmed_tokens=None,
            backend_confirmed_tokens=2048,
        ),
        _row(
            "p2",
            traffic_class=PROPAGATION_PROBE_CLASS,
            external_confirmed_tokens=512,
            backend_confirmed_tokens=2048,
        ),
    ]

    metrics = connector_audit_metrics(rows)

    assert metrics["propagation_cached_without_materialization_rate"] == 0.5
    assert "propagation_contamination_rate" not in metrics


def test_the_eviction_counter_is_surfaced_for_the_lane_two_quality_gate():
    """Section 4: until this counter exists and reads non-zero on a
    contaminated workload, every lane-2 quality number is unproven."""
    rows = [
        _row("i1", audit_prefix_blocks_evicted=32),
        _row("i2", audit_prefix_blocks_evicted=0),
    ]

    metrics = connector_audit_metrics(rows)

    assert metrics["prefix_blocks_evicted"] == 32
    assert metrics["rows_with_prefix_blocks_evicted"] == 1


# --------------------------------------------------------------------------
# Blocker 6 — the negative-control gate reads the median
# --------------------------------------------------------------------------


def _gate_result(tmp_path: Path, paired: dict, name: str = "result.json") -> str:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "quality_pass_rate": 1.0,
                    "semantic_placement_rate_by_request": 1.0,
                    "backend_confirmed_block_rate": 0.5,
                    "negative_control_backend_confirmed_rate": 0.0,
                    "negative_control_semantic_placement_rate": 0.0,
                },
                "paired": paired,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _control_paired(rows: list[RequestMetrics]) -> dict:
    summary = paired_summary(rows)
    assert summary is not None
    return summary


def _control_rows(ratios: list[float]) -> list[RequestMetrics]:
    """Negative-control pairs with the given cold/warm TTFT ratios."""
    rows: list[RequestMetrics] = []
    for index, ratio in enumerate(ratios):
        rows.append(
            _row(
                f"n{index}",
                arm="cold",
                negative_control=True,
                ttft_ms=100.0 * ratio,
                audit_joined=None,
            )
        )
        rows.append(_row(f"n{index}", arm="warm", negative_control=True, ttft_ms=100.0))
    return rows


def test_one_outlier_control_pair_does_not_flip_the_gate(tmp_path, capsys):
    """Nine controls at 1.0 and one at 6.0: the mean is 1.5 and fails a 0.20
    deviation gate, the median is 1.0 and passes. Ratios are heavy-tailed and
    the control is a claim about the typical pair."""
    paired = _control_paired(_control_rows([1.0] * 9 + [6.0]))

    assert paired["negative_control_ttft_speedup_mean"] > 1.20
    assert paired["negative_control_ttft_speedup_median"] == 1.0

    main(
        [
            "assert-result-gates",
            "--result",
            _gate_result(tmp_path, paired),
            "--max-negative-control-speedup-deviation",
            "0.20",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["observed"]["negative_control_ttft_speedup_median"] == 1.0
    # The CI is reported beside the point estimate: a control measured over
    # ten pairs has to be visibly a control measured over ten pairs.
    assert payload["observed"]["negative_control_ttft_speedup_median_ci"]["point"] == 1.0
    assert payload["observed"]["negative_control_pairs"] == 10


def test_a_control_that_really_moved_still_fails_the_gate(tmp_path):
    """The gate must still fire when the typical control pair deviates: the
    median is the estimator, not an excuse."""
    paired = _control_paired(_control_rows([2.0] * 6))

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "assert-result-gates",
                "--result",
                _gate_result(tmp_path, paired),
                "--max-negative-control-speedup-deviation",
                "0.20",
            ]
        )

    assert excinfo.value.code == 1


# --------------------------------------------------------------------------
# Minor 7 — the id the engine echoed, compared with the id that was sent
# --------------------------------------------------------------------------


def test_an_engine_id_built_from_the_sent_header_is_a_match():
    assert engine_id_echoes_sent("chatcmpl-sb-r1-i7", "sb-r1-i7") is True
    assert engine_id_echoes_sent("cmpl-sb-r1-i7-0", "sb-r1-i7") is True
    assert engine_id_echoes_sent("sb-r1-i7", "sb-r1-i7") is True
    assert engine_id_echoes_sent("chatcmpl-something-else", "sb-r1-i7") is False
    # Nothing observed is not a mismatch.
    assert engine_id_echoes_sent(None, "sb-r1-i7") is None
    assert engine_id_echoes_sent("chatcmpl-sb-r1-i7", None) is None


def test_the_echo_report_counts_mismatches_and_keeps_examples():
    rows = [
        _row("i1", engine_request_id="sb-i1", engine_response_id="chatcmpl-sb-i1"),
        _row("i2", engine_request_id="sb-i2", engine_response_id="chatcmpl-minted-by-proxy"),
        _row("i3", engine_request_id="sb-i3", engine_response_id=None),
    ]

    report = request_id_echo_report(rows)

    assert report["rows_checked"] == 2
    assert report["rows_id_echoed"] == 1
    assert report["rows_id_mismatched"] == 1
    assert report["rows_without_engine_response_id"] == 1
    assert report["mismatch_examples"] == [
        {"sent": "sb-i2", "engine_returned": "chatcmpl-minted-by-proxy"}
    ]


def test_a_header_stripping_front_end_is_reported_in_the_result(tmp_path, monkeypatch, capsys):
    """An unjoinable arm and an arm that materialized nothing are the same
    null downstream. The mismatch count is what separates them."""
    monkeypatch.setattr(gateway_live, "_chat_completion", _Gateway(echo_id=False))
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_item("i1", traffic_class="no_reuse")])
    output = tmp_path / "result.json"

    main(
        [
            "run-live-gateway",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--gateway-url",
            GATEWAY,
            "--model",
            "qwen",
            "--skip-verify",
        ]
    )
    capsys.readouterr()

    echo = json.loads(output.read_text(encoding="utf-8"))["config"]["request_id_echo"]
    assert echo["rows_id_mismatched"] == 1
    assert echo["rows_id_echoed"] == 0


def test_an_engine_that_honours_the_header_reports_no_mismatch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gateway_live, "_chat_completion", _Gateway())
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_item("i1", traffic_class="no_reuse")])
    output = tmp_path / "result.json"

    main(
        [
            "run-live-gateway",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--gateway-url",
            GATEWAY,
            "--model",
            "qwen",
            "--skip-verify",
        ]
    )
    capsys.readouterr()

    echo = json.loads(output.read_text(encoding="utf-8"))["config"]["request_id_echo"]
    assert echo["rows_id_mismatched"] == 0
    assert echo["rows_id_echoed"] == 1
