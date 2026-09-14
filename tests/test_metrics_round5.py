"""Round 5: the metrics against the payloads the connector actually writes.

Round 4 implemented section 4's formulas. This round pins them to the
connector's real audit payload and to the manifest, because both of the
failures it fixes look exactly like a measurement:

- **payload fidelity.** The fold read ``boundary`` off ``semantic_lookup_hit``.
  That event has no ``boundary`` key — it carries ``already_computed_tokens``
  — so every hit folded to a null boundary, every hit fell out of
  ``alignment_given_match``'s denominator, and the rate published itself as
  null or as a flattering ratio over the advertises alone. The three ways a
  span is declined *after* a hit had no branch at all, so a request that hit
  and was then declined — which is the definition of a misalignment — left no
  trace on its row.
- **unreachable classification.** ``donor_too_short`` was decided by
  ``stored_donor_tokens < longest raw span``, which the connector's own
  arithmetic makes impossible: every segment is trimmed to the captured window
  before ``raw_spans`` is built, and a segment that does not fit is dropped and
  counted in ``segments_beyond_capture``. The bucket could never fire, so
  every short-donor miss was published as a misalignment.

And three things section 4 defines that were missing or wrong:

- **M4** (miss tax) was not implemented at all;
- **denominators** were taken over the rows a run produced rather than over
  the manifest items section 4 names, so a run that lost rows scored better;
- **M1's numerator** was silently restricted to the opportunity classes, and
  **M7's** was divided by the probes that could be scored rather than by the
  probe set — both of which move the number in the flattering direction.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sembench import gateway_live, sglang_live
from sembench.cli import main
from sembench.connector_audit import (
    MISS_BELOW_MIN_SEMANTIC_SPAN,
    MISS_DONOR_NOT_CAPTURED,
    MISS_DONOR_TOO_SHORT,
    MISS_TRUE_MISALIGNMENT,
    MISS_UNCLASSIFIED,
    boundary_miss_reason,
    join_audit_file,
    load_audit,
)
from sembench.results import (
    ALIGNMENT_OPPORTUNITY_CLASSES,
    DENOMINATOR_FROM_MANIFEST,
    DENOMINATOR_FROM_ROWS_PRESENT,
    PHASE0_ARM_PAIRS,
    PROPAGATION_PROBE_CLASS,
    aggregate_metrics,
    arm_pair,
    connector_audit_metrics,
    lookup_cost_from_engine,
    paired_summary,
    result_manifest_class_counts,
)
from sembench.schema import (
    RequestMetrics,
    WorkloadItem,
    manifest_class_counts,
    manifest_expectations,
    write_jsonl,
)
from sembench.sglang_live import LiveSglangConfig

GATEWAY = "http://gateway:8000"
SAME_DOC = "same_doc_new_instruction"
REVISED = "revised_doc"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _event(event: str, *, request_id: str, **fields) -> str:
    """One audit line, with the join key the connector stamps on every event."""
    record: dict[str, object] = {
        "schema_version": 2,
        "event": event,
        "source": "semblend_vllm_connector",
        "connector_id": fields.pop("connector_id", "sched-1"),
        "mode": "semantic_span_experimental",
        "request_id": request_id,
        "request_seq": fields.pop("request_seq", 1),
        "event_seq": fields.pop("event_seq", 0),
        "time_unix_s": 1_700_000_000.0,
    }
    record.update(fields)
    return json.dumps(record, sort_keys=True)


def _lookup_hit(request_id: str, *, boundary: int, **fields) -> str:
    """A `semantic_lookup_hit` exactly as connector.py writes one.

    Field for field: the boundary rides on `already_computed_tokens`, and
    there is no `boundary` key.
    """
    return _event(
        "semantic_lookup_hit",
        request_id=request_id,
        attempt=fields.pop("attempt", 0),
        donor_id=fields.pop("donor_id", "d1"),
        namespace="ns",
        similarity=0.91,
        materialization_kind="semantic_span",
        reusable_tokens=fields.pop("reusable_tokens", 3776),
        reason="exact_run",
        latency_ms=4,
        already_computed_tokens=boundary,
        confidence_tier="high",
        **fields,
    )


def _audit(tmp_path: Path, lines: list[str], name: str = "audit.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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


def _item(item_id: str, *, traffic_class: str, **metadata) -> WorkloadItem:
    return WorkloadItem(
        item_id=item_id,
        dataset="longbench-hotpotqa",
        source_id=f"doc-{item_id}",
        transform=traffic_class,
        donor_prompts=[],
        recipient_prompt=f"recipient-{item_id}",
        answers=["Northern Ireland"],
        metadata={"traffic_class": traffic_class, **metadata},
    )


class _Gateway:
    """Chat endpoint double: one answer per prompt, echoing the sent id."""

    def __call__(self, *, base_url, model, prompt, max_tokens, tenant, template, timeout_seconds):
        from sembench.request_ids import current_request_id

        sent = current_request_id()
        return {
            "output_text": "42",
            "usage": {"prompt_tokens": 3000},
            "headers": {},
            "ttft_ms": 5.0,
            "latency_ms": 10.0,
            "request_id": sent,
            "response_id": f"chatcmpl-{sent}",
        }


# --------------------------------------------------------------------------
# 1 — payload fidelity for M1
# --------------------------------------------------------------------------


def test_a_lookup_hit_carries_its_boundary_in_already_computed_tokens(tmp_path):
    """The connector writes `already_computed_tokens`, never `boundary`.

    Reading the wrong key produced None for every hit ever written, which
    emptied `alignment_given_match`'s denominator without a word.
    """
    path = _audit(
        tmp_path,
        [
            _lookup_hit("chatcmpl-r1", boundary=1024, event_seq=0),
            _event(
                "semantic_span_load_advertised",
                request_id="chatcmpl-r1",
                event_seq=1,
                attempt=0,
                donor_id="d1",
                namespace="ns",
                token_count=3776,
                donor_start=1024,
                target_start=1024,
                boundary=1024,
                snapped_spans=[{"target_start": 1024, "target_end": 4800, "donor_start": 1024}],
            ),
        ],
    )

    rows, _ = join_audit_file([_row("i1", engine_request_id="r1")], path)

    assert rows[0].audit_lookup_hit_boundary == 1024
    metrics = connector_audit_metrics(rows)
    assert metrics["alignment_given_match_denominator"] == 1
    assert metrics["alignment_given_match"] == 1.0


def test_a_boundary_key_on_a_lookup_hit_is_not_invented(tmp_path):
    """The fold reads the connector's field, not a plausible one.

    A hit carrying only `boundary` is a payload the connector does not
    produce; treating it as the boundary would be reading an assumption back
    as a measurement.
    """
    path = _audit(
        tmp_path,
        [
            _event(
                "semantic_lookup_hit",
                request_id="chatcmpl-r1",
                donor_id="d1",
                boundary=1024,
                reusable_tokens=3776,
            )
        ],
    )

    rows, _ = join_audit_file([_row("i1", engine_request_id="r1")], path)

    assert rows[0].audit_semantic_lookup_hit is True
    assert rows[0].audit_lookup_hit_boundary is None


def test_the_reusable_token_count_on_the_hit_reaches_the_row(tmp_path):
    path = _audit(tmp_path, [_lookup_hit("chatcmpl-r1", boundary=512, reusable_tokens=2048)])

    rows, _ = join_audit_file([_row("i1", engine_request_id="r1")], path)

    assert rows[0].audit_lookup_reusable_tokens == 2048


@pytest.mark.parametrize(
    ("event", "fields"),
    [
        (
            "semantic_span_declined_unaligned_boundary",
            {"boundary": 1008, "block_size": 16},
        ),
        (
            "semantic_span_declined_below_min_after_clamp",
            {"boundary": 1024, "token_count": 16, "min_semantic_span": 512},
        ),
        (
            "semantic_span_boundary_missed",
            {
                "boundary": 1008,
                "block_size": 16,
                "stored_donor_tokens": 4096,
                "n_segments": 1,
                "n_raw_segments": 1,
                "segments_wrong_donor": 0,
                "segments_beyond_capture": 0,
                "raw_spans": [{"target_start": 32, "length": 1024, "donor_start": 16}],
                "snapped_spans": [{"target_start": 48, "target_end": 1024, "donor_start": 32}],
            },
        ),
    ],
)
def test_a_span_declined_after_a_hit_stays_in_the_match_denominator(tmp_path, event, fields):
    """A request that hit and was then declined IS the misalignment M1 counts.

    Without a branch for the decline the row carried no boundary anywhere, the
    request left `alignment_given_match`'s denominator, and the rate reported
    only the requests that went well.
    """
    path = _audit(
        tmp_path,
        [
            _lookup_hit("chatcmpl-r1", boundary=fields["boundary"], event_seq=0),
            _event(event, request_id="chatcmpl-r1", event_seq=1, donor_id="d1", **fields),
        ],
    )

    rows, _ = join_audit_file([_row("i1", engine_request_id="r1")], path)

    row = rows[0]
    assert row.audit_span_decline_event == event
    assert row.audit_span_declined_at == fields["boundary"]
    assert row.audit_observed_boundary == fields["boundary"]
    metrics = connector_audit_metrics(rows)
    assert metrics["alignment_given_match_denominator"] == 1
    assert metrics["alignment_given_match_numerator"] == 0
    assert metrics["alignment_given_match"] == 0.0
    assert metrics["span_decline_breakdown"] == {event: 1}


def test_a_decline_supplies_the_boundary_a_hit_payload_omitted(tmp_path):
    """An older hit payload without the field still lands in the denominator.

    The connector is asked about exactly one boundary per attempt, so the
    decline recorded for the same request is the same number.
    """
    path = _audit(
        tmp_path,
        [
            _event("semantic_lookup_hit", request_id="chatcmpl-r1", event_seq=0, donor_id="d1"),
            _event(
                "semantic_span_declined_unaligned_boundary",
                request_id="chatcmpl-r1",
                event_seq=1,
                donor_id="d1",
                boundary=1008,
                block_size=16,
            ),
        ],
    )

    rows, _ = join_audit_file([_row("i1", engine_request_id="r1")], path)

    assert rows[0].audit_lookup_hit_boundary == 1008
    assert connector_audit_metrics(rows)["alignment_given_match_denominator"] == 1


def test_a_supply_clamp_is_not_a_decline(tmp_path):
    """`semantic_span_supply_clamped` trims the count and carries on."""
    path = _audit(
        tmp_path,
        [
            _lookup_hit("chatcmpl-r1", boundary=1024, event_seq=0),
            _event(
                "semantic_span_supply_clamped",
                request_id="chatcmpl-r1",
                event_seq=1,
                donor_id="d1",
                boundary=1024,
                prompt_tokens=3000,
                requested_tokens=3776,
                clamped_tokens=1920,
            ),
            _event(
                "semantic_span_load_advertised",
                request_id="chatcmpl-r1",
                event_seq=2,
                donor_id="d1",
                token_count=1920,
                donor_start=1024,
                target_start=1024,
                boundary=1024,
                snapped_spans=[{"target_start": 1024, "target_end": 2944, "donor_start": 1024}],
            ),
        ],
    )

    rows, _ = join_audit_file([_row("i1", engine_request_id="r1")], path)

    assert rows[0].audit_span_decline_event is None
    assert rows[0].audit_advertised_tokens == 1920
    assert connector_audit_metrics(rows)["alignment_given_match"] == 1.0


# --------------------------------------------------------------------------
# 2 — donor_too_short, from the fields the connector actually emits
# --------------------------------------------------------------------------


def test_a_capture_shortfall_is_read_off_segments_beyond_capture():
    """The connector's own report of "the donor was not captured that far"."""
    assert (
        boundary_miss_reason(
            {
                "stored_donor_tokens": 2048,
                "n_segments": 3,
                "n_raw_segments": 1,
                "segments_wrong_donor": 0,
                "segments_beyond_capture": 2,
                "raw_spans": [{"target_start": 32, "length": 512, "donor_start": 16}],
                "snapped_spans": [],
            }
        )
        == MISS_DONOR_TOO_SHORT
    )


def test_the_unreachable_stored_tokens_under_span_rule_is_gone():
    """`stored_donor_tokens < longest raw span` cannot happen.

    Every segment is trimmed with
    `length = min(seg.token_count, stored_tokens - seg.donor_start)` before it
    reaches `raw_spans`, so a raw span longer than the stored donor is a shape
    the connector never writes. Classifying on it meant the bucket never fired
    and every short-donor miss was published as a misalignment.
    """
    payload = {
        "stored_donor_tokens": 512,
        "n_segments": 1,
        "n_raw_segments": 1,
        "segments_wrong_donor": 0,
        "segments_beyond_capture": 0,
        "raw_spans": [{"target_start": 32, "length": 2048, "donor_start": 16}],
        "snapped_spans": [],
    }

    assert boundary_miss_reason(payload) != MISS_DONOR_TOO_SHORT
    assert boundary_miss_reason(payload) == MISS_BELOW_MIN_SEMANTIC_SPAN


def test_every_segment_dropped_without_the_counter_is_still_a_short_donor():
    """A payload from before the per-cause counters existed."""
    assert (
        boundary_miss_reason({"stored_donor_tokens": 1024, "n_segments": 2, "n_raw_segments": 0})
        == MISS_DONOR_TOO_SHORT
    )


def test_a_wrong_donor_drop_is_not_a_short_donor():
    """The planner returned another donor's spans; the donor's length is not
    the finding, so the diagnosis stays a misalignment."""
    assert (
        boundary_miss_reason(
            {
                "stored_donor_tokens": 4096,
                "n_segments": 2,
                "n_raw_segments": 0,
                "segments_wrong_donor": 2,
            }
        )
        == MISS_TRUE_MISALIGNMENT
    )


def test_the_uncaptured_and_unclassified_buckets_still_hold():
    assert boundary_miss_reason({"stored_donor_tokens": 0}) == MISS_DONOR_NOT_CAPTURED
    assert boundary_miss_reason({}) == MISS_UNCLASSIFIED


# --------------------------------------------------------------------------
# 3 — M4, the miss tax
# --------------------------------------------------------------------------


def _tax_pair(item_id: str, *, cold_ttft: float, warm_ttft: float, advertised: int | None):
    return [
        _row(item_id, arm="cold", ttft_ms=cold_ttft, audit_joined=None),
        _row(item_id, arm="warm", ttft_ms=warm_ttft, audit_advertised_tokens=advertised),
    ]


def test_the_miss_tax_is_the_ttft_gap_on_pairs_that_advertised_nothing():
    """Section 4::

    median ttft_ms(A4 | supplied == 0) - median ttft_ms(A1)
    """
    rows = [
        *_tax_pair("m1", cold_ttft=100.0, warm_ttft=112.0, advertised=None),
        *_tax_pair("m2", cold_ttft=120.0, warm_ttft=128.0, advertised=0),
        # Served, so outside the population: its latency prices a kept promise.
        *_tax_pair("s1", cold_ttft=200.0, warm_ttft=40.0, advertised=3776),
    ]

    paired = paired_summary(rows)

    assert paired is not None
    assert paired["miss_tax_pairs"] == 2
    assert paired["miss_tax_pairs_advertising_excluded"] == 1
    assert paired["miss_tax_cold_ttft_p50_ms"] == 110.0
    assert paired["miss_tax_warm_ttft_p50_ms"] == 120.0
    assert paired["miss_tax_ms"] == 10.0
    # The paired form of the same question, over the same pairs.
    assert paired["miss_tax_ms_median_of_differences"] == 10.0


def test_the_miss_tax_excludes_negative_controls_and_says_so():
    rows = [
        *_tax_pair("m1", cold_ttft=100.0, warm_ttft=110.0, advertised=None),
        _row("n1", arm="cold", ttft_ms=100.0, negative_control=True, audit_joined=None),
        _row("n1", arm="warm", ttft_ms=400.0, negative_control=True, audit_advertised_tokens=0),
    ]

    paired = paired_summary(rows)

    assert paired is not None
    assert paired["miss_tax_pairs"] == 1
    assert paired["miss_tax_pairs_negative_control_excluded"] == 1
    assert paired["miss_tax_ms"] == 10.0


def test_the_lookup_leg_is_null_when_the_engine_window_does_not_carry_it():
    """Stock vLLM exposes neither counter, so the leg is unmeasured — and a
    0.0 would subtract a cost nobody measured from a published tax."""
    assert lookup_cost_from_engine(None) == {
        "miss_tax_lookup_latency_ms_sum": None,
        "miss_tax_lookups_total": None,
        "miss_tax_lookup_ms_per_lookup": None,
        "miss_tax_lookup_cost_source": None,
    }
    assert (
        lookup_cost_from_engine({"prometheus": {"delta": {"external_prefix_cache_hits_delta": 5}}})[
            "miss_tax_lookup_ms_per_lookup"
        ]
        is None
    )


def test_the_lookup_leg_comes_from_the_engine_window_when_it_carries_it():
    """Section 4 names it: lookup_latency_ms_sum / lookups_total."""
    engine = {
        "prometheus": {
            "delta": {"lookup_latency_ms_sum_delta": 4500.0, "lookups_total_delta": 300.0}
        }
    }

    paired = paired_summary(
        _tax_pair("m1", cold_ttft=100.0, warm_ttft=115.0, advertised=0), engine=engine
    )

    assert paired is not None
    assert paired["miss_tax_lookup_ms_per_lookup"] == 15.0
    assert paired["miss_tax_lookup_cost_source"] == "engine_window"


def _arm_document(
    tmp_path: Path,
    name: str,
    *,
    arm: str,
    rows: list[RequestMetrics],
    backend_id: str = "",
    class_counts: dict[str, int] | None = None,
    engine: dict | None = None,
) -> str:
    payload = {
        "run": {
            "run_id": name,
            "arm": arm,
            "manifest_sha256": "deadbeef",
            "backend_id": backend_id,
            "baseline_id": backend_id,
        },
        "config": ({"manifest_class_counts": class_counts} if class_counts else {}),
        "engine": engine,
        "requests": [row.to_dict() for row in rows],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_merge_results_records_the_arm_pair_section_four_names(tmp_path, capsys):
    """`--pair` is how a merged document says WHICH of section 4's
    comparisons it is; without it every merge is an unlabelled cold/warm
    join and an A3-vs-A4 capture leg reads like the M3 headline."""
    cold = _arm_document(
        tmp_path,
        "cold",
        arm="cold",
        backend_id="A3 conn_discovery",
        rows=[_row("i1", arm="cold", ttft_ms=100.0, audit_joined=None)],
    )
    warm = _arm_document(
        tmp_path,
        "warm",
        arm="warm",
        backend_id="A4 conn_span",
        rows=[_row("i1", arm="warm", ttft_ms=120.0, audit_advertised_tokens=0)],
        class_counts={SAME_DOC: 300, REVISED: 150, PROPAGATION_PROBE_CLASS: 50},
        engine={
            "prometheus": {
                "delta": {"lookup_latency_ms_sum_delta": 900.0, "lookups_total_delta": 90.0}
            }
        },
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
            "m4_capture",
        ]
    )
    capsys.readouterr()

    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["config"]["arm_pair"]["baseline_arm"] == "A3"
    assert merged["config"]["arm_pair"]["treatment_arm"] == "A4"
    assert merged["config"]["arm_pair"]["metric"] == "M4"
    # The capture leg, measured: 20 ms slower on a pair that advertised nothing.
    assert merged["paired"]["miss_tax_ms"] == 20.0
    assert merged["paired"]["miss_tax_lookup_ms_per_lookup"] == 10.0
    # The manifest's counts survived the merge, so the denominators did too.
    assert merged["aggregate"]["alignment_given_opportunity_denominator"] == 450
    assert merged["paired"]["propagation_contamination_denominator"] == 50


def test_merge_results_refuses_a_pair_the_arms_contradict(tmp_path, capsys):
    cold = _arm_document(
        tmp_path,
        "cold",
        arm="cold",
        backend_id="A3 conn_discovery",
        rows=[_row("i1", arm="cold", ttft_ms=100.0)],
    )
    warm = _arm_document(
        tmp_path,
        "warm",
        arm="warm",
        backend_id="A4 conn_span",
        rows=[_row("i1", arm="warm", ttft_ms=50.0)],
    )

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "merge-results",
                "--cold",
                cold,
                "--warm",
                warm,
                "--output",
                str(tmp_path / "merged.json"),
                # M3 is A1 vs A4; this cold arm is A3.
                "--pair",
                "m3_ttft",
            ]
        )

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["merged"] is False
    assert any("m3_ttft" in problem for problem in payload["arm_label_conflicts"])


def test_every_pair_section_four_names_is_reachable():
    for name in ("m3_ttft", "m6_noise_floor", "m4_capture", "m4_instrumentation", "m7_propagation"):
        assert name in PHASE0_ARM_PAIRS
        assert arm_pair(name).name == name
    with pytest.raises(ValueError, match="section 4 names"):
        arm_pair("A4_vs_whatever")


# --------------------------------------------------------------------------
# 4 — denominators from the manifest, not from the rows present
# --------------------------------------------------------------------------


def test_the_opportunity_denominator_is_the_manifests_item_count():
    """A run that lost rows must still divide by what it was asked to serve.

    Two rows survived out of 450 opportunity items; dividing by two would
    publish 0.5 for a run that served one item in 450.
    """
    rows = [
        _row("i1", audit_advertised_tokens=3776, audit_observed_boundary=1024),
        _row("i2", traffic_class=REVISED),
    ]

    metrics = connector_audit_metrics(
        rows, manifest_class_counts={SAME_DOC: 300, REVISED: 150, "no_reuse": 563}
    )

    assert metrics["alignment_given_opportunity_denominator"] == 450
    assert metrics["alignment_given_opportunity_denominator_source"] == DENOMINATOR_FROM_MANIFEST
    assert metrics["alignment_given_opportunity"] == pytest.approx(1 / 450)
    # The rows-present figure is published beside it, under its own name.
    assert metrics["alignment_given_opportunity_rows_present_denominator"] == 2
    assert metrics["alignment_given_opportunity_rows_present"] == 0.5


def test_without_manifest_counts_the_rows_present_are_used_and_named():
    rows = [_row("i1", audit_advertised_tokens=3776, audit_observed_boundary=1024)]

    metrics = connector_audit_metrics(rows)

    assert metrics["alignment_given_opportunity_denominator"] == 1
    assert (
        metrics["alignment_given_opportunity_denominator_source"] == DENOMINATOR_FROM_ROWS_PRESENT
    )


def test_manifest_class_counts_count_the_manifest():
    items = [
        _item("a", traffic_class=SAME_DOC),
        _item("b", traffic_class=SAME_DOC),
        _item("c", traffic_class=REVISED),
        _item("d", traffic_class=PROPAGATION_PROBE_CLASS),
    ]

    assert manifest_class_counts(items) == {
        PROPAGATION_PROBE_CLASS: 1,
        REVISED: 1,
        SAME_DOC: 2,
    }


def test_the_runner_carries_the_manifest_counts_into_the_result(tmp_path, monkeypatch, capsys):
    """The runner is the only stage that reads the manifest, so the counts
    have to leave it or every class-scoped denominator silently becomes "the
    rows that survived"."""
    monkeypatch.setattr(gateway_live, "_chat_completion", _Gateway())
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            _item("a", traffic_class=SAME_DOC),
            _item("b", traffic_class=REVISED),
            _item("c", traffic_class=PROPAGATION_PROBE_CLASS),
            _item("d", traffic_class="no_reuse"),
        ],
    )
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

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["manifest_class_counts"] == {
        PROPAGATION_PROBE_CLASS: 1,
        REVISED: 1,
        SAME_DOC: 1,
        "no_reuse": 1,
    }
    assert payload["aggregate"]["alignment_given_opportunity_denominator"] == 2
    assert (
        payload["aggregate"]["alignment_given_opportunity_denominator_source"]
        == DENOMINATOR_FROM_MANIFEST
    )
    assert result_manifest_class_counts(payload) == payload["config"]["manifest_class_counts"]


# --------------------------------------------------------------------------
# 5 — the shared numerator, with its excess named
# --------------------------------------------------------------------------


def test_the_two_alignment_rates_share_one_numerator():
    """Section 4 writes `alignment_given_opportunity = same numerator`.

    A `rope_delta_sweep` item carries a donor and advertises too, so the
    shared numerator CAN exceed a two-class denominator. Restricting it
    silently would answer a question section 4 did not ask; publishing the
    excess under its own name lets the reader see why a rate is above 1.0.
    """
    rows = [
        _row("in-class", audit_advertised_tokens=3776, audit_observed_boundary=1024),
        _row(
            "sweep",
            traffic_class="rope_delta_sweep",
            audit_advertised_tokens=2048,
            audit_observed_boundary=512,
        ),
        _row(
            "repeat",
            traffic_class="exact_repeat",
            audit_advertised_tokens=1024,
            audit_observed_boundary=512,
        ),
    ]

    metrics = connector_audit_metrics(rows, manifest_class_counts={SAME_DOC: 1, REVISED: 1})

    assert metrics["alignment_given_match_numerator"] == 3
    assert metrics["alignment_given_opportunity_numerator"] == 3
    assert metrics["alignment_given_opportunity_numerator_outside_classes"] == 2
    # 3 advertises over a 2-item opportunity denominator: above 1.0, and the
    # reason is published rather than hidden by trimming the numerator.
    assert metrics["alignment_given_opportunity"] == 1.5
    assert metrics["alignment_given_opportunity"] > 1.0


def test_the_outside_class_count_is_zero_when_every_advertise_was_in_class():
    rows = [_row("i1", audit_advertised_tokens=3776, audit_observed_boundary=1024)]

    metrics = connector_audit_metrics(rows, manifest_class_counts={SAME_DOC: 2})

    assert metrics["alignment_given_opportunity_numerator_outside_classes"] == 0
    assert metrics["alignment_given_opportunity"] == 0.5


def test_the_opportunity_classes_are_the_two_section_four_names():
    assert ALIGNMENT_OPPORTUNITY_CLASSES == (SAME_DOC, REVISED)


# --------------------------------------------------------------------------
# 6 — M7 divides by the probe set
# --------------------------------------------------------------------------


def _probe_rows(*, probe_answer: str, probe_kw: dict | None = None) -> list[RequestMetrics]:
    probe_kw = probe_kw or {}
    return [
        _row(
            "parent", arm="cold", ttft_ms=200.0, output_text="Northern Ireland", audit_joined=None
        ),
        _row("parent", arm="warm", ttft_ms=100.0, output_text="Donaghadee"),
        _row(
            "probe",
            arm="cold",
            ttft_ms=200.0,
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id="parent",
            output_text="Northern Ireland",
            audit_joined=None,
            **probe_kw,
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


def test_m7_divides_by_the_manifest_probe_set():
    """Section 4: "A4 vs A6 on the 50 propagation_probe items".

    One probe propagated out of a 50-item set is a 2% contamination rate, not
    a 100% one. Dividing by the probes that could be scored turns every
    failure to score into a better contamination number.
    """
    paired = paired_summary(
        _probe_rows(probe_answer="Donaghadee"),
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50, SAME_DOC: 300},
    )

    assert paired is not None
    assert paired["propagation_contamination_numerator"] == 1
    assert paired["propagation_contamination_denominator"] == 50
    assert paired["propagation_probe_set_source"] == DENOMINATOR_FROM_MANIFEST
    assert paired["propagation_contamination_rate"] == pytest.approx(1 / 50)
    # The scored-only rate keeps its own name and its own denominator.
    assert paired["propagation_contamination_scored_denominator"] == 1
    assert paired["propagation_contamination_rate_scored_only"] == 1.0
    # 49 probes the manifest declares and this run holds no row for.
    assert paired["propagation_probes_absent_from_run"] == 49


def test_a_probe_whose_pair_was_not_clean_is_counted_not_dropped():
    """An excluded probe is a probe that was not read, which is not the same
    as a probe that came back clean."""
    rows = _probe_rows(probe_answer="Donaghadee", probe_kw={"flush_contaminated": True})

    paired = paired_summary(rows, manifest_class_counts={PROPAGATION_PROBE_CLASS: 2})

    assert paired is not None
    assert paired["propagation_probes_excluded_unclean_pair"] == 1
    assert paired["propagation_probe_pairs"] == 0
    assert paired["propagation_contamination_denominator"] == 2
    assert paired["propagation_contamination_numerator"] == 0
    assert paired["propagation_contamination_rate"] == 0.0
    assert paired["propagation_contamination_rate_scored_only"] is None


def test_without_manifest_counts_the_probe_set_is_the_probes_present():
    paired = paired_summary(_probe_rows(probe_answer="Donaghadee"))

    assert paired is not None
    assert paired["propagation_contamination_denominator"] == 1
    assert paired["propagation_probe_set_source"] == DENOMINATOR_FROM_ROWS_PRESENT
    assert paired["propagation_probes_absent_from_run"] == 0


# --------------------------------------------------------------------------
# 7 — M2 folds both sides with one rule
# --------------------------------------------------------------------------


def test_only_the_materialization_after_the_last_advertise_counts(tmp_path):
    """One rule for both sums: the last advertise, and what followed it.

    Summing every materialization against the last advertise alone made a
    re-advertised request report 133% of its promise — 256 + 768 over 768 —
    with no extra KV reused.
    """
    path = _audit(
        tmp_path,
        [
            _event(
                "semantic_span_load_advertised",
                request_id="chatcmpl-r1",
                event_seq=0,
                donor_id="d1",
                token_count=256,
                donor_start=512,
                target_start=512,
                boundary=512,
                snapped_spans=[{"target_start": 512, "target_end": 768, "donor_start": 512}],
            ),
            _event(
                "runtime_materialized",
                request_id="chatcmpl-r1",
                connector_id="wrk-1",
                event_seq=0,
                donor_id="d1",
                tokens=256,
                materialization_kind="semantic_span",
            ),
            _event(
                "semantic_span_load_advertised",
                request_id="chatcmpl-r1",
                event_seq=1,
                donor_id="d1",
                token_count=768,
                donor_start=1024,
                target_start=1024,
                boundary=1024,
                snapped_spans=[{"target_start": 1024, "target_end": 1792, "donor_start": 1024}],
            ),
            _event(
                "runtime_materialized",
                request_id="chatcmpl-r1",
                connector_id="wrk-1",
                event_seq=1,
                donor_id="d1",
                tokens=768,
                materialization_kind="semantic_span",
            ),
        ],
    )

    rows, _ = join_audit_file([_row("i1", engine_request_id="r1")], path)

    assert rows[0].audit_advertised_tokens == 768
    assert rows[0].external_confirmed_tokens == 768
    assert rows[0].audit_superseded_materialized_tokens == 256
    metrics = connector_audit_metrics(rows)
    assert metrics["materialized_reuse_advertised_tokens"] == 768
    assert metrics["materialized_reuse_tokens"] == 768
    assert metrics["materialized_reuse_rate"] == 1.0
    # The superseded mass is published, not dropped: KV was written, just not
    # against the promise the denominator holds.
    assert metrics["materialized_reuse_superseded_tokens"] == 256


def test_a_single_advertise_and_materialization_is_unchanged(tmp_path):
    path = _audit(
        tmp_path,
        [
            _event(
                "semantic_span_load_advertised",
                request_id="chatcmpl-r1",
                event_seq=0,
                donor_id="d1",
                token_count=512,
                donor_start=1024,
                target_start=1024,
                boundary=1024,
                snapped_spans=[{"target_start": 1024, "target_end": 1536, "donor_start": 1024}],
            ),
            _event(
                "runtime_materialized",
                request_id="chatcmpl-r1",
                connector_id="wrk-1",
                event_seq=0,
                donor_id="d1",
                tokens=512,
            ),
        ],
    )

    rows, _ = join_audit_file([_row("i1", engine_request_id="r1")], path)

    assert rows[0].external_confirmed_tokens == 512
    assert rows[0].audit_superseded_materialized_tokens == 0
    assert connector_audit_metrics(rows)["materialized_reuse_rate"] == 1.0


def test_a_materialization_without_any_advertise_is_still_counted(tmp_path):
    """Evidence of written KV is never dropped for want of a promise."""
    path = _audit(
        tmp_path,
        [
            _event(
                "runtime_materialized",
                request_id="chatcmpl-r1",
                connector_id="wrk-1",
                donor_id="d1",
                tokens=512,
            )
        ],
    )

    rows, _ = join_audit_file([_row("i1", engine_request_id="r1")], path)

    assert rows[0].external_confirmed_tokens == 512
    assert load_audit(path).records["chatcmpl-r1"].superseded_materialized_tokens == 0


# --------------------------------------------------------------------------
# 8 — the minors: RoPE bucket, sglang stamping, stream position
# --------------------------------------------------------------------------


def test_the_rope_delta_bucket_is_stamped_from_the_manifest():
    item = _item("rd-001", traffic_class="rope_delta_sweep", rope_delta_bucket=2048)

    assert manifest_expectations(item)["rope_delta_bucket"] == 2048
    assert (
        manifest_expectations(_item("nr-1", traffic_class="no_reuse"))["rope_delta_bucket"] is None
    )


def test_quality_is_reported_per_rope_delta_bucket():
    """Section 4 (M6): "A quality result gathered only at |delta| <= 11 does
    not transfer." A blended mean over a stream that is 93% delta~0 cannot
    show damage that only appears at delta 2048."""
    rows = [
        _row("a", traffic_class="rope_delta_sweep", rope_delta_bucket=0, quality_f1=0.9),
        _row("b", traffic_class="rope_delta_sweep", rope_delta_bucket=0, quality_f1=0.8),
        _row("c", traffic_class="rope_delta_sweep", rope_delta_bucket=2048, quality_f1=0.2),
        _row("d", traffic_class=SAME_DOC, quality_f1=0.95),
    ]

    buckets = aggregate_metrics(rows)["quality_by_rope_delta_bucket"]

    assert set(buckets) == {"0", "2048"}
    assert buckets["0"]["requests"] == 2
    assert buckets["0"]["mean_quality_f1"] == pytest.approx(0.85)
    assert buckets["2048"]["mean_quality_f1"] == pytest.approx(0.2)


def test_no_bucket_declared_reports_null_rather_than_an_empty_split():
    assert aggregate_metrics([_row("a", quality_f1=0.9)])["quality_by_rope_delta_bucket"] is None


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [1] * len(text.split())


def test_the_sglang_row_constructor_stamps_the_manifest_expectations():
    """The docstring said "every live row constructor stamps these" while the
    sglang runner stamped none, so an sglang row was outside every
    class-scoped metric."""
    item = _item(
        "sd-001",
        traffic_class=SAME_DOC,
        expected_supplied_tokens=3776,
        expected_span_target_start=32,
        parent_item_id="sd-001-seed",
        rope_delta_bucket=512,
        stream_position=7,
    )

    row = sglang_live._metrics_from_live_item(
        item=item,
        tokenizer=_Tokenizer(),
        config=LiveSglangConfig(manifest="m", output="o", base_url="u", model="qwen"),
        response={"output_text": "42"},
        arm="warm",
    )

    assert row.traffic_class == SAME_DOC
    assert row.expected_supplied_tokens == 3776
    assert row.expected_span_target_start == 32
    assert row.propagation_parent_item_id == "sd-001-seed"
    assert row.rope_delta_bucket == 512
    assert row.stream_position == 7


def test_the_gateway_row_constructor_stamps_the_new_manifest_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_live, "_chat_completion", _Gateway())
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [_item("rd-1", traffic_class="rope_delta_sweep", rope_delta_bucket=128, stream_position=4)],
    )

    rows = gateway_live.run_live_gateway(
        gateway_live.LiveGatewayConfig(
            manifest=str(manifest),
            output=str(tmp_path / "result.json"),
            gateway_url=GATEWAY,
            model="qwen",
            run_id="run-1",
        )
    )

    assert rows[0].rope_delta_bucket == 128
    assert rows[0].stream_position == 4


def test_twins_at_different_stream_positions_are_excluded_and_counted():
    """Section 4 pairs M3 "per item_id, at the same stream position".

    Two twins at different positions did not replay the same stream, and
    their ratio measures the reorder rather than the cache.
    """
    rows = [
        _row("i1", arm="cold", ttft_ms=400.0, stream_position=3, audit_joined=None),
        _row("i1", arm="warm", ttft_ms=50.0, stream_position=9),
        _row("i2", arm="cold", ttft_ms=100.0, stream_position=4, audit_joined=None),
        _row("i2", arm="warm", ttft_ms=100.0, stream_position=4),
    ]

    paired = paired_summary(rows)

    assert paired is not None
    assert paired["pairs_stream_position_mismatched"] == 1
    assert paired["pairs_used"] == 1
    assert paired["blended_ttft_speedup_median"] == 1.0


def test_a_row_declaring_no_stream_position_makes_no_claim():
    rows = [
        _row("i1", arm="cold", ttft_ms=200.0, audit_joined=None),
        _row("i1", arm="warm", ttft_ms=100.0, stream_position=9),
    ]

    paired = paired_summary(rows)

    assert paired is not None
    assert paired["pairs_stream_position_mismatched"] == 0
    assert paired["pairs_used"] == 1


def test_a_cold_row_without_a_warm_twin_is_counted():
    rows = [
        _row("i1", arm="cold", ttft_ms=200.0, audit_joined=None),
        _row("i2", arm="cold", ttft_ms=200.0, audit_joined=None),
        _row("i1", arm="warm", ttft_ms=100.0),
    ]

    paired = paired_summary(rows)

    assert paired is not None
    assert paired["pairs_unpaired"] == 1
    assert paired["pairs_used"] == 1


def test_the_audit_join_survives_a_row_carrying_every_new_field(tmp_path):
    """A round-trip guard: the row dataclass grew six fields this round and
    every one of them has to survive `replace` in the join."""
    path = _audit(tmp_path, [_lookup_hit("chatcmpl-r1", boundary=1024)])
    original = _row("i1", engine_request_id="r1", rope_delta_bucket=512, stream_position=3)

    rows, _ = join_audit_file([original], path)

    assert rows[0].rope_delta_bucket == 512
    assert rows[0].stream_position == 3
    assert replace(rows[0], arm="cold").stream_position == 3


# --------------------------------------------------------------------------
# The docs are part of the contract
# --------------------------------------------------------------------------

ROUND5_KEYS = (
    # M1: the shared numerator, the manifest denominator, the declines
    "alignment_given_opportunity_numerator_outside_classes",
    "alignment_given_opportunity_denominator_source",
    "alignment_given_opportunity_rows_present",
    "alignment_given_opportunity_rows_present_denominator",
    "span_decline_breakdown",
    "manifest_class_counts",
    # M2: one fold rule for both sums
    "materialized_reuse_superseded_tokens",
    # M4: the miss tax
    "miss_tax_ms",
    "miss_tax_ms_median_of_differences",
    "miss_tax_warm_ttft_p50_ms",
    "miss_tax_cold_ttft_p50_ms",
    "miss_tax_pairs",
    "miss_tax_pairs_without_ttft",
    "miss_tax_pairs_advertising_excluded",
    "miss_tax_pairs_negative_control_excluded",
    "miss_tax_definition",
    "miss_tax_lookup_latency_ms_sum",
    "miss_tax_lookups_total",
    "miss_tax_lookup_ms_per_lookup",
    "miss_tax_lookup_cost_source",
    # M7: the probe set
    "propagation_probe_set_source",
    "propagation_contamination_rate_scored_only",
    "propagation_contamination_scored_denominator",
    "propagation_probes_excluded_unclean_pair",
    "propagation_probes_absent_from_run",
    # M6 and the pairing
    "quality_by_rope_delta_bucket",
    "pairs_unpaired",
    "pairs_stream_position_mismatched",
    # Manifest and row fields
    "rope_delta_bucket",
    "stream_position",
    "audit_lookup_reusable_tokens",
    "audit_span_decline_event",
    "audit_span_decline_reason",
    "audit_span_declined_at",
    "audit_superseded_materialized_tokens",
    "arm_pair",
)

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("key", ROUND5_KEYS)
def test_every_key_this_round_adds_is_documented(key):
    """A key nobody can look up is a key nobody can check a claim against."""
    assert key in (_ROOT / "README.md").read_text(encoding="utf-8"), key
    assert key in (_ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8"), key


def test_the_docs_state_the_two_payload_facts_this_round_turns_on():
    """Both fixes are invisible in the output and have to be written down.

    A fold reading the wrong field name publishes a null, and a partition on
    an unreachable condition publishes the wrong bucket; neither raises.
    """
    metrics = (_ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8")
    assert "`semantic_lookup_hit` has no `boundary` key" in metrics
    assert "already_computed_tokens" in metrics
    assert "A `stored_donor_tokens < span` test\nis unreachable" in metrics
    assert "segments_beyond_capture" in metrics


def test_the_docs_say_the_denominators_come_from_the_manifest():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    metrics = (_ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8")
    assert "config.manifest_class_counts" in readme
    assert "count manifest items" in readme or "counts manifest items" in readme
    assert "The manifest's denominators" in metrics
    assert "The denominator is the probe set" in metrics
