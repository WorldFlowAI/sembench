"""B10: the connector-audit join, and the three metrics that depend on it.

The audit stream is the only per-request evidence that semantic KV was
actually written. These tests pin the join on a synthetic stream with the
shapes the live one produces — a served request, a boundary miss, a
re-advertise that supersedes an earlier promise, a declined load, and a
request the connector never said anything about — and pin the thing that
matters most about the metrics: with no audit they are null, not zero.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sembench.connector_audit import (
    AuditError,
    join_audit_file,
    join_requests,
    load_audit,
    normalized_ids,
)
from sembench.results import (
    aggregate_metrics,
    connector_audit_metrics,
    is_reuse_hit,
    paired_summary,
    semantic_reuse_tokens,
)
from sembench.schema import EXTERNAL_SOURCE_CONNECTOR_AUDIT, RequestMetrics

SCHED = "sched-1"
WORKER = "wrk-1"

# Client ids, as the runner stamps them into X-Request-Id.
HIT = "sb-r1-warm-i1"
MISS = "sb-r1-warm-i2"
READVERTISED = "sb-r1-warm-i3"
SILENT = "sb-r1-warm-i4"
PROBE_SERVED = "sb-r1-warm-p2"
PROBE_CACHED = "sb-r1-warm-p1"


def _event(event: str, *, request_id: str | None = None, **fields) -> str:
    record: dict[str, object] = {
        "schema_version": 2,
        "event": event,
        "source": "semblend_vllm_connector",
        "connector_id": fields.pop("connector_id", SCHED),
        "mode": "semantic_span_experimental",
        "time_unix_s": 1_700_000_000.0,
    }
    if request_id is not None:
        record["request_id"] = request_id
        record.setdefault("request_seq", fields.pop("request_seq", 1))
        record.setdefault("event_seq", fields.pop("event_seq", 0))
    record.update(fields)
    return json.dumps(record, sort_keys=True)


def _span(target_start: int, target_end: int, donor_start: int) -> dict[str, int]:
    return {"target_start": target_start, "target_end": target_end, "donor_start": donor_start}


def _audit_lines() -> list[str]:
    """One arm's audit stream: both roles appending to the same file."""
    return [
        _event("connector_initialized", block_size=16),
        # Served end to end: advertise -> allocate -> materialize.
        _event("request_first_seen", request_id=f"chatcmpl-{HIT}", event_seq=0, prompt_tokens=3000),
        _event(
            "semantic_span_load_advertised",
            request_id=f"chatcmpl-{HIT}",
            event_seq=1,
            attempt=0,
            donor_id="d1",
            namespace="ns",
            token_count=512,
            donor_start=1024,
            target_start=1024,
            boundary=1024,
            snapped_spans=[_span(1024, 1536, 1024)],
        ),
        _event(
            "load_allocated",
            request_id=f"chatcmpl-{HIT}",
            event_seq=2,
            donor_id="d1",
            namespace="ns",
            tokens=512,
            materialization_kind="semantic_span",
            block_id_count=32,
        ),
        _event(
            "prefix_cache_blocks_evicted",
            request_id=f"chatcmpl-{HIT}",
            event_seq=3,
            donor_id="d1",
            phase="load_allocated",
            blocks_evicted=32,
        ),
        _event(
            "runtime_materialized",
            request_id=f"chatcmpl-{HIT}",
            connector_id=WORKER,
            event_seq=0,
            donor_id="d1",
            namespace="ns",
            tokens=512,
            materialization_kind="semantic_span",
            layers_materialized=28,
        ),
        # The boundary fell outside every span. Completion-shaped engine id.
        _event("request_first_seen", request_id=f"cmpl-{MISS}-0", event_seq=0, prompt_tokens=3000),
        _event(
            "semantic_span_boundary_missed",
            request_id=f"cmpl-{MISS}-0",
            event_seq=1,
            attempt=0,
            donor_id="d2",
            boundary=1008,
            block_size=16,
            stored_donor_tokens=0,
            n_raw_segments=0,
            raw_spans=[],
            snapped_spans=[],
            prompt_tokens=3000,
        ),
        # Advertised at 512, re-advertised at 1024, then the worker declined.
        _event(
            "request_first_seen",
            request_id=f"chatcmpl-{READVERTISED}",
            event_seq=0,
            prompt_tokens=3000,
        ),
        _event(
            "semantic_span_load_advertised",
            request_id=f"chatcmpl-{READVERTISED}",
            event_seq=1,
            attempt=0,
            donor_id="d3",
            token_count=256,
            donor_start=512,
            target_start=512,
            boundary=512,
            snapped_spans=[_span(512, 768, 512)],
        ),
        _event(
            "semantic_span_load_advertised",
            request_id=f"chatcmpl-{READVERTISED}",
            event_seq=2,
            attempt=1,
            donor_id="d3",
            token_count=768,
            donor_start=1024,
            target_start=1024,
            boundary=1024,
            snapped_spans=[_span(1024, 1792, 1024)],
        ),
        _event(
            "load_allocated",
            request_id=f"chatcmpl-{READVERTISED}",
            event_seq=3,
            donor_id="d3",
            tokens=768,
            materialization_kind="semantic_span",
            block_id_count=48,
        ),
        _event(
            "runtime_materialization_declined",
            request_id=f"chatcmpl-{READVERTISED}",
            connector_id=WORKER,
            event_seq=0,
            donor_id="d3",
            tokens=768,
            materialization_kind="semantic_span",
            declined_reason="no_kv_layers_materialized",
        ),
        # A probe the connector served, and a probe it only skipped a lookup
        # for — the one that can still be warmed by the exact prefix cache.
        _event(
            "semantic_span_load_advertised",
            request_id=f"chatcmpl-{PROBE_SERVED}",
            event_seq=0,
            donor_id="d4",
            token_count=512,
            donor_start=1024,
            target_start=1024,
            boundary=1024,
            snapped_spans=[_span(1024, 1536, 1024)],
        ),
        _event(
            "runtime_materialized",
            request_id=f"chatcmpl-{PROBE_SERVED}",
            connector_id=WORKER,
            event_seq=0,
            donor_id="d4",
            tokens=512,
            materialization_kind="semantic_span",
            layers_materialized=28,
        ),
        _event(
            "lookup_skipped_exact_ratio",
            request_id=f"chatcmpl-{PROBE_CACHED}",
            event_seq=0,
            attempt=0,
            boundary=2048,
            exact_ratio=0.98,
            threshold=0.9,
        ),
        # Junk the join must survive: a truncated line, an event from an older
        # schema, and a line that is not JSON at all.
        '{"schema_version": 2, "event": "runtime_materialized", "request_id": "chatcmpl-tr',
        _event("runtime_materialized", request_id="chatcmpl-old", tokens=4096, schema_version=1),
        "not json at all",
    ]


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    path = tmp_path / "connector-audit.jsonl"
    path.write_text("\n".join(_audit_lines()) + "\n", encoding="utf-8")
    return path


def _row(item_id: str, *, request_id: str | None = None, **kw) -> RequestMetrics:
    fields: dict[str, object] = {
        "item_id": item_id,
        "dataset": "longbench",
        "transform": "instruction_variant",
        "negative_control": False,
        "donor_count": 1,
        "prompt_tokens": 3000,
        "total_blocks": 188,
        "exact_hit_blocks": 0,
        "exact_hit_tokens": 0,
        "semantic_candidate_blocks": 0,
        "semantic_candidate_tokens": 0,
        "semantic_eligible_blocks": 0,
        "semantic_eligible_tokens": 0,
        "engine_request_id": request_id,
        "arm": "warm",
        "ttft_ms": 40.0,
    }
    fields.update(kw)
    return RequestMetrics(**fields)


def _warm_rows() -> list[RequestMetrics]:
    """One row per audited request, plus one the audit never mentions."""
    return [
        _row(
            "i1",
            request_id=HIT,
            expected_supplied_tokens=512,
            expected_span_target_start=1024,
            traffic_class="same_doc_new_instruction",
            backend_confirmed_tokens=1536,
        ),
        _row(
            "i2",
            request_id=MISS,
            expected_supplied_tokens=512,
            expected_span_target_start=1024,
            traffic_class="same_doc_new_instruction",
            backend_confirmed_tokens=1008,
        ),
        _row(
            "i3",
            request_id=READVERTISED,
            expected_supplied_tokens=768,
            expected_span_target_start=1024,
            traffic_class="revised_doc",
            backend_confirmed_tokens=1024,
        ),
        _row(
            "i4",
            request_id=SILENT,
            expected_supplied_tokens=512,
            expected_span_target_start=1024,
            traffic_class="revised_doc",
            backend_confirmed_tokens=0,
        ),
        # no_reuse: the manifest says no donor existed, so it is outside M1.
        _row(
            "i5",
            request_id="sb-r1-warm-i5",
            expected_supplied_tokens=0,
            expected_span_target_start=None,
            traffic_class="no_reuse",
        ),
        _row(
            "p1",
            request_id=PROBE_CACHED,
            expected_supplied_tokens=0,
            traffic_class="propagation_probe",
            backend_confirmed_tokens=2048,
        ),
        _row(
            "p2",
            request_id=PROBE_SERVED,
            expected_supplied_tokens=0,
            traffic_class="propagation_probe",
            backend_confirmed_tokens=1536,
        ),
    ]


def _joined(audit_path: Path):
    rows, report = join_audit_file(_warm_rows(), audit_path)
    return {row.item_id: row for row in rows}, report


def test_materialized_tokens_become_per_request_external_reuse(audit_path):
    rows, _ = _joined(audit_path)

    served = rows["i1"]
    assert served.external_confirmed_tokens == 512
    assert served.external_confirmed_tokens_source == EXTERNAL_SOURCE_CONNECTOR_AUDIT
    assert served.audit_joined is True
    assert served.audit_load_allocated is True
    assert served.audit_materialized is True
    assert served.audit_declined_reasons is None


def test_a_declined_load_leaves_external_tokens_null(audit_path):
    """Advertised and allocated is a promise and a destination, not KV."""
    rows, _ = _joined(audit_path)

    declined = rows["i3"]
    assert declined.audit_load_allocated is True
    assert declined.audit_materialized is False
    assert declined.external_confirmed_tokens is None
    assert declined.external_confirmed_tokens_source is None
    assert declined.audit_declined_reasons == ["no_kv_layers_materialized"]


def test_the_last_advertise_supersedes_the_earlier_one(audit_path):
    rows, _ = _joined(audit_path)

    readvertised = rows["i3"]
    assert readvertised.audit_advertised_tokens == 768
    assert readvertised.audit_advertised_target_start == 1024
    assert readvertised.audit_observed_boundary == 1024
    assert readvertised.audit_boundary_at_span_start is True


def test_a_boundary_miss_keeps_its_boundary_and_advertises_nothing(audit_path):
    rows, _ = _joined(audit_path)

    missed = rows["i2"]
    assert missed.audit_joined is True
    assert missed.audit_observed_boundary == 1008
    assert missed.audit_advertised_tokens is None
    assert missed.audit_boundary_at_span_start is None
    assert missed.audit_materialized is False
    assert missed.external_confirmed_tokens is None


def test_a_request_the_audit_never_mentions_joins_as_false(audit_path):
    """False, not None: the audit was read and held nothing for this row."""
    rows, report = _joined(audit_path)

    silent = rows["i4"]
    assert silent.audit_joined is False
    assert silent.audit_observed_boundary is None
    assert silent.audit_advertised_tokens is None
    assert silent.audit_materialized is None
    assert silent.external_confirmed_tokens is None
    assert report.rows_unmatched == 2  # the silent row and the no_reuse row


def test_both_openai_id_shapes_join_to_the_header_the_runner_sent(audit_path):
    """chatcmpl-<id> and cmpl-<id>-0 are the same client request id."""
    rows, report = _joined(audit_path)

    assert rows["i1"].audit_joined is True  # chatcmpl-<id>
    assert rows["i2"].audit_joined is True  # cmpl-<id>-0
    assert report.rows_matched_normalized == 5
    assert report.rows_matched_exact == 0


def test_normalized_ids_strip_one_prefix_and_one_subrequest_suffix():
    """Every intermediate form is a candidate, so a client id that itself
    ends in a number still matches and a collision is still visible."""
    assert normalized_ids("cmpl-sb-r1-i7-0") == ("cmpl-sb-r1-i7", "sb-r1-i7", "sb-r1-i7-0")
    assert normalized_ids("chatcmpl-sb-r1-i7") == ("sb-r1-i7",)


def test_two_subrequests_of_one_id_are_refused_not_guessed(tmp_path):
    """A multi-prompt HTTP request cannot be attributed to one row."""
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "\n".join(
            [
                _event("runtime_materialized", request_id="cmpl-amb-0", tokens=256),
                _event("runtime_materialized", request_id="cmpl-amb-1", tokens=256),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, report = join_audit_file([_row("i1", request_id="amb")], path)

    assert report.rows_ambiguous == 1
    assert rows[0].audit_joined is False
    assert rows[0].external_confirmed_tokens is None


def test_malformed_and_foreign_schema_lines_are_counted_not_fatal(audit_path):
    index = load_audit(audit_path)

    assert index.stats.malformed_lines == 2  # the truncated line and the junk
    assert index.stats.other_schema_versions == 1
    assert index.stats.engine_scope_events == 1  # connector_initialized
    # Five request ids left events behind; the sixth row's request never did.
    assert index.stats.requests == 5


def test_a_missing_audit_file_fails_loudly(tmp_path):
    with pytest.raises(AuditError, match="could not be read"):
        load_audit(tmp_path / "nope.jsonl")


def test_a_row_without_a_request_id_is_counted_not_joined(audit_path):
    index = load_audit(audit_path)

    rows, report = join_requests([_row("i1", request_id=None)], index)

    assert report.rows_without_request_id == 1
    assert rows[0].audit_joined is False


def test_the_join_returns_new_rows(audit_path):
    index = load_audit(audit_path)
    original = _row("i1", request_id=HIT)

    joined, _ = join_requests([original], index)

    assert joined[0] is not original
    assert original.external_confirmed_tokens is None
    assert original.audit_joined is None


def test_an_audited_non_materialization_is_a_confirmed_miss(audit_path):
    """The audit looked and nothing was materialized, so the row is a measured
    zero — it must not fall through to a cached_tokens-based legacy hit even
    though the engine reported 1024 cached tokens for it."""
    rows, _ = _joined(audit_path)

    declined = rows["i3"]
    assert declined.backend_confirmed_tokens == 1024
    assert semantic_reuse_tokens(declined) == 0
    assert is_reuse_hit(declined) is False


def test_an_unjoined_row_stays_unmeasured_rather_than_a_miss(audit_path):
    rows, _ = _joined(audit_path)

    silent = rows["i4"]
    assert silent.audit_joined is False
    assert semantic_reuse_tokens(silent) is None
    assert is_reuse_hit(silent) is None


def test_alignment_given_opportunity_is_over_the_two_manifest_classes(audit_path):
    rows, _ = _joined(audit_path)
    metrics = connector_audit_metrics(list(rows.values()))

    # Opportunity classes are same_doc_new_instruction (i1, i2) and
    # revised_doc (i3, i4). Of those, i1 and i3 advertised a non-zero span at
    # a non-zero boundary; i2 missed at 1008 and i4 was never advertised at
    # all. i5 (no_reuse) and the two probes are outside the denominator.
    # No manifest counts reached this document, so the denominator is the
    # opportunity rows present and says so.
    assert metrics["alignment_given_opportunity_denominator"] == 4
    assert metrics["alignment_given_opportunity_denominator_source"] == "rows_present"
    # Section 4's numerator is SHARED with alignment_given_match and is not
    # restricted to the two classes: p2 (a propagation probe) advertised too.
    # The excess is named rather than trimmed away in silence.
    assert metrics["alignment_given_opportunity_numerator"] == 3
    assert metrics["alignment_given_match_numerator"] == 3
    assert metrics["alignment_given_opportunity_numerator_outside_classes"] == 1
    assert metrics["alignment_given_opportunity"] == 0.75
    # The documented headline alias is the same number and nothing else.
    assert metrics["boundary_alignment_rate"] == metrics["alignment_given_opportunity"]


def test_alignment_given_match_is_null_without_lookup_hit_events(audit_path):
    """Its denominator is |{semantic_lookup_hit, boundary>0}|, so a connector
    that does not emit that event leaves the rate unmeasurable — null, never
    a flattering 1.0 over the advertises it did emit."""
    rows, _ = _joined(audit_path)
    metrics = connector_audit_metrics(list(rows.values()))

    assert metrics["alignment_given_match_denominator"] == 0
    assert metrics["alignment_given_match"] is None
    # The numerator is still counted, over every audited class: i1, i3, p2.
    assert metrics["alignment_given_match_numerator"] == 3


def test_materialized_reuse_headline_is_token_weighted(audit_path):
    rows, _ = _joined(audit_path)
    metrics = connector_audit_metrics(list(rows.values()))

    # Advertised: i1 (512), i3 (768), p2 (512). Materialized: i1 + p2 = 1024.
    assert metrics["materialized_reuse_advertised_tokens"] == 512 + 768 + 512
    assert metrics["materialized_reuse_tokens"] == 1024
    assert metrics["materialized_reuse_rate"] == pytest.approx(1024 / 1792)
    assert metrics["materialized_reuse_token_rate"] == metrics["materialized_reuse_rate"]
    # The request-count question keeps its own name.
    assert metrics["materialized_reuse_request_denominator"] == 3
    assert metrics["materialized_reuse_request_numerator"] == 2
    assert metrics["materialized_reuse_request_rate"] == pytest.approx(2 / 3)


def test_probes_cached_without_a_materialization_are_a_supporting_signal(audit_path):
    rows, _ = _joined(audit_path)
    metrics = connector_audit_metrics(list(rows.values()))

    # p1 materialized nothing of its own but was served 2048 cached tokens;
    # p2 got its own semantic load and is not propagation. This is NOT M7 —
    # M7 is the cross-arm answer comparison in paired_summary.
    assert metrics["propagation_cached_without_materialization_denominator"] == 2
    assert metrics["propagation_cached_without_materialization_numerator"] == 1
    assert metrics["propagation_cached_without_materialization_rate"] == 0.5
    assert "propagation_contamination_rate" not in metrics


def test_a_probe_class_declared_on_transform_is_still_a_probe(audit_path):
    rows, _ = _joined(audit_path)
    legacy = replace(rows["p1"], traffic_class=None, transform="propagation_probe")

    metrics = connector_audit_metrics([legacy])

    assert metrics["propagation_cached_without_materialization_denominator"] == 1
    assert metrics["propagation_cached_without_materialization_numerator"] == 1


def test_the_eviction_counter_reaches_the_row_and_the_metrics(audit_path):
    """Section 4 gates every lane-2 quality number on this counter reading
    non-zero, so it has to leave the audit stream."""
    rows, _ = _joined(audit_path)
    metrics = connector_audit_metrics(list(rows.values()))

    assert rows["i1"].audit_prefix_blocks_evicted == 32
    assert rows["i2"].audit_prefix_blocks_evicted == 0
    assert metrics["prefix_blocks_evicted"] == 32
    assert metrics["rows_with_prefix_blocks_evicted"] == 1


def test_an_absent_audit_yields_nulls_not_zeros():
    """The trap this whole module exists to avoid: with no audit every row's
    external mass is null for want of measurement, and a rate computed over
    that would publish a perfect propagation score for a run that measured
    nothing."""
    metrics = connector_audit_metrics(_warm_rows())

    assert metrics["connector_audit_present"] is False
    assert metrics["connector_audit_rows_joined"] is None
    for key in (
        "boundary_alignment_rate",
        "alignment_given_match",
        "alignment_given_match_numerator",
        "alignment_given_opportunity",
        "alignment_given_opportunity_numerator",
        "boundary_miss_breakdown",
        "materialized_reuse_rate",
        "materialized_reuse_token_rate",
        "materialized_reuse_tokens",
        "materialized_reuse_advertised_tokens",
        "materialized_reuse_request_rate",
        "materialized_reuse_request_numerator",
        "propagation_cached_without_materialization_rate",
        "propagation_cached_without_materialization_numerator",
        "prefix_blocks_evicted",
    ):
        assert metrics[key] is None, key
    # No row could have been audited, and the counters say exactly that
    # rather than leaving a denominator to be read as a measured zero.
    assert metrics["connector_audit_rows_considered"] == 0
    assert metrics["connector_audit_rows_excluded_not_joined"] == 7
    assert metrics["alignment_given_opportunity_denominator"] == 0


def test_rates_are_null_rather_than_zero_over_an_empty_denominator(audit_path):
    index = load_audit(audit_path)
    rows, _ = join_requests([_row("i5", request_id="sb-r1-warm-i5")], index)

    metrics = connector_audit_metrics(rows)

    assert metrics["connector_audit_present"] is True
    assert metrics["alignment_given_opportunity_denominator"] == 0
    assert metrics["alignment_given_opportunity"] is None
    assert metrics["boundary_alignment_rate"] is None
    assert metrics["materialized_reuse_request_denominator"] == 0
    assert metrics["materialized_reuse_rate"] is None
    assert metrics["propagation_cached_without_materialization_rate"] is None


def test_the_aggregate_and_paired_summary_both_publish_the_three_rates(audit_path):
    warm, _ = join_audit_file(_warm_rows(), audit_path)
    cold = [
        replace(
            row,
            arm="cold",
            ttft_ms=200.0,
            audit_joined=None,
            external_confirmed_tokens=None,
            external_confirmed_tokens_source=None,
            audit_advertised_tokens=None,
            audit_materialized=None,
        )
        for row in warm
    ]

    aggregate = aggregate_metrics(warm)
    paired = paired_summary(cold + warm)

    assert aggregate["boundary_alignment_rate"] == 0.75
    assert aggregate["connector_audit_rows_joined"] == 5
    # The paired block reports the warm (connector) arm's audit metrics.
    assert paired["boundary_alignment_rate"] == 0.75
    assert paired["materialized_reuse_request_numerator"] == 2
    assert paired["propagation_cached_without_materialization_denominator"] == 2
    assert paired["connector_audit_present"] is True


def test_a_merged_documents_cold_arm_is_not_in_the_audit_denominators(audit_path):
    """merge-results holds both arms in one list. Computing M1/M2 over that
    list doubles every denominator with requests no connector ever saw."""
    warm, _ = join_audit_file(_warm_rows(), audit_path)
    cold = [replace(row, arm="cold", audit_joined=None) for row in warm]

    both_arms = aggregate_metrics(cold + warm)
    warm_only = aggregate_metrics(warm)

    assert both_arms["connector_audit_rows_excluded_cold_arm"] == 7
    assert (
        both_arms["alignment_given_opportunity_denominator"]
        == warm_only["alignment_given_opportunity_denominator"]
        == 4
    )
    assert both_arms["boundary_alignment_rate"] == warm_only["boundary_alignment_rate"] == 0.75


def test_the_negative_control_gate_can_read_a_median(audit_path):
    cold = [
        _row("n1", arm="cold", ttft_ms=200.0, negative_control=True),
        _row("n2", arm="cold", ttft_ms=100.0, negative_control=True),
    ]
    warm = [
        _row("n1", arm="warm", ttft_ms=100.0, negative_control=True),
        _row("n2", arm="warm", ttft_ms=25.0, negative_control=True),
    ]

    paired = paired_summary(cold + warm)

    # Ratios 2.0 and 4.0: the median is 3.0 and the tail-sensitive mean 3.0
    # here, but the median is the key the deviation gate must read.
    assert paired["negative_control_pairs"] == 2
    assert paired["negative_control_ttft_speedup_median"] == 3.0
    assert paired["negative_control_ttft_speedup_median_ci"]["point"] == 3.0
