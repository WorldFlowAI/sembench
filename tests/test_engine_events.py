from __future__ import annotations

import json

from sembench.engine_events import (
    parse_engine_events,
    parse_sglang_logs,
    parse_trtllm_audit_jsonl,
    parse_vllm_logs,
)


def test_parse_vllm_logs_confirms_materialized_reuse() -> None:
    logs = "\n".join(
        [
            "INFO connector.py:619: SemBlend donor registered request_id=d1 namespace=vllm:abc tokens=898 blocks=57",
            "INFO connector.py:374: SemBlend semantic lookup hit request_id=r1 donor_id=d1 similarity=0.9670 kind=discovery_only reusable_tokens=0 reason=semblend_discovery latency_ms=30",
            "INFO connector.py:414: SemBlend request-local experimental load advertised request_id=r1 donor_id=d1 tokens=880",
            "INFO connector.py:526: SemBlend materializing request_id=r1 donor_id=d1 tokens=880",
            "INFO connector.py:557: SemBlend materialized request_id=r1 donor_id=d1 tokens=880",
        ]
    )

    summary = parse_vllm_logs(logs)

    assert summary.engine == "vllm"
    assert summary.donor_registrations == 1
    assert summary.semantic_hits == 1
    assert summary.load_advertisements == 1
    assert summary.materialization_events == 1
    assert summary.materialized_tokens == 880
    assert summary.materialized_semantic_kv_reuse is True


def test_parse_trtllm_audit_jsonl_counts_boundaries_rope_and_units() -> None:
    audit = "\n".join(
        [
            json.dumps({"event": "lookup", "found": False, "rejection_reason": "no_donor_match"}),
            json.dumps({"event": "lookup", "found": True, "similarity": 0.90}),
            json.dumps({"event": "engine_blend_boundary", "boundary_layer": 24}),
            json.dumps(
                {
                    "event": "materialized",
                    "materialized": 552,
                    "requires_rope_correction": True,
                }
            ),
            json.dumps({"event": "donor_registered", "request_id": 2049}),
        ]
    )

    summary = parse_trtllm_audit_jsonl(audit)

    assert summary.engine == "trtllm"
    assert summary.semantic_hits == 1
    assert summary.materialization_events == 1
    assert summary.materialized_units == 552
    assert summary.rope_corrections == 1
    assert summary.engine_boundaries == 1
    assert summary.rejection_reasons == {"no_donor_match": 1}
    assert summary.materialized_semantic_kv_reuse is True


def test_parse_sglang_logs_separates_hits_from_realized_materialization() -> None:
    logs = "\n".join(
        [
            "[FUZZY] register_donor_async: ok request_id=d1 tokens=899 embed=394.1ms total=400.7ms",
            "[FUZZY] semantic hit similarity=0.91 matched_tokens=512 request_id=r1 donor_id=d1",
            "[FUZZY RADIX] Fuzzy match success: rid=r1 cached=512, prompt=1024, offset=0, quality_cosine=0.910",
            "[FUZZY] Realized 512 fuzzy tokens (contiguous): copied donor KV with RoPE correction",
        ]
    )

    summary = parse_sglang_logs(logs)

    assert summary.engine == "sglang"
    assert summary.donor_registrations == 1
    assert summary.semantic_hits == 2
    assert summary.materialized_tokens == 512
    assert summary.materialization_events == 1
    assert summary.materialized_semantic_kv_reuse is True


def test_parse_sglang_candidate_hit_is_not_materialization() -> None:
    logs = "\n".join(
        [
            "[FUZZY] register_donor_async: ok request_id=d1 tokens=899 embed=394.1ms total=400.7ms",
            "[FUZZY] semantic hit similarity=0.91 matched_tokens=512 request_id=r1 donor_id=d1",
        ]
    )

    summary = parse_sglang_logs(logs)

    assert summary.semantic_hits == 1
    assert summary.materialization_events == 0
    assert summary.materialized_tokens == 0
    assert summary.materialized_semantic_kv_reuse is False


def test_parse_sglang_segmented_phases_are_backend_materialization() -> None:
    logs = "\n".join(
        [
            "[FUZZY] register_donor_async: ok request_id=d1 tokens=3905 embed=394.1ms total=400.7ms",
            "[FUZZY RADIX] Fuzzy match success: rid=r1 cached=3894, prompt=4024, offset=0, quality_cosine=0.964",
            "[FUZZY RADIX] segmented phased paged prefill active: backend=phased_paged rid=r1 fresh_tokens=130 donor_tokens=3894 prompt_tokens=4024 phases=13 direct_paged_kv=True",
            "[FUZZY RADIX] segmented prefill realized donor phase: backend=phased_paged rid=r1 donor_tokens=608 target=[21,629) prefix_len=629 direct_paged_kv=True",
            "[FUZZY RADIX] segmented prefill realized donor phase: backend=phased_paged rid=r1 donor_tokens=656 target=[644,1300) prefix_len=1300 direct_paged_kv=True",
            "[FUZZY] Realized 1264 fuzzy tokens (2 segments)",
        ]
    )

    summary = parse_sglang_logs(logs)

    assert summary.semantic_hits == 1
    assert summary.materialization_events == 2
    assert summary.materialized_tokens == 1264
    assert summary.events["segmented_phases"][0]["target_start"] == "21"
    assert summary.materialized_semantic_kv_reuse is True


def test_parse_sglang_runtime_shape_warnings_are_not_silent() -> None:
    logs = (
        "2026-06-22 [E:onnxruntime:, sequential_executor.cc:572 ExecuteKernel] "
        "Non-zero status code returned while running LayerNormalization node. "
        "Shape mismatch attempting to re-use buffer. {1,256,384} != {5,256,384}."
    )

    summary = parse_sglang_logs(logs)

    assert summary.errors == []
    assert len(summary.runtime_warnings) == 1


def test_parse_engine_events_dispatches_aliases() -> None:
    summary = parse_engine_events(
        "trt-llm",
        json.dumps({"event": "lookup", "found": True}) + "\n",
    )

    assert summary.engine == "trtllm"
    assert summary.semantic_hits == 1
