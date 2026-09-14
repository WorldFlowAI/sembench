"""The per-request metrics chunk captured off the wire is the parser's fixture.

Captured during phase-0 E5 (vLLM 0.29.0, 2026-09-14) by run-live-gateway
--metrics-chunk-output; it replaces the shape that was derived from source.
"""

from __future__ import annotations

import json
from pathlib import Path

from sembench.engine_metrics import engine_metrics_from_chunk

FIXTURES = Path(__file__).parent / "fixtures"
CAPTURED = FIXTURES / "vllm_0290_stream_chunk_captured.json"
DERIVED = FIXTURES / "vllm_0290_stream_chunk.json"


def _captured() -> dict:
    return json.loads(CAPTURED.read_text(encoding="utf-8"))


def test_the_captured_fixture_says_where_it_came_from():
    payload = _captured()
    assert "CAPTURED VERBATIM" in payload["_provenance"]
    assert payload["_captured"]["sembench_symbol"] == "sembench.gateway_live._chat_completion"
    assert payload["chunk"]["system_fingerprint"].startswith("vllm-0.29.0")
    assert payload["chunk"]["object"] == "chat.completion.chunk"


def test_the_raw_payload_parses_to_the_recorded_chunk():
    payload = _captured()
    assert json.loads(payload["raw_sse_data"]) == payload["chunk"]


def test_the_parser_reads_the_live_wire_shape():
    chunk = _captured()["chunk"]
    parsed = engine_metrics_from_chunk(chunk)
    assert parsed is not None
    assert parsed["time_to_first_token_ms"] == chunk["metrics"]["time_to_first_token_ms"]
    assert parsed["queue_time_ms"] == chunk["metrics"]["queue_time_ms"]


def test_the_live_wire_carries_no_external_token_field():
    """The only per-request cache fields vLLM 0.29 sends are the engine's own
    cached_tokens and created_cache_tokens; the external split must come from
    the connector audit join, never from the wire."""
    details = _captured()["chunk"]["usage"]["prompt_tokens_details"]
    assert set(details) == {"cached_tokens", "created_cache_tokens"}


def test_the_derived_shape_was_right():
    """Every key the derived fixture predicted is on the wire (created and
    model are per-run values the derivation did not try to pin)."""
    derived = json.loads(DERIVED.read_text(encoding="utf-8"))["chunk"]
    captured = _captured()["chunk"]
    assert set(derived) - set(captured) <= {"created", "model"}
    assert set(derived["metrics"]) <= set(captured["metrics"]) | {"speculative_decoding"}
    assert (
        "Replaced on 2026-09-14" in json.loads(DERIVED.read_text(encoding="utf-8"))["_provenance"]
    )
