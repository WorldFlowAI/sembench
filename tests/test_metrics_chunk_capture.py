"""`--metrics-chunk-output`: the run captures the wire format instead of asserting it.

`tests/fixtures/vllm_0290_stream_chunk.json` is derived from vLLM source, not
captured, and the phase-0 handoff makes replacing it a deliverable of the E5
GPU smoke. These tests cover the capture path that produces the replacement:
that it writes the RAW payload rather than a re-encoding, that it writes once
and only once (including across threads), that a donor ping cannot be the
chunk that gets captured, and that the file it writes is shaped like the
fixture it replaces.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from sembench import gateway_live
from sembench.cli import build_parser
from sembench.engine_metrics import engine_metrics_from_chunk
from sembench.gateway_live import LiveGatewayConfig
from sembench.metrics_chunk import MetricsChunkCapture, capture_for
from sembench.schema import DonorPrompt, WorkloadItem, write_jsonl

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GATEWAY = "http://gw:8000"

# The exact bytes a vLLM 0.29.0 final usage chunk arrives as, spacing and key
# order included, so "verbatim" can actually be asserted: a re-encoding would
# normalize both.
RAW_METRICS_CHUNK = (
    '{"id":"chatcmpl-x", "object":"chat.completion.chunk", "choices":[], '
    '"usage":{"prompt_tokens":4096,"completion_tokens":256,'
    '"prompt_tokens_details":{"cached_tokens":3072}}, '
    '"metrics":{"time_to_first_token_ms":412.7,"queue_time_ms":18.2}}'
)


class FakeResponse:
    """Minimal streaming HTTP response over a scripted list of SSE lines."""

    def __init__(self, body: list[bytes], headers: dict | None = None) -> None:
        self.body = body
        self.headers = headers or {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def __iter__(self):
        return iter(self.body)


def _stream_body() -> list[bytes]:
    return [
        b'data: {"id":"chatcmpl-x","choices":[{"delta":{"content":"4"}}]}\n',
        f"data: {RAW_METRICS_CHUNK}\n".encode(),
        b"data: [DONE]\n",
    ]


def _item(item_id: str) -> WorkloadItem:
    return WorkloadItem(
        item_id=item_id,
        dataset="fixture",
        source_id="src",
        transform="no_reuse",
        donor_prompts=[],
        recipient_prompt="prompt body",
        input="q",
        answers=["a"],
    )


# --------------------------------------------------------------------------
# the capture object itself
# --------------------------------------------------------------------------


def test_the_capture_writes_the_raw_payload_not_a_re_encoding(tmp_path):
    path = tmp_path / "metrics-chunk.json"
    capture = MetricsChunkCapture(path=str(path), run_id="r1", pairing_arm="single")

    assert capture.offer(raw=RAW_METRICS_CHUNK, chunk=json.loads(RAW_METRICS_CHUNK)) is True

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["raw_sse_data"] == RAW_METRICS_CHUNK
    # json.dumps of the parsed object would not carry the engine's spacing.
    assert json.dumps(document["chunk"]) != document["raw_sse_data"]


def test_the_capture_is_a_drop_in_for_the_derived_fixture(tmp_path):
    """The fixture's readers take `payload["chunk"]`; so must the captured file."""
    derived = json.loads((FIXTURES / "vllm_0290_stream_chunk.json").read_text(encoding="utf-8"))
    path = tmp_path / "metrics-chunk.json"
    MetricsChunkCapture(path=str(path)).offer(
        raw=RAW_METRICS_CHUNK, chunk=json.loads(RAW_METRICS_CHUNK)
    )
    captured = json.loads(path.read_text(encoding="utf-8"))

    assert set(derived["chunk"]) - set(captured["chunk"]) <= {"created", "model"}
    assert engine_metrics_from_chunk(captured["chunk"]) is not None
    assert "CAPTURED VERBATIM" in captured["_provenance"]
    # It says what the replacement costs, so nobody replaces the fixture and
    # then wonders why an assertion about it went red.
    assert "DERIVED FROM SOURCE, NOT CAPTURED" in captured["_provenance"]


def test_only_the_first_chunk_is_captured(tmp_path):
    path = tmp_path / "metrics-chunk.json"
    capture = MetricsChunkCapture(path=str(path))

    assert capture.offer(raw=RAW_METRICS_CHUNK, chunk={"metrics": {"a": 1}}) is True
    assert capture.offer(raw="second", chunk={"metrics": {"a": 2}}) is False
    assert json.loads(path.read_text(encoding="utf-8"))["raw_sse_data"] == RAW_METRICS_CHUNK


def test_concurrent_offers_write_exactly_one_file(tmp_path):
    """The dispatcher runs requests on worker threads; without the lock two of
    them could both see "not yet written" and race on the same path."""
    path = tmp_path / "metrics-chunk.json"
    capture = MetricsChunkCapture(path=str(path))
    start = threading.Barrier(8)
    wrote: list[bool] = []
    lock = threading.Lock()

    def offer(index: int) -> None:
        start.wait()
        result = capture.offer(raw=f"chunk-{index}", chunk={"metrics": {"i": index}})
        with lock:
            wrote.append(result)

    threads = [threading.Thread(target=offer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert wrote.count(True) == 1
    assert path.exists()


def test_no_path_means_no_capture():
    assert capture_for(None) is None
    assert capture_for("") is None
    assert capture_for("/tmp/x").path == "/tmp/x"


# --------------------------------------------------------------------------
# wired through the runner
# --------------------------------------------------------------------------


def test_a_run_captures_the_first_metrics_chunk_off_the_wire(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gateway_live, "urlopen", lambda req, timeout=None: FakeResponse(_stream_body())
    )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_item("i1"), _item("i2")])
    chunk_path = tmp_path / "metrics-chunk-a4.json"

    gateway_live.run_live_gateway(
        LiveGatewayConfig(
            manifest=str(manifest),
            output=str(tmp_path / "result.json"),
            gateway_url=GATEWAY,
            model="qwen",
            run_id="phase0-a4_conn_span-20260914T000000Z",
            metrics_chunk_output=str(chunk_path),
        )
    )

    document = json.loads(chunk_path.read_text(encoding="utf-8"))
    assert document["raw_sse_data"] == RAW_METRICS_CHUNK
    # The PHASE-0 arm is not a field of LiveGatewayConfig; it reaches the run
    # as --backend-id and inside --run-id, which is what the capture records.
    assert document["_captured"]["run_id"] == "phase0-a4_conn_span-20260914T000000Z"
    assert document["_captured"]["pairing_arm"] == "single"
    assert document["chunk"]["metrics"]["time_to_first_token_ms"] == 412.7


def test_a_run_without_the_flag_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gateway_live, "urlopen", lambda req, timeout=None: FakeResponse(_stream_body())
    )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_item("i1")])

    gateway_live.run_live_gateway(
        LiveGatewayConfig(
            manifest=str(manifest),
            output=str(tmp_path / "result.json"),
            gateway_url=GATEWAY,
            model="qwen",
        )
    )

    assert list(tmp_path.glob("metrics-chunk*")) == []


def test_a_donor_ping_is_never_the_captured_chunk(tmp_path, monkeypatch):
    """A donor asks for one output token. Its final chunk carries a metrics
    object too, and it would describe a generation nothing in phase 0 measures."""
    seen: list[str] = []

    def fake_chat_completion(**kwargs):
        seen.append("metrics_chunk" if kwargs.get("metrics_chunk") is not None else "none")
        return {"output_text": "", "usage": {}, "metrics": {}, "headers": {}}

    monkeypatch.setattr(gateway_live, "_chat_completion", fake_chat_completion)
    item = WorkloadItem(
        item_id="i1",
        dataset="fixture",
        source_id="src",
        transform="same_doc_new_instruction",
        donor_prompts=[DonorPrompt(donor_id="d1", text="donor", label="d")],
        recipient_prompt="recipient",
        input="q",
        answers=["a"],
    )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [item])

    gateway_live.run_live_gateway(
        LiveGatewayConfig(
            manifest=str(manifest),
            output=str(tmp_path / "result.json"),
            gateway_url=GATEWAY,
            model="qwen",
            metrics_chunk_output=str(tmp_path / "metrics-chunk.json"),
        )
    )

    assert seen == ["none", "metrics_chunk"]


def test_the_cli_exposes_the_flag_and_threads_it_into_the_config():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-live-gateway",
            "--manifest",
            "m.jsonl",
            "--output",
            "o.json",
            "--gateway-url",
            GATEWAY,
            "--model",
            "qwen",
            "--metrics-chunk-output",
            "/results/metrics-chunk-a4.json",
        ]
    )
    assert args.metrics_chunk_output == "/results/metrics-chunk-a4.json"
    assert "metrics_chunk_output" in LiveGatewayConfig.__dataclass_fields__


@pytest.mark.parametrize("doc", ["README.md", "docs/METRICS.md"])
def test_the_flag_is_documented(doc):
    root = Path(__file__).resolve().parents[1]
    assert "--metrics-chunk-output" in (root / doc).read_text(encoding="utf-8")
