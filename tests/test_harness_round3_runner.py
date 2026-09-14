"""Round-3 gaps in the live gateway runner and its CLI (B10, plus three bugs).

What these cover, and why each one was a hole rather than a nicety:

- **B10, the deterministic join key.** The runner sent no request id at all, so
  a connector audit event and a result row had nothing to join on and M1/M2/M7
  were unobtainable no matter how good the audit was. The id is now derived
  from (run id, arm, item, role, stream position) and sent as ``X-Request-Id``
  *and* as the request body's ``request_id``.
- **``_engine_metrics_urls`` spliced ``--worker-url`` raw**, so the comma form
  its own help text advertises produced one scrape target named
  ``http://a:8000,http://b:8000``. The arm then scraped nothing and looked
  like an engine that reports no external-KV counters.
- **``--connector-audit`` had no CLI surface**, so the audit join could only
  be run by hand and a published result carried no record of which audit file
  it was joined against.
- **The engine per-request metrics wire format was an assumption.** The parser
  read ``chunk["metrics"]`` without anyone checking what vLLM 0.29.0 actually
  emits on a streamed chat completion. It is now pinned against a fixture
  derived from the vLLM source, with the citations in the fixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sembench import gateway_live
from sembench.cli import _engine_metrics_urls, build_parser, main
from sembench.gateway_live import (
    LiveGatewayConfig,
    engine_metrics_from_chunk,
    engine_timing,
    run_live_gateway,
)
from sembench.request_ids import (
    ENGINE_REQUEST_ID_PREFIX,
    current_request_id,
    deterministic_request_id,
    donor_role,
    engine_request_id,
    row_request_id_field,
    sending_request_id,
    stamp_request_id,
)
from sembench.schema import DonorPrompt, RequestMetrics, WorkloadItem, write_jsonl

GATEWAY = "http://gateway:8000"
WORKERS = ("http://worker-0:8000", "http://worker-1:8000")
FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


class SpyGateway:
    """Chat endpoint double that records the id in flight for each call.

    The id travels beside the call rather than in its signature, so a double
    reads it the same way the real transport does.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, base_url, model, prompt, max_tokens, tenant, template, timeout_seconds):
        self.calls.append({"prompt": prompt, "request_id": current_request_id()})
        return {
            "output_text": "42",
            "usage": {"prompt_tokens": 64},
            "headers": {},
            "ttft_ms": 5.0,
            "latency_ms": 10.0,
        }

    def request_id_for(self, prompt: str) -> str | None:
        for call in self.calls:
            if call["prompt"] == prompt:
                return call["request_id"]
        raise AssertionError(f"{prompt!r} was never sent")


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


def _item(item_id: str, *, donors: int = 0) -> WorkloadItem:
    return WorkloadItem(
        item_id=item_id,
        dataset="fixture",
        source_id=f"s-{item_id}",
        transform="instruction_variant",
        donor_prompts=[
            DonorPrompt(donor_id=f"{item_id}-d{n}", text=f"DONOR {item_id}-{n}", label="doc")
            for n in range(donors)
        ],
        recipient_prompt=f"recipient-{item_id}",
        answers=["42"],
    )


def _config(tmp_path: Path, items: list[WorkloadItem], **overrides) -> LiveGatewayConfig:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, items)
    return LiveGatewayConfig(
        manifest=str(manifest),
        output=str(tmp_path / "result.json"),
        gateway_url=GATEWAY,
        model="qwen",
        **overrides,
    )


def _captured_request(monkeypatch, **kwargs) -> dict:
    """Run the real `_chat_completion` against a scripted stream; return the Request."""
    sent: dict = {}
    body = kwargs.pop(
        "body", [b'data: {"choices":[{"delta":{"content":"4"}}]}\n', b"data: [DONE]\n"]
    )

    def fake_urlopen(req, timeout=None):
        sent["request"] = req
        return FakeResponse(body)

    monkeypatch.setattr(gateway_live, "urlopen", fake_urlopen)
    call = {
        "base_url": GATEWAY,
        "model": "qwen",
        "prompt": "p",
        "max_tokens": 8,
        "tenant": "t",
        "template": "tpl",
        "timeout_seconds": 5.0,
    }
    call.update(kwargs)
    sent["response"] = gateway_live._chat_completion(**call)
    return sent


def _fixture_chunk() -> dict:
    payload = json.loads((FIXTURES / "vllm_0290_stream_chunk.json").read_text(encoding="utf-8"))
    return payload["chunk"]


# --------------------------------------------------------------------------
# B10: the id is derived, not random
# --------------------------------------------------------------------------


def test_the_request_id_is_a_pure_function_of_the_five_identifying_fields():
    args = {
        "run_id": "r1",
        "arm": "warm",
        "item_id": "item-7",
        "role": "recipient",
        "stream_position": 12,
    }
    assert deterministic_request_id(**args) == deterministic_request_id(**args)


@pytest.mark.parametrize(
    "changed",
    [
        {"run_id": "r2"},
        {"arm": "cold"},
        {"item_id": "item-8"},
        {"role": "donor-0"},
        {"stream_position": 13},
    ],
)
def test_changing_any_identifying_field_changes_the_request_id(changed):
    """Otherwise two arms of one run collide in a shared audit file."""
    base = {
        "run_id": "r1",
        "arm": "warm",
        "item_id": "item-7",
        "role": "recipient",
        "stream_position": 12,
    }
    assert deterministic_request_id(**base) != deterministic_request_id(**{**base, **changed})


def test_the_request_id_is_header_safe_whatever_the_item_id_is():
    """An id that cannot be encoded into a header takes the whole request down."""
    request_id = deterministic_request_id(
        run_id="r1",
        arm="warm",
        item_id="longbench/qasper — 文書 #3",
        role="recipient",
        stream_position=1,
    )
    assert request_id.encode("latin-1")
    assert set(request_id) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")


def test_the_readable_head_names_the_arm_and_the_stream_position():
    """So a human can find one request in an audit file without re-deriving it."""
    request_id = deterministic_request_id(
        run_id="r1", arm="cold", item_id="i", role=donor_role(2), stream_position=9
    )
    assert request_id.startswith("sembench-cold-000009-donor-2-")


def test_the_engine_form_is_the_sent_id_under_vllms_prefix():
    """vLLM's chat handler builds chatcmpl-<header>; the audit records that."""
    assert engine_request_id("abc") == f"{ENGINE_REQUEST_ID_PREFIX}abc" == "chatcmpl-abc"


# --------------------------------------------------------------------------
# B10: the id actually reaches the wire
# --------------------------------------------------------------------------


def test_the_request_id_is_sent_as_a_header_and_as_a_body_field(monkeypatch):
    """The header is what vLLM prefers; the body field survives a front end
    that strips unknown headers."""
    with sending_request_id("sembench-warm-000001-recipient-abc"):
        sent = _captured_request(monkeypatch)

    request = sent["request"]
    assert request.get_header("X-request-id") == "sembench-warm-000001-recipient-abc"
    assert json.loads(request.data)["request_id"] == "sembench-warm-000001-recipient-abc"


def test_no_request_id_means_no_header_and_no_body_field(monkeypatch):
    """An id-less call must not send an empty one: vLLM would adopt it."""
    sent = _captured_request(monkeypatch)

    assert sent["request"].get_header("X-request-id") is None
    assert "request_id" not in json.loads(sent["request"].data)


def test_usage_reporting_stays_on_because_the_metrics_ride_the_usage_chunk(monkeypatch):
    sent = _captured_request(monkeypatch)
    assert json.loads(sent["request"].data)["stream_options"] == {"include_usage": True}


def test_the_engines_own_id_comes_back_on_the_response(monkeypatch):
    """Captured so a gateway that rewrites ids in flight is visible."""
    body = [
        b'data: {"id":"chatcmpl-sembench-x","choices":[{"delta":{"content":"4"}}]}\n',
        b"data: [DONE]\n",
    ]
    sent = _captured_request(monkeypatch, body=body)
    assert sent["response"]["response_id"] == "chatcmpl-sembench-x"


def test_a_failed_request_still_reports_the_id_it_was_issued_under(monkeypatch):
    from urllib.error import URLError

    def fake_urlopen(req, timeout=None):
        raise URLError("connection refused")

    monkeypatch.setattr(gateway_live, "urlopen", fake_urlopen)
    with sending_request_id("sembench-warm-000001-recipient-abc"):
        response = gateway_live._chat_completion(
            base_url=GATEWAY,
            model="qwen",
            prompt="p",
            max_tokens=8,
            tenant="t",
            template="tpl",
            timeout_seconds=5.0,
        )
    assert response["error"]
    assert response["request_id"] == "sembench-warm-000001-recipient-abc"


def test_the_id_does_not_leak_out_of_its_block():
    """A pooled worker thread must not carry one request's id into the next."""
    with sending_request_id("a"):
        assert current_request_id() == "a"
    assert current_request_id() is None


# --------------------------------------------------------------------------
# B10: the runner derives one id per request and stamps the row
# --------------------------------------------------------------------------


def test_every_replayed_request_is_issued_under_its_own_derived_id(tmp_path, monkeypatch):
    spy = SpyGateway()
    monkeypatch.setattr(gateway_live, "_chat_completion", spy)

    run_live_gateway(_config(tmp_path, [_item("a", donors=2), _item("b")], run_id="run-1"))

    ids = [call["request_id"] for call in spy.calls]
    assert all(ids), "every request must be issued under an id"
    assert len(set(ids)) == len(ids), "two requests shared an id; the audit join would be wrong"


def test_donor_and_recipient_ids_are_distinct_per_role(tmp_path, monkeypatch):
    """A capture event must be attributable to the request that seeded it."""
    spy = SpyGateway()
    monkeypatch.setattr(gateway_live, "_chat_completion", spy)

    run_live_gateway(_config(tmp_path, [_item("a", donors=2)], run_id="run-1"))

    for index, prompt in enumerate(("DONOR a-0", "DONOR a-1")):
        assert spy.request_id_for(prompt) == deterministic_request_id(
            run_id="run-1", arm="single", item_id="a", role=donor_role(index), stream_position=0
        )
    assert spy.request_id_for("recipient-a") == deterministic_request_id(
        run_id="run-1", arm="single", item_id="a", role="recipient", stream_position=0
    )


def test_the_row_carries_the_id_the_runner_sent(tmp_path, monkeypatch):
    """The join key on the row is the header id, which is what
    sembench.connector_audit resolves an audited engine id back to."""
    spy = SpyGateway()
    monkeypatch.setattr(gateway_live, "_chat_completion", spy)

    rows = run_live_gateway(_config(tmp_path, [_item("a"), _item("b")], run_id="run-1"))

    field = row_request_id_field(RequestMetrics)
    assert field is not None, "the row schema has no field for the audit join key"
    for row, sent in zip(rows, ("recipient-a", "recipient-b"), strict=True):
        assert getattr(row, field) == spy.request_id_for(sent)


def test_the_two_arms_of_a_paired_run_get_different_ids(tmp_path, monkeypatch):
    """Cold and warm replay the same item; one id for both would collapse them."""
    spy = SpyGateway()
    monkeypatch.setattr(gateway_live, "_chat_completion", spy)
    monkeypatch.setattr(gateway_live, "reset_engine_caches", lambda *a, **k: True)

    rows = run_live_gateway(
        _config(
            tmp_path,
            [_item("a", donors=1)],
            run_id="run-1",
            paired=True,
            reset_urls=("http://worker-0:8000/reset_prefix_cache",),
        )
    )

    field = row_request_id_field(RequestMetrics)
    ids = [getattr(row, field) for row in rows]
    assert len(rows) == 2
    assert len(set(ids)) == 2


def test_the_row_id_survives_a_row_schema_without_the_field():
    """The runner must not take a run down over a field it can re-derive."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Bare:
        item_id: str

    row = Bare(item_id="a")
    assert stamp_request_id(row, "sembench-x") == row


def test_stamping_builds_a_new_row_rather_than_mutating_one():
    row = RequestMetrics(
        item_id="a",
        dataset="d",
        transform="t",
        negative_control=False,
        donor_count=0,
        prompt_tokens=1,
        total_blocks=1,
        exact_hit_blocks=0,
        exact_hit_tokens=0,
        semantic_candidate_blocks=0,
        semantic_candidate_tokens=0,
        semantic_eligible_blocks=0,
        semantic_eligible_tokens=0,
    )
    stamped = stamp_request_id(row, "sembench-x")

    field = row_request_id_field(RequestMetrics)
    assert getattr(stamped, field) == "sembench-x"
    assert getattr(row, field) is None


def test_the_cli_stamps_the_runs_own_id_onto_the_requests(tmp_path, monkeypatch):
    """A run whose result says one run id while its requests were derived from
    another cannot be joined to its audit."""
    spy = SpyGateway()
    monkeypatch.setattr(gateway_live, "_chat_completion", spy)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_item("a")])
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
            "--run-id",
            "run-xyz",
            "--skip-verify",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run"]["run_id"] == "run-xyz"
    assert payload["config"]["run_id"] == "run-xyz"
    expected = deterministic_request_id(
        run_id="run-xyz", arm="single", item_id="a", role="recipient", stream_position=0
    )
    assert spy.request_id_for("recipient-a") == expected


def test_the_cli_generates_a_run_id_when_none_is_given(tmp_path, monkeypatch):
    """The generated id has to reach the requests, not just the result."""
    spy = SpyGateway()
    monkeypatch.setattr(gateway_live, "_chat_completion", spy)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_item("a")])
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

    payload = json.loads(output.read_text(encoding="utf-8"))
    run_id = payload["run"]["run_id"]
    assert run_id
    assert payload["config"]["run_id"] == run_id
    assert spy.request_id_for("recipient-a") == deterministic_request_id(
        run_id=run_id, arm="single", item_id="a", role="recipient", stream_position=0
    )


def test_the_sent_id_resolves_against_an_audit_that_recorded_the_engine_form(tmp_path):
    """The end the whole key exists for: a row id joins an audited chatcmpl- id."""
    from sembench.connector_audit import load_audit

    sent = deterministic_request_id(
        run_id="run-1", arm="warm", item_id="a", role="recipient", stream_position=0
    )
    audit = tmp_path / "audit.jsonl"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "event": "runtime_materialized",
                "connector_id": "c-1",
                "request_id": engine_request_id(sent),
                "request_seq": 1,
                "event_seq": 0,
                "tokens": 512,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_audit(audit).resolve(sent).record is not None


# --------------------------------------------------------------------------
# --connector-audit
# --------------------------------------------------------------------------


def _audit_file(tmp_path: Path, request_ids: list[str], *, tokens: int = 512) -> Path:
    path = tmp_path / "audit.jsonl"
    lines = []
    for seq, request_id in enumerate(request_ids, start=1):
        lines.append(
            json.dumps(
                {
                    "schema_version": 2,
                    "event": "runtime_materialized",
                    "connector_id": "c-1",
                    "request_id": engine_request_id(request_id),
                    "request_seq": seq,
                    "event_seq": 0,
                    "tokens": tokens,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("command", ["run-live-gateway", "merge-results"])
def test_both_commands_accept_the_connector_audit_flag(command):
    subparsers = next(
        action
        for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    options = {
        option
        for action in subparsers.choices[command]._actions
        for option in action.option_strings
    }
    assert "--connector-audit" in options


def test_a_missing_audit_file_is_refused_before_the_arm_runs(tmp_path, monkeypatch):
    """A typo must not read downstream as an arm that materialized nothing."""
    spy = SpyGateway()
    monkeypatch.setattr(gateway_live, "_chat_completion", spy)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_item("a")])

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "run-live-gateway",
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "result.json"),
                "--gateway-url",
                GATEWAY,
                "--model",
                "qwen",
                "--skip-verify",
                "--connector-audit",
                str(tmp_path / "nope.jsonl"),
            ]
        )

    assert "connector audit" in str(excinfo.value)
    assert not spy.calls, "the arm ran before the audit path was checked"


def test_the_audit_is_joined_onto_the_rows_and_recorded_in_the_result(tmp_path, monkeypatch):
    spy = SpyGateway()
    monkeypatch.setattr(gateway_live, "_chat_completion", spy)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_item("a")])
    output = tmp_path / "result.json"
    sent = deterministic_request_id(
        run_id="run-xyz", arm="single", item_id="a", role="recipient", stream_position=0
    )
    audit = _audit_file(tmp_path, [sent])

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
            "--run-id",
            "run-xyz",
            "--skip-verify",
            "--connector-audit",
            str(audit),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["connector_audit"] == str(audit)
    assert payload["config"]["connector_audit_join"]["rows_total"] == 1
    assert payload["config"]["connector_audit_join"]["rows_unmatched"] == 0
    assert payload["requests"][0]["audit_joined"] is True
    assert payload["requests"][0]["external_confirmed_tokens"] == 512


def test_without_the_flag_nothing_about_the_audit_is_claimed(tmp_path, monkeypatch):
    """No audit means null, not zero: a run that measured nothing must not
    publish a clean score."""
    monkeypatch.setattr(gateway_live, "_chat_completion", SpyGateway())
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [_item("a")])
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

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["connector_audit"] is None
    assert payload["config"]["connector_audit_join"] is None
    assert payload["requests"][0]["audit_joined"] is None
    assert payload["aggregate"]["materialized_reuse_rate"] is None


# --------------------------------------------------------------------------
# _engine_metrics_urls: the comma form its own help text advertises
# --------------------------------------------------------------------------


def _gateway_args(*extra: str):
    return build_parser().parse_args(
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
            *extra,
        ]
    )


def test_a_comma_separated_worker_list_becomes_separate_scrape_targets():
    args = _gateway_args("--worker-url", f"{WORKERS[0]},{WORKERS[1]}")
    assert _engine_metrics_urls(args) == [WORKERS[0], WORKERS[1], GATEWAY]


def test_the_repeatable_and_comma_forms_scrape_the_same_endpoints():
    """They are documented as equivalent; they now are."""
    comma = _gateway_args("--worker-url", f"{WORKERS[0]},{WORKERS[1]}")
    repeated = _gateway_args("--worker-url", WORKERS[0], "--worker-url", WORKERS[1])
    assert _engine_metrics_urls(comma) == _engine_metrics_urls(repeated)


def test_a_comma_separated_metrics_url_is_split_too():
    args = _gateway_args("--metrics-url", f"{WORKERS[0]},{WORKERS[1]}")
    assert _engine_metrics_urls(args) == [WORKERS[0], WORKERS[1]]


def test_a_duplicated_endpoint_is_scraped_once():
    args = _gateway_args("--worker-url", f"{WORKERS[0]},{WORKERS[0]}", "--donor-url", WORKERS[0])
    assert _engine_metrics_urls(args) == [WORKERS[0], GATEWAY]


def test_an_absent_donor_url_contributes_no_scrape_target():
    args = _gateway_args("--worker-url", WORKERS[0])
    assert _engine_metrics_urls(args) == [WORKERS[0], GATEWAY]


# --------------------------------------------------------------------------
# The engine per-request metrics wire format, pinned to the vLLM source
# --------------------------------------------------------------------------


def test_the_fixture_says_it_is_derived_and_must_be_replaced():
    """The fixture is an argument from source, not evidence. Say so in the file."""
    payload = json.loads((FIXTURES / "vllm_0290_stream_chunk.json").read_text(encoding="utf-8"))
    assert "DERIVED FROM SOURCE, NOT CAPTURED" in payload["_provenance"]
    assert "E5" in payload["_provenance"]
    assert payload["_source"]["metrics_model"].startswith(
        "vllm/entrypoints/generate/base/protocol.py:55-62"
    )


def test_the_fixture_carries_exactly_the_documented_metric_fields():
    """PerRequestMetrics, vllm/entrypoints/generate/base/protocol.py:55-62.
    A field invented here would be a parser tuned to fiction."""
    assert set(_fixture_chunk()["metrics"]) <= {
        "time_to_first_token_ms",
        "generation_time_ms",
        "queue_time_ms",
        "mean_itl_ms",
        "tokens_per_second",
        "speculative_decoding",
    }


def test_the_metrics_object_is_top_level_on_the_final_usage_chunk():
    """Not nested under usage, and on a chunk that carries no choices."""
    chunk = _fixture_chunk()
    assert chunk["choices"] == []
    assert chunk["usage"]
    assert "metrics" not in chunk["usage"]
    assert engine_metrics_from_chunk(chunk) is chunk["metrics"]


def test_the_parser_reads_the_documented_shape():
    timing = engine_timing({"metrics": _fixture_chunk()["metrics"]})
    assert timing["engine_ttft_ms"] == pytest.approx(412.7)
    assert timing["queue_time_ms"] == pytest.approx(18.2)


def test_the_engine_ttft_excludes_the_queue_wait_it_is_reported_beside():
    """vllm/entrypoints/generate/base/serving.py:78-85: TTFT is measured from
    scheduling, so the two numbers are disjoint and must not be summed."""
    timing = engine_timing({"metrics": _fixture_chunk()["metrics"]})
    assert timing["engine_ttft_ms"] > timing["queue_time_ms"]


def test_the_whole_documented_chunk_parses_off_a_stream(monkeypatch):
    """End to end: the fixture serialized as SSE, through the real parser."""
    chunk = _fixture_chunk()
    body = [
        b'data: {"id":"chatcmpl-x","choices":[{"delta":{"content":"4"}}]}\n',
        b"data: " + json.dumps(chunk).encode("utf-8") + b"\n",
        b"data: [DONE]\n",
    ]
    sent = _captured_request(monkeypatch, body=body)

    response = sent["response"]
    assert response["metrics"] == chunk["metrics"]
    assert response["usage"]["prompt_tokens_details"]["cached_tokens"] == 3072
    assert engine_timing(response)["engine_ttft_ms"] == pytest.approx(412.7)


def test_a_non_mapping_metrics_field_is_not_stored_as_metrics(monkeypatch):
    """ "The engine reported timings this runner could not parse" is a
    different claim from "the engine reported none", and only one of them is
    true when a gateway puts a string there."""
    body = [
        b'data: {"choices":[{"delta":{"content":"4"}}],"metrics":"unavailable"}\n',
        b"data: [DONE]\n",
    ]
    sent = _captured_request(monkeypatch, body=body)

    assert sent["response"]["metrics"] == {}
    assert engine_metrics_from_chunk({"metrics": ["not", "a", "mapping"]}) is None


def test_a_metrics_object_with_only_speculative_decoding_yields_no_timings():
    """exclude_none=True drops the timing fields when their timestamps were
    unavailable, leaving the object present but timing-free."""
    timing = engine_timing({"metrics": {"speculative_decoding": {"num_drafts": 4}}})
    assert timing == {"engine_ttft_ms": None, "queue_time_ms": None}


# --------------------------------------------------------------------------
# The docs: what they promise has to be what the code does
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


def _flat(name: str) -> str:
    """A doc with its line wrapping collapsed, so a reflow is not a test failure."""
    return " ".join((ROOT / name).read_text(encoding="utf-8").split())


FLAT_README = _flat("README.md")
FLAT_METRICS = _flat("docs/METRICS.md")


def test_the_docs_no_longer_claim_cached_tokens_reaches_no_hit_rate_and_no_gate():
    """It reaches both: `hit_rate` falls back to it, and
    --min-backend-confirmed-block-rate gates a rate derived from it."""
    assert "neither the hit rate nor any gate is computed from `cached_tokens`" not in FLAT_README
    assert "The hit rate, `reuse_mechanism`, and every speedup gate read" not in FLAT_METRICS


def test_the_docs_name_the_strict_hit_rate_and_disown_the_legacy_one():
    for text in (FLAT_README, FLAT_METRICS):
        assert "hit_rate_external_confirmed" in text
        assert "must not be quoted for a vLLM run with prefix caching on" in text


def test_the_docs_say_the_strict_hit_rate_is_null_until_the_audit_lands():
    """Null, not zero: a gate that refuses null must fail rather than pass."""
    assert "`null` — never `0.0` — until the connector audit" in FLAT_README
    assert "is `None` — not `0.0` — when no pair carried the split" in FLAT_METRICS


def test_the_docs_name_the_gate_that_does_read_cached_tokens():
    for text in (FLAT_README, FLAT_METRICS):
        assert "--min-backend-confirmed-block-rate" in text
        assert "backend_confirmed_block_rate" in text


def test_the_connector_audit_flag_is_documented_and_accepted():
    assert "--connector-audit" in FLAT_README
    assert "--connector-audit" in FLAT_METRICS
    subparsers = next(
        action
        for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    for command in ("run-live-gateway", "merge-results"):
        options = {
            option
            for action in subparsers.choices[command]._actions
            for option in action.option_strings
        }
        assert "--connector-audit" in options


def test_the_documented_audit_metric_keys_are_the_keys_the_code_emits():
    """The drift this whole round exists to stop: a metric named in the docs
    that the result does not carry."""
    from sembench.results import connector_audit_metrics, paired_summary

    emitted = set(connector_audit_metrics([]))
    for key in (
        "alignment_given_match",
        "alignment_given_opportunity",
        "boundary_miss_breakdown",
        "boundary_alignment_rate",
        "materialized_reuse_rate",
        "materialized_reuse_token_rate",
        "materialized_reuse_request_rate",
        "propagation_cached_without_materialization_rate",
        "prefix_blocks_evicted",
        "connector_audit_present",
        "connector_audit_rows_joined",
    ):
        assert key in emitted, f"{key} is documented but the result does not carry it"
        assert key in FLAT_README
        assert key in FLAT_METRICS

    # M7 is a cross-arm comparison and lives in the paired block, not in the
    # per-arm audit metrics.
    paired = paired_summary(_paired_rows_for_key_check())
    assert paired is not None
    for key in ("propagation_contamination_rate", "propagation_probes_without_served_answer"):
        assert key in paired, f"{key} is documented but the paired summary does not carry it"
        assert key in FLAT_README
        assert key in FLAT_METRICS


def _paired_rows_for_key_check() -> list[RequestMetrics]:
    """One cold/warm pair, enough for paired_summary to return a document."""
    return [
        RequestMetrics(**_arm_row("i1", ttft=200.0, arm="cold")),
        RequestMetrics(**_arm_row("i1", ttft=100.0, arm="warm")),
    ]


def test_the_docs_map_each_audit_metric_to_its_spec_number():
    for text in (FLAT_README, FLAT_METRICS):
        assert "M1" in text and "M2" in text and "M7" in text


def test_the_docs_record_the_service_latency_change():
    """A reader comparing a round-1 latency with a round-3 one has to be told."""
    assert "service" in FLAT_README.lower()
    assert "latency_ms" in FLAT_README
    assert "SERVICE latency" in FLAT_METRICS
    assert "no longer the item's wall span" in FLAT_METRICS


def test_the_docs_describe_the_join_key_the_runner_actually_sends():
    for text in (FLAT_README, FLAT_METRICS):
        assert "X-Request-Id" in text
        assert "chatcmpl-" in text
    assert "run_id, arm, item_id, role, stream_position" in FLAT_README
    assert "(run_id, arm, item_id, role, stream_position)" in FLAT_METRICS


def test_the_docs_cite_the_vllm_source_for_the_metrics_wire_format():
    """The claim is only as good as the file:line behind it."""
    assert "chat_completion/protocol.py:180" in FLAT_METRICS
    assert "generate/base/protocol.py:55-62" in FLAT_METRICS
    assert "generate/base/serving.py:78-85" in FLAT_METRICS
    assert "tests/fixtures/vllm_0290_stream_chunk.json" in FLAT_METRICS


# --------------------------------------------------------------------------
# --connector-audit on merge-results, where the two arms ran as separate
# server processes and only the warm one has an audit
# --------------------------------------------------------------------------


def _arm_result(path: Path, rows: list[dict], *, run_id: str) -> str:
    path.write_text(
        json.dumps(
            {
                "result_version": "sembench.result.v2",
                "run": {
                    "run_id": run_id,
                    "engine": "gateway",
                    "manifest_path": "m.jsonl",
                    "manifest_sha256": "c" * 64,
                    "arm": "single",
                },
                "config": {"mode": "live-gateway"},
                "requests": rows,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _arm_row(item_id: str, *, ttft: float, arm: str, request_id: str | None = None) -> dict:
    row = RequestMetrics(
        item_id=item_id,
        dataset="t",
        transform="instruction_variant",
        negative_control=False,
        donor_count=1,
        prompt_tokens=64,
        total_blocks=4,
        exact_hit_blocks=0,
        exact_hit_tokens=0,
        semantic_candidate_blocks=0,
        semantic_candidate_tokens=0,
        semantic_eligible_blocks=0,
        semantic_eligible_tokens=0,
        ttft_ms=ttft,
        latency_ms=ttft * 1.2,
        output_text="the answer is 42",
        arm=arm,
    ).to_dict()
    field = row_request_id_field(RequestMetrics)
    if request_id is not None and field is not None:
        row[field] = request_id
    return row


def test_merge_results_joins_the_warm_arms_audit_onto_the_merged_rows(tmp_path):
    warm_id = deterministic_request_id(
        run_id="warm-run", arm="warm", item_id="i1", role="recipient", stream_position=0
    )
    cold = _arm_result(
        tmp_path / "cold.json", [_arm_row("i1", ttft=200.0, arm="cold")], run_id="cold-run"
    )
    warm = _arm_result(
        tmp_path / "warm.json",
        [_arm_row("i1", ttft=50.0, arm="warm", request_id=warm_id)],
        run_id="warm-run",
    )
    output = tmp_path / "paired.json"

    main(
        [
            "merge-results",
            "--cold",
            cold,
            "--warm",
            warm,
            "--output",
            str(output),
            "--connector-audit",
            str(_audit_file(tmp_path, [warm_id])),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    report = payload["config"]["connector_audit_join"]
    # Normalized, not exact: the row carries the header id the runner sent and
    # the audit carries vLLM's chatcmpl- form of it, which is the asymmetry
    # sembench.connector_audit exists to resolve.
    assert report["rows_matched_normalized"] == 1
    assert report["rows_unmatched"] == 0
    # The cold arm carries no id and must be reported as such, not as a miss
    # the audit ruled on.
    assert report["rows_without_request_id"] == 1
    assert payload["aggregate"]["connector_audit_present"] is True


def test_merge_results_refuses_a_missing_audit_before_it_merges(tmp_path):
    cold = _arm_result(
        tmp_path / "cold.json", [_arm_row("i1", ttft=200.0, arm="cold")], run_id="cold-run"
    )
    warm = _arm_result(
        tmp_path / "warm.json", [_arm_row("i1", ttft=50.0, arm="warm")], run_id="warm-run"
    )
    output = tmp_path / "paired.json"

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "merge-results",
                "--cold",
                cold,
                "--warm",
                warm,
                "--output",
                str(output),
                "--connector-audit",
                str(tmp_path / "nope.jsonl"),
            ]
        )

    assert "connector audit" in str(excinfo.value)
    assert not output.exists()
