"""Round-2 gaps in the live gateway runner (B11, B12, B15, B16).

Four of these are the difference between a flag existing and a flag working:
``--worker-url`` was unreachable from the CLI, the engine-side per-request
metrics the protocol mandates were never read off the stream,
``--min-donor-gap-requests`` was silently inert at the default concurrency of
1 -- which is the width every TTFT and quality arm runs at -- and ``--paired``
would happily manufacture a pair out of a manifest with no donors to seed.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from sembench import gateway_live
from sembench.cli import _engine_metrics_urls, build_parser, main
from sembench.gateway_live import (
    LiveGatewayConfig,
    engine_timing,
    run_live_gateway,
    run_live_gateway_measured,
)
from sembench.schema import DonorPrompt, WorkloadItem, write_jsonl

GATEWAY = "http://gateway:8000"
WORKERS = ("http://worker-0:8000", "http://worker-1:8000", "http://worker-2:8000")


class Recorder:
    """Scripted chat endpoint that records what it was asked and when.

    Keeps the completion count at the moment each request started, which is
    the unit the donor gap is measured in.
    """

    def __init__(self, *, service_seconds: float = 0.0, metrics: dict | None = None) -> None:
        self.service_seconds = service_seconds
        self.metrics = metrics
        self._lock = threading.Lock()
        self.calls: list[dict] = []
        self.started: list[tuple[str, int, float]] = []
        self.completed: list[str] = []

    def __call__(self, *, base_url, model, prompt, max_tokens, tenant, template, timeout_seconds):
        with self._lock:
            self.started.append((prompt, len(self.completed), time.perf_counter()))
            self.calls.append({"base_url": base_url, "prompt": prompt, "max_tokens": max_tokens})
        if self.service_seconds:
            time.sleep(self.service_seconds)
        with self._lock:
            self.completed.append(prompt)
        response = {
            "output_text": "42",
            "usage": {"prompt_tokens": 64, "completion_tokens": 4},
            "headers": {},
            "ttft_ms": 5.0,
            "latency_ms": 10.0,
        }
        if self.metrics is not None:
            response["metrics"] = dict(self.metrics)
        return response

    def base_urls(self, kind: str) -> list[str]:
        return [call["base_url"] for call in self.calls if _kind(call["prompt"]) == kind]

    def max_tokens(self, kind: str) -> list[int]:
        return [call["max_tokens"] for call in self.calls if _kind(call["prompt"]) == kind]

    def started_at(self, prompt: str) -> float:
        for sent, _, when in self.started:
            if sent == prompt:
                return when
        raise AssertionError(f"{prompt!r} was never sent")

    def completed_before(self, prompt: str) -> int:
        for sent, count, _ in self.started:
            if sent == prompt:
                return count
        raise AssertionError(f"{prompt!r} was never sent")

    def completed_rank(self, prompt: str) -> int:
        return self.completed.index(prompt) + 1


def _kind(prompt: str) -> str:
    return "donor" if prompt.startswith("DONOR") else "recipient"


class _FakeResponse:
    """Minimal streaming HTTP response over a scripted list of SSE lines."""

    def __init__(self, body: list[bytes], headers: dict | None = None) -> None:
        self.body = body
        self.headers = headers or {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def __iter__(self):
        return iter(self.body)


def _item(item_id: str, *, donors: int = 0, metadata: dict | None = None) -> WorkloadItem:
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
        metadata=dict(metadata or {}),
    )


def _manifest(tmp_path: Path, items: list[WorkloadItem]) -> str:
    path = tmp_path / "manifest.jsonl"
    write_jsonl(path, items)
    return str(path)


def _config(tmp_path: Path, items: list[WorkloadItem], **overrides) -> LiveGatewayConfig:
    return LiveGatewayConfig(
        manifest=_manifest(tmp_path, items),
        output=str(tmp_path / "result.json"),
        gateway_url=overrides.pop("gateway_url", GATEWAY),
        model="qwen",
        **overrides,
    )


# --------------------------------------------------------------------------
# B15: --worker-url has to be reachable from the CLI, not just from the config
# --------------------------------------------------------------------------


def test_cli_worker_url_is_repeatable_and_comma_separable():
    args = build_parser().parse_args(
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
            "--worker-url",
            WORKERS[0],
            "--worker-url",
            f"{WORKERS[1]},{WORKERS[2]}",
        ]
    )
    assert args.worker_url == [WORKERS[0], f"{WORKERS[1]},{WORKERS[2]}"]


def test_cli_worker_url_defaults_to_an_empty_fleet():
    args = build_parser().parse_args(
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
        ]
    )
    assert args.worker_url == []


def test_cli_threads_the_worker_fleet_into_the_runner_config(tmp_path, monkeypatch):
    captured: list[LiveGatewayConfig] = []
    monkeypatch.setattr(
        "sembench.cli.run_live_gateway", lambda config: captured.append(config) or []
    )
    monkeypatch.setattr("sembench.cli.scrape_all", lambda urls, **kw: [])
    manifest = _manifest(tmp_path, [_item("i0", donors=1)])

    main(
        [
            "run-live-gateway",
            "--manifest",
            manifest,
            "--output",
            str(tmp_path / "result.json"),
            "--gateway-url",
            GATEWAY,
            "--model",
            "qwen",
            "--skip-verify",
            "--worker-url",
            f"{WORKERS[0]}/",
            "--worker-url",
            f"{WORKERS[1]},{WORKERS[2]}",
        ]
    )

    assert captured[0].worker_urls == WORKERS


def test_cli_worker_urls_place_donors_across_the_fleet(tmp_path, monkeypatch):
    """The end the flag exists for: donors land on named workers, recipients
    still go through the gateway so the router's placement is what is measured."""
    recorder = Recorder()
    monkeypatch.setattr(gateway_live, "_chat_completion", recorder)
    monkeypatch.setattr("sembench.cli.scrape_all", lambda urls, **kw: [])
    manifest = _manifest(tmp_path, [_item(f"i{i}", donors=1) for i in range(6)])

    main(
        [
            "run-live-gateway",
            "--manifest",
            manifest,
            "--output",
            str(tmp_path / "result.json"),
            "--gateway-url",
            GATEWAY,
            "--model",
            "qwen",
            "--skip-verify",
            *[arg for url in WORKERS for arg in ("--worker-url", url)],
        ]
    )

    assert recorder.base_urls("recipient") == [GATEWAY] * 6
    assert set(recorder.base_urls("donor")) <= set(WORKERS)
    assert len(set(recorder.base_urls("donor"))) > 1


def test_metrics_are_scraped_from_the_worker_fleet_by_default():
    args = build_parser().parse_args(
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
            "--worker-url",
            WORKERS[0],
            "--worker-url",
            WORKERS[1],
        ]
    )
    assert _engine_metrics_urls(args) == [WORKERS[0], WORKERS[1], GATEWAY]


# --------------------------------------------------------------------------
# B12: engine-side per-request TTFT, which the flag gate already demands
# --------------------------------------------------------------------------


def test_chat_completion_captures_the_per_request_metrics_chunk(monkeypatch):
    body = [
        b'data: {"choices":[{"delta":{"content":"4"}}]}\n',
        b'data: {"usage":{"prompt_tokens":64},'
        b'"metrics":{"time_to_first_token_ms":31.0,"queue_time_ms":12.5}}\n',
        b"data: [DONE]\n",
    ]
    monkeypatch.setattr(gateway_live, "urlopen", lambda req, timeout=None: _FakeResponse(body))

    response = gateway_live._chat_completion(
        base_url=GATEWAY,
        model="qwen",
        prompt="p",
        max_tokens=8,
        tenant="t",
        template="tpl",
        timeout_seconds=5.0,
    )

    assert response["metrics"] == {"time_to_first_token_ms": 31.0, "queue_time_ms": 12.5}
    assert response["usage"]["prompt_tokens"] == 64


def test_chat_completion_reports_no_metrics_when_the_engine_emits_none(monkeypatch):
    body = [
        b'data: {"choices":[{"delta":{"content":"4"}}]}\n',
        b'data: {"usage":{"prompt_tokens":64}}\n',
        b"data: [DONE]\n",
    ]
    monkeypatch.setattr(gateway_live, "urlopen", lambda req, timeout=None: _FakeResponse(body))

    response = gateway_live._chat_completion(
        base_url=GATEWAY,
        model="qwen",
        prompt="p",
        max_tokens=8,
        tenant="t",
        template="tpl",
        timeout_seconds=5.0,
    )

    assert response["metrics"] == {}


def test_engine_timing_prefers_the_millisecond_keys():
    timing = engine_timing({"metrics": {"time_to_first_token_ms": 31.0, "queue_time_ms": 12.5}})
    assert timing == {"engine_ttft_ms": 31.0, "queue_time_ms": 12.5}


def test_engine_timing_converts_second_valued_keys():
    timing = engine_timing({"metrics": {"time_to_first_token": 0.031, "time_in_queue": 0.0125}})
    assert timing["engine_ttft_ms"] == pytest.approx(31.0)
    assert timing["queue_time_ms"] == pytest.approx(12.5)


def test_engine_timing_falls_back_to_the_engine_timestamps():
    """vLLM's own record is timestamps; TTFT is measured from the moment the
    request was scheduled, so the queue wait is not folded into it."""
    timing = engine_timing(
        {
            "metrics": {
                "arrival_time": 1000.0,
                "first_scheduled_time": 1000.5,
                "first_token_time": 1000.75,
            }
        }
    )
    assert timing["queue_time_ms"] == pytest.approx(500.0)
    assert timing["engine_ttft_ms"] == pytest.approx(250.0)


def test_engine_timing_is_none_without_a_metrics_chunk():
    assert engine_timing({}) == {"engine_ttft_ms": None, "queue_time_ms": None}
    assert engine_timing({"metrics": {}}) == {"engine_ttft_ms": None, "queue_time_ms": None}
    assert engine_timing({"metrics": "unavailable"}) == {
        "engine_ttft_ms": None,
        "queue_time_ms": None,
    }
    assert engine_timing({"metrics": {"finished_time": 1.0}}) == {
        "engine_ttft_ms": None,
        "queue_time_ms": None,
    }


def test_engine_ttft_and_queue_time_land_on_the_row(tmp_path, monkeypatch):
    recorder = Recorder(metrics={"time_to_first_token_ms": 31.0, "queue_time_ms": 12.5})
    monkeypatch.setattr(gateway_live, "_chat_completion", recorder)

    row = run_live_gateway(_config(tmp_path, [_item("i0", donors=1)]))[0]

    assert row.engine_ttft_ms == 31.0
    assert row.queue_time_ms == 12.5
    # The client-side measurement is untouched; the two are reported side by side.
    assert row.ttft_ms == 5.0


def test_a_row_from_an_engine_without_per_request_metrics_stays_null(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_live, "_chat_completion", Recorder())

    row = run_live_gateway(_config(tmp_path, [_item("i0")]))[0]

    assert row.engine_ttft_ms is None
    assert row.queue_time_ms is None


# --------------------------------------------------------------------------
# B16: the donor gap at concurrency 1, the width every TTFT arm runs at
# --------------------------------------------------------------------------


def _gap_run(tmp_path, monkeypatch, items, *, concurrency: int, gap: int):
    recorder = Recorder(service_seconds=0.002)
    monkeypatch.setattr(gateway_live, "_chat_completion", recorder)
    run = run_live_gateway_measured(
        _config(tmp_path, items, concurrency=concurrency, min_donor_gap_requests=gap)
    )
    return recorder, run


def _unsatisfiable_gap_items() -> list[WorkloadItem]:
    # i2 names i0 as its donor but sits two positions later, so no amount of
    # waiting can put five completed requests between them.
    return [_item("i0"), _item("i1"), _item("i2", metadata={"donor_item_id": "i0"})]


def test_an_unsatisfiable_donor_gap_is_counted_at_concurrency_one(tmp_path, monkeypatch):
    recorder, run = _gap_run(
        tmp_path, monkeypatch, _unsatisfiable_gap_items(), concurrency=1, gap=5
    )

    assert run.throughput["min_donor_gap_requests"] == 5
    assert run.throughput["gap_forced_stages"] == 1
    # Forced means issued anyway and recorded, never silently dropped.
    assert recorder.completed_before("recipient-i2") == 2


def test_the_donor_gap_behaves_identically_at_every_concurrency(tmp_path, monkeypatch):
    """The default width is 1; a gap that is inert there is inert for every
    TTFT and quality arm in the protocol."""
    _, serial = _gap_run(tmp_path, monkeypatch, _unsatisfiable_gap_items(), concurrency=1, gap=5)
    _, concurrent = _gap_run(
        tmp_path, monkeypatch, _unsatisfiable_gap_items(), concurrency=2, gap=5
    )

    assert serial.throughput["gap_forced_stages"] == concurrent.throughput["gap_forced_stages"]
    assert serial.throughput["min_donor_gap_requests"] == 5
    assert concurrent.throughput["min_donor_gap_requests"] == 5


def test_a_satisfiable_donor_gap_forces_nothing_at_concurrency_one(tmp_path, monkeypatch):
    items = [_item(f"i{i}") for i in range(8)]
    items.append(_item("i8", metadata={"donor_item_id": "i0"}))

    recorder, run = _gap_run(tmp_path, monkeypatch, items, concurrency=1, gap=5)

    assert run.throughput["gap_forced_stages"] == 0
    separation = recorder.completed_before("recipient-i8") - recorder.completed_rank("recipient-i0")
    assert separation >= 5


def test_a_per_item_gap_overrides_the_run_default_at_concurrency_one(tmp_path, monkeypatch):
    items = [_item(f"i{i}") for i in range(4)]
    items.append(_item("i4", metadata={"donor_item_id": "i0", "donor_gap_requests": 9}))

    _, run = _gap_run(tmp_path, monkeypatch, items, concurrency=1, gap=0)

    assert run.throughput["gap_forced_stages"] == 1


def test_a_serial_run_still_issues_one_request_at_a_time_in_manifest_order(tmp_path, monkeypatch):
    recorder = Recorder(service_seconds=0.002)
    monkeypatch.setattr(gateway_live, "_chat_completion", recorder)

    run = run_live_gateway_measured(_config(tmp_path, [_item(f"i{i}", donors=1) for i in range(3)]))

    assert [call["prompt"] for call in recorder.calls] == [
        "DONOR i0-0",
        "recipient-i0",
        "DONOR i1-0",
        "recipient-i1",
        "DONOR i2-0",
        "recipient-i2",
    ]
    assert run.throughput["concurrency"] == 1
    assert run.throughput["max_in_flight"] == 1
    assert [row.item_id for row in run.requests] == ["i0", "i1", "i2"]


# --------------------------------------------------------------------------
# The serial settle is owed by a step that actually sent donors
# --------------------------------------------------------------------------


def test_a_serial_step_with_no_donors_pays_no_settle(tmp_path, monkeypatch):
    """A self-seeding manifest sends no donors, so there is nothing for the
    engine to index and nothing to wait for."""
    monkeypatch.setattr(gateway_live, "_chat_completion", Recorder())

    run = run_live_gateway_measured(
        _config(tmp_path, [_item(f"i{i}") for i in range(3)], post_donor_delay_ms=300)
    )

    assert run.throughput["wall_seconds"] < 0.3
    assert run.throughput["idle_seconds"] < 0.05


def test_a_serial_step_with_donors_still_pays_the_settle(tmp_path, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(gateway_live, "_chat_completion", recorder)

    run_live_gateway_measured(_config(tmp_path, [_item("i0", donors=1)], post_donor_delay_ms=120))

    settle = recorder.started_at("recipient-i0") - recorder.started_at("DONOR i0-0")
    assert settle >= 0.12


# --------------------------------------------------------------------------
# B11: --paired cannot manufacture a pair out of a self-seeding manifest
# --------------------------------------------------------------------------


def test_paired_refuses_a_manifest_with_no_donor_prompts(tmp_path, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(gateway_live, "_chat_completion", recorder)
    monkeypatch.setattr(gateway_live, "reset_engine_caches", lambda urls, **kw: True)

    with pytest.raises(ValueError, match="merge-results"):
        run_live_gateway_measured(
            _config(
                tmp_path,
                [_item("i0"), _item("i1")],
                paired=True,
                reset_urls=("http://worker/reset",),
            )
        )

    # Refused before any traffic: the GPU time is the expensive part.
    assert recorder.calls == []


def test_paired_still_runs_when_the_manifest_carries_donors(tmp_path, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(gateway_live, "_chat_completion", recorder)
    monkeypatch.setattr(gateway_live, "reset_engine_caches", lambda urls, **kw: True)

    rows = run_live_gateway(
        _config(
            tmp_path,
            [_item("i0", donors=1)],
            paired=True,
            reset_urls=("http://worker/reset",),
        )
    )

    assert [row.arm for row in rows] == ["cold", "warm"]
    assert [call["prompt"] for call in recorder.calls] == [
        "recipient-i0",
        "DONOR i0-0",
        "recipient-i0",
    ]


def test_paired_resets_the_engine_once_per_arm_on_the_serial_path(tmp_path, monkeypatch):
    resets: list[tuple[str, ...]] = []
    monkeypatch.setattr(gateway_live, "_chat_completion", Recorder())
    monkeypatch.setattr(
        gateway_live, "reset_engine_caches", lambda urls, **kw: resets.append(tuple(urls)) or False
    )

    rows = run_live_gateway(
        _config(
            tmp_path,
            [_item("i0", donors=1)],
            paired=True,
            reset_urls=("http://worker/reset",),
        )
    )

    assert resets == [("http://worker/reset",), ("http://worker/reset",)]
    # A reset that did not take leaves the cold twin warm, and the row says so.
    assert rows[0].flush_contaminated is True


# --------------------------------------------------------------------------
# Recipient length: the quality arms score a real answer
# --------------------------------------------------------------------------


def test_recipient_max_tokens_defaults_to_256():
    config = LiveGatewayConfig(
        manifest="m.jsonl", output="o.json", gateway_url=GATEWAY, model="qwen"
    )
    assert config.recipient_max_tokens == 256

    args = build_parser().parse_args(
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
        ]
    )
    assert args.recipient_max_tokens == 256


def test_the_default_recipient_length_reaches_the_request(tmp_path, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(gateway_live, "_chat_completion", recorder)

    run_live_gateway(_config(tmp_path, [_item("i0", donors=1)]))

    assert recorder.max_tokens("recipient") == [256]
    # Donors are seeds, not answers; one token is still enough.
    assert recorder.max_tokens("donor") == [1]
