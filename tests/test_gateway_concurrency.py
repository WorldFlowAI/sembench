"""`run-live-gateway --concurrency`: the throughput arm.

Three things have to stay true when the runner stops being serial: TTFT is
still the streamed first token of each individual request, the stream order
and the donor gap survive, and requests-per-second is counted over the whole
arm rather than over whatever window happened to be busy. Concurrency 1 must
remain the serial path it is today, byte for byte, because every TTFT and
quality arm is measured on it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from sembench import gateway_live
from sembench.cli import build_parser, throughput_document_path
from sembench.gateway_live import (
    LiveGatewayConfig,
    run_live_gateway,
    run_live_gateway_measured,
)
from sembench.pairing import replay_plan
from sembench.replay_stages import build_stages, donor_item_refs
from sembench.schema import DonorPrompt, WorkloadItem, write_jsonl


class FakeGateway:
    """Records every request, with the completion count at the moment it started."""

    def __init__(self, *, service_seconds: float = 0.01, ttft_fraction: float = 0.25) -> None:
        self.service_seconds = service_seconds
        self.ttft_fraction = ttft_fraction
        self._lock = threading.Lock()
        self.started: list[tuple[str, int, float]] = []
        self.completed: list[str] = []
        self.peak_in_flight = 0
        self._in_flight = 0

    def __call__(self, *, base_url, model, prompt, max_tokens, tenant, template, timeout_seconds):
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
            self.started.append((prompt, len(self.completed), time.perf_counter()))
        time.sleep(self.service_seconds)
        with self._lock:
            self._in_flight -= 1
            self.completed.append(prompt)
        return {
            "output_text": "42",
            "usage": {"prompt_tokens": 64, "completion_tokens": 4},
            "headers": {},
            "ttft_ms": self.service_seconds * self.ttft_fraction * 1000,
            "latency_ms": self.service_seconds * 1000,
        }

    @property
    def prompts(self) -> list[str]:
        return [prompt for prompt, _, _ in self.started]

    def completed_before(self, prompt: str) -> int:
        for sent, count, _ in self.started:
            if sent == prompt:
                return count
        raise AssertionError(f"{prompt!r} was never sent")

    def started_at(self, prompt: str) -> float:
        for sent, _, when in self.started:
            if sent == prompt:
                return when
        raise AssertionError(f"{prompt!r} was never sent")


def _item(index: int, *, donors: int = 0, metadata: dict | None = None) -> WorkloadItem:
    return WorkloadItem(
        item_id=f"i{index}",
        dataset="fixture",
        source_id=f"s{index}",
        transform="instruction_variant",
        donor_prompts=[
            DonorPrompt(donor_id=f"i{index}-d{n}", text=f"donor-{index}-{n}", label="doc")
            for n in range(donors)
        ],
        recipient_prompt=f"recipient-{index}",
        answers=["42"],
        metadata=dict(metadata or {}),
    )


def _manifest(tmp_path: Path, items: list[WorkloadItem]) -> str:
    path = tmp_path / "manifest.jsonl"
    write_jsonl(path, items)
    return str(path)


def _config(manifest: str, tmp_path: Path, **overrides) -> LiveGatewayConfig:
    return LiveGatewayConfig(
        manifest=manifest,
        output=str(tmp_path / "result.json"),
        gateway_url="http://gateway.invalid",
        model="qwen",
        **overrides,
    )


def test_concurrency_one_is_still_one_stream_in_manifest_order(tmp_path, monkeypatch):
    fake = FakeGateway(service_seconds=0.005)
    monkeypatch.setattr(gateway_live, "_chat_completion", fake)
    manifest = _manifest(tmp_path, [_item(i, donors=1) for i in range(3)])

    rows = run_live_gateway(_config(manifest, tmp_path))

    assert fake.prompts == [
        "donor-0-0",
        "recipient-0",
        "donor-1-0",
        "recipient-1",
        "donor-2-0",
        "recipient-2",
    ]
    assert fake.peak_in_flight == 1
    assert [row.item_id for row in rows] == ["i0", "i1", "i2"]
    assert [row.donor_ids for row in rows] == [["i0-d0"], ["i1-d0"], ["i2-d0"]]


def test_run_live_gateway_still_returns_plain_request_rows(tmp_path, monkeypatch):
    """The existing return type is what every caller and result writer uses."""
    monkeypatch.setattr(gateway_live, "_chat_completion", FakeGateway(service_seconds=0.0))
    manifest = _manifest(tmp_path, [_item(0)])

    rows = run_live_gateway(_config(manifest, tmp_path))

    assert isinstance(rows, list)
    assert rows[0].item_id == "i0"
    assert rows[0].quality_score is not None


def test_concurrency_overlaps_streams_but_keeps_result_order(tmp_path, monkeypatch):
    fake = FakeGateway(service_seconds=0.02)
    monkeypatch.setattr(gateway_live, "_chat_completion", fake)
    manifest = _manifest(tmp_path, [_item(i) for i in range(8)])

    run = run_live_gateway_measured(_config(manifest, tmp_path, concurrency=4))

    assert [row.item_id for row in run.requests] == [f"i{i}" for i in range(8)]
    assert fake.peak_in_flight > 1
    assert fake.peak_in_flight <= 4
    assert run.throughput["max_in_flight"] <= 4
    # Eight 20ms requests four at a time cannot have taken the serial 160ms.
    assert run.throughput["wall_seconds"] < 0.16


def test_ttft_stays_the_streamed_first_token_under_concurrency(tmp_path, monkeypatch):
    """Not re-derived from the item's wall span, which under load is mostly
    queue time and would read as a TTFT regression that never happened."""
    fake = FakeGateway(service_seconds=0.02, ttft_fraction=0.25)
    monkeypatch.setattr(gateway_live, "_chat_completion", fake)
    manifest = _manifest(tmp_path, [_item(i) for i in range(6)])

    run = run_live_gateway_measured(_config(manifest, tmp_path, concurrency=3))

    assert {row.ttft_ms for row in run.requests} == {5.0}
    assert all(row.latency_ms == pytest.approx(20.0) for row in run.requests)
    assert run.throughput["recipient_ttft_ms"]["p50"] == 5.0
    assert run.throughput["recipient_ttft_ms"]["p99"] == 5.0


def test_item_latency_excludes_the_gap_the_harness_imposed(tmp_path, monkeypatch):
    """An item's latency is engine service time, not time it spent queued
    behind its own donor gap."""
    fake = FakeGateway(service_seconds=0.01)
    monkeypatch.setattr(gateway_live, "_chat_completion", fake)
    items = [_item(0, donors=1)] + [_item(i) for i in range(1, 4)]
    items.append(_item(4, metadata={"donor_item_id": "i0"}))
    manifest = _manifest(tmp_path, items)

    run = run_live_gateway_measured(
        _config(manifest, tmp_path, concurrency=2, min_donor_gap_requests=3)
    )

    last = run.requests[-1]
    assert last.item_id == "i4"
    assert last.latency_ms == pytest.approx(10.0)  # one recipient request, not the wait


def test_throughput_document_counts_every_request_over_the_whole_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_live, "_chat_completion", FakeGateway(service_seconds=0.01))
    manifest = _manifest(tmp_path, [_item(i, donors=1) for i in range(4)])

    run = run_live_gateway_measured(_config(manifest, tmp_path, concurrency=2))

    doc = run.throughput
    assert doc["items"] == 4
    assert doc["requests"] == 8  # four donors and four recipients
    assert doc["errors"] == 0
    assert doc["concurrency"] == 2
    # wall_seconds is published rounded to 3 decimals while the rate uses the
    # unrounded wall time, so reconstructing the rate from the rounded value
    # carries up to 0.0005 s of error -- over 1% on a fast runner's ~0.04 s.
    wall = doc["wall_seconds"]
    rounding = 8 * 0.0005 / (wall * (wall - 0.0005))
    assert doc["requests_per_second"] == pytest.approx(8 / wall, rel=1e-2, abs=rounding)
    assert doc["settle_seconds_est"] == 0.0
    assert doc["settle_excluded_basis"] == "measured_idle"
    assert doc["requests_per_second_excluding_settle"] >= doc["requests_per_second"]
    assert doc["output_tokens_per_second"] is not None
    assert doc["donor_ttft_ms"]["p50"] is not None
    assert doc["min_donor_gap_requests"] == 0
    assert doc["gap_forced_stages"] == 0


def test_settle_excluded_rate_removes_the_delay_the_harness_added(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_live, "_chat_completion", FakeGateway(service_seconds=0.0))
    manifest = _manifest(tmp_path, [_item(i, donors=1) for i in range(4)])

    run = run_live_gateway_measured(
        _config(manifest, tmp_path, concurrency=2, post_donor_delay_ms=50)
    )

    doc = run.throughput
    assert doc["settle_seconds_est"] == pytest.approx(0.1)  # 4 items x 50ms / 2 lanes
    assert doc["requests_per_second_excluding_settle"] > doc["requests_per_second"]


def test_a_serial_run_also_reports_its_throughput(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_live, "_chat_completion", FakeGateway(service_seconds=0.0))
    manifest = _manifest(tmp_path, [_item(i) for i in range(3)])

    run = run_live_gateway_measured(_config(manifest, tmp_path))

    assert run.throughput["concurrency"] == 1
    assert run.throughput["requests"] == 3
    assert len(run.recipient_records) == 3
    assert run.donor_records == ()


def test_a_recipient_waits_for_its_manifest_donor_and_the_configured_gap(tmp_path, monkeypatch):
    """metadata.donor_item_id names an earlier request in the same stream; the
    gap between them is the experiment and must hold under concurrency."""
    fake = FakeGateway(service_seconds=0.01)
    monkeypatch.setattr(gateway_live, "_chat_completion", fake)
    items = [_item(i) for i in range(14)]
    items.append(_item(14, metadata={"donor_item_id": "i0"}))
    manifest = _manifest(tmp_path, items)

    run = run_live_gateway_measured(
        _config(manifest, tmp_path, concurrency=4, min_donor_gap_requests=5)
    )

    assert "recipient-0" in fake.completed
    completed_when_sent = fake.completed_before("recipient-14")
    donor_rank = fake.completed.index("recipient-0") + 1
    assert completed_when_sent - donor_rank >= 5
    assert run.throughput["gap_forced_stages"] == 0
    assert run.throughput["min_donor_gap_requests"] == 5


def test_a_per_item_gap_overrides_the_run_default(tmp_path, monkeypatch):
    fake = FakeGateway(service_seconds=0.005)
    monkeypatch.setattr(gateway_live, "_chat_completion", fake)
    items = [_item(i) for i in range(14)]
    items.append(_item(14, metadata={"donor_item_id": "i0", "donor_gap_requests": 6}))
    manifest = _manifest(tmp_path, items)

    run_live_gateway_measured(_config(manifest, tmp_path, concurrency=4))

    donor_rank = fake.completed.index("recipient-0") + 1
    assert fake.completed_before("recipient-14") - donor_rank >= 6


def test_the_post_donor_settle_does_not_occupy_a_worker_slot(tmp_path, monkeypatch):
    """Sleeping inside the worker thread is what made the old run-load rate
    meaningless: the slot was held while nothing was being served."""
    fake = FakeGateway(service_seconds=0.0)
    monkeypatch.setattr(gateway_live, "_chat_completion", fake)
    manifest = _manifest(tmp_path, [_item(i, donors=1) for i in range(4)])

    run = run_live_gateway_measured(
        _config(manifest, tmp_path, concurrency=4, post_donor_delay_ms=100)
    )

    for index in range(4):
        gap = fake.started_at(f"recipient-{index}") - fake.started_at(f"donor-{index}-0")
        assert gap >= 0.1
    # Four 100ms settles, all four in flight together: well under 400ms.
    assert run.throughput["wall_seconds"] < 0.3


def test_a_donor_failure_still_skips_its_recipient_under_concurrency(tmp_path, monkeypatch):
    fake = FakeGateway(service_seconds=0.0)

    def flaky(**kwargs):
        if kwargs["prompt"] == "donor-1-0":
            raise ConnectionResetError("worker went away")
        return fake(**kwargs)

    monkeypatch.setattr(gateway_live, "_chat_completion", flaky)
    manifest = _manifest(tmp_path, [_item(i, donors=1) for i in range(3)])

    run = run_live_gateway_measured(_config(manifest, tmp_path, concurrency=3))

    assert "recipient-1" not in fake.prompts
    failed = [row for row in run.requests if row.item_id == "i1"][0]
    assert failed.error == "ConnectionResetError: worker went away"
    assert [row.error for row in run.requests if row.item_id != "i1"] == [None, None]


def test_concurrency_with_cache_resets_is_refused(tmp_path, monkeypatch):
    """A per-step reset fired mid-flight flushes other requests' KV, so the
    arm would measure the harness rather than the engine."""
    monkeypatch.setattr(gateway_live, "_chat_completion", FakeGateway(service_seconds=0.0))
    manifest = _manifest(tmp_path, [_item(0)])

    with pytest.raises(ValueError, match="reset_urls"):
        run_live_gateway_measured(
            _config(
                manifest,
                tmp_path,
                concurrency=8,
                reset_urls=("http://worker.invalid/reset_prefix_cache",),
            )
        )


def test_build_stages_puts_the_recipient_behind_its_own_donors():
    plan = replay_plan([_item(0, donors=2)])
    stages = build_stages(plan, settle_seconds=1.5)

    assert [stage.kind for stage in stages] == ["donors", "recipient"]
    assert stages[0].requests == 2  # two donor requests, not one stage
    assert stages[1].depends_on[0].key == stages[0].key
    assert stages[1].depends_on[0].min_seconds == 1.5


def test_build_stages_skips_a_donor_reference_that_points_forward():
    """Waiting on something that has not been submitted yet would deadlock the
    head of the queue; an unresolvable reference is dropped instead."""
    plan = replay_plan([_item(0, metadata={"donor_item_id": "i1"}), _item(1)])
    stages = build_stages(plan, min_donor_gap_requests=4)

    assert stages[0].depends_on == ()
    assert stages[1].depends_on == ()


def test_build_stages_keys_stages_by_stream_position():
    """A paired plan replays the same item_id twice; the two must not collide."""
    plan = replay_plan([_item(0), _item(0)])
    stages = build_stages(plan)

    assert len({stage.key for stage in stages}) == 2


def test_donor_item_refs_accepts_one_id_or_many():
    assert donor_item_refs(_item(0)) == ()
    assert donor_item_refs(_item(0, metadata={"donor_item_id": "i7"})) == ("i7",)
    assert donor_item_refs(_item(0, metadata={"donor_item_ids": ["a", "b"]})) == ("a", "b")
    assert donor_item_refs(_item(0, metadata={"donor_item_ids": []})) == ()


def test_cli_parses_concurrency_gap_and_throughput_output():
    args = build_parser().parse_args(
        [
            "run-live-gateway",
            "--manifest",
            "m.jsonl",
            "--output",
            "o.json",
            "--gateway-url",
            "http://127.0.0.1:1",
            "--model",
            "m",
            "--concurrency",
            "32",
            "--min-donor-gap-requests",
            "20",
        ]
    )
    assert args.concurrency == 32
    assert args.min_donor_gap_requests == 20
    assert args.throughput_output is None


def test_cli_concurrency_defaults_to_one():
    args = build_parser().parse_args(
        [
            "run-live-gateway",
            "--manifest",
            "m.jsonl",
            "--output",
            "o.json",
            "--gateway-url",
            "http://127.0.0.1:1",
            "--model",
            "m",
        ]
    )
    assert args.concurrency == 1
    assert args.min_donor_gap_requests == 0


def test_throughput_document_lands_beside_its_result():
    assert throughput_document_path("runs/a4-c32.json") == "runs/a4-c32.throughput.json"
    assert throughput_document_path("runs/a4.json", "elsewhere.json") == "elsewhere.json"


def test_run_live_gateway_writes_a_throughput_document_under_concurrency(tmp_path, monkeypatch):
    from sembench.cli import main

    monkeypatch.setattr(gateway_live, "_chat_completion", FakeGateway(service_seconds=0.0))
    manifest = _manifest(tmp_path, [_item(i) for i in range(3)])
    output = tmp_path / "result.json"

    main(
        [
            "run-live-gateway",
            "--manifest",
            manifest,
            "--output",
            str(output),
            "--gateway-url",
            "http://gateway.invalid",
            "--model",
            "qwen",
            "--concurrency",
            "3",
            "--skip-verify",
        ]
    )

    import json

    throughput = json.loads((tmp_path / "result.throughput.json").read_text())
    assert throughput["requests"] == 3
    assert throughput["concurrency"] == 3
    assert throughput["requests_per_second"] is not None
    result = json.loads(output.read_text())
    assert result["config"]["concurrency"] == 3
    assert len(result["requests"]) == 3
