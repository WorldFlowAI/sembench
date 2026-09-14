"""Multi-worker gateway runner: donor placement and per-request route capture."""

import json

import pytest

from sembench import gateway_live
from sembench.gateway_live import (
    LiveGatewayConfig,
    normalize_route_outcome,
    parse_worker_urls,
    run_live_gateway,
    select_worker_url,
)
from sembench.schema import DonorPrompt, WorkloadItem, write_jsonl

GATEWAY = "http://gateway:8000"
WORKERS = ("http://worker-0:8000", "http://worker-1:8000", "http://worker-2:8000")

SSE_BODY = [
    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
    b'data: {"usage":{"prompt_tokens":64,"prompt_tokens_details":{"cached_tokens":32}}}\n',
    b"data: [DONE]\n",
]


class _FakeResponse:
    def __init__(self, headers: dict) -> None:
        self.headers = headers

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def __iter__(self):
        return iter(SSE_BODY)


class _FakeFleet:
    """Records the base URL of every request and replays scripted headers."""

    def __init__(self, headers_by_base: dict | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._headers_by_base = headers_by_base or {}

    def __call__(self, req, timeout=None) -> _FakeResponse:
        base = req.full_url.rsplit("/v1/chat/completions", 1)[0]
        payload = json.loads(req.data.decode("utf-8"))
        prompt = payload["messages"][0]["content"]
        kind = "donor" if prompt.startswith("DONOR") else "recipient"
        self.calls.append((kind, base))
        return _FakeResponse(dict(self._headers_by_base.get(base, {})))

    def bases(self, kind: str) -> list[str]:
        return [base for call_kind, base in self.calls if call_kind == kind]


def _item(item_id: str = "i1", donor_ids: tuple[str, ...] = ("d1",)) -> WorkloadItem:
    return WorkloadItem(
        item_id=item_id,
        dataset="fixture",
        source_id="s1",
        transform="instruction_variant",
        donor_prompts=[
            DonorPrompt(donor_id=donor_id, text=f"DONOR {donor_id} text " * 8, label="base")
            for donor_id in donor_ids
        ],
        recipient_prompt="recipient prompt " * 8,
        answers=["42"],
    )


def _manifest(tmp_path, items) -> str:
    path = tmp_path / "manifest.jsonl"
    write_jsonl(path, items)
    return str(path)


def _run(monkeypatch, tmp_path, items, fleet: _FakeFleet, **overrides):
    monkeypatch.setattr(gateway_live, "urlopen", fleet)
    config = LiveGatewayConfig(
        manifest=_manifest(tmp_path, items),
        output=str(tmp_path / "out.json"),
        gateway_url=overrides.pop("gateway_url", GATEWAY),
        model="test-model",
        **overrides,
    )
    return run_live_gateway(config)


def test_parse_worker_urls_accepts_repeated_and_comma_forms():
    assert parse_worker_urls(["http://a:8000/", "http://b:8000,http://c:8000"]) == (
        "http://a:8000",
        "http://b:8000",
        "http://c:8000",
    )
    assert parse_worker_urls("http://a:8000") == ("http://a:8000",)
    assert parse_worker_urls(None) == ()
    assert parse_worker_urls(["", "  ", ","]) == ()


def test_parse_worker_urls_dedupes_and_keeps_order():
    assert parse_worker_urls(["http://b:8000", "http://a:8000", "http://b:8000/"]) == (
        "http://b:8000",
        "http://a:8000",
    )


def test_select_worker_url_is_stable_and_not_a_round_robin_counter():
    first = select_worker_url(WORKERS, "donor-7")
    for other in ("donor-1", "donor-2", "donor-3", "donor-4"):
        select_worker_url(WORKERS, other)
        assert select_worker_url(WORKERS, "donor-7") == first


def test_select_worker_url_spreads_across_the_fleet():
    placed = {select_worker_url(WORKERS, f"donor-{i}") for i in range(60)}
    assert placed == set(WORKERS)


def test_select_worker_url_rejects_an_empty_fleet():
    with pytest.raises(ValueError):
        select_worker_url((), "donor-1")


def test_donors_go_to_workers_and_recipients_go_to_the_gateway(monkeypatch, tmp_path):
    fleet = _FakeFleet()
    items = [_item(f"i{i}", donor_ids=(f"d{i}",)) for i in range(6)]
    rows = _run(monkeypatch, tmp_path, items, fleet, worker_urls=WORKERS)

    assert fleet.bases("recipient") == [GATEWAY] * len(items)
    assert set(fleet.bases("donor")) <= set(WORKERS)
    assert len(set(fleet.bases("donor"))) > 1
    for row, item in zip(rows, items):
        expected = select_worker_url(WORKERS, item.donor_prompts[0].donor_id)
        assert row.donor_worker_ids == [expected]


def test_donor_placement_is_identical_across_replays(monkeypatch, tmp_path):
    items = [_item(f"i{i}", donor_ids=(f"d{i}",)) for i in range(8)]
    first = _run(monkeypatch, tmp_path, items, _FakeFleet(), worker_urls=WORKERS)
    second = _run(monkeypatch, tmp_path, items, _FakeFleet(), worker_urls=WORKERS)
    assert [r.donor_worker_ids for r in first] == [r.donor_worker_ids for r in second]


def test_multi_donor_placement_is_parallel_to_donor_ids(monkeypatch, tmp_path):
    fleet = _FakeFleet()
    item = _item("i1", donor_ids=("d-alpha", "d-beta", "d-gamma"))
    rows = _run(monkeypatch, tmp_path, [item], fleet, worker_urls=WORKERS)

    row = rows[0]
    assert row.donor_ids == ["d-alpha", "d-beta", "d-gamma"]
    assert row.donor_worker_ids == [select_worker_url(WORKERS, d) for d in row.donor_ids]
    assert fleet.bases("donor") == row.donor_worker_ids


def test_worker_urls_are_normalized_before_placement(monkeypatch, tmp_path):
    fleet = _FakeFleet()
    rows = _run(
        monkeypatch,
        tmp_path,
        [_item()],
        fleet,
        worker_urls=("http://worker-0:8000/", "http://worker-0:8000"),
    )
    assert fleet.bases("donor") == ["http://worker-0:8000"]
    assert rows[0].donor_worker_ids == ["http://worker-0:8000"]


def test_single_url_path_is_unchanged(monkeypatch, tmp_path):
    fleet = _FakeFleet()
    rows = _run(monkeypatch, tmp_path, [_item()], fleet, donor_url="http://donor:8000/")

    assert fleet.bases("donor") == ["http://donor:8000"]
    assert fleet.bases("recipient") == [GATEWAY]
    row = rows[0]
    assert row.donor_worker_ids == ["http://donor:8000"]
    assert row.worker_id is None
    assert row.route_outcome is None
    assert row.route_endpoint_id is None
    assert row.route_semantic_score is None
    assert row.gateway_route_header is None


def test_gateway_only_path_still_sends_donors_to_the_gateway(monkeypatch, tmp_path):
    fleet = _FakeFleet()
    rows = _run(monkeypatch, tmp_path, [_item()], fleet)
    assert fleet.bases("donor") == [GATEWAY]
    assert rows[0].donor_worker_ids == [GATEWAY]


def test_synapse_route_headers_are_captured(monkeypatch, tmp_path):
    fleet = _FakeFleet(
        {
            GATEWAY: {
                "X-Synapse-Route-Outcome": "semantic_placement",
                "X-Synapse-Route-Worker": "vllm-worker-1",
                "X-Synapse-Route-Similarity": "0.83",
            }
        }
    )
    row = _run(monkeypatch, tmp_path, [_item()], fleet, worker_urls=WORKERS)[0]

    assert row.route_outcome == "semantic_placement"
    assert row.route_endpoint_id == "vllm-worker-1"
    assert row.worker_id == "vllm-worker-1"
    assert row.route_semantic_score == pytest.approx(0.83)
    assert row.gateway_route_header == "semantic_placement"


def test_epp_shorthand_outcomes_fold_onto_long_form_labels(monkeypatch, tmp_path):
    fleet = _FakeFleet({GATEWAY: {"x-synapse-route": "semantic"}})
    row = _run(monkeypatch, tmp_path, [_item()], fleet, worker_urls=WORKERS)[0]

    assert row.route_outcome == "semantic_placement"
    assert row.gateway_route_header == "semantic"
    assert row.worker_id is None


def test_legacy_route_headers_still_read(monkeypatch, tmp_path):
    fleet = _FakeFleet(
        {
            GATEWAY: {
                "x-semblend-routing-path": "cold_route",
                "x-semblend-routing-worker": "worker-9",
                "x-semblend-routing-similarity": "0.4",
            }
        }
    )
    row = _run(monkeypatch, tmp_path, [_item()], fleet)[0]

    assert row.route_outcome == "cold_route"
    assert row.route_endpoint_id == "worker-9"
    assert row.route_semantic_score == pytest.approx(0.4)


def test_semantic_route_header_still_reaches_gateway_route_header(monkeypatch, tmp_path):
    fleet = _FakeFleet({GATEWAY: {"x-semantic-route": "hit"}})
    row = _run(monkeypatch, tmp_path, [_item()], fleet)[0]
    assert row.gateway_route_header == "hit"
    assert row.route_outcome == "hit"


def test_unparseable_similarity_does_not_fail_the_request(monkeypatch, tmp_path):
    fleet = _FakeFleet(
        {GATEWAY: {"x-synapse-route-outcome": "cold_route", "x-synapse-route-similarity": "n/a"}}
    )
    row = _run(monkeypatch, tmp_path, [_item()], fleet)[0]

    assert row.route_outcome == "cold_route"
    assert row.route_semantic_score is None
    assert row.error is None


def test_worker_id_falls_back_to_a_directly_addressed_worker(monkeypatch, tmp_path):
    fleet = _FakeFleet()
    rows = _run(
        monkeypatch,
        tmp_path,
        [_item()],
        fleet,
        gateway_url=WORKERS[0],
        worker_urls=WORKERS,
    )
    assert rows[0].worker_id == WORKERS[0]
    assert rows[0].route_endpoint_id is None


def test_normalize_route_outcome_labels():
    assert normalize_route_outcome("semantic") == "semantic_placement"
    assert normalize_route_outcome("COLD") == "cold_route"
    assert normalize_route_outcome("exact") == "exact_route"
    assert normalize_route_outcome("fail_open") == "error_fallback"
    assert normalize_route_outcome("semantic_discovery_only") == "semantic_discovery_only"
    assert normalize_route_outcome("") is None
    assert normalize_route_outcome(None) is None
