"""Staged prefill: stages are token-exact prefixes, and the wait starts at the first.

A row carrying ``metadata.staged_prefill`` (token ids plus cut points, planned
by the builder from the row's linked donor) is sent as one completions request
per cut, each a prefix of the final prompt and tagged ``semblend_stage``, and
then the full prompt as token ids. The TTFT recorded is the user's: from the
first stage to the first token of the last request.
"""

from __future__ import annotations

import json
from dataclasses import replace

from test_harness_round3_runner import FakeResponse
from test_system_turn import GATEWAY, _config, _item

from sembench import gateway_live
from sembench.request_ids import sending_request_id

TOKENS = list(range(100, 140))
PLAN = {"token_ids": TOKENS, "cuts": [16, 32], "basis": "linked_donor_maximal_runs"}


def test_no_plan_is_replayed_unless_the_run_asks():
    assert gateway_live._stage_plan_for_item(_item(staged_prefill=PLAN), _config()) is None


def test_a_valid_plan_is_replayed():
    config = replace(_config(), staged_prefill=True)
    plan = gateway_live._stage_plan_for_item(_item(staged_prefill=PLAN), config)
    assert plan == {"token_ids": TOKENS, "cuts": [16, 32]}


def test_a_malformed_plan_falls_back_to_the_plain_request():
    config = replace(_config(), staged_prefill=True)
    for bad in ({"token_ids": TOKENS, "cuts": [0]}, {"token_ids": TOKENS, "cuts": [40]}, {}):
        assert gateway_live._stage_plan_for_item(_item(staged_prefill=bad), config) is None


def test_stages_are_prefixes_then_the_full_prompt_as_tokens(monkeypatch):
    sent: list = []

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, json.loads(req.data.decode("utf-8")), dict(req.headers)))
        if req.full_url.endswith("/v1/completions") and len(sent) <= 2:
            return FakeResponse([b'{"choices":[{"text":"x"}]}'])
        return FakeResponse([b'data: {"choices":[{"text":"4"}]}\n', b"data: [DONE]\n"])

    monkeypatch.setattr(gateway_live, "urlopen", fake_urlopen)
    config = replace(_config(), staged_prefill=True)
    with sending_request_id("req-1"):
        result = gateway_live._recipient_request(
            item=_item(staged_prefill=PLAN), config=config, base_url=GATEWAY
        )

    (u0, p0, _), (u1, p1, _), (u2, p2, _) = sent
    assert u0 == u1 == u2 == f"{GATEWAY}/v1/completions"
    assert (p0["prompt"], p1["prompt"], p2["prompt"]) == (TOKENS[:16], TOKENS[:32], TOKENS)
    assert p0["vllm_xargs"] == {"semblend_stage": "1"} and p0["max_tokens"] == 1
    assert (p0["request_id"], p1["request_id"], p2["request_id"]) == (
        "req-1-stage0",
        "req-1-stage1",
        "req-1",
    )
    assert "vllm_xargs" not in p2 and "messages" not in p2
    staged = result["staged_prefill"]
    assert staged["stages"] == 2 and staged["stage_errors"] == []
    assert result["ttft_ms"] == staged["final_ttft_ms"] + staged["stage_ms"]
    assert result["output_text"] == "4"


def test_a_failed_stage_is_recorded_and_the_request_still_runs(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise gateway_live.URLError("refused")
        if calls["n"] == 2:
            return FakeResponse([b"{}"])
        return FakeResponse([b'data: {"choices":[{"text":"4"}]}\n', b"data: [DONE]\n"])

    monkeypatch.setattr(gateway_live, "urlopen", fake_urlopen)
    config = replace(_config(), staged_prefill=True)
    result = gateway_live._recipient_request(
        item=_item(staged_prefill=PLAN), config=config, base_url=GATEWAY
    )
    assert result["staged_prefill"]["stage_errors"] == ["refused"]
    assert result["output_text"] == "4"
