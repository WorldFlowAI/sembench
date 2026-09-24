"""The harness sends the system turn a manifest row asks for.

The builder places the instruction wrapper in ``metadata.system_prompt`` and
measures every expectation on the row against ``messages=[system, user]``.
Before 0.2.1 the runner sent the user turn alone and read no system field, so
on stream B all eight wrappers tokenized to one 16-token prefix, both halves of
every "same document, new instruction" pair were byte-identical on the wire,
and the boundary sat at 16 on every GPU tried. The sidecar had recorded this
under ``harness_message_shape.directive`` before the runs were made.
"""

from __future__ import annotations

import json
from pathlib import Path

from test_harness_round3_runner import FakeResponse

from sembench import gateway_live
from sembench.schema import WorkloadItem

GATEWAY = "http://gw:8080"
SYSTEM = "You are a careful analyst. Answer in at most one sentence."


def _sent_payload(monkeypatch, **kwargs) -> dict:
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["request"] = req
        return FakeResponse([b'data: {"choices":[{"delta":{"content":"4"}}]}\n', b"data: [DONE]\n"])

    monkeypatch.setattr(gateway_live, "urlopen", fake_urlopen)
    call = {
        "base_url": GATEWAY,
        "model": "qwen",
        "prompt": "the document",
        "max_tokens": 8,
        "tenant": "t",
        "template": "tpl",
        "timeout_seconds": 5.0,
    }
    call.update(kwargs)
    gateway_live._chat_completion(**call)
    return json.loads(sent["request"].data.decode("utf-8"))


def test_a_system_turn_is_sent_ahead_of_the_user_turn(monkeypatch):
    payload = _sent_payload(monkeypatch, system=SYSTEM)

    assert payload["messages"] == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "the document"},
    ]


def test_without_a_system_turn_the_call_is_unchanged(monkeypatch):
    """Every existing test double is written against this shape."""
    payload = _sent_payload(monkeypatch)

    assert payload["messages"] == [{"role": "user", "content": "the document"}]


def test_an_empty_system_string_sends_no_system_turn(monkeypatch):
    payload = _sent_payload(monkeypatch, system="")

    assert payload["messages"] == [{"role": "user", "content": "the document"}]


def _config() -> gateway_live.LiveGatewayConfig:
    return gateway_live.LiveGatewayConfig(
        manifest=Path("m.jsonl"),
        output=Path("o.json"),
        gateway_url=GATEWAY,
        model="qwen",
        recipient_max_tokens=8,
        donor_max_tokens=8,
    )


def _item(**metadata) -> WorkloadItem:
    return WorkloadItem(
        item_id="sd-000-recip",
        dataset="d",
        source_id="s",
        transform="same_doc_new_instruction",
        donor_prompts=[],
        recipient_prompt="the document",
        metadata=metadata,
    )


def test_the_row_system_prompt_reaches_the_wire():
    assert gateway_live._system_turn_for_item(_item(system_prompt=SYSTEM)) == SYSTEM


def test_a_row_without_a_system_prompt_asks_for_none():
    """A builder that folded the wrapper into the user turn records None, and
    sending anything extra would put the wrapper on the wire twice."""
    assert gateway_live._system_turn_for_item(_item()) is None
    assert gateway_live._system_turn_for_item(_item(system_prompt=None)) is None
    assert gateway_live._system_turn_for_item(_item(system_prompt="")) is None
    assert gateway_live._system_turn_for_item(_item(system_prompt=17)) is None


def test_the_recipient_request_threads_the_system_turn(monkeypatch):
    seen: dict = {}

    def fake_chat_completion(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(gateway_live, "_chat_completion", fake_chat_completion)
    config = _config()

    gateway_live._recipient_request(
        item=_item(system_prompt=SYSTEM), config=config, base_url=GATEWAY
    )

    assert seen["system"] == SYSTEM
    assert seen["prompt"] == "the document"


def test_a_row_without_a_system_prompt_makes_the_call_it_always_made(monkeypatch):
    """The kwarg is absent, not None, so a strict double rejecting unknown
    keyword arguments keeps passing."""
    seen: dict = {}

    def fake_chat_completion(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(gateway_live, "_chat_completion", fake_chat_completion)
    config = _config()

    gateway_live._recipient_request(item=_item(), config=config, base_url=GATEWAY)

    assert "system" not in seen


# --- the capture hint --------------------------------------------------------


def test_a_hinted_role_carries_the_capture_flag_in_the_body(monkeypatch):
    payload = _sent_payload(monkeypatch, extra_body={"vllm_xargs": {"semblend_capture": "1"}})
    assert payload["vllm_xargs"] == {"semblend_capture": "1"}
    assert payload["messages"] == [{"role": "user", "content": "the document"}]


def test_the_hint_is_only_built_for_the_configured_role():
    config = _config()
    assert gateway_live._capture_hint_for_item(_item(role="seed"), config) is None  # unset
    hinted = gateway_live.LiveGatewayConfig(
        manifest=Path("m.jsonl"),
        output=Path("o.json"),
        gateway_url=GATEWAY,
        model="qwen",
        recipient_max_tokens=8,
        donor_max_tokens=8,
        capture_hint_role="seed",
    )
    assert gateway_live._capture_hint_for_item(_item(role="seed"), hinted) == {
        "vllm_xargs": {"semblend_capture": "1"}
    }
    assert gateway_live._capture_hint_for_item(_item(role="recipient"), hinted) is None
    assert gateway_live._capture_hint_for_item(_item(), hinted) is None


def test_an_unhinted_row_makes_the_call_it_always_made(monkeypatch):
    seen: dict = {}

    def fake_chat_completion(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(gateway_live, "_chat_completion", fake_chat_completion)
    hinted = gateway_live.LiveGatewayConfig(
        manifest=Path("m.jsonl"),
        output=Path("o.json"),
        gateway_url=GATEWAY,
        model="qwen",
        recipient_max_tokens=8,
        donor_max_tokens=8,
        capture_hint_role="seed",
    )
    gateway_live._recipient_request(item=_item(role="recipient"), config=hinted, base_url=GATEWAY)
    assert "extra_body" not in seen
    gateway_live._recipient_request(item=_item(role="seed"), config=hinted, base_url=GATEWAY)
    assert seen["extra_body"] == {"vllm_xargs": {"semblend_capture": "1"}}
