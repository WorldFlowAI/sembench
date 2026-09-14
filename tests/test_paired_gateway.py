"""B11: paired arms on the vLLM/OpenAI gateway path.

Before this, `run-live-gateway` never set `RequestMetrics.arm`, so every row
kept the default "single" and `paired_summary` returned None for every result
the vLLM path ever produced — no speedup, no hit rate, no warm-vs-cold ROUGE.
These tests pin the pairing being deterministic per item and per arm.
"""

import json
from dataclasses import replace

import pytest

from sembench import gateway_live
from sembench.gateway_live import LiveGatewayConfig, cold_arm_contaminated, run_live_gateway
from sembench.pairing import COLD_ARM, WARM_ARM, replay_plan
from sembench.results import paired_summary
from sembench.schema import (
    EXTERNAL_SOURCE_CONNECTOR_AUDIT,
    DonorPrompt,
    WorkloadItem,
    write_jsonl,
)


def _item(item_id: str, *, donors: int = 1, negative: bool = False) -> WorkloadItem:
    return WorkloadItem(
        item_id=item_id,
        dataset="t",
        source_id=f"src-{item_id}",
        transform="instruction_variant",
        donor_prompts=[
            DonorPrompt(donor_id=f"{item_id}-d{n}", text=f"DONOR {item_id} " * 12, label="base")
            for n in range(donors)
        ],
        recipient_prompt=f"recipient {item_id} " * 12,
        answers=["42 minutes"],
        negative_control=negative,
    )


def _with_external_split(rows):
    """Stamp the warm rows with the external split the audit join supplies.

    The fake gateway reports `cached_tokens`, which on a prefix-caching-on
    vLLM arm is local cache + external transfer; the warm twin's hit has to
    be judged on the connector-confirmed external mass alone.
    """
    return [
        replace(
            row,
            external_confirmed_tokens=96,
            external_confirmed_tokens_source=EXTERNAL_SOURCE_CONNECTOR_AUDIT,
        )
        if row.arm == WARM_ARM
        else row
        for row in rows
    ]


def _manifest(tmp_path, items) -> str:
    path = tmp_path / "manifest.jsonl"
    write_jsonl(path, items)
    return str(path)


class FakeGateway:
    """Scripted OpenAI-compatible endpoint: donor-seeded recipients are fast
    and report cached tokens; unseeded ones are slow and cold."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.resets: list[str] = []
        self.seeded = False
        self.reset_ok = True

    def chat_completion(self, *, base_url, model, prompt, max_tokens, **_) -> dict:
        kind = "donor" if prompt.startswith("DONOR") else "recipient"
        self.sent.append((kind, prompt[:18]))
        if kind == "donor":
            self.seeded = True
            return {"output_text": "ok", "usage": {}, "headers": {}, "ttft_ms": 90.0}
        warm = self.seeded
        return {
            "output_text": "the answer is 42 minutes",
            "usage": {
                "prompt_tokens": 64,
                "prompt_tokens_details": {"cached_tokens": 96 if warm else 0},
            },
            "headers": {},
            "ttft_ms": 40.0 if warm else 200.0,
        }

    def reset(self, urls, *, timeout_seconds=60.0) -> bool:
        self.resets.extend(urls)
        self.seeded = False
        return self.reset_ok


@pytest.fixture
def fake_gateway(monkeypatch):
    fake = FakeGateway()
    monkeypatch.setattr(gateway_live, "_chat_completion", fake.chat_completion)
    monkeypatch.setattr(gateway_live, "reset_engine_caches", fake.reset)
    return fake


def _config(tmp_path, items, **kw) -> LiveGatewayConfig:
    return LiveGatewayConfig(
        manifest=_manifest(tmp_path, items),
        output=str(tmp_path / "result.json"),
        gateway_url="http://gw",
        model="m",
        **kw,
    )


def test_replay_plan_pairs_every_item_once_per_arm():
    items = [_item("i1"), _item("i2")]
    plan = replay_plan(items, paired=True)

    assert [(step.item.item_id, step.arm, step.stream_position) for step in plan] == [
        ("i1", COLD_ARM, 0),
        ("i1", WARM_ARM, 1),
        ("i2", COLD_ARM, 2),
        ("i2", WARM_ARM, 3),
    ]
    # The cold twin must not be seeded; the warm twin must.
    assert [step.seed_donors for step in plan] == [False, True, False, True]


def test_replay_plan_is_stable_across_calls():
    items = [_item("i1"), _item("i2"), _item("i3")]
    first = [(s.item.item_id, s.arm, s.stream_position) for s in replay_plan(items, paired=True)]
    second = [(s.item.item_id, s.arm, s.stream_position) for s in replay_plan(items, paired=True)]
    assert first == second


def test_replay_plan_rejects_duplicate_item_ids_for_paired_arms():
    items = [_item("i1"), _item("i1")]
    with pytest.raises(ValueError, match="unique"):
        replay_plan(items, paired=True)
    with pytest.raises(ValueError, match="unique"):
        replay_plan(items, arm=COLD_ARM)
    # A single-arm run is not joined by item_id, so it stays permissive.
    assert len(replay_plan(items)) == 2


def test_replay_plan_rejects_paired_with_a_pinned_arm():
    with pytest.raises(ValueError, match="do not also pin"):
        replay_plan([_item("i1")], paired=True, arm=WARM_ARM)


def test_gateway_paired_run_stamps_both_arms_per_item(tmp_path, fake_gateway):
    rows = run_live_gateway(
        _config(tmp_path, [_item("i1"), _item("i2")], paired=True, reset_urls=("http://w/reset",))
    )

    assert [(row.item_id, row.arm) for row in rows] == [
        ("i1", COLD_ARM),
        ("i1", WARM_ARM),
        ("i2", COLD_ARM),
        ("i2", WARM_ARM),
    ]
    # Exactly one cold twin per warm row, and vice versa.
    assert sorted(r.item_id for r in rows if r.arm == COLD_ARM) == ["i1", "i2"]
    assert sorted(r.item_id for r in rows if r.arm == WARM_ARM) == ["i1", "i2"]


def test_gateway_paired_cold_twin_sends_no_donors(tmp_path, fake_gateway):
    run_live_gateway(_config(tmp_path, [_item("i1")], paired=True, reset_urls=("http://w/reset",)))

    # cold recipient, then donor, then warm recipient.
    assert [kind for kind, _ in fake_gateway.sent] == ["recipient", "donor", "recipient"]
    # One reset per arm, so the warm twin's only warmth is its own donor.
    assert len(fake_gateway.resets) == 2


def test_gateway_paired_result_has_a_real_paired_summary(tmp_path, fake_gateway):
    rows = run_live_gateway(
        _config(tmp_path, [_item("i1"), _item("i2")], paired=True, reset_urls=("http://w/reset",))
    )
    summary = paired_summary(_with_external_split(rows))

    assert summary is not None
    assert summary["pairs_used"] == 2
    assert summary["blended_ttft_speedup_mean"] == 200.0 / 40.0
    assert summary["hit_rate"] == 1.0
    assert summary["hit_rate_external_confirmed"] == 1.0
    assert summary["pairs_external_unconfirmed"] == 0
    assert summary["ttft_cold_p50_ms"] == 200.0
    assert summary["ttft_warm_p50_ms"] == 40.0


def test_gateway_paired_flags_a_cold_twin_whose_reset_failed(tmp_path, fake_gateway):
    fake_gateway.reset_ok = False
    rows = run_live_gateway(
        _config(tmp_path, [_item("i1")], paired=True, reset_urls=("http://w/reset",))
    )

    cold = next(row for row in rows if row.arm == COLD_ARM)
    assert cold.flush_contaminated is True
    assert paired_summary(rows)["pairs_contaminated"] == 1


def test_gateway_paired_requires_a_reset_url(tmp_path, fake_gateway):
    with pytest.raises(ValueError, match="reset_urls"):
        run_live_gateway(_config(tmp_path, [_item("i1")], paired=True))


def test_gateway_single_arm_run_stamps_the_requested_arm(tmp_path, fake_gateway):
    rows = run_live_gateway(_config(tmp_path, [_item("i1"), _item("i2")], arm=COLD_ARM))

    assert [row.arm for row in rows] == [COLD_ARM, COLD_ARM]
    # A whole-run cold arm still replays the stream in full, donors included,
    # so both arms see the same prompts at the same stream positions.
    assert [kind for kind, _ in fake_gateway.sent] == [
        "donor",
        "recipient",
        "donor",
        "recipient",
    ]
    # No reset was performed, so contamination is undecidable — not "clean".
    assert [row.flush_contaminated for row in rows] == [None, None]


def test_gateway_default_run_is_unchanged(tmp_path, fake_gateway):
    rows = run_live_gateway(_config(tmp_path, [_item("i1")]))

    assert [row.arm for row in rows] == ["single"]
    assert rows[0].flush_contaminated is None
    assert paired_summary(rows) is None
    assert fake_gateway.resets == []


def test_cold_arm_contamination_is_undecidable_without_a_reset():
    # The A1 baseline runs with prefix caching ON and is *supposed* to hit its
    # own cache; calling that contamination would drop the pairs the baseline
    # exists to provide.
    assert (
        cold_arm_contaminated(arm=COLD_ARM, cache_reset=None, confirmed_tokens=4096, block_size=16)
        is None
    )
    assert (
        cold_arm_contaminated(arm=COLD_ARM, cache_reset=True, confirmed_tokens=8, block_size=16)
        is False
    )
    assert (
        cold_arm_contaminated(arm=COLD_ARM, cache_reset=True, confirmed_tokens=64, block_size=16)
        is True
    )
    assert (
        cold_arm_contaminated(arm=WARM_ARM, cache_reset=True, confirmed_tokens=4096, block_size=16)
        is None
    )


def test_cli_paired_flags_reach_the_runner(tmp_path, fake_gateway):
    from sembench.cli import build_parser, main

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
            "--paired",
            "--reset-url",
            "http://w1/reset_prefix_cache?reset_external=true",
            "--reset-url",
            "http://w2/reset_prefix_cache?reset_external=true",
        ]
    )
    assert args.paired is True
    assert len(args.reset_urls) == 2

    output = tmp_path / "out.json"
    main(
        [
            "run-live-gateway",
            "--manifest",
            _manifest(tmp_path, [_item("i1")]),
            "--output",
            str(output),
            "--gateway-url",
            "http://127.0.0.1:1",
            "--model",
            "m",
            "--skip-verify",
            "--paired",
            "--reset-url",
            "http://w1/reset",
        ]
    )
    written = json.loads(output.read_text())
    assert written["config"]["paired"] is True
    assert written["config"]["reset_urls"] == ["http://w1/reset"]
    assert [row["arm"] for row in written["requests"]] == [COLD_ARM, WARM_ARM]
    assert written["paired"] is not None


def test_cli_rejects_paired_without_a_reset_url(tmp_path):
    from sembench.cli import main

    with pytest.raises(SystemExit, match="--reset-url"):
        main(
            [
                "run-live-gateway",
                "--manifest",
                _manifest(tmp_path, [_item("i1")]),
                "--output",
                str(tmp_path / "out.json"),
                "--gateway-url",
                "http://127.0.0.1:1",
                "--model",
                "m",
                "--skip-verify",
                "--paired",
            ]
        )


def test_cli_rejects_paired_combined_with_a_pinned_arm(tmp_path):
    from sembench.cli import main

    with pytest.raises(SystemExit, match="drop --arm"):
        main(
            [
                "run-live-gateway",
                "--manifest",
                _manifest(tmp_path, [_item("i1")]),
                "--output",
                str(tmp_path / "out.json"),
                "--gateway-url",
                "http://127.0.0.1:1",
                "--model",
                "m",
                "--skip-verify",
                "--paired",
                "--reset-url",
                "http://w/reset",
                "--arm",
                "warm",
            ]
        )
