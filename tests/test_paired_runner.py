import asyncio

from sembench.results import paired_summary
from sembench.schema import DonorPrompt, WorkloadItem
from sembench.sglang_live import LiveSglangConfig, replay_items
from sembench.tokenization import load_tokenizer


class FakeTransport:
    """Scripted transport: cold recipients slow/uncached, warm fast/cached."""

    def __init__(self, *, contaminate_cold: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.flushed = 0
        self._contaminate = contaminate_cold
        self._seeded = False

    async def flush(self) -> None:
        self.flushed += 1
        self._seeded = False

    async def generate(self, text: str, max_new_tokens: int):
        kind = "donor" if text.startswith("DONOR") else "recipient"
        self.calls.append((kind, text[:20]))
        if kind == "donor":
            self._seeded = True
            return {
                "prompt_tokens": 64,
                "cached_tokens": 0,
                "ttft_ms": 100.0,
                "latency_ms": 120.0,
                "output_text": "ok",
                "error": None,
            }
        if self._seeded:
            return {
                "prompt_tokens": 64,
                "cached_tokens": 48,
                "ttft_ms": 40.0,
                "latency_ms": 60.0,
                "output_text": "the answer is 42 minutes",
                "error": None,
            }
        cached = 48 if self._contaminate else 0
        return {
            "prompt_tokens": 64,
            "cached_tokens": cached,
            "ttft_ms": 200.0,
            "latency_ms": 240.0,
            "output_text": "the answer is 42 minutes",
            "error": None,
        }


def _item(item_id: str = "i1") -> WorkloadItem:
    return WorkloadItem(
        item_id=item_id,
        dataset="t",
        source_id="s1",
        transform="instruction_variant",
        donor_prompts=[DonorPrompt(donor_id="d", text="DONOR text " * 12, label="base")],
        recipient_prompt="recipient prompt " * 12,
        answers=["42 minutes"],
    )


def _run(config: LiveSglangConfig, transport: FakeTransport, items=None):
    return asyncio.run(
        replay_items(
            items=items or [_item()],
            transport=transport,
            config=config,
            tokenizer=load_tokenizer(None),
        )
    )


def _config(**kw) -> LiveSglangConfig:
    return LiveSglangConfig(manifest="x", output="y", base_url="http://h", model="m", **kw)


def test_paired_mode_runs_cold_then_warm_per_item():
    transport = FakeTransport()
    rows = _run(_config(paired=True), transport)
    assert [r.arm for r in rows] == ["cold", "warm"]
    # cold arm must not seed donors
    donor_calls_before_first_recipient = [k for k, _ in transport.calls[:1]]
    assert donor_calls_before_first_recipient == ["recipient"]
    # both arms flushed
    assert transport.flushed == 2


def test_paired_summary_speedup_and_similarity():
    rows = _run(_config(paired=True), FakeTransport())
    summary = paired_summary(rows)
    assert summary["pairs_used"] == 1
    assert summary["ttft_speedup_mean"] == 200.0 / 40.0
    assert summary["warm_vs_cold_output_rouge_l_mean"] == 1.0  # identical outputs


def test_contaminated_cold_arm_is_excluded_and_counted():
    rows = _run(_config(paired=True), FakeTransport(contaminate_cold=True))
    cold = [r for r in rows if r.arm == "cold"][0]
    assert cold.flush_contaminated is True
    summary = paired_summary(rows)
    assert summary["pairs_contaminated"] == 1
    assert summary["pairs_used"] == 0


def test_warmup_requests_precede_measurement():
    transport = FakeTransport()
    _run(_config(paired=True, warmup_requests=3), transport)
    assert len([c for c in transport.calls if c[1].startswith("warmup")]) == 3
    assert transport.calls[0][1].startswith("warmup")


def test_single_mode_unchanged_shape():
    rows = _run(_config(), FakeTransport())
    assert [r.arm for r in rows] == ["single"]
    assert rows[0].flush_contaminated is None
    assert paired_summary(rows) is None


def test_negative_control_pairs_tracked_separately():
    neg = WorkloadItem(
        item_id="neg1",
        dataset="t",
        source_id="s2",
        transform="negative_control",
        donor_prompts=[DonorPrompt(donor_id="d", text="DONOR unrelated " * 12, label="u")],
        recipient_prompt="unrelated recipient " * 12,
        negative_control=True,
    )
    rows = _run(_config(paired=True), FakeTransport(), items=[_item(), neg])
    summary = paired_summary(rows)
    assert summary["negative_control_pairs"] == 1
    # regular pair: 200/40 = 5.0 blended; hit (cached_tokens 48 > 0) → hit_rate 1.0
    assert summary["blended_ttft_speedup_mean"] == 5.0
    assert summary["hit_rate"] == 1.0
    assert summary["hit_only_ttft_speedup_mean"] == 5.0
    # FakeTransport warms every seeded recipient, so the neg control also
    # speeds up — exactly what the deviation gate exists to catch.
    assert summary["negative_control_ttft_speedup_mean"] == 5.0
    assert summary["ttft_cold_p50_ms"] == 200.0
    assert summary["ttft_warm_p50_ms"] == 40.0
