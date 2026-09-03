"""Traffic-mix ROI model for semantic KV reuse.

Answers the operator's question directly: given a fleet's prompt-length
distribution and the fraction of traffic that semantically matches a
resident donor, does turning the layer on reduce time-to-first-token and
GPU prefill seconds overall, or does the miss-path overhead eat the gain?

Every input is explicit and overridable. The defaults are measured
values from our validation runs (A10G, Qwen2.5-7B, fp16) and should be
replaced with the operator's own numbers. Nothing here assumes reuse is
free: lookup cost applies to EVERY request above the minimum prompt
length, capture cost applies to every donor-eligible request, and hits
still pay for the unmatched remainder of the prompt plus the
materialization copy.

Usage:
    python -m sembench.roi --histogram "512:0.60,2048:0.25,4096:0.10,16384:0.05" \
        --hit-rate 0.30 --qps 50
    python -m sembench.roi --histogram traffic.csv --hit-rate-by-bucket "512:0.05,2048:0.2,4096:0.35,16384:0.5"
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CostModel:
    """Per-request cost coefficients (milliseconds unless noted)."""

    # Cold prefill: TTFT ~= fixed + per_token * prompt_tokens. A10G/7B fp16
    # measured ~ (120 ms + 0.62 ms/token); H100-class hardware is ~5x faster
    # per token, so pass your own numbers.
    prefill_fixed_ms: float = 120.0
    prefill_per_token_ms: float = 0.62
    # Miss path: one embedding + catalog lookup per eligible request.
    # Measured 2026-09-03 on vLLM 0.26 + connector, A10G, 3.5K-token
    # prompts, MiniLM on CPU, chunk fast path off: p50 188 ms. With the
    # ONNX GPU embedder the embedding drops to single-digit ms; pass
    # --lookup-ms 15 to model that deployment.
    lookup_ms: float = 190.0
    # Donor capture: block-aligned KV copy to the donor store, paid by every
    # donor-eligible request when capture is on. Measured 2026-09-03 (vLLM
    # 0.26 + connector, A10G, 28 layers, 3.5K-token prompts): the zero-match
    # arm ran +418 ms p50 over cold, of which ~190 ms is lookup, so capture
    # is ~230 ms / 3500 tokens. Set 0 for lookup-only mode
    # (register_donors=false) or when capture is moved off the request path.
    capture_per_token_ms: float = 0.065
    # Hit path: fraction of the prompt served from the donor, and the copy
    # cost of realizing donor KV into the recipient's blocks.
    served_fraction_on_hit: float = 0.85
    materialize_per_token_ms: float = 0.02
    # Below this prompt length the layer does not engage at all.
    min_prompt_tokens: int = 256


@dataclass(frozen=True)
class Bucket:
    prompt_tokens: int
    traffic_share: float
    hit_rate: float


@dataclass
class BucketResult:
    prompt_tokens: int
    traffic_share: float
    hit_rate: float
    cold_ttft_ms: float
    miss_ttft_ms: float
    hit_ttft_ms: float
    expected_ttft_ms: float
    expected_delta_ms: float
    breakeven_hit_rate: float


@dataclass
class RoiReport:
    buckets: list[BucketResult] = field(default_factory=list)
    cold_ttft_ms: float = 0.0
    expected_ttft_ms: float = 0.0
    net_ttft_delta_ms: float = 0.0
    net_ttft_delta_pct: float = 0.0
    prefill_gpu_seconds_saved_per_hour: float = 0.0
    fleet_breakeven_hit_rate: float = 0.0
    verdict: str = ""


def cold_ttft(tokens: int, c: CostModel) -> float:
    return c.prefill_fixed_ms + c.prefill_per_token_ms * tokens


def miss_ttft(tokens: int, c: CostModel) -> float:
    if tokens < c.min_prompt_tokens:
        return cold_ttft(tokens, c)
    return cold_ttft(tokens, c) + c.lookup_ms + c.capture_per_token_ms * tokens


def hit_ttft(tokens: int, c: CostModel) -> float:
    if tokens < c.min_prompt_tokens:
        return cold_ttft(tokens, c)
    served = int(tokens * c.served_fraction_on_hit)
    remainder = tokens - served
    return (
        c.prefill_fixed_ms
        + c.prefill_per_token_ms * remainder
        + c.lookup_ms
        + c.materialize_per_token_ms * served
        + c.capture_per_token_ms * tokens
    )


def breakeven_hit_rate(tokens: int, c: CostModel) -> float:
    """Hit rate at which expected TTFT equals cold TTFT for this bucket."""
    cold = cold_ttft(tokens, c)
    miss = miss_ttft(tokens, c)
    hit = hit_ttft(tokens, c)
    if miss <= cold:
        return 0.0
    if hit >= miss:
        return 1.0
    return min(1.0, (miss - cold) / (miss - hit))


def evaluate(buckets: list[Bucket], c: CostModel, qps: float = 0.0) -> RoiReport:
    report = RoiReport()
    total_share = sum(b.traffic_share for b in buckets) or 1.0
    saved_ms_per_request = 0.0
    for b in buckets:
        share = b.traffic_share / total_share
        cold = cold_ttft(b.prompt_tokens, c)
        miss = miss_ttft(b.prompt_tokens, c)
        hit = hit_ttft(b.prompt_tokens, c)
        expected = b.hit_rate * hit + (1.0 - b.hit_rate) * miss
        report.buckets.append(
            BucketResult(
                prompt_tokens=b.prompt_tokens,
                traffic_share=share,
                hit_rate=b.hit_rate,
                cold_ttft_ms=cold,
                miss_ttft_ms=miss,
                hit_ttft_ms=hit,
                expected_ttft_ms=expected,
                expected_delta_ms=expected - cold,
                breakeven_hit_rate=breakeven_hit_rate(b.prompt_tokens, c),
            )
        )
        report.cold_ttft_ms += share * cold
        report.expected_ttft_ms += share * expected
        saved_ms_per_request += share * (cold - expected)
    report.net_ttft_delta_ms = report.expected_ttft_ms - report.cold_ttft_ms
    report.net_ttft_delta_pct = (
        100.0 * report.net_ttft_delta_ms / report.cold_ttft_ms if report.cold_ttft_ms else 0.0
    )
    report.prefill_gpu_seconds_saved_per_hour = saved_ms_per_request / 1000.0 * qps * 3600.0
    # Fleet breakeven: the uniform hit rate at which the traffic-weighted
    # expected TTFT equals cold.
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        exp_mid = sum(
            (b.traffic_share / total_share)
            * (mid * hit_ttft(b.prompt_tokens, c) + (1 - mid) * miss_ttft(b.prompt_tokens, c))
            for b in buckets
        )
        if exp_mid > report.cold_ttft_ms:
            lo = mid
        else:
            hi = mid
    report.fleet_breakeven_hit_rate = hi
    if report.net_ttft_delta_ms < 0:
        report.verdict = "net win: expected TTFT below cold across the mix"
    else:
        report.verdict = (
            "net loss at this hit rate: raise min_prompt_tokens, restrict "
            "capture to long-context tenants, or route only eligible traffic"
        )
    return report


def parse_histogram(spec: str) -> list[tuple[int, float]]:
    if spec.endswith(".csv"):
        out = []
        with open(spec, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                out.append((int(row[0]), float(row[1])))
        return out
    pairs = []
    for part in spec.split(","):
        tokens, share = part.split(":")
        pairs.append((int(tokens), float(share)))
    return pairs


def parse_rates(spec: str | None, default: float, tokens: list[int]) -> dict[int, float]:
    rates = {t: default for t in tokens}
    if spec:
        for part in spec.split(","):
            t, r = part.split(":")
            rates[int(t)] = float(r)
    return rates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--histogram", required=True, help="tokens:share,... or a CSV of tokens,share")
    ap.add_argument("--hit-rate", type=float, default=0.0, help="uniform semantic hit rate")
    ap.add_argument("--hit-rate-by-bucket", default=None, help="tokens:rate,... overrides")
    ap.add_argument("--qps", type=float, default=0.0, help="fleet requests/s for GPU-second savings")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    for f in CostModel.__dataclass_fields__:
        ap.add_argument(f"--{f.replace('_', '-')}", type=float, default=None)
    args = ap.parse_args(argv)

    overrides = {
        f: getattr(args, f) for f in CostModel.__dataclass_fields__ if getattr(args, f) is not None
    }
    if "min_prompt_tokens" in overrides:
        overrides["min_prompt_tokens"] = int(overrides["min_prompt_tokens"])
    cost = CostModel(**overrides)
    hist = parse_histogram(args.histogram)
    rates = parse_rates(args.hit_rate_by_bucket, args.hit_rate, [t for t, _ in hist])
    buckets = [Bucket(t, s, rates[t]) for t, s in hist]
    report = evaluate(buckets, cost, qps=args.qps)

    if args.json:
        print(json.dumps({"cost_model": asdict(cost), "report": asdict(report)}, indent=2))
        return 0
    print(f"{'tokens':>8} {'share':>6} {'hit%':>5} {'cold':>8} {'miss':>8} {'hit':>8} {'expect':>8} {'delta':>8} {'breakeven':>9}")
    for b in report.buckets:
        print(
            f"{b.prompt_tokens:>8} {b.traffic_share:>6.2f} {100 * b.hit_rate:>5.0f} "
            f"{b.cold_ttft_ms:>8.0f} {b.miss_ttft_ms:>8.0f} {b.hit_ttft_ms:>8.0f} "
            f"{b.expected_ttft_ms:>8.0f} {b.expected_delta_ms:>+8.0f} {100 * b.breakeven_hit_rate:>8.1f}%"
        )
    print(
        f"\nfleet: cold TTFT {report.cold_ttft_ms:.0f} ms -> expected {report.expected_ttft_ms:.0f} ms "
        f"({report.net_ttft_delta_pct:+.1f}%), fleet breakeven hit rate {100 * report.fleet_breakeven_hit_rate:.1f}%"
    )
    if args.qps:
        print(f"prefill GPU-seconds saved per hour at {args.qps:g} qps: {report.prefill_gpu_seconds_saved_per_hour:,.0f}")
    print(f"verdict: {report.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
