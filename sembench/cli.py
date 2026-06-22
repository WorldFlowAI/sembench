"""CLI for SemBench."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sembench.engine_events import parse_engine_events
from sembench.gateway_live import LiveGatewayConfig, run_live_gateway
from sembench.longbench import DEFAULT_LONGBENCH_V1_DATASETS, load_source_records
from sembench.offline import OfflineConfig, run_offline
from sembench.results import write_result
from sembench.schema import write_jsonl
from sembench.sglang_live import LiveSglangConfig, run_live_sglang_sync
from sembench.transforms import DEFAULT_TRANSFORMS, TransformConfig, build_workload


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        cmd_build(args)
    elif args.command == "run-offline":
        cmd_run_offline(args)
    elif args.command == "run-live-sglang":
        cmd_run_live_sglang(args)
    elif args.command == "run-live-gateway":
        cmd_run_live_gateway(args)
    elif args.command == "summarize-engine-events":
        cmd_summarize_engine_events(args)
    elif args.command == "collect-k8s-engine-events":
        cmd_collect_k8s_engine_events(args)
    elif args.command == "assert-result-gates":
        cmd_assert_result_gates(args)
    else:
        parser.print_help()
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sembench",
        description="Semantic KV cache benchmark suite",
    )
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build", help="Build a local workload manifest")
    build.add_argument("--profile", choices=("fixture", "longbench-v1", "longbench-v2"), required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--datasets", nargs="*", default=None)
    build.add_argument("--max-items-per-dataset", type=int, default=None)
    build.add_argument("--transforms", nargs="*", default=list(DEFAULT_TRANSFORMS))
    build.add_argument("--max-segments", type=int, default=4)
    build.add_argument("--min-segment-chars", type=int, default=400)

    offline = sub.add_parser("run-offline", help="Run offline exact-vs-SemBlend metrics")
    offline.add_argument("--manifest", required=True)
    offline.add_argument("--output", required=True)
    offline.add_argument("--block-size", type=int, default=16)
    offline.add_argument("--tokenizer", default=None)
    offline.add_argument("--semblend-path", default=None)
    offline.add_argument("--semblend-embedder", default="jaccard")
    offline.add_argument("--semblend-min-similarity", type=float, default=0.0)
    offline.add_argument("--semblend-min-reuse-ratio", type=float, default=0.25)
    offline.add_argument("--disable-multi-donor", action="store_true")
    offline.add_argument("--max-items", type=int, default=None)

    live = sub.add_parser("run-live-sglang", help="Replay a manifest against SGLang")
    live.add_argument("--manifest", required=True)
    live.add_argument("--output", required=True)
    live.add_argument("--base-url", required=True)
    live.add_argument("--model", required=True)
    live.add_argument("--block-size", type=int, default=16)
    live.add_argument("--tokenizer", default=None)
    live.add_argument("--max-items", type=int, default=None)
    live.add_argument("--donor-max-new-tokens", type=int, default=1)
    live.add_argument("--recipient-max-new-tokens", type=int, default=16)
    live.add_argument("--post-donor-delay-ms", type=int, default=0)
    live.add_argument("--no-flush-per-item", action="store_true")
    live.add_argument("--timeout-seconds", type=int, default=3600)
    live.add_argument("--quality-threshold", type=float, default=0.60)

    gateway = sub.add_parser("run-live-gateway", help="Replay a manifest through an OpenAI-compatible gateway")
    gateway.add_argument("--manifest", required=True)
    gateway.add_argument("--output", required=True)
    gateway.add_argument("--gateway-url", required=True)
    gateway.add_argument("--model", required=True)
    gateway.add_argument("--donor-url", default=None)
    gateway.add_argument("--tenant", default="tenant-a")
    gateway.add_argument("--template", default="rag-template-v1")
    gateway.add_argument("--block-size", type=int, default=16)
    gateway.add_argument("--tokenizer", default=None)
    gateway.add_argument("--max-items", type=int, default=None)
    gateway.add_argument("--donor-max-tokens", type=int, default=1)
    gateway.add_argument("--recipient-max-tokens", type=int, default=24)
    gateway.add_argument("--timeout-seconds", type=float, default=900.0)
    gateway.add_argument("--quality-threshold", type=float, default=0.60)

    events = sub.add_parser(
        "summarize-engine-events",
        help="Summarize backend log/audit evidence for semantic KV reuse",
    )
    events.add_argument("--engine", choices=("vllm", "sglang", "trtllm"), required=True)
    events.add_argument("--input", required=True)
    events.add_argument("--output", default=None)

    collect = sub.add_parser(
        "collect-k8s-engine-events",
        help="Collect pod logs and summarize backend-confirmed semantic KV reuse",
    )
    collect.add_argument("--engine", choices=("vllm", "sglang", "trtllm"), required=True)
    collect.add_argument("--namespace", required=True)
    collect.add_argument("--pod", required=True)
    collect.add_argument("--container", default=None)
    collect.add_argument("--tail", type=int, default=2000)
    collect.add_argument("--since")
    collect.add_argument("--since-time")
    collect.add_argument("--output-log", required=True)
    collect.add_argument("--output-summary", required=True)

    gates = sub.add_parser(
        "assert-result-gates",
        help="Fail unless replay and engine audit artifacts meet quality/reuse gates",
    )
    gates.add_argument("--result", required=True)
    gates.add_argument("--engine-summary", action="append", default=[])
    gates.add_argument("--min-quality-pass-rate", type=float, default=0.0)
    gates.add_argument("--min-semantic-placement-rate", type=float, default=0.0)
    gates.add_argument("--min-backend-confirmed-block-rate", type=float, default=0.0)
    gates.add_argument("--min-materialization-events", type=int, default=0)
    gates.add_argument("--min-materialized-tokens", type=int, default=0)
    gates.add_argument("--min-materialized-units", type=int, default=0)
    gates.add_argument("--max-negative-control-confirmed-rate", type=float, default=0.0)
    gates.add_argument("--require-materialized-reuse", action="store_true")
    gates.add_argument("--require-no-engine-errors", action="store_true")

    return parser


def cmd_build(args) -> None:
    datasets = args.datasets
    if args.profile == "longbench-v1" and not datasets:
        datasets = list(DEFAULT_LONGBENCH_V1_DATASETS)
    records = load_source_records(
        profile=args.profile,
        datasets=datasets,
        max_items_per_dataset=args.max_items_per_dataset,
    )
    config = TransformConfig(
        transforms=tuple(args.transforms),
        max_segments=args.max_segments,
        min_segment_chars=args.min_segment_chars,
    )
    items = build_workload(records, config)
    write_jsonl(args.output, items)

    summary = {
        "output": str(Path(args.output)),
        "profile": args.profile,
        "source_records": len(records),
        "workload_items": len(items),
        "datasets": sorted({item.dataset for item in items}),
        "transforms": sorted({item.transform for item in items}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_run_offline(args) -> None:
    config = OfflineConfig(
        manifest=args.manifest,
        output=args.output,
        block_size=args.block_size,
        tokenizer=args.tokenizer,
        semblend_path=args.semblend_path,
        semblend_embedder=args.semblend_embedder,
        semblend_min_similarity=args.semblend_min_similarity,
        semblend_min_reuse_ratio=args.semblend_min_reuse_ratio,
        enable_multi_donor=not args.disable_multi_donor,
        max_items=args.max_items,
    )
    requests = run_offline(config)
    write_result(
        args.output,
        requests=requests,
        config={
            "mode": "offline",
            **config.__dict__,
        },
    )
    print(json.dumps({"output": args.output, "requests": len(requests)}, indent=2))


def cmd_run_live_sglang(args) -> None:
    config = LiveSglangConfig(
        manifest=args.manifest,
        output=args.output,
        base_url=args.base_url,
        model=args.model,
        block_size=args.block_size,
        tokenizer=args.tokenizer,
        max_items=args.max_items,
        donor_max_new_tokens=args.donor_max_new_tokens,
        recipient_max_new_tokens=args.recipient_max_new_tokens,
        post_donor_delay_ms=args.post_donor_delay_ms,
        flush_per_item=not args.no_flush_per_item,
        timeout_seconds=args.timeout_seconds,
        quality_threshold=args.quality_threshold,
    )
    requests = run_live_sglang_sync(config)
    write_result(
        args.output,
        requests=requests,
        config={
            "mode": "live-sglang",
            **config.__dict__,
        },
    )
    print(json.dumps({"output": args.output, "requests": len(requests)}, indent=2))


def cmd_run_live_gateway(args) -> None:
    config = LiveGatewayConfig(
        manifest=args.manifest,
        output=args.output,
        gateway_url=args.gateway_url,
        model=args.model,
        donor_url=args.donor_url,
        tenant=args.tenant,
        template=args.template,
        block_size=args.block_size,
        tokenizer=args.tokenizer,
        max_items=args.max_items,
        donor_max_tokens=args.donor_max_tokens,
        recipient_max_tokens=args.recipient_max_tokens,
        timeout_seconds=args.timeout_seconds,
        quality_threshold=args.quality_threshold,
    )
    requests = run_live_gateway(config)
    write_result(
        args.output,
        requests=requests,
        config={
            "mode": "live-gateway",
            **config.__dict__,
        },
    )
    print(json.dumps({"output": args.output, "requests": len(requests)}, indent=2))


def cmd_summarize_engine_events(args) -> None:
    text = Path(args.input).read_text(encoding="utf-8")
    summary = parse_engine_events(args.engine, text).to_dict()
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def cmd_collect_k8s_engine_events(args) -> None:
    command = [
        "kubectl",
        "-n",
        args.namespace,
        "logs",
        args.pod,
        "--tail",
        str(args.tail),
    ]
    if args.since:
        command.extend(["--since", args.since])
    if args.since_time:
        command.extend(["--since-time", args.since_time])
    if args.container:
        command.extend(["-c", args.container])
    proc = subprocess.run(command, check=True, text=True, capture_output=True)  # noqa: S603
    log_path = Path(args.output_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    summary = parse_engine_events(args.engine, proc.stdout).to_dict()
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"log": str(log_path), "summary": summary},
            indent=2,
            sort_keys=True,
        )
    )


def cmd_assert_result_gates(args) -> None:
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    aggregate = result.get("aggregate") or {}
    engine_summaries = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.engine_summary
    ]
    failures: list[str] = []

    def require(name: str, ok: bool, detail: str) -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    quality = aggregate.get("quality_pass_rate")
    if quality is not None:
        require(
            "quality_pass_rate",
            float(quality) >= args.min_quality_pass_rate,
            f"{quality} < {args.min_quality_pass_rate}",
        )
    semantic_placement = float(aggregate.get("semantic_placement_rate_by_request") or 0.0)
    require(
        "semantic_placement_rate_by_request",
        semantic_placement >= args.min_semantic_placement_rate,
        f"{semantic_placement} < {args.min_semantic_placement_rate}",
    )
    backend_rate = aggregate.get("backend_confirmed_block_rate")
    if backend_rate is not None:
        require(
            "backend_confirmed_block_rate",
            float(backend_rate) >= args.min_backend_confirmed_block_rate,
            f"{backend_rate} < {args.min_backend_confirmed_block_rate}",
        )
    negative_rate = aggregate.get("negative_control_backend_confirmed_rate")
    if negative_rate is not None:
        require(
            "negative_control_backend_confirmed_rate",
            float(negative_rate) <= args.max_negative_control_confirmed_rate,
            f"{negative_rate} > {args.max_negative_control_confirmed_rate}",
        )

    materialization_events = sum(
        int(summary.get("materialization_events") or 0)
        for summary in engine_summaries
    )
    materialized_tokens = sum(
        int(summary.get("materialized_tokens") or 0)
        for summary in engine_summaries
    )
    materialized_units = sum(
        int(summary.get("materialized_units") or 0)
        for summary in engine_summaries
    )
    materialized_reuse = any(
        bool(summary.get("materialized_semantic_kv_reuse"))
        for summary in engine_summaries
    )
    engine_errors = [
        error
        for summary in engine_summaries
        for error in summary.get("errors", [])
    ]

    require(
        "materialization_events",
        materialization_events >= args.min_materialization_events,
        f"{materialization_events} < {args.min_materialization_events}",
    )
    require(
        "materialized_tokens",
        materialized_tokens >= args.min_materialized_tokens,
        f"{materialized_tokens} < {args.min_materialized_tokens}",
    )
    require(
        "materialized_units",
        materialized_units >= args.min_materialized_units,
        f"{materialized_units} < {args.min_materialized_units}",
    )
    if args.require_materialized_reuse:
        require(
            "materialized_reuse",
            materialized_reuse,
            "no engine summary proved materialization",
        )
    if args.require_no_engine_errors:
        require("engine_errors", not engine_errors, f"{len(engine_errors)} errors present")

    payload = {
        "result": args.result,
        "engine_summaries": args.engine_summary,
        "passed": not failures,
        "failures": failures,
        "observed": {
            "quality_pass_rate": quality,
            "semantic_placement_rate_by_request": semantic_placement,
            "backend_confirmed_block_rate": backend_rate,
            "negative_control_backend_confirmed_rate": negative_rate,
            "materialization_events": materialization_events,
            "materialized_tokens": materialized_tokens,
            "materialized_units": materialized_units,
            "materialized_reuse": materialized_reuse,
            "engine_error_count": len(engine_errors),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
