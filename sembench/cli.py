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
    elif args.command == "checksum-manifest":
        cmd_checksum_manifest(args)
    elif args.command == "verify-endpoint":
        cmd_verify_endpoint(args)
    elif args.command == "freeze":
        cmd_freeze(args)
    elif args.command == "verify-frozen":
        cmd_verify_frozen(args)
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


def _add_run_identity_args(parser: argparse.ArgumentParser) -> None:
    """Run-identity flags shared by every runner (result schema v2)."""
    parser.add_argument("--run-id", default=None, help="Stable id for this run (default: random)")
    parser.add_argument(
        "--arm",
        choices=("cold", "warm", "single"),
        default="single",
        help="Which arm of a paired cold/warm comparison this run is",
    )
    parser.add_argument(
        "--backend-id", default="", help="Cache backend under test, e.g. sglang-fuzzy-pr31057"
    )
    parser.add_argument(
        "--baseline-id", default="", help="Baseline label when this run is a baseline arm"
    )
    parser.add_argument(
        "--engine-version", default="", help="Engine version string (client cannot always detect)"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the endpoint pre-flight check (unverified runs are not leaderboard-eligible)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sembench",
        description="Semantic KV cache benchmark suite",
    )
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build", help="Build a local workload manifest")
    build.add_argument(
        "--profile", choices=("fixture", "longbench-v1", "longbench-v2"), default=None
    )
    build.add_argument("--frozen", default=None, help="Build from a frozen spec (e.g. v1)")
    build.add_argument("--hf-revision", default=None, help="Pin the HF dataset revision")
    build.add_argument("--output", required=True)
    build.add_argument("--datasets", nargs="*", default=None)
    build.add_argument("--max-items-per-dataset", type=int, default=None)
    build.add_argument("--transforms", nargs="*", default=list(DEFAULT_TRANSFORMS))
    build.add_argument("--max-segments", type=int, default=4)
    build.add_argument("--min-segment-chars", type=int, default=400)

    checksum = sub.add_parser("checksum-manifest", help="Print the SHA256 of a manifest file")
    checksum.add_argument("--manifest", required=True)

    verify = sub.add_parser(
        "verify-endpoint", help="Pre-flight check a live endpoint (reachability, model identity)"
    )
    verify.add_argument("--engine", choices=("sglang", "gateway"), required=True)
    verify.add_argument("--base-url", required=True)
    verify.add_argument("--expect-model", default=None)

    freeze = sub.add_parser("freeze", help="Build a frozen spec's manifest and record its checksum")
    freeze.add_argument("--spec", required=True)
    freeze.add_argument("--manifests-dir", default="manifests")

    verify_frozen = sub.add_parser(
        "verify-frozen", help="Rebuild a frozen spec and compare against recorded checksums"
    )
    verify_frozen.add_argument("--spec", required=True)
    verify_frozen.add_argument("--manifests-dir", default="manifests")

    offline = sub.add_parser("run-offline", help="Run offline exact-vs-SemBlend metrics")
    _add_run_identity_args(offline)
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
    _add_run_identity_args(live)
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

    gateway = sub.add_parser(
        "run-live-gateway", help="Replay a manifest through an OpenAI-compatible gateway"
    )
    _add_run_identity_args(gateway)
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
    gates.add_argument("--max-negative-control-semantic-placement-rate", type=float, default=1.0)
    gates.add_argument("--require-materialized-reuse", action="store_true")
    gates.add_argument("--require-no-engine-errors", action="store_true")

    return parser


def _build_items(
    *,
    profile: str,
    datasets: list[str] | None,
    max_items_per_dataset: int | None,
    transforms: tuple[str, ...],
    max_segments: int,
    min_segment_chars: int,
    revision: str | None,
):
    if profile == "longbench-v1" and not datasets:
        datasets = list(DEFAULT_LONGBENCH_V1_DATASETS)
    records = load_source_records(
        profile=profile,
        datasets=datasets,
        max_items_per_dataset=max_items_per_dataset,
        revision=revision,
    )
    config = TransformConfig(
        transforms=transforms,
        max_segments=max_segments,
        min_segment_chars=min_segment_chars,
    )
    return records, build_workload(records, config)


def cmd_build(args) -> None:
    if args.frozen is not None:
        from sembench.frozen import get_frozen_spec

        overridden = [
            flag
            for flag, given in (
                ("--profile", args.profile is not None),
                ("--datasets", bool(args.datasets)),
                ("--hf-revision", args.hf_revision is not None),
            )
            if given
        ]
        if overridden:
            raise SystemExit(
                f"--frozen pins these inputs; drop {', '.join(overridden)} "
                "(a frozen build must not be overridable)"
            )
        spec = get_frozen_spec(args.frozen)
        profile = spec.profile
        datasets = list(spec.datasets)
        max_items_per_dataset = spec.max_items_per_dataset
        transforms = spec.transforms
        max_segments = spec.max_segments
        min_segment_chars = spec.min_segment_chars
        revision = spec.hf_revision
    else:
        if args.profile is None:
            raise SystemExit("one of --profile or --frozen is required")
        profile = args.profile
        datasets = args.datasets
        max_items_per_dataset = args.max_items_per_dataset
        transforms = tuple(args.transforms)
        max_segments = args.max_segments
        min_segment_chars = args.min_segment_chars
        revision = args.hf_revision

    records, items = _build_items(
        profile=profile,
        datasets=datasets,
        max_items_per_dataset=max_items_per_dataset,
        transforms=transforms,
        max_segments=max_segments,
        min_segment_chars=min_segment_chars,
        revision=revision,
    )
    write_jsonl(args.output, items)

    summary = {
        "output": str(Path(args.output)),
        "profile": profile,
        "frozen": args.frozen,
        "source_records": len(records),
        "workload_items": len(items),
        "datasets": sorted({item.dataset for item in items}),
        "transforms": sorted({item.transform for item in items}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_frozen_manifest(spec, output_path: Path) -> int:
    _, items = _build_items(
        profile=spec.profile,
        datasets=list(spec.datasets),
        max_items_per_dataset=spec.max_items_per_dataset,
        transforms=spec.transforms,
        max_segments=spec.max_segments,
        min_segment_chars=spec.min_segment_chars,
        revision=spec.hf_revision,
    )
    write_jsonl(output_path, items)
    return len(items)


def cmd_freeze(args) -> None:
    from sembench.frozen import get_frozen_spec, write_checksums
    from sembench.schema import manifest_sha256

    spec = get_frozen_spec(args.spec)
    manifest_path = Path(args.manifests_dir) / spec.manifest_filename()
    item_count = _build_frozen_manifest(spec, manifest_path)
    digest = manifest_sha256(manifest_path)
    checksums = write_checksums(
        args.manifests_dir, spec, manifest_sha256=digest, workload_items=item_count
    )
    print(
        json.dumps(
            {
                "spec": spec.name,
                "manifest": str(manifest_path),
                "sha256": digest,
                "workload_items": item_count,
                "checksums": str(checksums),
            },
            indent=2,
            sort_keys=True,
        )
    )


def cmd_verify_frozen(args) -> None:
    import tempfile

    from sembench.frozen import get_frozen_spec, read_checksums
    from sembench.schema import manifest_sha256

    spec = get_frozen_spec(args.spec)
    recorded = read_checksums(args.manifests_dir, spec)
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = Path(tmp) / spec.manifest_filename()
        item_count = _build_frozen_manifest(spec, rebuilt)
        digest = manifest_sha256(rebuilt)
    passed = digest == recorded["sha256"] and item_count == recorded["workload_items"]
    print(
        json.dumps(
            {
                "spec": spec.name,
                "recorded_sha256": recorded["sha256"],
                "rebuilt_sha256": digest,
                "recorded_items": recorded["workload_items"],
                "rebuilt_items": item_count,
                "passed": passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


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
        run=_run_metadata(args, engine="offline"),
    )
    print(json.dumps({"output": args.output, "requests": len(requests)}, indent=2))


def _run_metadata(args, *, engine: str, engine_version: str = ""):
    from sembench.schema import collect_run_metadata

    return collect_run_metadata(
        engine=engine,
        manifest=args.manifest,
        run_id=args.run_id,
        arm=args.arm,
        engine_version=args.engine_version or engine_version,
        backend_id=args.backend_id,
        baseline_id=args.baseline_id,
    )


def _preflight(args, *, engine: str, base_url: str) -> str:
    """Verify the endpoint before live traffic; returns detected engine version.

    Fails the run (exit 3) on unreachable endpoint or model mismatch unless
    --skip-verify is set.
    """
    if args.skip_verify:
        return ""
    from sembench.verify import verify_endpoint

    report = verify_endpoint(engine=engine, base_url=base_url, expect_model=args.model)
    print(json.dumps({"preflight": report.to_dict()}, indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(3)
    return report.engine_version


def cmd_verify_endpoint(args) -> None:
    from sembench.verify import verify_endpoint

    report = verify_endpoint(
        engine=args.engine,
        base_url=args.base_url,
        expect_model=args.expect_model,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(3)


def cmd_checksum_manifest(args) -> None:
    from sembench.schema import manifest_sha256

    print(
        json.dumps(
            {"manifest": args.manifest, "sha256": manifest_sha256(args.manifest)},
            indent=2,
            sort_keys=True,
        )
    )


def cmd_run_live_sglang(args) -> None:
    detected_version = _preflight(args, engine="sglang", base_url=args.base_url)
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
        run=_run_metadata(args, engine="sglang", engine_version=detected_version),
    )
    print(json.dumps({"output": args.output, "requests": len(requests)}, indent=2))


def cmd_run_live_gateway(args) -> None:
    detected_version = _preflight(args, engine="gateway", base_url=args.gateway_url)
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
        run=_run_metadata(args, engine="gateway", engine_version=detected_version),
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
        json.loads(Path(path).read_text(encoding="utf-8")) for path in args.engine_summary
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
    negative_semantic_placement = aggregate.get("negative_control_semantic_placement_rate")
    if negative_semantic_placement is not None:
        require(
            "negative_control_semantic_placement_rate",
            float(negative_semantic_placement) <= args.max_negative_control_semantic_placement_rate,
            f"{negative_semantic_placement} > {args.max_negative_control_semantic_placement_rate}",
        )

    materialization_events = sum(
        int(summary.get("materialization_events") or 0) for summary in engine_summaries
    )
    materialized_tokens = sum(
        int(summary.get("materialized_tokens") or 0) for summary in engine_summaries
    )
    materialized_units = sum(
        int(summary.get("materialized_units") or 0) for summary in engine_summaries
    )
    materialized_reuse = any(
        bool(summary.get("materialized_semantic_kv_reuse")) for summary in engine_summaries
    )
    engine_errors = [error for summary in engine_summaries for error in summary.get("errors", [])]

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
            "negative_control_semantic_placement_rate": negative_semantic_placement,
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
