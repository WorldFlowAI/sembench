"""CLI for SemBench."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sembench.cli_audit import (
    add_connector_audit_arg,
    connector_audit_or_exit,
    join_connector_audit,
    request_id_echo,
)
from sembench.engine_config import (
    attach_engine_document,
    engine_document,
    load_engine_flags,
    phase0_flag_violations,
    snapshot_document,
    snapshots_from_document,
)
from sembench.engine_events import parse_engine_events
from sembench.gates_cli import add_gates_parser, cmd_assert_result_gates
from sembench.gateway_live import (
    LiveGatewayConfig,
    parse_worker_urls,
    run_live_gateway_measured,
)
from sembench.manifest_cli import (
    add_manifest_parsers,
    cmd_audit_manifest,
    cmd_build,
    cmd_checksum_manifest,
    cmd_freeze,
    cmd_verify_frozen,
)
from sembench.merge import add_merge_parser, cmd_merge_results
from sembench.metrics_chunk import add_metrics_chunk_arg
from sembench.offline import OfflineConfig, run_offline
from sembench.prometheus import MetricsWindow, scrape_all
from sembench.results import write_result
from sembench.sglang_live import (
    LiveSglangConfig,
    manifest_class_counts_for,
    run_live_sglang_sync,
)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        cmd_build(args)
    elif args.command == "checksum-manifest":
        cmd_checksum_manifest(args)
    elif args.command == "verify-endpoint":
        cmd_verify_endpoint(args)
    elif args.command == "audit-manifest":
        cmd_audit_manifest(args)
    elif args.command == "freeze":
        cmd_freeze(args)
    elif args.command == "verify-frozen":
        cmd_verify_frozen(args)
    elif args.command == "calibrate-noise-floor":
        cmd_calibrate_noise_floor(args)
    elif args.command == "run-offline":
        cmd_run_offline(args)
    elif args.command == "run-live-sglang":
        cmd_run_live_sglang(args)
    elif args.command == "run-live-gateway":
        cmd_run_live_gateway(args)
    elif args.command == "run-load":
        cmd_run_load(args)
    elif args.command == "merge-results":
        cmd_merge_results(args)
    elif args.command == "engine-snapshot":
        cmd_engine_snapshot(args)
    elif args.command == "engine-window":
        cmd_engine_window(args)
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


def _add_engine_config_args(parser: argparse.ArgumentParser) -> None:
    """Engine-identity flags for a live arm.

    A TTFT result whose prefix-caching state, batching limits, CUDA graph
    mode and --kv-transfer-config were never recorded cannot be compared
    with another arm; the serve line is part of the measurement.
    """
    parser.add_argument(
        "--engine-serve-command",
        default=None,
        help='Verbatim serve line for this arm, e.g. "VLLM_SERVER_DEV_MODE=1 vllm serve ..."',
    )
    parser.add_argument(
        "--engine-serve-command-file",
        default=None,
        help="File holding the serve line (for lines too long to pass inline)",
    )
    parser.add_argument(
        "--engine-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment that changes engine behaviour, e.g. VLLM_SERVER_DEV_MODE=1",
    )
    parser.add_argument(
        "--metrics-url",
        action="append",
        default=[],
        help="Worker base URL to scrape /metrics from before and after the arm "
        "(repeatable; defaults to the donor/gateway URL)",
    )
    parser.add_argument(
        "--require-phase0-flags",
        action="store_true",
        help="Refuse to run (exit 3) when the serve line is missing a phase-0 flag",
    )


def _engine_metrics_urls(args) -> list[str]:
    """Endpoints to scrape for this arm, deduped and order-preserving.

    Every source goes through ``parse_worker_urls``, which is what makes the
    comma-separated form ``--worker-url`` advertises work here too. Splicing
    the raw argv values in instead produced a single scrape target literally
    named ``http://a:8000,http://b:8000``: the run then scraped nothing, the
    engine counter window came back empty, and the arm looked like an engine
    that reports no external-KV counters rather than like a client bug.
    """
    if args.metrics_url:
        return list(parse_worker_urls(args.metrics_url))
    return list(
        parse_worker_urls(
            [
                *(getattr(args, "worker_url", None) or []),
                getattr(args, "donor_url", None) or "",
                getattr(args, "gateway_url", None) or "",
            ]
        )
    )


def _engine_flags_or_exit(args):
    try:
        return load_engine_flags(
            serve_command=args.engine_serve_command,
            serve_command_file=args.engine_serve_command_file,
            env_assignments=args.engine_env,
        )
    except ValueError as exc:
        raise SystemExit(f"engine config: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sembench",
        description="Semantic KV cache benchmark suite",
    )
    sub = parser.add_subparsers(dest="command")

    add_manifest_parsers(sub)

    verify = sub.add_parser(
        "verify-endpoint", help="Pre-flight check a live endpoint (reachability, model identity)"
    )
    verify.add_argument("--engine", choices=("sglang", "gateway"), required=True)
    verify.add_argument("--base-url", required=True)
    verify.add_argument("--expect-model", default=None)

    calibrate = sub.add_parser(
        "calibrate-noise-floor",
        help="Measure cold/cold ROUGE-L self-agreement for a live endpoint",
    )
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--base-url", required=True)
    calibrate.add_argument("--model", required=True)
    calibrate.add_argument("--max-items", type=int, default=200)
    calibrate.add_argument("--max-new-tokens", type=int, default=64)
    calibrate.add_argument("--cooldown-ms", type=int, default=0)
    calibrate.add_argument("--warmup-requests", type=int, default=2)
    calibrate.add_argument("--timeout-seconds", type=int, default=3600)
    calibrate.add_argument("--skip-verify", action="store_true")

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
    live.add_argument(
        "--paired",
        action="store_true",
        help="Run cold and warm arms per item on the same recipient prompt",
    )
    live.add_argument("--warmup-requests", type=int, default=0)
    live.add_argument("--cooldown-ms", type=int, default=0)
    live.add_argument(
        "--capture-logprobs",
        action="store_true",
        help="Capture per-token top-k logprobs for warm-vs-cold KL analysis",
    )
    live.add_argument("--top-logprobs-num", type=int, default=8)
    live.add_argument(
        "--resume",
        action="store_true",
        help="Skip items already completed in <output>.partial.jsonl from an interrupted run",
    )

    gateway = sub.add_parser(
        "run-live-gateway", help="Replay a manifest through an OpenAI-compatible gateway"
    )
    _add_run_identity_args(gateway)
    _add_engine_config_args(gateway)
    gateway.add_argument("--manifest", required=True)
    gateway.add_argument("--output", required=True)
    gateway.add_argument("--gateway-url", required=True)
    gateway.add_argument("--model", required=True)
    gateway.add_argument("--donor-url", default=None)
    gateway.add_argument(
        "--worker-url",
        action="append",
        default=[],
        metavar="URL",
        help="Fleet worker endpoint donors are seeded on directly (repeatable or comma-"
        "separated); recipients still go through --gateway-url, so placement is measured",
    )
    gateway.add_argument("--tenant", default="tenant-a")
    gateway.add_argument("--template", default="rag-template-v1")
    # Oracle: the harness knows the roles, so this bounds capture_policy=hinted.
    gateway.add_argument("--capture-hint-role", default=None, help="hint capture on this role")
    gateway.add_argument("--block-size", type=int, default=16)
    gateway.add_argument("--tokenizer", default=None)
    gateway.add_argument("--max-items", type=int, default=None)
    gateway.add_argument("--donor-max-tokens", type=int, default=1)
    gateway.add_argument("--recipient-max-tokens", type=int, default=256)
    gateway.add_argument("--timeout-seconds", type=float, default=900.0)
    gateway.add_argument("--quality-threshold", type=float, default=0.60)
    gateway.add_argument("--post-donor-delay-ms", type=int, default=0)
    gateway.add_argument(
        "--concurrency",
        type=int, default=1,
        help="Request streams in flight at once (default 1 = serial, unchanged). "
        "Above 1 the run also writes a throughput document; TTFT stays per-request, "
        "measured at the streamed first token",
    )
    gateway.add_argument(
        "--min-donor-gap-requests",
        type=int, default=0,
        help="Minimum donor->recipient separation in COMPLETED requests, for manifests "
        "that name their donor in metadata.donor_item_id. Under concurrency a gap "
        "measured in stream positions does not hold",
    )
    gateway.add_argument(
        "--throughput-output",
        default=None,
        metavar="PATH",
        help="Where to write the throughput document (default: <output> with a "
        ".throughput.json suffix, written whenever --concurrency > 1)",
    )
    gateway.add_argument(
        "--paired",
        action="store_true",
        help="Run a cold and a warm twin per item, adjacent in the stream (requires --reset-url)",
    )
    gateway.add_argument(
        "--reset-url",
        action="append",
        dest="reset_urls",
        default=[],
        metavar="URL",
        help="Engine cache-reset endpoint POSTed before each arm; repeat per worker "
        "(vLLM: http://host:8000/reset_prefix_cache?reset_external=true)",
    )
    add_connector_audit_arg(gateway)
    add_metrics_chunk_arg(gateway)

    load = sub.add_parser(
        "run-load",
        help="Drive donor->recipient streams concurrently; report throughput and TTFT under load",
    )
    load.add_argument("--manifest", required=True)
    load.add_argument("--output", required=True)
    load.add_argument("--gateway-url", required=True)
    load.add_argument("--model", required=True)
    load.add_argument("--tenant", default="tenant-a")
    load.add_argument("--template", default="rag-template-v1")
    load.add_argument("--concurrency", type=int, default=4)
    load.add_argument("--max-items", type=int, default=None)
    load.add_argument("--donor-max-tokens", type=int, default=1)
    load.add_argument("--recipient-max-tokens", type=int, default=32)
    load.add_argument("--post-donor-delay-ms", type=int, default=1000)
    load.add_argument("--timeout-seconds", type=float, default=1800.0)
    load.add_argument("--run-id", default="load")
    load.add_argument("--min-donor-gap-requests", type=int, default=0)

    snapshot = sub.add_parser(
        "engine-snapshot",
        help="Read the engine's external-KV Prometheus counters (run before and after an arm)",
    )
    snapshot.add_argument("--metrics-url", action="append", default=[], required=True)
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument("--timeout-seconds", type=float, default=15.0)

    window = sub.add_parser(
        "engine-window",
        help="Combine two engine snapshots plus the serve line into a result's engine block",
    )
    _add_engine_config_args(window)
    window.add_argument("--before", required=True, help="Snapshot taken before the arm")
    window.add_argument("--after", required=True, help="Snapshot taken after the arm")
    window.add_argument("--arm", default="single")
    window.add_argument("--output", default=None, help="Write the engine block here")
    window.add_argument(
        "--result",
        default=None,
        help="Result JSON to splice the engine block into (for arms run by another driver)",
    )

    events = sub.add_parser(
        "summarize-engine-events",
        help="Summarize backend log/audit evidence for semantic KV reuse",
    )
    events.add_argument("--engine", choices=("vllm", "sglang", "trtllm", "lmcache"), required=True)
    events.add_argument("--input", required=True)
    events.add_argument("--output", default=None)

    collect = sub.add_parser(
        "collect-k8s-engine-events",
        help="Collect pod logs and summarize backend-confirmed semantic KV reuse",
    )
    collect.add_argument("--engine", choices=("vllm", "sglang", "trtllm", "lmcache"), required=True)
    collect.add_argument("--namespace", required=True)
    collect.add_argument("--pod", required=True)
    collect.add_argument("--container", default=None)
    collect.add_argument("--tail", type=int, default=2000)
    collect.add_argument("--since")
    collect.add_argument("--since-time")
    collect.add_argument("--output-log", required=True)
    collect.add_argument("--output-summary", required=True)

    add_merge_parser(sub)
    add_gates_parser(sub)

    return parser


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
        paired=args.paired,
        warmup_requests=args.warmup_requests,
        cooldown_ms=args.cooldown_ms,
        capture_logprobs=args.capture_logprobs,
        top_logprobs_num=args.top_logprobs_num,
        resume=args.resume,
    )
    requests = run_live_sglang_sync(config)
    # The manifest's per-class item counts: section 4's class-scoped
    # denominators count manifest items, not the rows a run produced.
    class_counts = manifest_class_counts_for(config)
    write_result(
        args.output,
        requests=requests,
        config={
            "mode": "live-sglang",
            **config.__dict__,
            "manifest_class_counts": class_counts,
        },
        run=_run_metadata(args, engine="sglang", engine_version=detected_version),
        manifest_class_counts=class_counts,
    )
    print(json.dumps({"output": args.output, "requests": len(requests)}, indent=2))


def cmd_run_live_gateway(args) -> None:
    # Validate the arm wiring before any traffic: a run that stamps the wrong
    # arm is only discoverable after the GPU time is already spent.
    if args.paired and args.arm != "single":
        raise SystemExit(
            "--paired stamps cold and warm per item; drop --arm "
            f"(got --arm {args.arm}). Use --arm for two separate runs joined "
            "by `sembench merge-results`."
        )
    if args.paired and not args.reset_urls:
        raise SystemExit(
            "--paired requires --reset-url: without an engine cache reset the cold "
            "twin is warmed by the arm before it. For stock vLLM, serve with "
            "VLLM_SERVER_DEV_MODE=1 and pass "
            "--reset-url http://<worker>/reset_prefix_cache?reset_external=true"
        )
    connector_audit = connector_audit_or_exit(args)
    detected_version = _preflight(args, engine="gateway", base_url=args.gateway_url)
    # Resolved before the config, not at write time: the runner derives every
    # X-Request-Id from this id, and a run whose result says one run id while
    # its requests were stamped with another cannot be joined to its audit.
    run_metadata = _run_metadata(args, engine="gateway", engine_version=detected_version)
    config = LiveGatewayConfig(
        manifest=args.manifest,
        output=args.output,
        gateway_url=args.gateway_url,
        model=args.model,
        run_id=run_metadata.run_id,
        donor_url=args.donor_url,
        worker_urls=parse_worker_urls(args.worker_url),
        tenant=args.tenant,
        template=args.template,
        capture_hint_role=args.capture_hint_role,
        block_size=args.block_size,
        tokenizer=args.tokenizer,
        max_items=args.max_items,
        donor_max_tokens=args.donor_max_tokens,
        recipient_max_tokens=args.recipient_max_tokens,
        timeout_seconds=args.timeout_seconds,
        quality_threshold=args.quality_threshold,
        post_donor_delay_ms=args.post_donor_delay_ms,
        arm=args.arm,
        paired=args.paired,
        reset_urls=tuple(args.reset_urls),
        concurrency=args.concurrency,
        min_donor_gap_requests=args.min_donor_gap_requests,
        metrics_chunk_output=args.metrics_chunk_output,
    )
    engine_flags = _engine_flags_or_exit(args)
    violations = (
        phase0_flag_violations(engine_flags)
        if engine_flags is not None
        else ["no serve command recorded for this arm (--engine-serve-command)"]
    )
    if violations and args.require_phase0_flags:
        print(json.dumps({"phase0_flag_violations": violations}, indent=2))
        raise SystemExit(3)
    # Counters are read around the arm so the result carries a delta, not a
    # process-lifetime total inherited from whatever ran before it.
    metrics_urls = _engine_metrics_urls(args)
    before = tuple(scrape_all(metrics_urls))
    run = run_live_gateway_measured(config)
    after = tuple(scrape_all(metrics_urls))
    requests, audit_join = join_connector_audit(list(run.requests), connector_audit)
    engine = engine_document(
        arm=args.arm,
        flags=engine_flags,
        window=MetricsWindow(before=before, after=after),
    )
    # The runner is the only stage that reads the manifest, so the per-class
    # item counts section 4's denominators are taken over travel with the
    # result rather than being re-derived from the rows that survived.
    class_counts = dict(run.manifest_class_counts)
    write_result(
        args.output,
        requests=requests,
        config={
            "mode": "live-gateway",
            **config.__dict__,
            "connector_audit": connector_audit,
            "connector_audit_join": audit_join,
            "request_id_echo": request_id_echo(requests),
            "manifest_class_counts": class_counts,
        },
        run=run_metadata,
        engine=engine,
        manifest_class_counts=class_counts,
    )
    summary = {"output": args.output, "requests": len(requests)}
    # A throughput document is what the concurrency arms are for; a serial arm
    # writes one only when asked.
    if args.concurrency > 1 or args.throughput_output:
        throughput_path = write_throughput_document(
            output=args.output,
            explicit_path=args.throughput_output,
            document=run.throughput,
        )
        summary["throughput_output"] = throughput_path
        summary["requests_per_second"] = run.throughput.get("requests_per_second")
        summary["requests_per_second_excluding_settle"] = run.throughput.get(
            "requests_per_second_excluding_settle"
        )
    print(json.dumps(summary, indent=2))


def throughput_document_path(output: str, explicit_path: str | None = None) -> str:
    """Where a run's throughput document goes: beside its result, by default."""
    if explicit_path:
        return explicit_path
    return str(Path(output).with_suffix(".throughput.json"))


def write_throughput_document(
    *,
    output: str,
    explicit_path: str | None,
    document: dict,
) -> str:
    path = Path(throughput_document_path(output, explicit_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def cmd_engine_snapshot(args) -> None:
    document = snapshot_document(scrape_all(list(args.metrics_url), timeout=args.timeout_seconds))
    encoded = json.dumps(document, indent=2, sort_keys=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def cmd_engine_window(args) -> None:
    engine_flags = _engine_flags_or_exit(args)
    try:
        before = snapshots_from_document(json.loads(Path(args.before).read_text(encoding="utf-8")))
        after = snapshots_from_document(json.loads(Path(args.after).read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"engine window: {exc}") from exc

    document = engine_document(
        arm=args.arm,
        flags=engine_flags,
        window=MetricsWindow(before=before, after=after),
    )
    if args.result:
        try:
            attach_engine_document(args.result, document)
        except ValueError as exc:
            raise SystemExit(f"engine window: {exc}") from exc
    encoded = json.dumps(document, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if args.require_phase0_flags and not document["phase0_flags_ok"]:
        raise SystemExit(3)


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


def cmd_calibrate_noise_floor(args) -> None:
    import asyncio

    from sembench.calibration import build_artifact, run_calibration
    from sembench.schema import read_jsonl
    from sembench.verify import verify_endpoint

    engine_version = ""
    if not args.skip_verify:
        report = verify_endpoint(engine="sglang", base_url=args.base_url, expect_model=args.model)
        print(json.dumps({"preflight": report.to_dict()}, indent=2, sort_keys=True))
        if not report.passed:
            raise SystemExit(3)
        engine_version = report.engine_version

    items = read_jsonl(args.manifest, max_items=args.max_items)

    async def _run():
        import aiohttp

        from sembench.sglang_live import _HttpTransport

        timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            transport = _HttpTransport(session=session, base_url=args.base_url.rstrip("/"))
            return await run_calibration(
                items=items,
                transport=transport,
                max_new_tokens=args.max_new_tokens,
                cooldown_ms=args.cooldown_ms,
                warmup_requests=args.warmup_requests,
            )

    pairs, contaminated, errored = asyncio.run(_run())
    artifact = build_artifact(
        pairs=pairs,
        contaminated=contaminated,
        errored=errored,
        model=args.model,
        engine="sglang",
        engine_version=engine_version,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(artifact.to_dict(), indent=2, sort_keys=True))


def cmd_run_load(args) -> None:
    import json as _json

    from sembench.load import LoadConfig, run_load

    config = LoadConfig(
        manifest=args.manifest,
        output=args.output,
        gateway_url=args.gateway_url,
        model=args.model,
        tenant=args.tenant,
        template=args.template,
        concurrency=args.concurrency,
        max_items=args.max_items,
        donor_max_tokens=args.donor_max_tokens,
        recipient_max_tokens=args.recipient_max_tokens,
        post_donor_delay_ms=args.post_donor_delay_ms,
        timeout_seconds=args.timeout_seconds,
        run_id=args.run_id,
        min_donor_gap_requests=args.min_donor_gap_requests,
    )
    doc = run_load(config)
    with open(args.output, "w", encoding="utf-8") as handle:
        _json.dump(doc, handle)
    print(
        _json.dumps({k: v for k, v in doc.items() if k not in ("donors", "recipients")}, indent=2)
    )


if __name__ == "__main__":
    main()
