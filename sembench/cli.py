"""CLI for SemBench."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sembench.engine_config import (
    attach_engine_document,
    engine_document,
    load_engine_flags,
    phase0_flag_violations,
    snapshot_document,
    snapshots_from_document,
)
from sembench.engine_events import parse_engine_events
from sembench.gateway_live import (
    LiveGatewayConfig,
    parse_worker_urls,
    run_live_gateway,
    run_live_gateway_measured,
)
from sembench.longbench import DEFAULT_LONGBENCH_V1_DATASETS, load_source_records
from sembench.offline import OfflineConfig, run_offline
from sembench.prometheus import MetricsWindow, scrape_all
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


def _add_connector_audit_arg(parser: argparse.ArgumentParser) -> None:
    """The connector audit JSONL this result is to be joined against.

    The audit is the only artifact that can say a load was *materialized*
    rather than advertised, so M1/M2/M7 are unobtainable without it.
    """
    parser.add_argument(
        "--connector-audit",
        default=None,
        metavar="PATH",
        help="SemBlend vLLM connector audit JSONL (the connector's audit_path) to join this "
        "result against by request_id; source of boundary_alignment_rate (M1), "
        "materialized_reuse_rate (M2) and propagation_rate (M7)",
    )


def _connector_audit_or_exit(args) -> str | None:
    """The audit path, checked to exist before an arm spends any GPU time.

    A mistyped path is otherwise indistinguishable downstream from an arm that
    genuinely produced no materialization events, and that confusion reads as
    a clean zero rather than as a missing measurement.
    """
    path = getattr(args, "connector_audit", None)
    if not path:
        return None
    if not Path(path).is_file():
        raise SystemExit(
            f"connector audit: {path} does not exist. Point --connector-audit at the file the "
            "connector's audit_path / SEMBLEND_VLLM_AUDIT_PATH writes, or drop the flag; a "
            "missing audit is not an arm with no materialization"
        )
    return str(path)


def _join_connector_audit(requests: list, connector_audit: str | None) -> tuple[list, dict | None]:
    """Stamp the audit's facts onto the rows before the result is written.

    Returns the rows and the join report. With no audit the rows pass through
    untouched and every audit-derived metric stays null, which is the honest
    reading of a run that measured nothing rather than a zero.
    """
    if not connector_audit:
        return list(requests), None
    from sembench.connector_audit import AuditError, join_audit_file

    try:
        joined, report = join_audit_file(requests, connector_audit)
    except AuditError as exc:
        raise SystemExit(f"connector audit: {exc}") from exc
    return list(joined), report.to_dict()


def _engine_flags_or_exit(args):
    try:
        return load_engine_flags(
            serve_command=args.engine_serve_command,
            serve_command_file=args.engine_serve_command_file,
            env_assignments=args.engine_env,
        )
    except ValueError as exc:
        raise SystemExit(f"engine config: {exc}") from exc


class _GateThreshold(argparse.Action):
    """A gate threshold, remembering that the caller actually asked for it.

    Gates whose metric is absent used to be skipped, which reads as a pass in
    CI: that is how a reuse gate goes green on a run that never measured
    reuse. A gate the caller explicitly requested now fails when its metric is
    missing, so `requested_gates` has to survive parsing.
    """

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)
        requested = getattr(namespace, "requested_gates", None)
        if requested is None:
            requested = set()
            setattr(namespace, "requested_gates", requested)
        requested.add(self.dest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sembench",
        description="Semantic KV cache benchmark suite",
    )
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build", help="Build a local workload manifest")
    build.add_argument(
        "--profile",
        choices=("fixture", "synthetic-v1", "longbench-v1", "longbench-v2"),
        default=None,
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

    audit = sub.add_parser(
        "audit-manifest", help="Check a manifest for phantom cross-item block collisions"
    )
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--block-size", type=int, default=16)
    audit.add_argument("--tokenizer", default=None)

    freeze = sub.add_parser("freeze", help="Build a frozen spec's manifest and record its checksum")
    freeze.add_argument("--spec", required=True)
    freeze.add_argument("--manifests-dir", default="manifests")
    freeze.add_argument("--block-size", type=int, default=16)

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
        help="Fleet worker endpoint donors are seeded on directly; repeat per worker (or "
        "pass a comma-separated list). Recipients still go through --gateway-url, so what "
        "is measured is the router's placement decision",
    )
    gateway.add_argument("--tenant", default="tenant-a")
    gateway.add_argument("--template", default="rag-template-v1")
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
        type=int,
        default=1,
        help="Request streams in flight at once (default 1 = serial, unchanged). "
        "Above 1 the run also writes a throughput document; TTFT stays per-request, "
        "measured at the streamed first token",
    )
    gateway.add_argument(
        "--min-donor-gap-requests",
        type=int,
        default=0,
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
    _add_connector_audit_arg(gateway)

    merge = sub.add_parser(
        "merge-results",
        help="Join a cold and a warm single-arm result into one paired result by item_id",
    )
    merge.add_argument("--cold", required=True, help="Result JSON for the cold (baseline) arm")
    merge.add_argument("--warm", required=True, help="Result JSON for the warm (treatment) arm")
    merge.add_argument("--output", required=True)
    merge.add_argument("--run-id", default=None)
    merge.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Merge anyway when the arms do not join one-to-one (the pairing report still "
        "records every unpaired row; the paired summary is then a subset, not the run)",
    )
    _add_connector_audit_arg(merge)

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

    gates = sub.add_parser(
        "assert-result-gates",
        help="Fail unless replay and engine audit artifacts meet quality/reuse gates",
    )
    gates.add_argument("--result", required=True)
    gates.add_argument("--engine-summary", action="append", default=[])
    gates.add_argument("--min-quality-pass-rate", type=float, default=0.0, action=_GateThreshold)
    gates.add_argument(
        "--min-semantic-placement-rate", type=float, default=0.0, action=_GateThreshold
    )
    gates.add_argument(
        "--min-backend-confirmed-block-rate", type=float, default=0.0, action=_GateThreshold
    )
    gates.add_argument("--min-materialization-events", type=int, default=0)
    gates.add_argument("--min-materialized-tokens", type=int, default=0)
    gates.add_argument("--min-materialized-units", type=int, default=0)
    gates.add_argument(
        "--max-negative-control-confirmed-rate", type=float, default=0.0, action=_GateThreshold
    )
    gates.add_argument(
        "--max-negative-control-semantic-placement-rate",
        type=float,
        default=1.0,
        action=_GateThreshold,
    )
    gates.add_argument("--require-materialized-reuse", action="store_true")
    gates.add_argument("--require-no-engine-errors", action="store_true")
    gates.add_argument(
        "--min-blended-ttft-speedup", type=float, default=None, action=_GateThreshold
    )
    gates.add_argument(
        "--max-negative-control-speedup-deviation",
        type=float,
        default=None,
        action=_GateThreshold,
        help="Max allowed |negative-control speedup - 1.0| (cache must not fire on unrelated content)",
    )
    gates.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip an explicitly requested gate whose metric is absent instead of failing "
        "(a skipped gate reads as a pass: say so on purpose)",
    )
    gates.add_argument(
        "--require-contamination-check",
        action="store_true",
        help="Fail if any paired cold arm was flush-contaminated",
    )
    gates.add_argument(
        "--noise-floor-calibration",
        default=None,
        help="Calibration artifact from calibrate-noise-floor",
    )
    gates.add_argument(
        "--max-warm-vs-cold-rouge-drop",
        type=float,
        default=None,
        action=_GateThreshold,
        help="Max allowed drop of paired warm-vs-cold ROUGE-L below the calibrated floor (requires --noise-floor-calibration)",
    )

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
    negative_selection: str = "cross_domain",
):
    if profile == "longbench-v1" and not datasets:
        datasets = list(DEFAULT_LONGBENCH_V1_DATASETS)
    real_data = profile.startswith("longbench")
    fetch_cap = max_items_per_dataset
    if real_data and fetch_cap is not None:
        # Over-fetch so dropping duplicated documents doesn't shrink the corpus.
        fetch_cap = fetch_cap * 2
    records = load_source_records(
        profile=profile,
        datasets=datasets,
        max_items_per_dataset=fetch_cap,
        revision=revision,
    )
    if real_data:
        from sembench.dedupe import drop_overlapping_sources, trim_per_dataset

        records, dedupe_report = drop_overlapping_sources(records)
        records = trim_per_dataset(records, max_items_per_dataset)
        if dedupe_report.dropped:
            print(json.dumps({"dedupe_dropped": dedupe_report.dropped}, indent=2, sort_keys=True))
    config = TransformConfig(
        transforms=transforms,
        max_segments=max_segments,
        min_segment_chars=min_segment_chars,
        negative_selection=negative_selection,
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
        negative_selection = spec.negative_selection
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
        negative_selection = "cross_domain"

    records, items = _build_items(
        profile=profile,
        datasets=datasets,
        max_items_per_dataset=max_items_per_dataset,
        transforms=transforms,
        max_segments=max_segments,
        min_segment_chars=min_segment_chars,
        revision=revision,
        negative_selection=negative_selection,
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
        negative_selection=spec.negative_selection,
    )
    write_jsonl(output_path, items)
    return len(items)


def cmd_audit_manifest(args) -> None:
    from sembench.collision_audit import audit_manifest_items
    from sembench.schema import read_jsonl

    report = audit_manifest_items(
        read_jsonl(args.manifest),
        block_size=args.block_size,
        tokenizer_name=args.tokenizer,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


def cmd_freeze(args) -> None:
    from sembench.collision_audit import audit_manifest_items
    from sembench.frozen import get_frozen_spec, write_checksums
    from sembench.schema import manifest_sha256, read_jsonl

    spec = get_frozen_spec(args.spec)
    manifest_path = Path(args.manifests_dir) / spec.manifest_filename()
    item_count = _build_frozen_manifest(spec, manifest_path)
    audit = audit_manifest_items(read_jsonl(manifest_path), block_size=args.block_size)
    if not audit.passed:
        print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
        raise SystemExit(
            f"collision audit failed: {len(audit.violations)} violations — not freezing"
        )
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
        paired=args.paired,
        warmup_requests=args.warmup_requests,
        cooldown_ms=args.cooldown_ms,
        capture_logprobs=args.capture_logprobs,
        top_logprobs_num=args.top_logprobs_num,
        resume=args.resume,
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
    connector_audit = _connector_audit_or_exit(args)
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
    # A run that was asked for a throughput document needs the measured
    # variant; a plain serial arm is the same replay either way.
    wants_throughput = args.concurrency > 1 or bool(args.throughput_output)
    run = run_live_gateway_measured(config) if wants_throughput else None
    replayed = list(run.requests) if run is not None else run_live_gateway(config)
    after = tuple(scrape_all(metrics_urls))
    requests, audit_join = _join_connector_audit(replayed, connector_audit)
    write_result(
        args.output,
        requests=requests,
        config={
            "mode": "live-gateway",
            **config.__dict__,
            "connector_audit": connector_audit,
            "connector_audit_join": audit_join,
        },
        run=run_metadata,
        engine=engine_document(
            arm=args.arm,
            flags=engine_flags,
            window=MetricsWindow(before=before, after=after),
        ),
    )
    summary = {"output": args.output, "requests": len(requests)}
    if run is not None:
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


def _arm_provenance(payload: dict) -> dict:
    """What a merged result keeps about one source arm."""
    return {
        "run": payload.get("run"),
        "config": payload.get("config"),
        "engine": payload.get("engine"),
    }


def cmd_merge_results(args) -> None:
    """Join two single-arm results into one paired result.

    Arms usually run as separate server processes (stock baseline vs
    connector), so the pairing has to be reconstructed. It is reconstructed by
    item_id and nothing else: both arms must have replayed the same manifest
    bytes, each item must appear exactly once per arm, and the two rows must
    carry the same manifest-derived fingerprint. Anything else is reported and
    refused rather than quietly summarized.

    The cold/warm roles come from the command line, so swapping the two flags
    would invert every speedup in the merged document without a word of
    complaint. Each arm's own ``run.arm`` label is checked against the role it
    was passed as, and a contradiction is refused before anything is joined.
    """
    from sembench.pairing import (
        join_arms,
        merged_run_metadata,
        requests_from_result,
        result_manifest_sha256,
    )
    from sembench.results import arm_label_conflicts

    connector_audit = _connector_audit_or_exit(args)
    try:
        cold_payload = json.loads(Path(args.cold).read_text(encoding="utf-8"))
        warm_payload = json.loads(Path(args.warm).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"merge-results: {exc}") from exc

    conflicts = arm_label_conflicts(cold_payload, warm_payload)
    if conflicts:
        print(
            json.dumps(
                {"merged": False, "output": None, "arm_label_conflicts": conflicts},
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1)

    rows, report = join_arms(
        requests_from_result(cold_payload),
        requests_from_result(warm_payload),
        cold_manifest_sha256=result_manifest_sha256(cold_payload),
        warm_manifest_sha256=result_manifest_sha256(warm_payload),
    )
    if not report.ok and not args.allow_unpaired:
        print(
            json.dumps(
                {"merged": False, "output": None, "pairing": report.to_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1)

    joined_rows, audit_join = _join_connector_audit(rows, connector_audit)
    write_result(
        args.output,
        requests=joined_rows,
        config={
            "mode": "merge-results",
            "cold_result": args.cold,
            "warm_result": args.warm,
            "allow_unpaired": bool(args.allow_unpaired),
            "pairing": report.to_dict(),
            "connector_audit": connector_audit,
            "connector_audit_join": audit_join,
            "arms": {
                "cold": _arm_provenance(cold_payload),
                "warm": _arm_provenance(warm_payload),
            },
        },
        run=merged_run_metadata(cold_payload, warm_payload, run_id=args.run_id),
    )
    print(
        json.dumps(
            {"merged": True, "output": args.output, "pairing": report.to_dict()},
            indent=2,
            sort_keys=True,
        )
    )


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


def cmd_assert_result_gates(args) -> None:
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    aggregate = result.get("aggregate") or {}
    paired = result.get("paired") or {}
    engine_summaries = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in args.engine_summary
    ]
    failures: list[str] = []
    missing_metrics: list[str] = []
    requested = set(getattr(args, "requested_gates", None) or ())
    allow_missing = bool(getattr(args, "allow_missing", False))

    def require(name: str, ok: bool, detail: str) -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    def require_metric(name: str, dest: str, value, ok, detail: str) -> None:
        """Gate a metric that may be absent from the result.

        An absent metric is not a pass. When the caller explicitly asked for
        the gate, a missing metric FAILS: silently skipping is how an
        unmeasured reuse gate goes green. --allow-missing opts back into the
        skip, on the record.
        """
        if value is None:
            missing_metrics.append(name)
            if dest in requested and not allow_missing:
                option = "--" + dest.replace("_", "-")
                require(
                    name,
                    False,
                    f"metric absent from the result but {option} was requested "
                    "(pass --allow-missing to skip it deliberately)",
                )
            return
        require(name, ok(float(value)), detail)

    quality = aggregate.get("quality_pass_rate")
    require_metric(
        "quality_pass_rate",
        "min_quality_pass_rate",
        quality,
        lambda value: value >= args.min_quality_pass_rate,
        f"{quality} < {args.min_quality_pass_rate}",
    )
    semantic_placement = aggregate.get("semantic_placement_rate_by_request")
    require_metric(
        "semantic_placement_rate_by_request",
        "min_semantic_placement_rate",
        semantic_placement,
        lambda value: value >= args.min_semantic_placement_rate,
        f"{semantic_placement} < {args.min_semantic_placement_rate}",
    )
    backend_rate = aggregate.get("backend_confirmed_block_rate")
    require_metric(
        "backend_confirmed_block_rate",
        "min_backend_confirmed_block_rate",
        backend_rate,
        lambda value: value >= args.min_backend_confirmed_block_rate,
        f"{backend_rate} < {args.min_backend_confirmed_block_rate}",
    )
    negative_rate = aggregate.get("negative_control_backend_confirmed_rate")
    require_metric(
        "negative_control_backend_confirmed_rate",
        "max_negative_control_confirmed_rate",
        negative_rate,
        lambda value: value <= args.max_negative_control_confirmed_rate,
        f"{negative_rate} > {args.max_negative_control_confirmed_rate}",
    )
    negative_semantic_placement = aggregate.get("negative_control_semantic_placement_rate")
    require_metric(
        "negative_control_semantic_placement_rate",
        "max_negative_control_semantic_placement_rate",
        negative_semantic_placement,
        lambda value: value <= args.max_negative_control_semantic_placement_rate,
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

    # Every gate below reads `paired`, which is null unless the run produced
    # both arms. A null paired block used to make those gates evaporate, so a
    # single-arm result could satisfy a speedup gate it never measured.
    paired_present = result.get("paired") is not None
    paired_gates = requested & {
        "min_blended_ttft_speedup",
        "max_negative_control_speedup_deviation",
        "max_warm_vs_cold_rouge_drop",
    }
    if args.require_contamination_check:
        paired_gates = paired_gates | {"require_contamination_check"}
    if paired_gates and not paired_present and not allow_missing:
        missing_metrics.append("paired")
        require(
            "paired_summary",
            False,
            "result has no paired summary, so "
            f"{', '.join(sorted(paired_gates))} cannot be evaluated: run the gateway "
            "with --paired, or join two single-arm results with `sembench merge-results`",
        )

    # The gate reads the MEDIAN of the paired ratios, never the mean. Speedups
    # are ratios, so one pair whose cold arm stalled carries a mean over a
    # threshold the typical pair never reached. The mean stays in the document
    # as a secondary number; it is not gateable. A result that carries only the
    # mean has no median to evaluate, and an absent metric is not a pass.
    blended = paired.get("blended_ttft_speedup_median")
    if args.min_blended_ttft_speedup is not None:
        require_metric(
            "blended_ttft_speedup_median",
            "min_blended_ttft_speedup",
            blended,
            lambda value: value >= args.min_blended_ttft_speedup,
            f"{blended} < {args.min_blended_ttft_speedup} (median of paired cold/warm ratios)",
        )
    if args.max_negative_control_speedup_deviation is not None:
        negative_speedup = paired.get("negative_control_ttft_speedup_mean")
        require_metric(
            "negative_control_ttft_speedup",
            "max_negative_control_speedup_deviation",
            negative_speedup,
            lambda value: abs(value - 1.0) <= args.max_negative_control_speedup_deviation,
            f"|{negative_speedup} - 1.0| > {args.max_negative_control_speedup_deviation}",
        )
    if args.require_contamination_check:
        require(
            "paired_contamination",
            paired.get("pairs_contaminated") == 0,
            f"{paired.get('pairs_contaminated')} contaminated pairs present",
        )

    warm_vs_cold = paired.get("warm_vs_cold_output_rouge_l_mean")
    if args.max_warm_vs_cold_rouge_drop is not None:
        if args.noise_floor_calibration is None:
            raise SystemExit(
                "--max-warm-vs-cold-rouge-drop requires --noise-floor-calibration: "
                "quality gates are calibration-relative, never absolute"
            )
        from sembench.calibration import load_calibration

        calibration = load_calibration(args.noise_floor_calibration)
        floor = calibration.rouge_l_mean - args.max_warm_vs_cold_rouge_drop
        require(
            "warm_vs_cold_rouge_vs_floor",
            warm_vs_cold is not None and float(warm_vs_cold) >= floor,
            f"{warm_vs_cold} < calibrated floor {calibration.rouge_l_mean:.4f} "
            f"- allowed drop {args.max_warm_vs_cold_rouge_drop}",
        )
        # A warm-vs-cold similarity ABOVE the cold/cold self-agreement band is
        # itself suspicious (suggests the cold arm was warm): flag, don't pass silently.
        ceiling = min(1.0, calibration.rouge_l_ci.get("hi", 1.0) + 0.10)
        if warm_vs_cold is not None and float(warm_vs_cold) > ceiling:
            require(
                "warm_vs_cold_rouge_above_plausible_band",
                False,
                f"{warm_vs_cold} > {ceiling:.4f} — implausibly high; check cold-arm contamination",
            )

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
            "paired_present": paired_present,
            "pairs_used": paired.get("pairs_used"),
            "blended_ttft_speedup_median": blended,
            "blended_ttft_speedup_mean": paired.get("blended_ttft_speedup_mean"),
        },
        "requested_gates": sorted(requested),
        "missing_metrics": sorted(set(missing_metrics)),
        "allow_missing": allow_missing,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


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
