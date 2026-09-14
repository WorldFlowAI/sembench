"""``sembench assert-result-gates`` — the CI gate over a written result.

The gate definitions, the threshold action that remembers what the caller
actually asked for, and the command that evaluates them live together: a gate
whose metric is absent is a FAILURE unless the caller opted into skipping it,
and that rule is only legible with the flags next to the evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class GateThreshold(argparse.Action):
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


def add_gates_parser(sub) -> argparse.ArgumentParser:
    """Register ``assert-result-gates`` on the CLI's subparsers."""
    gates = sub.add_parser(
        "assert-result-gates",
        help="Fail unless replay and engine audit artifacts meet quality/reuse gates",
    )
    gates.add_argument("--result", required=True)
    gates.add_argument("--engine-summary", action="append", default=[])
    gates.add_argument("--min-quality-pass-rate", type=float, default=0.0, action=GateThreshold)
    gates.add_argument(
        "--min-semantic-placement-rate", type=float, default=0.0, action=GateThreshold
    )
    gates.add_argument(
        "--min-backend-confirmed-block-rate", type=float, default=0.0, action=GateThreshold
    )
    gates.add_argument("--min-materialization-events", type=int, default=0)
    gates.add_argument("--min-materialized-tokens", type=int, default=0)
    gates.add_argument("--min-materialized-units", type=int, default=0)
    gates.add_argument(
        "--max-negative-control-confirmed-rate", type=float, default=0.0, action=GateThreshold
    )
    gates.add_argument(
        "--max-negative-control-semantic-placement-rate",
        type=float,
        default=1.0,
        action=GateThreshold,
    )
    gates.add_argument("--require-materialized-reuse", action="store_true")
    gates.add_argument("--require-no-engine-errors", action="store_true")
    gates.add_argument("--min-blended-ttft-speedup", type=float, default=None, action=GateThreshold)
    gates.add_argument(
        "--max-negative-control-speedup-deviation",
        type=float,
        default=None,
        action=GateThreshold,
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
        action=GateThreshold,
        help="Max allowed drop of paired warm-vs-cold ROUGE-L below the calibrated floor (requires --noise-floor-calibration)",
    )

    return gates


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
    # The control gate reads the MEDIAN, for the same reason the headline gate
    # does: these are ratios. One control pair whose cold arm hit a slow
    # prefill carries a mean far enough from 1.0 to fail a deviation gate on
    # its own, which reports contamination that did not happen — and the same
    # arithmetic in the other direction hides one that did. The CI is
    # published beside the point estimate so a "passing" control measured over
    # four pairs is visibly a control measured over four pairs.
    negative_speedup = paired.get("negative_control_ttft_speedup_median")
    negative_speedup_ci = paired.get("negative_control_ttft_speedup_median_ci")
    if args.max_negative_control_speedup_deviation is not None:
        require_metric(
            "negative_control_ttft_speedup_median",
            "max_negative_control_speedup_deviation",
            negative_speedup,
            lambda value: abs(value - 1.0) <= args.max_negative_control_speedup_deviation,
            f"|{negative_speedup} - 1.0| > {args.max_negative_control_speedup_deviation} "
            f"(median of the negative-control pairs; 95% CI {negative_speedup_ci})",
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
            "negative_control_pairs": paired.get("negative_control_pairs"),
            "negative_control_ttft_speedup_median": negative_speedup,
            "negative_control_ttft_speedup_median_ci": negative_speedup_ci,
            # Secondary and not gateable, published so the two can be compared
            # when they disagree.
            "negative_control_ttft_speedup_mean": paired.get("negative_control_ttft_speedup_mean"),
        },
        "requested_gates": sorted(requested),
        "missing_metrics": sorted(set(missing_metrics)),
        "allow_missing": allow_missing,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
