"""``sembench merge-results`` — join a cold and a warm arm into one document.

The command and its flags live together, and next to M7's cold-reference
wiring, which is the part of the merge that decides whether section 4's
contamination number can be published at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sembench.cli_audit import (
    add_connector_audit_arg,
    connector_audit_or_exit,
    join_connector_audit,
    request_id_echo,
)
from sembench.results import PHASE0_ARM_PAIRS, PROPAGATION_COLD_REFERENCE_ARM, write_result


def add_merge_parser(sub) -> argparse.ArgumentParser:
    """Register ``merge-results`` on the CLI's subparsers."""
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
    merge.add_argument(
        "--pair",
        default=None,
        choices=sorted(PHASE0_ARM_PAIRS),
        help="Which of the phase-0 plan's section-4 comparisons this merge is "
        "(m3_ttft = A1 vs A4, m6_noise_floor = A1 vs A2, m4_capture = A3 vs A4, "
        "m4_instrumentation = A5 vs A4, m7_propagation = A4 vs A6). Recorded on the "
        "merged document and checked against each arm's --backend-id/--baseline-id, so "
        "an A3-vs-A5 merge cannot be published as the M3 headline",
    )
    merge.add_argument(
        "--cold-reference",
        default=None,
        metavar="RESULT",
        help="Result JSON for the A1 reference arm, used as M7's cold answer. Section 4 "
        "compares the treatment answer against the COLD (A1) output, and on an "
        "m7_propagation merge the baseline arm is A4 — which can be contaminated on the "
        "same probe. propagation_contamination_rate is published only when some run is "
        "identified as A1: this flag, --pair m3_ttft/m6_noise_floor, or a cold arm whose "
        "--backend-id names A1. A merge that identifies no arm (the default, since "
        "--backend-id defaults to empty) publishes null with "
        "propagation_cold_reference_missing true and _arm 'undeclared'. The cold arm's "
        "RUN ID is never read as an arm declaration, however suggestively it is named",
    )
    merge.add_argument(
        "--cold-reference-arm",
        default=PROPAGATION_COLD_REFERENCE_ARM,
        help="Which arm --cold-reference holds. Checked against that result's own "
        "--backend-id and refused on a contradiction; the merged document records the "
        "arm the reference DECLARES, and a reference that declares none is recorded as "
        f"undeclared rather than as an asserted {PROPAGATION_COLD_REFERENCE_ARM} "
        f"(default {PROPAGATION_COLD_REFERENCE_ARM})",
    )
    add_connector_audit_arg(merge)
    return merge


def _arm_provenance(payload: dict) -> dict:
    """What a merged result keeps about one source arm."""
    return {
        "run": payload.get("run"),
        "config": payload.get("config"),
        "engine": payload.get("engine"),
    }


def _refuse(reason_key: str, reasons) -> None:
    """Print the refusal a merge stops on and exit non-zero."""
    print(
        json.dumps({"merged": False, "output": None, reason_key: reasons}, indent=2, sort_keys=True)
    )
    raise SystemExit(1)


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
    from sembench.results import (
        arm_label_conflicts,
        arm_pair,
        arm_pair_conflicts,
        cold_reference_arm_conflicts,
        cold_reference_conflicts,
        result_backend_arm,
        result_manifest_class_counts,
    )

    connector_audit = connector_audit_or_exit(args)
    try:
        pair = arm_pair(args.pair) if args.pair else None
    except ValueError as exc:
        raise SystemExit(f"merge-results: {exc}") from exc
    reference_path = getattr(args, "cold_reference", None)
    try:
        cold_payload = json.loads(Path(args.cold).read_text(encoding="utf-8"))
        warm_payload = json.loads(Path(args.warm).read_text(encoding="utf-8"))
        reference_payload = (
            json.loads(Path(reference_path).read_text(encoding="utf-8")) if reference_path else None
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"merge-results: {exc}") from exc

    conflicts = arm_label_conflicts(cold_payload, warm_payload)
    if pair is not None:
        conflicts.extend(arm_pair_conflicts(pair, cold_payload, warm_payload))
    # M7's reference provenance: --cold-reference-arm is an operator assertion
    # that defaults to A1, so it is checked against the reference document's
    # own arm, and a reference that can answer none of this merge's items is
    # refused rather than published as a scored-nothing zero.
    reference_rows = requests_from_result(reference_payload) if reference_payload else None
    reference_arm = result_backend_arm(reference_payload) if reference_payload else ""
    if reference_payload is not None and reference_rows is not None:
        conflicts.extend(cold_reference_arm_conflicts(reference_payload, args.cold_reference_arm))
        conflicts.extend(
            cold_reference_conflicts(
                reference_rows,
                requests_from_result(warm_payload),
                reference_path=str(reference_path),
            )
        )
    if conflicts:
        _refuse("arm_label_conflicts", conflicts)

    rows, report = join_arms(
        requests_from_result(cold_payload),
        requests_from_result(warm_payload),
        cold_manifest_sha256=result_manifest_sha256(cold_payload),
        warm_manifest_sha256=result_manifest_sha256(warm_payload),
    )
    if not report.ok and not args.allow_unpaired:
        _refuse("pairing", report.to_dict())

    joined_rows, audit_join = join_connector_audit(rows, connector_audit)
    # Both arms replayed the same manifest (the pairing check above refuses
    # anything else), so either arm's counts are the workload's. The warm arm
    # is preferred only because it is the one whose metrics they scope.
    class_counts = result_manifest_class_counts(warm_payload) or result_manifest_class_counts(
        cold_payload
    )
    # M4's per-lookup cost is read off the treatment arm's engine window.
    warm_engine = warm_payload.get("engine")
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
            "request_id_echo": request_id_echo(joined_rows),
            "manifest_class_counts": class_counts,
            # M7's cold reference: which document answered as A1, and which
            # arm that document DECLARES — not the flag, which is an assertion
            # with no evidence. Null means the merged arms answered for
            # themselves, which section 4 allows only when the baseline IS A1.
            "cold_reference_result": reference_path,
            "cold_reference_arm": reference_arm or None,
            # Which of section 4's comparisons this document is, when the
            # operator named one. Null means "an unlabelled cold/warm join".
            "arm_pair": pair.to_dict() if pair is not None else None,
            "arms": {
                "cold": _arm_provenance(cold_payload),
                "warm": _arm_provenance(warm_payload),
            },
        },
        run=merged_run_metadata(cold_payload, warm_payload, run_id=args.run_id),
        engine=warm_engine,
        manifest_class_counts=class_counts,
        cold_reference=reference_rows,
        cold_reference_arm=reference_arm or None,
    )
    print(
        json.dumps(
            {"merged": True, "output": args.output, "pairing": report.to_dict()},
            indent=2,
            sort_keys=True,
        )
    )
