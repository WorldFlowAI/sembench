"""Connector-audit plumbing shared by the CLI commands that write a result.

Both a live gateway run and ``merge-results`` join the connector's audit JSONL
onto their rows before writing, and both echo back whether the engine adopted
the request ids the runner sent. The helpers live here so the merge command can
move out of ``sembench.cli`` without either importing the other.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_connector_audit_arg(parser: argparse.ArgumentParser) -> None:
    """The connector audit JSONL this result is to be joined against.

    The audit is the only artifact that can say a load was *materialized*
    rather than advertised, so M1/M2/M7 are unobtainable without it.
    """
    parser.add_argument(
        "--connector-audit",
        default=None,
        metavar="PATH",
        help="SemBlend vLLM connector audit JSONL (the connector's audit_path) to join this "
        "result against by request_id; source of alignment_given_match / "
        "alignment_given_opportunity / boundary_miss_breakdown (M1), materialized_reuse_rate "
        "(M2, token-weighted) and prefix_blocks_evicted (M7's gating counter)",
    )


def connector_audit_or_exit(args) -> str | None:
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


def join_connector_audit(requests: list, connector_audit: str | None) -> tuple[list, dict | None]:
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


def request_id_echo(requests: list) -> dict:
    """Did the engine adopt the ids the runner sent?

    Reported whether or not an audit was joined: a front end that strips
    ``X-Request-Id`` makes vLLM mint its own id, and every audit-derived
    metric then reads as an arm that materialized nothing. The mismatch count
    is the difference between "no reuse happened" and "the join key never
    arrived".
    """
    from sembench.connector_audit import request_id_echo_report

    return request_id_echo_report(requests)
