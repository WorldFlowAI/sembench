"""Result aggregation and JSON writing — the public surface.

The metric blocks are split along their three seams and this module is the
entry point that keeps every import path the CLI and the tests already use:

- :mod:`sembench.arm_matrix` — section 3's eight arms, section 4's five named
  comparisons, and the checks that refuse a mislabelled merge;
- :mod:`sembench.per_arm_metrics` — the per-arm section-4 blocks (the
  aggregate, M1, M2, M7's inputs), computable from one arm's rows;
- :mod:`sembench.paired_metrics` — the blocks that need both arms (M3, M4, M6's
  paired quality, M7);

with :mod:`sembench.traffic_classes`, :mod:`sembench.reuse_signals` and
:mod:`sembench.metric_math` underneath them. Nothing here re-implements a
metric: this module re-exports them and writes the document.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sembench.arm_matrix import (
    M4_CAPTURE_PAIR,
    M7_PROPAGATION_PAIR,
    PHASE0_ARM_PAIRS,
    PHASE0_ARMS,
    Arm,
    ArmPair,
    arm_id_of,
    arm_label_conflicts,
    arm_pair,
    arm_pair_conflicts,
    cold_reference_arm_conflicts,
    result_arm,
    result_arm_pair_name,
    result_backend_arm,
    result_manifest_class_counts,
)
from sembench.paired_metrics import (
    LOOKUP_LATENCY_SUM_KEY,
    LOOKUPS_TOTAL_KEY,
    MISS_TAX_POPULATION_CAPTURE,
    MISS_TAX_POPULATION_DEFAULT,
    MISS_TAX_POPULATION_UNIDENTIFIED,
    MISS_TAX_SOURCE_AUDIT,
    MISS_TAX_SOURCE_UNIDENTIFIED,
    lookup_cost_from_engine,
    paired_summary,
)
from sembench.per_arm_metrics import (
    aggregate_by_transform,
    aggregate_metrics,
    connector_audit_metrics,
    quality_by_rope_delta_bucket,
)
from sembench.propagation import (
    COLD_REFERENCE_ARM_REQUIRED,
    COLD_REFERENCE_ARM_UNDECLARED,
    COLD_REFERENCE_BASELINE_ARMS,
    COLD_REFERENCE_FROM_BASELINE_ARM,
    COLD_REFERENCE_FROM_REFERENCE_ARM,
    COLD_REFERENCE_FROM_REFERENCE_ARM_UNDECLARED,
    PROPAGATION_COLD_REFERENCE_ARM,
    ColdReference,
    cold_reference_conflicts,
    cold_reference_for,
)
from sembench.reuse_signals import (
    REUSE_HIT_THRESHOLD_TOKENS,
    SEMANTIC_MECHANISMS,
    AuditableRows,
    audit_measured_row,
    audit_was_joined,
    audit_was_read,
    auditable_rows,
    external_token_sources,
    external_tokens_are_per_request,
    is_reuse_hit,
    reuse_mechanism,
    semantic_reuse_tokens,
)
from sembench.schema import RESULT_VERSION, RequestMetrics, RunMetadata
from sembench.traffic_classes import (
    AD_HOC_WRAPPER_STRATUM,
    ALIGNMENT_OPPORTUNITY_CLASSES,
    DENOMINATOR_FROM_MANIFEST,
    DENOMINATOR_FROM_ROWS_PRESENT,
    EXACT_REPEAT_CLASS,
    NO_REUSE_CLASS,
    PROPAGATION_PROBE_CLASS,
    REVISED_DOC_CLASS,
    REWORDED_DOC_CLASS,
    ROPE_DELTA_SWEEP_CLASS,
    SAME_DOC_NEW_INSTRUCTION_CLASS,
    SHARED_WRAPPER_MAX_RANK,
    SHARED_WRAPPER_STRATUM,
    TRAFFIC_CLASSES,
    UNSTRATIFIED_WRAPPER,
    WRAPPER_STRATA,
    WRAPPER_STRATUM_RULE,
    manifest_class_total,
    traffic_class_of,
    wrapper_head_share,
    wrapper_stratum_of,
)

__all__ = [
    "AD_HOC_WRAPPER_STRATUM",
    "ALIGNMENT_OPPORTUNITY_CLASSES",
    "Arm",
    "ArmPair",
    "AuditableRows",
    "COLD_REFERENCE_ARM_REQUIRED",
    "COLD_REFERENCE_ARM_UNDECLARED",
    "COLD_REFERENCE_BASELINE_ARMS",
    "COLD_REFERENCE_FROM_BASELINE_ARM",
    "COLD_REFERENCE_FROM_REFERENCE_ARM",
    "COLD_REFERENCE_FROM_REFERENCE_ARM_UNDECLARED",
    "ColdReference",
    "DENOMINATOR_FROM_MANIFEST",
    "DENOMINATOR_FROM_ROWS_PRESENT",
    "EXACT_REPEAT_CLASS",
    "LOOKUPS_TOTAL_KEY",
    "LOOKUP_LATENCY_SUM_KEY",
    "M4_CAPTURE_PAIR",
    "M7_PROPAGATION_PAIR",
    "MISS_TAX_POPULATION_CAPTURE",
    "MISS_TAX_POPULATION_DEFAULT",
    "MISS_TAX_POPULATION_UNIDENTIFIED",
    "MISS_TAX_SOURCE_AUDIT",
    "MISS_TAX_SOURCE_UNIDENTIFIED",
    "NO_REUSE_CLASS",
    "PHASE0_ARMS",
    "PHASE0_ARM_PAIRS",
    "PROPAGATION_COLD_REFERENCE_ARM",
    "PROPAGATION_PROBE_CLASS",
    "REUSE_HIT_THRESHOLD_TOKENS",
    "REVISED_DOC_CLASS",
    "REWORDED_DOC_CLASS",
    "ROPE_DELTA_SWEEP_CLASS",
    "SAME_DOC_NEW_INSTRUCTION_CLASS",
    "SEMANTIC_MECHANISMS",
    "SHARED_WRAPPER_MAX_RANK",
    "SHARED_WRAPPER_STRATUM",
    "TRAFFIC_CLASSES",
    "UNSTRATIFIED_WRAPPER",
    "WRAPPER_STRATA",
    "WRAPPER_STRATUM_RULE",
    "aggregate_by_transform",
    "aggregate_metrics",
    "arm_id_of",
    "arm_label_conflicts",
    "arm_pair",
    "arm_pair_conflicts",
    "audit_measured_row",
    "audit_was_joined",
    "audit_was_read",
    "auditable_rows",
    "cold_reference_arm_conflicts",
    "cold_reference_conflicts",
    "cold_reference_for",
    "connector_audit_metrics",
    "external_token_sources",
    "external_tokens_are_per_request",
    "is_reuse_hit",
    "lookup_cost_from_engine",
    "manifest_class_total",
    "paired_summary",
    "quality_by_rope_delta_bucket",
    "result_arm",
    "result_arm_pair_name",
    "result_backend_arm",
    "result_manifest_class_counts",
    "reuse_mechanism",
    "semantic_reuse_tokens",
    "traffic_class_of",
    "wrapper_head_share",
    "wrapper_stratum_of",
    "write_result",
]


def write_result(
    path: str | Path,
    *,
    requests: list[RequestMetrics],
    config: dict[str, Any],
    run: RunMetadata | None = None,
    engine: dict[str, Any] | None = None,
    manifest_class_counts: dict[str, int] | None = None,
    cold_reference: Sequence[RequestMetrics] | None = None,
    cold_reference_arm: str | None = None,
) -> None:
    """Write one arm's result document.

    ``engine`` carries the arm's serve line, the parsed launch flags, and the
    before/after engine counter window (see ``sembench.engine_config``). The
    key is always present — null when the arm was run without it — so a
    reader can tell "no engine config was recorded" from "these were the
    flags", rather than assuming.

    ``manifest_class_counts`` is the workload's per-traffic-class item count.
    Section 4's class-scoped denominators are counts of manifest items, and
    only the runner ever sees the manifest, so the count travels with the
    document; without it the metrics fall back to the rows present and say so.

    ``cold_reference`` is an A1 reference run's rows for M7's cold answer
    (``merge-results --cold-reference``). The comparison this document is
    (``--pair``) is read back out of ``config``, so a document always applies
    the population rules of the pair it declares — and when it declares no
    pair, ``run.baseline_arm_declared`` still says which arm supplied the cold
    rows, so M7 reads that rather than assuming the baseline is A1. It is that
    field and not ``run.baseline_id``: the latter is run identity and falls
    back to the cold RUN ID on a merge, and a run id that happens to contain
    "a1" is not an operator saying the baseline was A1.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_version": RESULT_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": run.to_dict() if run is not None else None,
        "config": config,
        "engine": engine,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host": platform.node(),
        },
        "aggregate": aggregate_metrics(requests, manifest_class_counts=manifest_class_counts),
        "paired": paired_summary(
            requests,
            manifest_class_counts=manifest_class_counts,
            engine=engine,
            arm_pair=result_arm_pair_name(config),
            baseline_arm=(run.baseline_arm_declared if run is not None else None),
            cold_reference=cold_reference,
            cold_reference_arm=cold_reference_arm,
        ),
        "by_transform": aggregate_by_transform(requests),
        "requests": [r.to_dict() for r in requests],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
