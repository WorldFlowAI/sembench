"""Round 6, third pass: the holes the verifier found in the second pass.

- **M7's permissive branch was the DEFAULT branch.** Round 6 stopped an
  unlabelled A4-vs-A6 merge from reading its own contaminated baseline by
  checking ``run.baseline_id`` — but ``--backend-id`` defaults to the empty
  string, ``merged_run_metadata`` then falls back to the cold *run id*, and a
  backend id that names a build (``vllm-0.29-span``) resolves to no arm
  either. All three land on "declares no arm", which the code read as
  permission to publish. A fully contaminated workload merged with default
  flags still printed ``propagation_contamination_rate: 0.0``.
- **``propagation_cold_reference_arm`` asserted A1 where no arm answered.**
  The ``missing`` branch returned the arm that *should* have answered, so a
  reader saw ``"A1"`` beside a null rate and could conclude A1 was consulted.
- **M1's pinned-boundary rationale stated a mechanism that is false here.**
  "a donor built with the same wrapper is routinely already resident" — in
  M1's opportunity population the donor's wrapper differs from the
  recipient's by construction, in every item that names a donor.
- **The doc contract asserted a hardcoded list.** It could only catch keys the
  author remembered to list, and it missed one this pass added.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sembench.results import (
    COLD_REFERENCE_ARM_REQUIRED,
    COLD_REFERENCE_ARM_UNDECLARED,
    COLD_REFERENCE_FROM_BASELINE_ARM,
    PROPAGATION_PROBE_CLASS,
    WRAPPER_STRATUM_RULE,
    arm_id_of,
    arm_pair,
    paired_summary,
    write_result,
)
from sembench.schema import RequestMetrics

_ROOT = Path(__file__).resolve().parents[1]
SAME_DOC = "same_doc_new_instruction"
NO_REUSE = "no_reuse"
SERVED = "Donaghadee, County Down"
COLD_ANSWER = "Northern Ireland"


def _row(item_id: str, **kw) -> RequestMetrics:
    fields: dict[str, object] = {
        "item_id": item_id,
        "dataset": "longbench",
        "transform": SAME_DOC,
        "negative_control": False,
        "donor_count": 1,
        "prompt_tokens": 3000,
        "total_blocks": 187,
        "exact_hit_blocks": 0,
        "exact_hit_tokens": 0,
        "semantic_candidate_blocks": 0,
        "semantic_candidate_tokens": 0,
        "semantic_eligible_blocks": 0,
        "semantic_eligible_tokens": 0,
        "arm": "warm",
        "audit_joined": True,
        "traffic_class": SAME_DOC,
        "ttft_ms": 50.0,
    }
    fields.update(kw)
    return RequestMetrics(**fields)


def _contaminated_probe_rows() -> list[RequestMetrics]:
    """A workload contaminated end to end: both arms answer as served."""
    return [
        _row("parent", arm="cold", ttft_ms=200.0, output_text=SERVED, audit_joined=None),
        _row("parent", arm="warm", ttft_ms=100.0, output_text=SERVED),
        _row(
            "probe",
            arm="cold",
            ttft_ms=200.0,
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id="parent",
            output_text=SERVED,
            audit_joined=None,
        ),
        _row(
            "probe",
            arm="warm",
            ttft_ms=100.0,
            traffic_class=PROPAGATION_PROBE_CLASS,
            propagation_parent_item_id="parent",
            output_text=SERVED,
        ),
    ]


# --------------------------------------------------------------------------
# 1 — publishing M7 takes an affirmative A1 signal
# --------------------------------------------------------------------------


def _arm_document(tmp_path: Path, name: str, *, arm: str, rows, backend_id: str = "") -> str:
    payload = {
        "run": {
            "run_id": name,
            "arm": arm,
            "manifest_sha256": "deadbeef",
            "backend_id": backend_id,
            "baseline_id": backend_id,
        },
        "config": {"manifest_class_counts": {PROPAGATION_PROBE_CLASS: 50}},
        "engine": None,
        "requests": [row.to_dict() for row in rows],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _merged_with_backend_ids(tmp_path: Path, *, cold_id: str, warm_id: str) -> dict:
    """Merge a contaminated workload with the flags an operator actually
    passes, and hand back the document the CLI wrote."""
    from sembench.cli import main

    rows = _contaminated_probe_rows()
    cold = _arm_document(
        tmp_path,
        "cold",
        arm="cold",
        backend_id=cold_id,
        rows=[row for row in rows if row.arm == "cold"],
    )
    warm = _arm_document(
        tmp_path,
        "warm",
        arm="warm",
        backend_id=warm_id,
        rows=[row for row in rows if row.arm == "warm"],
    )
    output = tmp_path / "merged.json"
    main(["merge-results", "--cold", cold, "--warm", warm, "--output", str(output)])
    return json.loads(output.read_text(encoding="utf-8"))


def test_a_merge_with_the_default_flags_publishes_no_contamination_rate(tmp_path, capsys):
    """``--backend-id`` defaults to '', which is not "this is A1".

    The round-6 fix read ``run.baseline_id`` — but with no ``--backend-id``,
    ``merged_run_metadata`` falls back to the cold RUN ID, which names no arm,
    and the permissive branch published ``0.0`` on a workload contaminated end
    to end. This is the plain CLI with no optional flag at all, so it was the
    ordinary path, not an edge case.
    """
    merged = _merged_with_backend_ids(tmp_path, cold_id="", warm_id="")
    capsys.readouterr()

    # The run id fallback: non-empty, and not an arm.
    assert merged["run"]["baseline_id"] == "cold"
    assert arm_id_of(merged["run"]["baseline_id"]) == ""
    assert merged["paired"]["propagation_contamination_rate"] is None
    assert merged["paired"]["propagation_contamination_numerator"] is None
    assert merged["paired"]["propagation_cold_reference_missing"] is True
    assert merged["paired"]["propagation_cold_reference_arm"] == COLD_REFERENCE_ARM_UNDECLARED
    # The arm-vs-arm diagnostic — the number the defect used to publish as
    # section 4's metric — is still there, under the name that is not it.
    assert merged["paired"]["propagation_contamination_rate_vs_baseline_arm"] == 0.0
    assert merged["paired"]["propagation_contamination_numerator_vs_baseline_arm"] == 0


def test_a_merge_labelled_with_a_build_name_publishes_no_contamination_rate(tmp_path, capsys):
    """A run labelled the way ``--backend-id``'s own help advertises.

    ``vllm-0.29-span`` and ``sglang-fuzzy-pr31057`` name builds, not arms, so
    they resolve to no arm and land in exactly the same branch as the empty
    default — a labelled run is not an identified one.
    """
    assert arm_id_of("vllm-0.29-span") == ""
    assert arm_id_of("sglang-fuzzy-pr31057") == ""

    merged = _merged_with_backend_ids(
        tmp_path, cold_id="vllm-0.29-span", warm_id="sglang-fuzzy-pr31057"
    )
    capsys.readouterr()

    assert merged["run"]["baseline_id"] == "vllm-0.29-span"
    assert merged["paired"]["propagation_contamination_rate"] is None
    assert merged["paired"]["propagation_cold_reference_arm"] == COLD_REFERENCE_ARM_UNDECLARED


def test_a_merge_that_names_the_a1_cold_arm_still_publishes_the_rate(tmp_path, capsys):
    """The legitimate A1-vs-A4 merge keeps its number: it says which arm was
    cold, which is the whole of what the guard asks for."""
    merged = _merged_with_backend_ids(tmp_path, cold_id="A1 stock_pc", warm_id="A4 conn_span")
    capsys.readouterr()

    assert merged["paired"]["propagation_cold_reference_missing"] is False
    assert merged["paired"]["propagation_cold_reference_arm"] == "A1"
    assert merged["paired"]["propagation_cold_reference_source"] == COLD_REFERENCE_FROM_BASELINE_ARM
    assert merged["paired"]["propagation_contamination_rate"] == 0.0
    assert merged["paired"]["propagation_contamination_numerator"] == 0


def test_a_library_caller_that_declares_nothing_gets_no_rate_either():
    """The guard lives in the metric, not in the CLI: a caller that assembles
    rows by hand is in the same position as a merge that labelled nothing."""
    paired = paired_summary(
        _contaminated_probe_rows(), manifest_class_counts={PROPAGATION_PROBE_CLASS: 50}
    )

    assert paired is not None
    assert paired["propagation_contamination_rate"] is None
    assert paired["propagation_cold_reference_missing"] is True
    assert paired["propagation_cold_reference_source"] is None


# --------------------------------------------------------------------------
# 2 — a missing reference never names the arm that did not answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("a labelled m7 merge", {"arm_pair": "m7_propagation"}),
        ("an unlabelled A4 baseline", {"baseline_arm": "A4 conn_span"}),
    ],
)
def test_a_missing_cold_reference_is_not_recorded_as_if_a1_answered(label, kwargs):
    """``propagation_cold_reference_arm: "A1"`` beside a null rate reads as
    "A1 was consulted and found nothing". No A1 run was present at all."""
    paired = paired_summary(
        _contaminated_probe_rows(),
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
        **kwargs,
    )

    assert paired is not None, label
    assert paired["propagation_cold_reference_missing"] is True
    assert paired["propagation_cold_reference_arm"] != "A1"
    assert paired["propagation_cold_reference_arm"] == COLD_REFERENCE_ARM_REQUIRED
    # The sentinel names what is still needed, so it cannot be read as
    # provenance, and it stays distinct from "nothing named an arm".
    assert COLD_REFERENCE_ARM_REQUIRED != COLD_REFERENCE_ARM_UNDECLARED
    assert paired["propagation_cold_reference_source"] is None


def test_both_missing_shapes_are_documented_with_the_values_they_emit():
    metrics = (_ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8")

    assert COLD_REFERENCE_ARM_REQUIRED in metrics
    assert f'"{COLD_REFERENCE_ARM_UNDECLARED}"' in metrics


# --------------------------------------------------------------------------
# 3 — the stratum rationale states a mechanism that is true here
# --------------------------------------------------------------------------

DONOR_IDENTITY_CLAIM = "donor built with the same wrapper"


@pytest.mark.parametrize(
    "source",
    ["sembench/traffic_classes.py", "docs/METRICS.md"],
)
def test_the_pinned_boundary_is_not_justified_by_a_donor_that_shares_the_wrapper(source):
    """M1's opportunity population is ``same_doc_new_instruction ∪
    revised_doc``, and both classes are wrapper-mismatched by construction —
    the donor never shares the recipient's wrapper, popular rank or not. The
    rationale the boundary is checked against has to be the one that is true:
    the RECIPIENT's own wrapper is resident, so its boundary is non-zero.
    """
    text = (_ROOT / source).read_text(encoding="utf-8")

    assert DONOR_IDENTITY_CLAIM not in text, source
    assert "recipient" in text.lower(), source


def test_the_published_rule_states_the_recipient_side_mechanism():
    """The rule travels in every document, so the correction travels too:
    the head is where the RECIPIENT's own wrapper is resident and its boundary
    is non-zero, not where its donor shares a wrapper."""
    assert "recipient's OWN wrapper" in WRAPPER_STRATUM_RULE
    assert "boundary is non-zero" in WRAPPER_STRATUM_RULE
    assert DONOR_IDENTITY_CLAIM not in WRAPPER_STRATUM_RULE


# --------------------------------------------------------------------------
# 4 — the doc contract is derived from the document, not from memory
# --------------------------------------------------------------------------

# Keys a written document emits that docs/METRICS.md does not spell out
# verbatim: each is covered there by a prose shorthand on its parent row
# ("+ _numerator / _denominator", "+ _ci") or by the block that introduces it.
# The list predates this pass and is deliberately explicit: a NEW key is
# either documented by name or added here on purpose, and nothing else passes.
UNDOCUMENTED_BY_NAME = (
    "alignment_given_match_denominator",
    "alignment_given_match_numerator",
    "blended_ttft_speedup_median_ci",
    "engine_ttft_speedup_median_ci",
    "exact_hit_tokens",
    "exact_token_weighted_reuse_rate",
    "expected_span_target_start_agreement_denominator",
    "expected_span_target_start_agreement_numerator",
    "expected_supplied_tokens_agreement_denominator",
    "expected_supplied_tokens_agreement_numerator",
    "external_confirmed_token_sources",
    "external_confirmed_token_weighted_reuse_rate",
    "external_confirmed_tokens_warm",
    "first_token_divergence_median",
    "fuzzy_confirmed_tokens_warm",
    "hit_only_ttft_speedup_ci",
    "hit_only_ttft_speedup_mean",
    "hit_only_ttft_speedup_median_ci",
    "kl_pairs",
    "materialized_reuse_request_denominator",
    "materialized_reuse_request_numerator",
    "mean_semblend_latency_ms",
    "miss_tax_ms_median_of_differences_ci",
    "negative_control_backend_confirmed_blocks",
    "negative_control_backend_confirmed_rate",
    "negative_control_blocks",
    "negative_control_count",
    "negative_control_pairs",
    "negative_control_semantic_eligible_blocks",
    "negative_control_semantic_placement_rate",
    "negative_control_semantic_placements",
    "negative_control_ttft_speedup_median_ci",
    "propagation_cached_without_materialization_denominator",
    "propagation_cached_without_materialization_numerator",
    "quality_f1_ci",
    "quality_rouge_l_ci",
    "request_count",
    "reuse_mechanisms",
    "semantic_candidate_token_weighted_reuse_rate",
    "semantic_candidate_tokens",
    "semantic_eligible_token_weighted_reuse_rate",
    "semantic_eligible_tokens",
    "semantic_mechanisms",
    "semblend_hit_rate_by_request",
    "ttft_cold_p50_ms",
    "ttft_cold_p95_ms",
    "ttft_warm_p50_ms",
    "ttft_warm_p95_ms",
    "warm_vs_cold_mean_kl_topk",
    "warm_vs_cold_mean_kl_topk_ci",
    "warm_vs_cold_output_rouge_l_ci",
)


def _document_rows() -> list[RequestMetrics]:
    """Enough shape to reach every branch of the metric blocks at once:
    an audited same-doc pair, a no_reuse pair (M4's capture leg) and a
    propagation probe with its parent (M7)."""
    audited = {
        "wrapper_id": "w1-terse",
        "wrapper_rank": 0,
        "audit_semantic_lookup_hit": True,
        "audit_lookup_hit_boundary": 1024,
        "audit_advertised_tokens": 512,
        "audit_advertised_target_start": 1024,
        "audit_materialized": 512,
        "audit_observed_boundary": 1024,
        "expected_supplied_tokens": 512,
        "expected_span_target_start": 1024,
        "quality_score": 0.9,
        "output_text": SERVED,
    }
    return [
        _row("i1", arm="cold", ttft_ms=200.0, audit_joined=None, **audited),
        _row("i1", arm="warm", ttft_ms=100.0, **audited),
        _row(
            "nr1",
            arm="cold",
            ttft_ms=200.0,
            traffic_class=NO_REUSE,
            audit_joined=None,
            **audited,
        ),
        _row("nr1", arm="warm", ttft_ms=100.0, traffic_class=NO_REUSE, **audited),
        *_contaminated_probe_rows(),
    ]


def _emitted_keys(tmp_path: Path) -> set[str]:
    """Every metric key a written document carries, across the pair labels
    that change which keys are emitted."""
    keys: set[str] = set()
    for name, pair in (("plain", None), ("capture", "m4_capture"), ("m7", "m7_propagation")):
        config: dict[str, object] = {"mode": "merge-results"}
        if pair is not None:
            config["arm_pair"] = arm_pair(pair).to_dict()
        path = tmp_path / f"{name}.json"
        write_result(
            path,
            requests=_document_rows(),
            config=config,
            manifest_class_counts={
                SAME_DOC: 300,
                "revised_doc": 150,
                NO_REUSE: 100,
                PROPAGATION_PROBE_CLASS: 50,
            },
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        for block in ("aggregate", "paired"):
            keys |= set((payload.get(block) or {}).keys())
    return keys


def test_every_key_a_written_document_emits_is_documented(tmp_path):
    """The doc contract, derived rather than remembered.

    The hardcoded 5-tuple this replaces asserted only what the author listed,
    and it missed ``propagation_contamination_numerator_vs_baseline_arm``: a
    key a reader can find nowhere in METRICS.md by name.
    """
    metrics = (_ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8")

    undocumented = sorted(
        key
        for key in _emitted_keys(tmp_path)
        if key not in metrics and key not in UNDOCUMENTED_BY_NAME
    )

    assert undocumented == [], undocumented


def test_the_undocumented_allowlist_holds_no_stale_entry(tmp_path):
    """An allowlist nobody prunes stops being a contract. Every entry has to
    be a key the document still emits and METRICS.md still omits."""
    emitted = _emitted_keys(tmp_path)
    metrics = (_ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8")

    assert sorted(UNDOCUMENTED_BY_NAME) == list(UNDOCUMENTED_BY_NAME)
    assert [key for key in UNDOCUMENTED_BY_NAME if key not in emitted] == []
    assert [key for key in UNDOCUMENTED_BY_NAME if key in metrics] == []
