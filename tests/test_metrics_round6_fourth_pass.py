"""Round 6, fourth pass: the holes the third-pass verifier found.

- **A free-text RUN ID could unlock M7.** The third pass made publishing M7
  take an affirmative A1 signal and read it off ``run.baseline_id`` — but
  ``merged_run_metadata`` falls back to the cold *run id* for that field when
  no ``--backend-id`` was given, and ``arm_id_of`` matches an arm name anywhere
  in a free-text id. A run called ``phase0-g5-a1-rack-cold`` therefore resolved
  to A1 and published ``0.0`` on a fully contaminated workload, through the
  same door one hinge over.
- **The scored-nothing guard was asymmetric.** A supplied reference that
  scored no probe nulled the rate; the identical "zero probes could be read"
  condition on the baseline-arm source published ``0.0``.
- **``ad_hoc`` overstated what it partitions.** Every wrapper in the stream
  comes from one shared 8-entry set, so ranks 2-7 are unpopular *shared*
  wrappers, not ad-hoc traffic.
- **``sembench/cli.py`` was over the 800-line ceiling and growing.**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sembench.merge import cmd_merge_results
from sembench.pairing import merged_run_metadata
from sembench.results import (
    COLD_REFERENCE_ARM_UNDECLARED,
    COLD_REFERENCE_FROM_BASELINE_ARM,
    PROPAGATION_PROBE_CLASS,
    WRAPPER_STRATUM_RULE,
    arm_id_of,
    paired_summary,
)
from sembench.schema import RequestMetrics

_ROOT = Path(__file__).resolve().parents[1]
SAME_DOC = "same_doc_new_instruction"
SERVED = "Donaghadee, County Down"
COLD_ANSWER = "Northern Ireland"
# A run id an operator would plausibly write for a phase-0 rack: it names the
# stream, the GPU box and the arm's ROLE, and it contains "a1" as a word.
A1_SHAPED_RUN_ID = "phase0-g5-a1-rack-cold"


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
# 1 — a run id is not an arm declaration
# --------------------------------------------------------------------------


def _arm_document(
    tmp_path: Path, name: str, *, arm: str, rows, backend_id: str = "", run_id: str | None = None
) -> str:
    payload = {
        "run": {
            "run_id": run_id or name,
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


def _merge(tmp_path: Path, *, cold_id: str = "", cold_run_id: str, warm_run_id: str) -> dict:
    """Merge a contaminated workload with the real CLI and hand back the
    document it wrote."""
    from sembench.cli import main

    rows = _contaminated_probe_rows()
    cold = _arm_document(
        tmp_path,
        "cold",
        arm="cold",
        backend_id=cold_id,
        run_id=cold_run_id,
        rows=[row for row in rows if row.arm == "cold"],
    )
    warm = _arm_document(
        tmp_path,
        "warm",
        arm="warm",
        run_id=warm_run_id,
        rows=[row for row in rows if row.arm == "warm"],
    )
    output = tmp_path / "merged.json"
    main(["merge-results", "--cold", cold, "--warm", warm, "--output", str(output)])
    return json.loads(output.read_text(encoding="utf-8"))


def test_the_matcher_really_does_read_an_arm_out_of_a_free_text_run_id():
    """The premise of the hole, stated on its own: ``arm_id_of`` is a
    word-boundary search over free text, so an id that merely CONTAINS an arm
    token resolves to that arm. That is correct for ``--backend-id``, which an
    operator writes as a declaration, and is exactly why a run id must never
    be fed to it."""
    assert arm_id_of(A1_SHAPED_RUN_ID) == "A1"
    assert arm_id_of("run-2026-09-13-a1") == "A1"
    assert arm_id_of("my-a0-baseline") == "A0"


def test_a_cold_run_id_that_names_an_arm_does_not_publish_m7(tmp_path, capsys):
    """The default flags with a suggestively named run: no ``--backend-id``,
    so nothing was DECLARED, and the merge must stay silent about M7 even
    though the cold run id resolves to A1."""
    merged = _merge(tmp_path, cold_run_id=A1_SHAPED_RUN_ID, warm_run_id="phase0-g5-a4-rack-warm")
    capsys.readouterr()

    # Run identity keeps the fallback, so the document still says which run
    # was the baseline...
    assert merged["run"]["baseline_id"] == A1_SHAPED_RUN_ID
    # ...and the arm declaration is empty, because nobody made one.
    assert merged["run"]["baseline_arm_declared"] == ""
    assert merged["paired"]["propagation_cold_reference_arm"] == COLD_REFERENCE_ARM_UNDECLARED
    assert merged["paired"]["propagation_cold_reference_missing"] is True
    assert merged["paired"]["propagation_cold_reference_source"] is None
    assert merged["paired"]["propagation_contamination_rate"] is None
    assert merged["paired"]["propagation_contamination_numerator"] is None


def test_a_declared_a1_cold_arm_still_publishes_m7(tmp_path, capsys):
    """The other direction, so the guard cannot be satisfied by refusing
    everything: ``--backend-id A1 stock_pc`` IS a declaration, and it keeps its
    number even when the run id names nothing at all."""
    merged = _merge(
        tmp_path,
        cold_id="A1 stock_pc",
        cold_run_id="phase0-g5-rack-cold",
        warm_run_id="phase0-g5-rack-warm",
    )
    capsys.readouterr()

    assert merged["run"]["baseline_arm_declared"] == "A1 stock_pc"
    assert merged["paired"]["propagation_cold_reference_arm"] == "A1"
    assert merged["paired"]["propagation_cold_reference_source"] == COLD_REFERENCE_FROM_BASELINE_ARM
    assert merged["paired"]["propagation_contamination_rate"] == 0.0


def test_merged_run_metadata_separates_identity_from_declaration():
    """The two fields, at the one place that writes them: ``baseline_id`` is
    which run, ``baseline_arm_declared`` is which arm it SAID it was."""
    cold = {"run": {"run_id": A1_SHAPED_RUN_ID, "backend_id": "", "baseline_id": ""}}
    warm = {"run": {"run_id": "warm", "backend_id": "A4 conn_span"}}

    undeclared = merged_run_metadata(cold, warm)
    declared = merged_run_metadata(
        {"run": {"run_id": "cold", "backend_id": "A1 stock_pc"}},
        warm,
    )

    assert undeclared.baseline_id == A1_SHAPED_RUN_ID
    assert undeclared.baseline_arm_declared == ""
    assert declared.baseline_id == "A1 stock_pc"
    assert declared.baseline_arm_declared == "A1 stock_pc"


# --------------------------------------------------------------------------
# 2 — scoring nothing publishes nothing, whatever the source
# --------------------------------------------------------------------------


def _unlinked_probe_rows(count: int) -> list[RequestMetrics]:
    """Probes that name no parent item: present, paired, and unscoreable."""
    rows: list[RequestMetrics] = []
    for index in range(count):
        for arm, ttft in (("cold", 200.0), ("warm", 100.0)):
            rows.append(
                _row(
                    f"probe{index}",
                    arm=arm,
                    ttft_ms=ttft,
                    traffic_class=PROPAGATION_PROBE_CLASS,
                    output_text=SERVED,
                    audit_joined=None if arm == "cold" else True,
                )
            )
    return rows


def test_the_baseline_arm_source_publishes_no_rate_when_nothing_was_scored():
    """The asymmetry the third-pass verifier found: an A1 baseline arm with 50
    probes and not one of them scoreable published ``0.0``, which reads as a
    clean run to anyone taking the headline key."""
    paired = paired_summary(
        _unlinked_probe_rows(50),
        baseline_arm="A1 stock_pc",
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None
    # A cold reference WAS available: this is not the missing-reference hole.
    assert paired["propagation_cold_reference_missing"] is False
    assert paired["propagation_cold_reference_source"] == COLD_REFERENCE_FROM_BASELINE_ARM
    assert paired["propagation_cold_reference_unusable"] is False
    # Nothing could be read, so nothing is published...
    assert paired["propagation_no_probe_scored"] is True
    assert paired["propagation_contamination_rate"] is None
    assert paired["propagation_contamination_numerator"] is None
    assert paired["propagation_contamination_rate_scored_only"] is None
    assert paired["propagation_contamination_scored_denominator"] is None
    # ...and the denominator and the exclusion counter still say what happened.
    assert paired["propagation_contamination_denominator"] == 50
    assert paired["propagation_probes_unlinked"] == 50


def test_a_supplied_reference_that_scored_nothing_keeps_its_narrower_flag():
    """The two states stay distinguishable: ``_unusable`` is "a reference was
    handed over and answered nothing", ``_no_probe_scored`` is the general
    condition it is one case of."""
    reference = [
        _row("another-manifest-item", arm="single", output_text=COLD_ANSWER, audit_joined=None)
    ]

    paired = paired_summary(
        _contaminated_probe_rows(),
        cold_reference=reference,
        cold_reference_arm="A1",
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None
    assert paired["propagation_cold_reference_unusable"] is True
    assert paired["propagation_no_probe_scored"] is True
    assert paired["propagation_contamination_rate"] is None
    assert paired["propagation_probes_without_cold_reference"] == 1


def test_a_document_that_scored_a_probe_still_publishes_its_rate():
    """The guard must not swallow the metric: one scoreable probe is one
    measurement, and it is published."""
    paired = paired_summary(
        _contaminated_probe_rows(),
        baseline_arm="A1 stock_pc",
        manifest_class_counts={PROPAGATION_PROBE_CLASS: 50},
    )

    assert paired is not None
    assert paired["propagation_no_probe_scored"] is False
    assert paired["propagation_contamination_scored_denominator"] == 1
    assert paired["propagation_contamination_rate"] == 0.0


def test_a_workload_with_no_probes_is_not_a_scored_nothing_document():
    """An EMPTY probe set read nothing because there was nothing to read. Its
    rate is null because 0/0 is, and the flag stays false so it cannot be
    reported as a run that failed to measure its probes."""
    paired = paired_summary(
        [
            _row("i1", arm="cold", ttft_ms=200.0, output_text=COLD_ANSWER, audit_joined=None),
            _row("i1", arm="warm", ttft_ms=100.0, output_text=COLD_ANSWER),
        ],
        baseline_arm="A1 stock_pc",
    )

    assert paired is not None
    assert paired["propagation_contamination_denominator"] == 0
    assert paired["propagation_no_probe_scored"] is False
    assert paired["propagation_contamination_rate"] is None


# --------------------------------------------------------------------------
# 3 — "ad_hoc" says what it actually partitions
# --------------------------------------------------------------------------

TAIL_DISCLOSURE = "no item carries a genuinely one-off wrapper"


@pytest.mark.parametrize("source", ["sembench/traffic_classes.py", "docs/METRICS.md"])
def test_the_ad_hoc_stratum_discloses_that_its_wrappers_are_shared(source):
    """Every wrapper in Stream B comes from one fixed 8-entry set, so the
    ad_hoc stratum is the unpopular TAIL of that set. A reader told "the ad-hoc
    stratum aligns at 1.03" would otherwise believe ad-hoc traffic was
    measured."""
    text = (_ROOT / source).read_text(encoding="utf-8")

    assert "one-off wrapper" in text, source
    assert "tail" in text.lower(), source


def test_the_disclosure_travels_in_the_published_rule():
    """The rule is republished on every document, so the caveat has to be in
    the rule and not only in the docs nobody ships with the JSON."""
    assert "TAIL" in WRAPPER_STRATUM_RULE
    assert "one shared fixed set" in WRAPPER_STRATUM_RULE
    assert "no item carries a genuinely one-off wrapper" in WRAPPER_STRATUM_RULE


def test_the_document_publishes_the_disclosure_with_the_strata(tmp_path):
    from sembench.results import write_result

    path = tmp_path / "result.json"
    write_result(
        path,
        requests=[
            _row("i1", arm="cold", wrapper_id="w1-terse", wrapper_rank=0, audit_joined=None),
            _row("i1", arm="warm", wrapper_id="w1-terse", wrapper_rank=0),
        ],
        config={"mode": "merge-results"},
    )
    aggregate = json.loads(path.read_text(encoding="utf-8"))["aggregate"]

    assert "one-off wrapper" in aggregate["wrapper_stratum_rule"]


# --------------------------------------------------------------------------
# 4 — the CLI is back under the file-size ceiling
# --------------------------------------------------------------------------

LINE_CEILING = 800


def test_the_merge_command_lives_in_its_own_module():
    """``cmd_merge_results`` and its flags moved out of ``sembench/cli.py``
    whole: the CLI dispatches to the module that owns the command."""
    from sembench import cli

    assert cli.cmd_merge_results is cmd_merge_results
    source = (_ROOT / "sembench" / "cli.py").read_text(encoding="utf-8")
    assert "def cmd_merge_results" not in source
    assert "def cmd_assert_result_gates" not in source


@pytest.mark.parametrize(
    "module",
    ["cli.py", "cli_audit.py", "merge.py", "gates_cli.py", "manifest_cli.py"],
)
def test_the_cli_modules_are_under_the_line_ceiling(module):
    """cli.py was 1,607 lines and growing; the extracted modules must not
    reintroduce the same pile under another name."""
    lines = (_ROOT / "sembench" / module).read_text(encoding="utf-8").splitlines()

    assert len(lines) <= LINE_CEILING, f"{module}: {len(lines)} lines"
