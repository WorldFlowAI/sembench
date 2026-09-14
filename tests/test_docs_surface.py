"""The docs are part of the contract: a flag or metric nobody can find is a
flag nobody uses, and a documented one that no longer exists is worse.

These checks are deliberately shallow. They do not grade the prose; they catch
the drift that silently happens — a subcommand added to the CLI and never
documented, a documented subcommand deleted from the CLI, a gate whose docs
still name the estimator it stopped reading.
"""

import argparse
import re
from pathlib import Path

from sembench.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
METRICS = (ROOT / "docs" / "METRICS.md").read_text(encoding="utf-8")


def _cli_subcommands() -> set[str]:
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("the CLI has no subparsers")


def _documented_subcommands() -> set[str]:
    """Subcommand names from the README table's first column.

    One row may cover a pair (`freeze` / `verify-frozen`), so every code span
    in the cell counts.
    """
    documented: set[str] = set()
    for line in README.splitlines():
        if not line.startswith("| `"):
            continue
        cell = line.split("|")[1]
        documented.update(re.findall(r"`([a-z0-9-]+)`", cell))
    return documented


def test_every_subcommand_is_in_the_readme_table():
    assert not _cli_subcommands() - _documented_subcommands()


def test_the_readme_table_invents_no_subcommands():
    assert not _documented_subcommands() - _cli_subcommands()


def test_the_new_subcommands_are_documented():
    """The three the round-2 spec names explicitly."""
    documented = _documented_subcommands()
    for command in ("merge-results", "engine-snapshot", "engine-window"):
        assert command in documented


def test_the_paired_runner_flags_are_documented():
    gateway = build_parser()._actions
    subparsers = next(
        action for action in gateway if isinstance(action, argparse._SubParsersAction)
    )
    options = {
        option
        for action in subparsers.choices["run-live-gateway"]._actions
        for option in action.option_strings
    }
    for flag in ("--paired", "--reset-url", "--worker-url", "--concurrency"):
        assert flag in options, f"{flag} is documented but the runner does not accept it"
        assert flag in README


def test_the_new_result_keys_are_documented():
    for key in (
        "external_confirmed_tokens",
        "engine_ttft_ms",
        "blended_ttft_speedup_median",
        "requests_per_second",
    ):
        assert key in README
        assert key in METRICS


def test_the_docs_say_the_gate_reads_the_median():
    assert "blended_ttft_speedup_median" in README
    assert "`--min-blended-ttft-speedup` gates `blended_ttft_speedup_median`" in README
    assert "gateable (`--min-blended-ttft-speedup`)" in METRICS


def test_the_docs_keep_cached_tokens_out_of_semantic_reuse():
    """The distinction the whole round exists to make: cached_tokens is local
    prefix cache plus external transfer, and only the external split is
    semantic reuse."""
    assert "local prefix cache" in METRICS
    assert "must never be gated on as semantic reuse" in METRICS
    assert "not `cached_tokens`, is the\n  semantic-reuse signal" in README
