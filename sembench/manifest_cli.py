"""``sembench build`` and the frozen-manifest commands.

Everything that turns source records into a manifest file, and everything that
pins one: building a workload, auditing it for phantom block collisions,
freezing a spec's checksum and rebuilding a frozen spec to compare against it.
They share ``_build_items`` and nothing else in the CLI shares them.
"""

from __future__ import annotations

import json
from pathlib import Path

from sembench.longbench import DEFAULT_LONGBENCH_V1_DATASETS, load_source_records
from sembench.schema import write_jsonl
from sembench.transforms import DEFAULT_TRANSFORMS, TransformConfig, build_workload


def add_manifest_parsers(sub) -> None:
    """Register the manifest-side subcommands on the CLI's subparsers."""
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
    verify_frozen = sub.add_parser(
        "verify-frozen", help="Rebuild a frozen spec and compare against recorded checksums"
    )
    verify_frozen.add_argument("--spec", required=True)
    verify_frozen.add_argument("--manifests-dir", default="manifests")


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


def cmd_checksum_manifest(args) -> None:
    from sembench.schema import manifest_sha256

    print(
        json.dumps(
            {"manifest": args.manifest, "sha256": manifest_sha256(args.manifest)},
            indent=2,
            sort_keys=True,
        )
    )
