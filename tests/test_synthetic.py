from collections import Counter

from sembench.collision_audit import audit_manifest_items
from sembench.synthetic import synthetic_records
from sembench.transforms import DEFAULT_TRANSFORMS, TransformConfig, build_workload


def test_generator_is_deterministic():
    a = synthetic_records(sources_per_domain=4)
    b = synthetic_records(sources_per_domain=4)
    assert [r.__dict__ for r in a] == [r.__dict__ for r in b]


def test_composition_and_bands():
    records = synthetic_records()
    assert len(records) == 240
    domains = Counter(r.metadata["domain"] for r in records)
    assert set(domains.values()) == {80}
    bands = Counter(r.metadata["length_band"] for r in records)
    long_share = (bands["8k"] + bands["12k"] + bands["16k"]) / len(records)
    assert long_share > 0.5, f"8k+ bands must dominate, got {long_share:.2f}"
    assert len({r.source_id for r in records}) == len(records)


def test_every_source_has_extractable_answer():
    for record in synthetic_records(sources_per_domain=6):
        assert record.answers and record.answers[0]
        assert record.input.endswith("?")


def _small_workload():
    records = synthetic_records(sources_per_domain=6)
    config = TransformConfig(transforms=tuple(DEFAULT_TRANSFORMS))
    return build_workload(records, config)


def test_transformed_workload_is_collision_clean():
    report = audit_manifest_items(_small_workload(), block_size=16)
    assert report.passed, report.violations[:3]


def test_no_sentence_is_shared_across_sources():
    """Every sentence is salted with a per-source tag, so no sentence —
    template or fact — may appear in two sources. This is the invariant that
    makes the corpus collision-clean even when fact VALUES birthday-collide
    (e.g. two contracts drawing the same topic and day count)."""
    records = synthetic_records(sources_per_domain=80)
    seen: dict[str, str] = {}
    for record in records:
        for sentence in {s.strip() for s in record.context.split(". ") if s.strip()}:
            owner = seen.setdefault(sentence, record.source_id)
            assert (
                owner == record.source_id
            ), f"sentence shared by {owner} and {record.source_id}: {sentence[:80]}"
