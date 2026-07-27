from sembench.dedupe import drop_overlapping_sources, trim_per_dataset
from sembench.schema import SourceRecord


def _rec(source_id: str, dataset: str, context: str) -> SourceRecord:
    return SourceRecord(source_id=source_id, dataset=dataset, context=context, input="q?")


UNIQUE_A = " ".join(f"apple{i}" for i in range(64))
UNIQUE_B = " ".join(f"berry{i}" for i in range(64))
SHARED = " ".join(f"shared{i}" for i in range(32))


def test_disjoint_sources_all_kept():
    kept, report = drop_overlapping_sources([_rec("a", "d1", UNIQUE_A), _rec("b", "d1", UNIQUE_B)])
    assert [r.source_id for r in kept] == ["a", "b"]
    assert not report.dropped


def test_later_duplicate_is_dropped_order_preserving():
    unique_c = " ".join(f"cedar{i}" for i in range(48))
    kept, report = drop_overlapping_sources(
        [
            _rec("a", "d1", UNIQUE_A + " " + SHARED),
            _rec("b", "d1", UNIQUE_B),
            _rec("c", "d2", SHARED + " " + unique_c),  # overlaps ONLY a, via SHARED
        ]
    )
    assert [r.source_id for r in kept] == ["a", "b"]
    assert report.dropped[0]["source_id"] == "c"
    assert report.dropped[0]["overlaps"] == "a"


def test_shifted_overlap_is_caught():
    """The duplicated span sits at different offsets in the two sources —
    aligned-block hashing would miss it; rolling windows must not."""
    prefix = " ".join(f"pad{i}" for i in range(7))  # shifts alignment by 7 tokens
    kept, report = drop_overlapping_sources(
        [
            _rec("a", "d1", UNIQUE_A + " " + SHARED),
            _rec("b", "d1", prefix + " " + SHARED + " " + UNIQUE_B),
        ]
    )
    assert [r.source_id for r in kept] == ["a"]
    assert report.dropped[0]["source_id"] == "b"


def test_short_contexts_never_match_anything():
    kept, report = drop_overlapping_sources(
        [_rec("a", "d1", "tiny context"), _rec("b", "d1", "tiny context")]
    )
    # Below one window, no hashes are produced; both kept.
    assert len(kept) == 2
    assert not report.dropped


def test_trim_per_dataset_caps_in_order():
    records = [_rec(f"{d}-{i}", d, UNIQUE_A) for d in ("d1", "d2") for i in range(4)]
    trimmed = trim_per_dataset(records, 2)
    assert [r.source_id for r in trimmed] == ["d1-0", "d1-1", "d2-0", "d2-1"]
    assert trim_per_dataset(records, None) == records
