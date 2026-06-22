from sembench.transforms import DEFAULT_TRANSFORMS, build_workload, fixture_records, split_context


def test_fixture_workload_has_all_transforms_and_stable_ids():
    records = fixture_records()
    first = build_workload(records)
    second = build_workload(records)

    assert len(first) == len(records) * len(DEFAULT_TRANSFORMS)
    assert [item.item_id for item in first] == [item.item_id for item in second]
    assert {item.transform for item in first} == set(DEFAULT_TRANSFORMS)
    assert any(item.negative_control for item in first)


def test_multi_donor_composite_uses_segment_donors():
    items = build_workload(fixture_records())
    item = next(i for i in items if i.transform == "multi_donor_composite")

    assert len(item.donor_prompts) > 1
    assert all(donor.label == "segment_donor" for donor in item.donor_prompts)
    assert "Cross-document enterprise analysis task" in item.recipient_prompt


def test_leading_evidence_transform_places_reusable_span_at_target_prefix():
    item = next(
        i for i in build_workload(fixture_records())
        if i.transform == "leading_evidence_new_task"
    )

    donor_text = item.donor_prompts[0].text
    evidence = donor_text.split("Reusable enterprise evidence cache entry\n\n", 1)[1]
    evidence = evidence.split("\n\nCached evidence summary:", 1)[0]

    assert item.metadata["materialization_shape"] == "target_prefix_boundary"
    assert donor_text.startswith("Reusable enterprise evidence cache entry")
    assert item.recipient_prompt.startswith(evidence)
    assert not item.recipient_prompt.startswith("Workspace evidence review")


def test_split_context_is_bounded_and_deterministic():
    context = ("Alpha sentence. Beta sentence. Gamma sentence. Delta sentence. " * 40).strip()

    a = split_context(context, max_segments=3, min_segment_chars=100)
    b = split_context(context, max_segments=3, min_segment_chars=100)

    assert a == b
    assert 1 <= len(a) <= 3
    assert all(segment for segment in a)
