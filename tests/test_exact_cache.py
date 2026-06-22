from sembench.exact_cache import (
    ExactBlockIndex,
    count_backend_eligible_blocks,
    count_full_reuse_blocks,
)


def test_exact_block_index_counts_full_blocks_only():
    index = ExactBlockIndex(block_size=4)
    index.add("donor-a", [1, 2, 3, 4, 5, 6, 7])

    lookup = index.lookup([1, 2, 3, 4, 9, 9, 9, 9, 5, 6, 7])

    assert lookup.total_blocks == 2
    assert lookup.hit_blocks == 1
    assert lookup.hit_block_indices == {0}


def test_count_full_reuse_blocks_requires_complete_block():
    reused = {0, 1, 2, 3, 4, 5, 6}

    assert count_full_reuse_blocks(reused, token_count=8, block_size=4) == 1


def test_backend_eligible_requires_one_contiguous_donor_span():
    actions = [
        {"action": "copy_from_donor", "targetPos": 0, "donorPos": 8, "donorId": "a"},
        {"action": "copy_from_donor", "targetPos": 1, "donorPos": 9, "donorId": "a"},
        {"action": "copy_from_donor", "targetPos": 2, "donorPos": 10, "donorId": "a"},
        {"action": "copy_from_donor", "targetPos": 3, "donorPos": 11, "donorId": "a"},
        {"action": "copy_from_donor", "targetPos": 4, "donorPos": 20, "donorId": "a"},
        {"action": "copy_from_donor", "targetPos": 5, "donorPos": 22, "donorId": "a"},
        {"action": "copy_from_donor", "targetPos": 6, "donorPos": 23, "donorId": "a"},
        {"action": "copy_from_donor", "targetPos": 7, "donorPos": 24, "donorId": "a"},
    ]

    assert count_backend_eligible_blocks(actions, token_count=8, block_size=4) == 1
