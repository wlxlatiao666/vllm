# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only tests for tree trigger and candidate-budget helpers."""

from types import SimpleNamespace

from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import TreeSearchParams


class FakeSequence:
    def __init__(self, token_ids, *, output_len=4, tree_depth=0):
        self._token_ids = list(token_ids)
        self._output_len = output_len
        self.tree_depth = tree_depth

    def get_token_ids(self):
        return self._token_ids

    def get_output_len(self):
        return self._output_len


def test_random_trigger_probability_boundaries_skip_entropy():
    engine = object.__new__(LLMEngine)

    def fail_if_entropy_is_computed(_):
        raise AssertionError("random trigger must not calculate entropy")

    engine._calculate_entropy = fail_if_entropy_is_computed
    sequence = FakeSequence([1, 2, 3, 4])

    for probability, expected in ((0.0, False), (1.0, True)):
        params = SimpleNamespace(
            seed=7,
            tree_search_params=TreeSearchParams(
                branch_trigger_mode="random",
                random_branch_probability=probability,
                min_seg_length=0,
            ),
        )
        assert engine._should_create_branches(
            sequence, logprobs=None, sampling_params=params) is expected


def test_random_trigger_hashes_full_branch_path():
    params = SimpleNamespace(seed=11)
    # Same length/depth and same last four tokens, but a different earlier path.
    first = FakeSequence([100, 2, 3, 4, 5])
    second = FakeSequence([200, 2, 3, 4, 5])
    first_value = LLMEngine._tree_random_branch_value(first, params)
    second_value = LLMEngine._tree_random_branch_value(second, params)
    assert first_value != second_value
    # The rolling cache must not change the deterministic value on a repeat.
    assert LLMEngine._tree_random_branch_value(first, params) == first_value


def test_leaf_budget_caps_each_split():
    token_ids = [10, 11, 12, 13]
    tree_params = TreeSearchParams(max_num_leaves=3)

    def group_with_leaf_count(count):
        return SimpleNamespace(assembled_seq_group=SimpleNamespace(
            seqs=[SimpleNamespace(is_leaf=True) for _ in range(count)]))

    assert LLMEngine._cap_tree_branch_token_ids(
        group_with_leaf_count(1), token_ids, tree_params) == token_ids[:3]
    assert LLMEngine._cap_tree_branch_token_ids(
        group_with_leaf_count(2), token_ids, tree_params) == token_ids[:2]
    assert LLMEngine._cap_tree_branch_token_ids(
        group_with_leaf_count(3), token_ids, tree_params) == []

    legacy_params = TreeSearchParams(max_num_leaves=None)
    assert LLMEngine._cap_tree_branch_token_ids(
        group_with_leaf_count(99), token_ids, legacy_params) == token_ids
