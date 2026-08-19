# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only tests for tree trigger and candidate-budget helpers."""

from types import SimpleNamespace

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sampling_params import SamplingParams, TreeSearchParams
from vllm.sequence import Sequence, SequenceStatus
from vllm.worker.model_runner import ModelRunner


class FakeSequence:
    def __init__(self, token_ids, *, output_len=4, tree_depth=0):
        self._token_ids = list(token_ids)
        self._output_len = output_len
        self.tree_depth = tree_depth

    def get_token_ids(self):
        return self._token_ids

    def get_output_len(self):
        return self._output_len

    def get_tree_segment_len(self):
        return self._output_len


def _sequence_with_output_len(output_len, *, branch_token_id=None):
    sequence = object.__new__(Sequence)
    sequence.data = SimpleNamespace(
        get_output_len=lambda: output_len,
        get_last_token_id=lambda: 99,
        get_len=lambda: output_len + 1,
    )
    sequence.new_branch_token_id = branch_token_id
    sequence.eos_token_id = 2
    sequence.status = SequenceStatus.RUNNING
    sequence.output_text = ""
    return sequence


def test_forced_branch_token_does_not_consume_child_output_budget():
    root = _sequence_with_output_len(7)
    child = _sequence_with_output_len(7, branch_token_id=123)

    # A forced branch token is already part of the child prompt.  It is
    # visible in the reconstructed segment, but must not count against the
    # child's generated-token budget or vLLM's delta-output offsets.
    assert root.get_output_len() == 7
    assert root.get_tree_segment_len() == 7
    assert child.get_output_len() == 7
    assert child.get_tree_segment_len() == 8


def test_stop_checker_counts_only_tokens_generated_by_child():
    checker = StopChecker(
        max_model_len=128,
        get_tokenizer_for_seq=lambda _: None,
    )
    params = SamplingParams(max_tokens=8, ignore_eos=True)

    one_slot_left = _sequence_with_output_len(7, branch_token_id=123)
    checker.maybe_stop_sequence(
        one_slot_left,
        new_char_count=0,
        sampling_params=params,
    )
    assert one_slot_left.status == SequenceStatus.RUNNING

    budget_consumed = _sequence_with_output_len(8, branch_token_id=123)
    checker.maybe_stop_sequence(
        budget_consumed,
        new_char_count=0,
        sampling_params=params,
    )
    assert budget_consumed.status == SequenceStatus.FINISHED_LENGTH_CAPPED


def test_immediate_tree_budget_is_preserved_across_depths():
    max_tokens = 32
    generated_before_each_split = [5, 7, 3]
    remaining_budget = max_tokens
    stitched_segment_lengths = []

    for depth, generated in enumerate(generated_before_each_split):
        sequence = _sequence_with_output_len(
            generated,
            branch_token_id=100 + depth if depth else None,
        )
        remaining_budget -= sequence.get_output_len()
        # Immediate branching removes the sampled trigger and replaces it
        # with the forced child token.  Root has no leading branch token;
        # every later segment does.
        stitched_segment_lengths.append(
            sequence.get_tree_segment_len() - 1)

    # The final child contributes its leading forced token plus every token
    # left in the generation budget.
    stitched_segment_lengths.append(1 + remaining_budget)
    assert sum(stitched_segment_lengths) == max_tokens


def test_deferred_tree_budget_keeps_the_existing_extra_slot():
    max_tokens = 32
    generated_before_split = 9
    # Deferred WAAD branching removes t1 and t2 from the parent and replaces
    # t1 with a forced child token, so the child receives one extra output
    # slot.  This is independent of the forced-token accounting fixed above.
    child_budget = max_tokens - generated_before_split + 1
    stitched_parent_length = generated_before_split - 2
    stitched_child_length = 1 + child_budget
    assert stitched_parent_length + stitched_child_length == max_tokens


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


def test_entropy_trigger_ignores_waad_score():
    engine = object.__new__(LLMEngine)
    sequence = FakeSequence([1, 2, 3, 4])
    params = SimpleNamespace(
        seed=0,
        tree_search_params=TreeSearchParams(
            enable_tree_search=True,
            entropy_threshold=1.0,
            min_seg_length=0,
            branch_trigger_mode="entropy",
            # A numeric legacy threshold must not turn an explicitly selected
            # entropy trigger back into the deferred WAAD path.
            tau_importance=100.0,
        ),
    )

    engine._calculate_entropy = lambda _: 2.0
    assert engine._should_create_branches(
        sequence,
        logprobs=None,
        sampling_params=params,
        importance_score=0.0,
    ) is True

    engine._calculate_entropy = lambda _: 0.5
    assert engine._should_create_branches(
        sequence,
        logprobs=None,
        sampling_params=params,
        importance_score=1000.0,
    ) is False


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


def test_entropy_stats_skip_attention_importance_path():
    runner = object.__new__(ModelRunner)
    params = SamplingParams(
        collect_threshold_stats=True,
        collect_importance_stats=False,
    )
    model_input = SimpleNamespace(
        attn_metadata=SimpleNamespace(num_prefills=0),
        sampling_metadata=SimpleNamespace(
            seq_groups=[SimpleNamespace(sampling_params=params)],
        ),
    )

    # Entropy-only collection must return before accessing runner.vllm_config
    # or any cached attention query used by WAAD.
    assert runner._compute_importance_if_needed(
        output=None,
        model_input=model_input,
        virtual_engine=0,
    ) is None


def test_entropy_tree_skip_attention_importance_path():
    runner = object.__new__(ModelRunner)
    params = SamplingParams(
        tree_search_params=TreeSearchParams(
            enable_tree_search=True,
            branch_trigger_mode="entropy",
            tau_importance=None,
        ),
    )
    model_input = SimpleNamespace(
        attn_metadata=SimpleNamespace(num_prefills=0),
        sampling_metadata=SimpleNamespace(
            seq_groups=[SimpleNamespace(sampling_params=params)],
        ),
    )

    # The real entropy-only tree request must also return before looking up
    # runner.vllm_config or an attention layer's cached query.
    assert runner._compute_importance_if_needed(
        output=None,
        model_input=model_input,
        virtual_engine=0,
    ) is None
