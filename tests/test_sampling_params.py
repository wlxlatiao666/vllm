# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SamplingParams class.
"""

import pytest

from vllm import SamplingParams
from vllm.config import ModelConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.sampling_params import TreeSearchParams

MODEL_NAME = "Qwen/Qwen1.5-7B"


def test_tree_search_trigger_mode_backward_compatibility():
    assert (TreeSearchParams(
        tau_importance=None).resolved_branch_trigger_mode() == "entropy")
    assert (TreeSearchParams(
        tau_importance=0.0).resolved_branch_trigger_mode()
            == "entropy_waad")
    assert (TreeSearchParams(
        tau_importance=1.0,
        branch_trigger_mode="random",
    ).resolved_branch_trigger_mode() == "random")
    assert (TreeSearchParams(
        tau_importance=1.0,
        branch_trigger_mode="entropy",
    ).resolved_branch_trigger_mode() == "entropy")


def test_importance_stats_can_be_disabled_without_changing_legacy_default():
    assert (SamplingParams(
        collect_threshold_stats=True).collect_importance_stats is True)
    assert SamplingParams(
        collect_threshold_stats=True,
        collect_importance_stats=False,
    ).collect_importance_stats is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"branch_trigger_mode": "unknown"},
        {"branch_trigger_mode": "entropy_waad"},
        {"random_branch_probability": -0.1},
        {"random_branch_probability": 1.1},
        {"random_branch_probability": "0.2"},
        {"random_branch_probability": None},
        {"random_branch_probability": True},
        {"max_num_leaves": 0},
        {"max_num_leaves": 1.5},
    ],
)
def test_tree_search_trigger_mode_validation(kwargs):
    with pytest.raises(ValueError):
        TreeSearchParams(**kwargs)


def test_max_tokens_none():
    """max_tokens=None should be allowed"""
    SamplingParams(temperature=0.01, top_p=0.1, max_tokens=None)


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(
        MODEL_NAME,
        task="auto",
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
    )


@pytest.fixture(scope="module")
def default_max_tokens():
    return 4096


def test_sampling_params_from_request_with_no_guided_decoding_backend(
        model_config, default_max_tokens):
    # guided_decoding_backend is not present at request level
    request = ChatCompletionRequest.model_validate({
        'messages': [{
            'role': 'user',
            'content': 'Hello'
        }],
        'model':
        MODEL_NAME,
        'response_format': {
            'type': 'json_object',
        },
    })

    sampling_params = request.to_sampling_params(
        default_max_tokens,
        model_config.logits_processor_pattern,
    )
    # we do not expect any backend to be present and the default
    # guided_decoding_backend at engine level will be used.
    assert sampling_params.guided_decoding.backend is None


@pytest.mark.parametrize("request_level_guided_decoding_backend,expected",
                         [("xgrammar", "xgrammar"),
                          ("lm-format-enforcer", "lm-format-enforcer"),
                          ("outlines", "outlines")])
def test_sampling_params_from_request_with_guided_decoding_backend(
        request_level_guided_decoding_backend: str, expected: str,
        model_config, default_max_tokens):

    request = ChatCompletionRequest.model_validate({
        'messages': [{
            'role': 'user',
            'content': 'Hello'
        }],
        'model':
        MODEL_NAME,
        'response_format': {
            'type': 'json_object',
        },
        'guided_decoding_backend':
        request_level_guided_decoding_backend,
    })

    sampling_params = request.to_sampling_params(
        default_max_tokens,
        model_config.logits_processor_pattern,
    )
    # backend correctly identified in resulting sampling_params
    assert sampling_params.guided_decoding.backend == expected
