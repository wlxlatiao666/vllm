# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.envs as envs
from vllm.attention import AttentionType
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.platforms import _Backend, current_platform
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.utils import validate_kv_sharing_target

def _compute_importance_score(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    attn_metadata: Any,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    W: int = 10,
) -> list[float]:
    num_seqs = query.shape[0]
    if key_cache.numel() == 0:
        return [0.0] * num_seqs

    block_size = key_cache.shape[1] if key_cache.dim() == 4 else key_cache.shape[3]
    importance_scores = []

    for i in range(num_seqs):
        seq_len = int(attn_metadata.seq_lens_tensor[i].item())
        if seq_len <= 1:
            importance_scores.append(0.0)
            continue

        block_table = attn_metadata.block_tables[i]
        token_indices = torch.arange(seq_len, device=key_cache.device)
        block_numbers = block_table[token_indices // block_size].long()
        block_offsets = (token_indices % block_size).long()

        if key_cache.dim() == 4:
            keys = key_cache[block_numbers, block_offsets, :, :]
            keys = keys.to(query.dtype)
        elif key_cache.dim() == 5:
            x = 16 // key_cache.element_size()
            keys = key_cache[block_numbers, :, :, block_offsets, :]
            keys = keys.reshape(seq_len, num_kv_heads, head_size).to(query.dtype)
        else:
            raise ValueError(f"Unsupported key_cache shape: {key_cache.shape}")

        num_queries_per_kv = num_heads // num_kv_heads
        if num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)

        q = query[i].unsqueeze(0).float()  # [1, num_heads, head_size]
        keys = keys.float()

        # attn_weights: [num_heads, 1, seq_len]
        attn_weights = scale * torch.einsum("qhd,khd->hqk", q, keys)
        # print("attn_weights.shape: ", attn_weights.shape)

        if alibi_slopes is not None:
            position_ids = torch.arange(seq_len, device=key_cache.device).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)
            attn_weights = attn_weights + alibi_bias

        # attn_probs: [num_heads, 1, seq_len]
        attn_probs = torch.softmax(attn_weights, dim=-1)
        # average over heads -> [seq_len]
        attn_avg = attn_probs.mean(dim=0).squeeze(0)
        attn_sum = attn_avg.sum(dim=-1)
        # print("attn_sum: ", attn_sum)
        # WAAD: weighted sum of attention from last token to all past tokens,
        # weight = min(distance, W)
        cur_idx = seq_len - 1
        past_indices = torch.arange(0, cur_idx, device=key_cache.device)
        deltas = (cur_idx - past_indices).float()
        weights = torch.clamp(deltas, max=W)
        attns_to_prev = attn_avg[:cur_idx]
        waad = float((attns_to_prev * weights).sum().item())

        importance_scores.append(waad)

    return importance_scores

class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        per_layer_sliding_window: Optional[int] = None,
        use_mla: bool = False,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            is_attention_free = cache_config.is_attention_free
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            is_attention_free = False
            calculate_kv_scales = False
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # The default k/v_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized k/v_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)
        # FlashAttn doesn't support quantizing the kv-cache only
        # but requires q to be quantized as well.
        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._prob_scale = torch.tensor(1.0, dtype=torch.float32)

        # We also keep the float32 versions of k/v_scale for attention
        # backends that don't support tensors (Flashinfer)
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0

        self.use_mla = use_mla
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window

        quant_method = quant_config.get_quant_method(
            self, prefix=prefix) if quant_config else None
        if quant_method is not None and not isinstance(
                quant_method, UnquantizedLinearMethod):
            assert isinstance(quant_method, BaseKVCacheMethod)
            # TODO (mgoin): kv cache dtype should be specified in the FP8
            # checkpoint config and become the "auto" behavior
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError("fp8_e5m2 kv-cache is not supported with "
                                 "fp8 checkpoints.")
            # If quantization is enabled, we make "k_scale" and "v_scale"
            # parameters so that it can be loaded from the model checkpoint.
            # The k/v_scale will then be converted back to native float32
            # values after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(head_size,
                                        dtype,
                                        kv_cache_dtype,
                                        block_size,
                                        is_attention_free,
                                        blocksparse_params is not None,
                                        use_mla=use_mla)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window, kv_cache_dtype,
                             blocksparse_params, logits_soft_cap, attn_type,
                             kv_sharing_target_layer_name, **extra_impl_args)
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        self.use_direct_call = not current_platform.is_cuda_alike(
        ) and not current_platform.is_cpu()

        self.use_output = attn_backend.accept_output_buffer
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix
        self.attn_type = attn_type

        # Determine if this is the last layer to optimize importance score calc
        self.is_last_layer = True # Default to true for safety
        try:
            vllm_config = get_current_vllm_config()
            hf_config = getattr(vllm_config.model_config, "hf_config", None)
            if hf_config is not None:
                num_layers = getattr(hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 0))
                if num_layers > 0:
                    import re
                    match = re.search(r'\.layers?\.(\d+)', prefix)
                    if match:
                        self.is_last_layer = (int(match.group(1)) == num_layers - 1)
        except Exception:
            pass

        if kv_sharing_target_layer_name is not None:
            if not envs.VLLM_USE_V1:
                raise NotImplementedError(
                    "Cross-layer KV sharing is not supported in V0.")

            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config(
            ).parallel_config.pipeline_parallel_size)
        ]

        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

        # Cache the last decode query for on-demand importance score computation
        self._cached_query: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales:
            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata.enable_kv_scales_calculation:
                self.calc_kv_scales(query, key, value)
        if self.use_output:
            output_shape = (output_shape
                            if output_shape is not None else query.shape)
            output = torch.empty(output_shape,
                                 dtype=query.dtype,
                                 device=query.device)
            hidden_size = output_shape[-1]
            # We skip reshaping query, key and value tensors for the MLA
            # backend since these tensors have different semantics and are
            # processed differently.
            if not self.use_mla:
                # Reshape the query, key, and value tensors.
                # NOTE(woosuk): We do this outside the custom op to minimize the
                # CPU overheads from the non-CUDA-graph regions.
                query = query.view(-1, self.num_heads, self.head_size)
                output = output.view(-1, self.num_heads, self.head_size)
                if key is not None:
                    key = key.view(-1, self.num_kv_heads, self.head_size)
                if value is not None:
                    value = value.view(-1, self.num_kv_heads, self.head_size)
            if self.use_direct_call:
                forward_context: ForwardContext = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                self.impl.forward(self,
                                  query,
                                  key,
                                  value,
                                  self_kv_cache,
                                  attn_metadata,
                                  output=output)
            else:
                torch.ops.vllm.unified_attention_with_output(
                    query, key, value, output, self.layer_name)

            # Cache query for on-demand importance score computation (decode only)
            attn_metadata_ = get_forward_context().attn_metadata
            if getattr(attn_metadata_, 'num_prefills', 0) == 0 and getattr(self, 'is_last_layer', True):
                self._cached_query = query.detach()

            return output.view(-1, hidden_size)
        else:
            if self.use_direct_call:
                forward_context = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                result = self.impl.forward(self, query, key, value,
                                         self_kv_cache, attn_metadata)
            else:
                result = torch.ops.vllm.unified_attention(
                    query, key, value, self.layer_name)

            # Cache query for on-demand importance score computation (decode only)
            attn_metadata_ = get_forward_context().attn_metadata
            if getattr(attn_metadata_, 'num_prefills', 0) == 0 and getattr(self, 'is_last_layer', True):
                self._cached_query = query.view(-1, self.num_heads, self.head_size).detach()

            return result

    def compute_importance_scores(self, attn_metadata) -> Optional[list]:
        """Compute importance scores using the cached query from the last decode step.

        The cached query corresponds to the token that was just decoded (new token),
        since it was the input query in the previous forward pass.
        """
        if self._cached_query is None:
            return None
        forward_context: ForwardContext = get_forward_context()
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        if len(self_kv_cache) == 0 or self_kv_cache[0].numel() == 0:
            return None
        scale_val = getattr(self.impl, 'scale', 1.0 / (self.head_size ** 0.5))
        alibi_slopes = getattr(self.impl, 'alibi_slopes', None)
        return _compute_importance_score(
            self._cached_query, self_kv_cache[0], attn_metadata,
            self.num_heads, self.num_kv_heads, self.head_size,
            scale_val, alibi_slopes
        )

    def calc_kv_scales(self, query, key, value):
        self._q_scale.copy_(torch.abs(query).max() / self.q_range)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range)
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        # We only calculate the scales once
        self.calculate_kv_scales = False

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(self.impl, "process_weights_after_loading"):
            self.impl.process_weights_after_loading(act_dtype)


class MultiHeadAttention(nn.Module):
    """Multi-headed attention without any cache, used for ViT."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        dtype = torch.get_default_dtype()
        attn_backend = get_attn_backend(head_size,
                                        dtype,
                                        kv_cache_dtype=None,
                                        block_size=16,
                                        is_attention_free=False)
        backend = backend_name_to_enum(attn_backend.get_name())
        if backend in {_Backend.FLASH_ATTN, _Backend.FLASH_ATTN_VLLM_V1}:
            backend = _Backend.XFORMERS

        self.attn_backend = backend if backend in {
            _Backend.TORCH_SDPA, _Backend.XFORMERS, _Backend.PALLAS_VLLM_V1
        } else _Backend.TORCH_SDPA

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: batch_size x seq_len x hidden_size"""
        # TODO(Isotr0py): Use existing backend implementations and support FA3
        bsz, q_len, _ = query.size()
        kv_len = key.size(1)

        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        if self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops

            out = xops.memory_efficient_attention_forward(query,
                                                          key,
                                                          value,
                                                          scale=self.scale)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            query, key, value = (x.transpose(1, 2)
                                 for x in (query, key, value))
            out = F.scaled_dot_product_attention(query,
                                                 key,
                                                 value,
                                                 scale=self.scale)
            out = out.transpose(1, 2)
        elif self.attn_backend == _Backend.PALLAS_VLLM_V1:
            query, key, value = (x.transpose(1, 2)
                                 for x in (query, key, value))
            from torch_xla.experimental.custom_kernel import flash_attention
            out = flash_attention(query, key, value, sm_scale=self.scale)
            out = out.transpose(1, 2)

        return out.reshape(bsz, q_len, -1)


def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: List[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.save_kv_layer(layer_name, kv_cache_layer,
                            attn_metadata[layer_name])


def unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    wait_for_kv_layer_from_connector(layer_name)

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    output = self.impl.forward(self, query, key, value, kv_cache,
                               attn_metadata)

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return output


def unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


direct_register_custom_op(
    op_name="unified_attention",
    op_func=unified_attention,
    mutates_args=[],
    fake_impl=unified_attention_fake,
    dispatch_key=current_platform.dispatch_key,
)


def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      output=output)

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key=current_platform.dispatch_key,
)
