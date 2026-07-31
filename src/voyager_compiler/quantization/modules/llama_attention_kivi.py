"""KIVI attention for LLaMA: a split prefill + residual KV cache.

``swap_llama_attention`` replaces every ``LlamaAttention`` with a variant whose
forward reads two caches — the quantized prefill cache and a short unquantized
residual — and concatenates their attention scores.

Copied from ``transformers`` ``modeling_llama`` and modified for the residual
cache; ``fmt: off`` keeps it in the upstream's formatting so a diff against
upstream stays readable.

Scheduled for retirement: ``codegen.transform.quant_folding`` now folds a
quantize into the cache write directly, which is what this module was written
to arrange by hand.
"""

import logging
from typing import Optional

import torch
from torch import nn
from torch.ao.quantization.fx.utils import assert_and_get_unique_device
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

logger = logging.getLogger(__name__)

__all__ = ["LlamaAttentionKIVI", "swap_llama_attention"]

# fmt: off

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    key_residual: torch.Tensor = None,
    value_residual: torch.Tensor = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # HACK manually add a slice nop to prevent compiler from folding the scale
    # computation into the param
    if module.num_key_value_groups == 1:
        value_states = value_states[:]

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if key_residual is not None:
        key_states_residual = repeat_kv(key_residual, module.num_key_value_groups)
        attn_weights_residual = torch.matmul(query, key_states_residual.transpose(2, 3)) * scaling
        attn_weights = torch.cat([attn_weights, attn_weights_residual], dim=-1)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : attn_weights.shape[-1]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if value_residual is not None:
        value_states_residual = repeat_kv(value_residual, module.num_key_value_groups)
        attn_output = torch.matmul(attn_weights[:, :, :, : value_states.shape[-2]], value_states)
        attn_output_residual = torch.matmul(attn_weights[:, :, :, value_states.shape[-2]:], value_states_residual)
        attn_output = attn_output + attn_output_residual
    else:
        attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttentionKIVI(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_values_residual = kwargs.get("past_key_values_residual", None)

        if past_key_values_residual is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position_residual")}
            key_states_residual, value_states_residual = past_key_values_residual.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                past_key_values.layers[self.layer_idx].keys,
                past_key_values.layers[self.layer_idx].values,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                key_residual=key_states_residual,
                value_residual=value_states_residual,
                **kwargs,
            )
        else:
            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def swap_llama_attention(model: PreTrainedModel) -> PreTrainedModel:
    """
    Swap the attention module in a LLaMA model with the custom LlamaAttention module.

    Args:
        model (`PreTrainedModel`): The pretrained LLaMA model to modify.
    Returns:
        `PreTrainedModel`: The modified model with the custom attention module.
    """

    logger.info("Using custom LlamaAttention module.")

    def swap_module(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LlamaAttention):
                device = assert_and_get_unique_device(child)
                dtype = next(child.parameters()).dtype
                new_attn = LlamaAttentionKIVI(child.config, child.layer_idx).to(device=device, dtype=dtype)
                new_attn.load_state_dict(child.state_dict(), strict=True)
                setattr(module, name, new_attn)
                logger.info(f"Replaced {full_name} with LlamaAttentionKIVI")
            else:
                swap_module(child, full_name)

        return module

    return swap_module(model)

# fmt: on
