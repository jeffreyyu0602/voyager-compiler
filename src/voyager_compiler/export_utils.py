"""Capture an FX graph, add to one, and read what capture recorded.

``export_model`` / ``get_aten_graph_module`` turn eager code into a
``GraphModule``; ``create_getattr_from_value`` materialises a tensor into an
existing graph as a buffer; ``get_node_name_to_scope`` /
``print_node_scope_tabular`` read the ``nn_module_stack`` provenance that
export stamps on every node.

``TorchExportableModuleWithStaticCache`` and
``convert_and_export_with_split_cache`` are adapted from the ``transformers``
/ ExecuTorch static-cache recipe.  The local change is the *split* cache: a
``prefill_length`` cache for the prompt alongside a ``max_new_tokens``
residual cache for generated tokens, each exposed as its own buffer, so the
two can carry different quantization.  Re-sync when the upstream
``StaticCache`` API moves.
"""

import logging
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.fx import Graph, GraphModule, Node
from transformers import GenerationConfig, PreTrainedModel
from transformers.cache_utils import StaticCache
from transformers.utils import is_torch_greater_or_equal

logger = logging.getLogger(__name__)

__all__ = [
    "TorchExportableModuleWithStaticCache",
    "convert_and_export_with_split_cache",
    "create_getattr_from_value",
    "export_model",
    "get_aten_graph_module",
    "get_node_name_to_scope",
    "print_node_scope_tabular",
]


def export_model(
    model: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    strict: bool = False,
):
    """Export ``model`` to a training-safe ``GraphModule``.

    Picks the newest export entry point the installed torch offers, and
    suppresses the ``_assert_tensor_metadata`` nodes that each ``.to(dtype)``
    would otherwise pin to the dtype seen at trace time.

    Args:
        model: Module to export.
        args: Positional example inputs.
        kwargs: Keyword example inputs.
        dynamic_shapes: Dynamic-shape spec forwarded to ``torch.export``.
        strict: Whether export runs under torchdynamo.

    Returns:
        The exported program's ``GraphModule``.

    Raises:
        RuntimeError: If the installed torch predates 2.0.
    """
    export_args = (model, args, kwargs)
    export_kwargs = {"dynamic_shapes": dynamic_shapes, "strict": strict}

    if is_torch_greater_or_equal("2.10"):
        from torch._export.utils import (
            _disable_aten_to_metadata_assertions,
        )

        with _disable_aten_to_metadata_assertions():
            gm = torch.export.export(*export_args, **export_kwargs)
        return gm.module(check_guards=False)
    elif is_torch_greater_or_equal("2.8"):
        from torch._export.utils import (
            _disable_aten_to_metadata_assertions,
        )

        with _disable_aten_to_metadata_assertions():
            gm = torch.export.export_for_training(*export_args, **export_kwargs)
        return gm.module()
    elif is_torch_greater_or_equal("2.5"):
        return torch.export.export_for_training(
            *export_args, **export_kwargs
        ).module()
    elif is_torch_greater_or_equal("2.0"):
        return torch._export.capture_pre_autograd_graph(
            model, args, kwargs, dynamic_shapes=dynamic_shapes
        )
    else:
        raise RuntimeError(f"Require torch>=2.0, but found {torch.__version__}")


def get_aten_graph_module(
    pattern: Callable,
    example_inputs: Tuple[Any, ...],
    example_kwargs: Dict[str, Any] = None,
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], None] = None,
    is_cuda: bool = False,
) -> GraphModule:
    """Convert ``pattern`` to an FX graph of decomposed aten ops.

    Args:
        pattern: Callable or module to trace.
        example_inputs: Positional example inputs.
        example_kwargs: Keyword example inputs.
        dynamic_shapes: Dynamic-shape spec forwarded to export.
        is_cuda: Move tensor inputs to CUDA before tracing.

    Returns:
        The traced pattern, dead code eliminated.
    """
    if is_cuda:
        example_inputs = tuple(
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in example_inputs
        )
    aten_pattern = export_model(
        pattern,
        example_inputs,
        example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    aten_pattern.graph.eliminate_dead_code()
    aten_pattern.recompile()
    return aten_pattern


def create_getattr_from_value(
    module: torch.nn.Module, graph: Graph, prefix: str, value: Any
) -> Node:
    """Register ``value`` as a buffer and return a ``get_attr`` node for it.

    Args:
        module: Module the buffer is registered on.
        graph: Graph the node is created in.
        prefix: Base attribute name; dots become underscores and a numeric
            suffix is appended until the name is unused (``s``, ``s_1``, …).
        value: Tensor or scalar to store.

    Returns:
        The ``get_attr`` node referencing the new buffer.
    """
    prefix = prefix.replace(".", "_")
    attr_name, i = prefix, 0
    while hasattr(module, attr_name):
        i += 1
        attr_name = f"{prefix}_{i}"

    new_value = (
        value.clone().detach()
        if isinstance(value, torch.Tensor)
        else torch.tensor(value)
    )
    module.register_buffer(attr_name, new_value)
    return graph.create_node("get_attr", attr_name)


def get_node_name_to_scope(
    model: GraphModule,
) -> Dict[str, Tuple[str, type, int]]:
    """Map each node's name to the module scope export recorded for it.

    Args:
        model: Exported graph module.

    Returns:
        ``node.name`` -> ``(module_path, module_type, call_index)`` taken from
        the innermost frame of the node's ``nn_module_stack``.
    """
    node_name_to_scope: Dict[str, Tuple[str, type]] = {}
    submodule_to_object_type_to_cur_idx: Dict[str, Dict[Callable, int]] = (
        defaultdict(lambda: defaultdict(int))
    )
    for n in model.graph.nodes:
        if (nn_module_stack := n.meta.get("nn_module_stack", None)) is None:
            node_name_to_scope[n.name] = [("", type(None))]
            continue

        current_scope = []
        for bt in nn_module_stack.values():
            module_path = bt[0]
            cur_object_type_idx = submodule_to_object_type_to_cur_idx[
                module_path
            ][n.target]
            submodule_to_object_type_to_cur_idx[module_path][n.target] += 1
            current_scope.append((module_path, bt[1], cur_object_type_idx))
        node_name_to_scope[n.name] = current_scope[-1]

    return node_name_to_scope


def print_node_scope_tabular(gm: GraphModule):
    """Print each node alongside the module scope it was traced from."""
    # Deferred: ``tabulate`` is only needed for this debugging printer.
    try:
        from tabulate import tabulate
    except ImportError:
        print(
            "`print_tabular` relies on the library `tabulate`, which could "
            "not be found on this machine. Run `pip install tabulate` to "
            "install the library."
        )
        raise

    node_name_to_scope = get_node_name_to_scope(gm)
    node_specs = [
        [n.op, n.name, n.target, node_name_to_scope[n.name]]
        for n in gm.graph.nodes
        if n.name in node_name_to_scope
    ]
    print(tabulate(node_specs, headers=["opcode", "name", "target", "scope"]))


# ---------------------------------------------------------------------------
# Vendored from the ``transformers`` / ExecuTorch static-cache export recipe.
#
# Kept in the upstream's formatting on purpose -- ``fmt: off`` shields it from
# black so a diff against upstream stays readable when the recipe is re-synced.
# The local change is the *split* cache: a ``prefill_length`` cache for the
# prompt alongside a ``max_new_tokens`` residual cache, each its own buffer, so
# the two can carry different quantization.
# ---------------------------------------------------------------------------
# fmt: off
def process_logits(scores: torch.Tensor, eos_token_id: torch.Tensor) -> torch.Tensor:
    vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
    eos_token_mask = torch.isin(vocab_tensor, eos_token_id)
    scores_processed = scores.clone()
    scores_processed = torch.where(eos_token_mask, -math.inf, scores)
    return scores_processed


def create_causal_mask_residual(
    target_length: int,
    prefill_length: int,
    max_length: int,
    cache_position: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((1, target_length), fill_value=min_dtype, dtype=dtype)
    position = torch.arange(target_length)
    causal_mask *= (
        ((position > prefill_length) & (position < max_length)) | (position > max_length + cache_position)
    ).reshape(1, -1)
    causal_mask = causal_mask[None, None, :, :].expand(1, 1, -1, -1)
    return causal_mask


class TorchExportableModuleWithStaticCache(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for decoder-only LM to `StaticCache`. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.

    Note:
        This class is specifically designed to support export process using `torch.export`
        in a way that ensures the model can be further lowered and run efficiently in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        prefill_length: int,
        max_new_tokens: int,
        batch_size: Optional[int] = None,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes the wrapper module with the pretrained model.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap. The model must have caching
                enabled and use a 'static' caching implementation.
            batch_size (`Optional[int]`): The batch size of the model. If not provided, we check if a value can be found
                in `generation_config.cache_config` and otherwise we raise a ValueError.
            max_cache_len (`Optional[int]`): The maximum cache length for generation. Same mechanism as `batch_size` if
                not provided.
            device (`Optional[torch.device]`): The device to use. If not provided, we check if a value can be found
                in `generation_config.cache_config` and otherwise we use `model.device` (no error is raised).

        Raises:
            AssertionError: If the pretrained model does not have caching enabled or if it does
            not use a 'static' caching implementation in `model.generation_config`.
            ValueError: If `batch_size` or `max_cache_len` is not provided, either as an argument or in `cache_config`.
        """
        super().__init__()

        config = model.config.get_text_config()
        generation_config = model.generation_config

        # Sanity checks
        if generation_config is None:
            raise AssertionError(
                "The model must have a generation config to be exported with static caching. "
                "Please set `generation_config` in `model`."
            )
        if not generation_config.use_cache:
            raise AssertionError(
                "The model must have caching enabled to be exported with static caching. "
                "Please set `generation_config.use_cache=True`."
            )
        if generation_config.cache_implementation != "static":
            raise AssertionError(
                "The model must use a 'static' caching implementation to be exported with static caching. "
                "Please set `generation_config.cache_implementation='static'`."
            )

        cache_config = {} if generation_config.cache_config is None else generation_config.cache_config

        # Ensure batch_size and max_cache_len are set
        if batch_size is None:
            batch_size = cache_config.get("batch_size", None)
            if batch_size is None:
                raise ValueError("batch_size must be provided, either as an argument or in cache_config.")
        if max_cache_len is None:
            max_cache_len = cache_config.get("max_cache_len", None)
            if max_cache_len is None:
                raise ValueError("max_cache_len must be provided, either as an argument or in cache_config.")
        # Infer device if not provided
        if device is None:
            device = cache_config.get("device", model.device)

        self.max_cache_len = max_cache_len

        self.model = model
        self.static_cache = StaticCache(max_cache_len=prefill_length, config=config)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        dtype = self.model.dtype
        # We need this call to initialize all the layers (otherwise it's done lazily, which is not exportable)
        self.static_cache.early_initialization(batch_size, num_heads, head_dim, dtype, device)
        for i in range(len(self.static_cache)):
            self.register_buffer(f"key_cache_{i}", self.static_cache.layers[i].keys, persistent=False)
            self.register_buffer(f"value_cache_{i}", self.static_cache.layers[i].values, persistent=False)
            self.register_buffer(
                f"cumulative_length_{i}",
                self.static_cache.layers[i].cumulative_length,
                persistent=False,
            )

        self.static_cache_residual = StaticCache(max_cache_len=max_new_tokens, config=config)
        self.static_cache_residual.early_initialization(batch_size, num_heads, head_dim, dtype, device)
        for i in range(len(self.static_cache_residual)):
            self.register_buffer(f"key_cache_residual_{i}", self.static_cache_residual.layers[i].keys, persistent=False)
            self.register_buffer(f"value_cache_residual_{i}", self.static_cache_residual.layers[i].values, persistent=False)
            self.register_buffer(
                f"cumulative_length_residual_{i}",
                self.static_cache_residual.layers[i].cumulative_length,
                persistent=False,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        cache_position_residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the module, which is compatible with the ExecuTorch runtime.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            inputs_embeds (`torch.Tensor`): Tensor representing current input embeddings to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.

        This forward adapter serves two primary purposes:

        1. **Making the Model `torch.export`-Compatible**:
            The adapter hides unsupported objects, such as the `Cache`, from the graph inputs and outputs,
            enabling the model to be exportable using `torch.export` without encountering issues.

        2. **Ensuring Compatibility with `ExecuTorch` runtime**:
            The adapter matches the model's forward signature with that in `executorch/extension/llm/runner`,
            ensuring that the exported model can be executed in `ExecuTorch` out-of-the-box.
        """
        # Start by resetting static cache (it's needed to be able to run several generations with the same exported program,
        # as otherwise it's mutated in-place indefinitely - we cannot call reset in-between the `generate` as the program was
        # already exported)
        for layer in self.static_cache.layers:
            layer.cumulative_length.copy_(cache_position[0])
        for layer in self.static_cache_residual.layers:
            layer.cumulative_length.copy_(cache_position_residual[0])

        past_key_values = self.static_cache

        outs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            past_key_values_residual=self.static_cache_residual,
            cache_position_residual=cache_position_residual,
        )
        if hasattr(outs, "logits"):
            # Returned outputs is `CausalLMOutputWithPast`
            return outs.logits
        else:
            # Returned the `last_hidden_state` from `BaseModelOutputWithPast`
            return outs.last_hidden_state

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    @staticmethod
    def generate(
        model: torch.nn.Module,
        prompt_token_ids: torch.Tensor,
        max_new_tokens: int,
        min_length: int = 0,
        eos_token_id: Union[int, List[int]] = None,
        model_decode: torch.fx.GraphModule = None,
        key_quantizer=None,
        value_quantizer=None,
    ):
        device = model.device
        generation_config = model.generation_config

        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_id = {eos_token_id}
            else:
                eos_token_id = set(eos_token_id)
        elif generation_config.eos_token_id is not None:
            eos_token_id = {generation_config.eos_token_id}
        else:
            eos_token_id = set()

        # Initial forward pass to get logits and prefill KV cache
        with torch.no_grad():
            outputs = model(prompt_token_ids)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        # pre-process distribution
        logits = process_logits(logits, torch.tensor(list(eos_token_id), device=device))
        # print("Prefill logits:", logits)

        current_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        response_tokens = prompt_token_ids[0].tolist() + [current_token.item()]

        seq_length = past_key_values.get_seq_length()
        # print(f"Prompt length: {seq_length}")

        for i, layer in enumerate(past_key_values.layers):
            cache_len = model_decode.get_buffer(f"value_cache_{i}").shape[2]
            assert seq_length <= cache_len, f"seq_length {seq_length} exceeds cache size {cache_len}"

            key_state = layer.keys
            if key_quantizer is not None:
                key_state = key_quantizer(key_state)
            model_decode.get_buffer(f"key_cache_{i}")[:, :, : key_state.shape[2], :] = key_state

            value_state = layer.values
            if value_quantizer is not None:
                value_state = value_quantizer(value_state)
            model_decode.get_buffer(f"value_cache_{i}")[:, :, : value_state.shape[2], :] = value_state

        for step in range(1, max_new_tokens):
            with torch.no_grad():
                cache_len = model_decode.get_buffer("value_cache_0").shape[2]
                residual_len = model_decode.get_buffer("value_cache_residual_0").shape[2]

                # TODO: create causal mask only once and update it incrementally
                causal_mask = create_causal_mask_residual(
                    target_length=cache_len + residual_len,
                    prefill_length=seq_length,
                    max_length=cache_len,
                    cache_position=step - 1,
                    dtype=next(model_decode.parameters()).dtype,
                )

                logits = model_decode(
                    input_ids=current_token.to(device),
                    cache_position=torch.tensor([len(response_tokens) - 1], dtype=torch.long, device=device),
                    cache_position_residual=torch.tensor([step - 1], dtype=torch.long, device=device),
                    attention_mask=causal_mask.to(device),
                )

                # print(f"Step {step} logits:", logits)

            if len(response_tokens) < min_length - 1:
                logits = process_logits(logits, torch.tensor(list(eos_token_id), device=device))

            current_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            response_tokens.append(current_token.item())

            if len(response_tokens) >= min_length and current_token.item() in eos_token_id:
                break

        return torch.tensor([response_tokens], dtype=torch.long, device=device)


def convert_and_export_with_split_cache(
    model: PreTrainedModel,
    max_len: int = 4096,
    max_new_tokens: int = 512,
    example_input_ids: Optional[torch.Tensor] = None,
    example_cache_position: Optional[torch.Tensor] = None,
    example_cache_position_residual: Optional[torch.Tensor] = None,
    example_attention_mask: Optional[torch.Tensor] = None,
    dynamic_shapes: Optional[dict] = None,
    strict: Optional[bool] = None,
):
    """
    Convert a `PreTrainedModel` into an exportable module and export it using `torch.export`,
    ensuring the exported model is compatible with `ExecuTorch`.

    Args:
        model (`PreTrainedModel`): The pretrained model to be exported.
        example_input_ids (`Optional[torch.Tensor]`): Example input token id used by `torch.export`.
        example_cache_position (`Optional[torch.Tensor]`): Example current cache position used by `torch.export`.
        dynamic_shapes(`Optional[dict]`): Dynamic shapes used by `torch.export`.
        strict(`Optional[bool]`): Flag to instruct `torch.export` to use `torchdynamo`.

    Returns:
        Exported program (`torch.export.ExportedProgram`): The exported program generated via `torch.export`.
    """
    if not is_torch_greater_or_equal("2.6", accept_dev=True):
        raise ImportError("torch >= 2.6 is required.")

    max_cache_len = max_len + max_new_tokens

    config_dict = model.generation_config.to_dict()

    config_dict.update({
        "use_cache": True,
        "cache_implementation": "static",
        "cache_config": {
            "batch_size": 1,
            "max_cache_len": max_cache_len,
            "device": str(model.device),
        }
    })

    model.generation_config = GenerationConfig(**config_dict)

    with torch.no_grad():
        # TODO: The default inputs only work for text models. We need to add support for vision/audio models.
        example_input_ids = (
            example_input_ids
            if example_input_ids is not None
            else torch.tensor([[1]], dtype=torch.long, device=model.device)
        )
        example_cache_position = (
            example_cache_position
            if example_cache_position is not None
            else torch.tensor([0], dtype=torch.long, device=model.device)
        )
        example_cache_position_residual = (
            example_cache_position_residual
            if example_cache_position_residual is not None
            else torch.tensor([0], dtype=torch.long, device=model.device)
        )
        example_attention_mask = (
            example_attention_mask
            if example_attention_mask is not None
            else torch.ones((1, max_cache_len), dtype=model.dtype, device=model.device)[None, None, :, :]
        )

        exported_program = torch.export.export(
            TorchExportableModuleWithStaticCache(model, max_len, max_new_tokens),
            args=(),
            kwargs={
                "input_ids": example_input_ids,
                "cache_position": example_cache_position,
                "cache_position_residual": example_cache_position_residual,
                "attention_mask": example_attention_mask,
            },
            dynamic_shapes=dynamic_shapes,
            strict=strict if strict is not None else True,
        )

        return exported_program

# fmt: on
