import collections.abc
import logging
import math
import re
from itertools import repeat
from typing import Any, Optional

import torch
from torch.fx import GraphModule, Node
from torch.fx.operator_schemas import normalize_function

# Re-exported: both predicates are generated from the Core ATen IR by
# ``tools/gen_aten_classifier.py`` and are imported from here across codegen.
from voyager_compiler.codegen.aten_classifier import (  # noqa: F401
    is_compute_op,
    is_elementwise_op,
)

logger = logging.getLogger(__name__)

aten = torch.ops.aten

# Sentinel ``aten.slice`` uses for ``end`` to mean "to the end of the dim".
INT64_MAX = torch.iinfo(torch.int64).max

# Arg names of the quantization lookup tables the quantized_ops
# quantize / dequantize / quantize_mx family takes.
QMAP_PARAMS = {"qmap", "scale_qmap", "input_qmap", "output_qmap"}
CODEBOOK_PARAMS = {"code", "input_code", "weight_code", "output_code"}
QUANT_PARAMS = QMAP_PARAMS | CODEBOOK_PARAMS

# Positional arg holding the axes a quantize blocks along, per op.
AXES_ARG_INDEX_MAP = {
    torch.ops.quantized_ops.dequantize.default: 3,
    torch.ops.quantized_ops.quantize.default: 3,
    torch.ops.quantized_ops.quantize_mx.default: 2,
}


# --------------------------------------------------------------------------
# Graph walking
# --------------------------------------------------------------------------


def get_anchor_node(node):
    """The compute anchor of ``node``: for a fused ``call_module`` (its submodule
    in ``node.meta['submodule']``, set at fusion), the GEMM/conv/pool/pointwise
    op inside the submodule (``None`` if it has none); for any other node,
    ``node`` itself.
    """
    if node.op != "call_module":
        return node
    submod = node.meta.get("submodule")
    if not isinstance(submod, GraphModule):
        return None
    anchor_node = None
    for n in submod.graph.nodes:
        if is_gemm_op(n):
            return n
        if n.op == "call_function" and (
            anchor_node is None
            or anchor_node.target == torch.ops.quantized_ops.dequantize.default
        ):
            anchor_node = n
    return anchor_node


def bound_operands(node, submod) -> dict:
    """Each ``submod`` placeholder -> the operand ``node`` binds to it.

    A call site binds operands to its region's placeholders positionally --
    the invariant emit resolves every reference by -- so the parent operand
    a placeholder stands for is derived from the call, never cached on the
    placeholder's meta (a cached pointer goes stale the moment a rewire
    replaces the operand).  ``submod`` is the graph object the caller walks
    (``node.meta['submodule']`` for a fused call), so the map's keys are the
    very placeholder objects the caller holds; ``None`` -> an empty map, the
    identity for a bare op whose operands are already top-level.
    """
    if not isinstance(submod, GraphModule):
        return {}
    placeholders = [n for n in submod.graph.nodes if n.op == "placeholder"]
    return dict(zip(placeholders, node.all_input_nodes))


def ancestors(node: Node) -> set:
    """Every transitive input node of ``node`` (its operand prelude): for a
    matmul ``Q @ Kᵀ`` this is the operand placeholders *and* the ``transpose``
    that builds ``Kᵀ``, so a pre-anchor relayout is not mistaken for a fused
    post-op."""
    if node is None:
        return set()
    result = set()
    stack = list(node.all_input_nodes)
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(current.all_input_nodes)
    return result


def quant_param_arg_nodes(node: Node, params: set = QUANT_PARAMS) -> set:
    """Tensor args of ``node`` that are quantization lookup tables, identified
    by schema arg name so positions need not be hardcoded.  ``params`` selects
    the family — ``QMAP_PARAMS``, ``CODEBOOK_PARAMS``, or both (the default)."""
    result = set()
    if getattr(node.target, "namespace", None) != "quantized_ops":
        return result
    for i, arg in enumerate(node.target._schema.arguments):
        if arg.name not in params:
            continue
        val = get_arg_value(node, i, arg.name)
        if isinstance(val, Node):
            result.add(val)
    return result


def require_allocation(node: torch.fx.Node) -> bool:
    """Whether ``node`` needs storage allocated for it.

    Three kinds do not, because they are programmed into the instruction that
    reads them rather than fetched from an address: a quantization lookup
    table, a non-tensor value, and a single-element parameter.
    """
    for user in node.users:
        gm = user.meta.get("submodule")
        bound = bound_operands(user, gm)
        for op in gm.graph.nodes if gm is not None else [user]:
            params = quant_param_arg_nodes(op)
            if any(bound.get(p, p) is node for p in params):
                return False

    if (val := getattr(node, "value", None)) is None:
        return True

    if not isinstance(val, torch.Tensor):
        return False

    if node.op in ["placeholder", "get_attr"] and val.numel() == 1:
        return False

    return True


# --------------------------------------------------------------------------
# Op-kind predicates
# --------------------------------------------------------------------------


def is_gemm_op(node: Node) -> bool:
    return is_conv2d(node) or is_linear(node) or is_matmul(node)


def is_conv2d(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.conv2d.default,
        torch.ops.quantized_ops.conv2d.default,
        torch.ops.quantized_ops.conv2d_mx.default,
    ]


def is_depthwise_conv(node: Node) -> bool:
    return is_conv2d(node) and get_arg_value(node, 6, "groups", 1) != 1


def is_linear(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.linear.default,
        torch.ops.quantized_ops.linear.default,
        torch.ops.quantized_ops.linear_mx.default,
    ]


def is_matmul(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.matmul.default,
        torch.ops.quantized_ops.matmul.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ]


def is_bmm(node: Node) -> bool:
    if is_matmul(node):
        input_shape = node.args[0].shape
        other_shape = node.args[1].shape
        return len(input_shape) > 2 or len(other_shape) > 2
    return False


def is_fully_connected(node: Node) -> bool:
    if is_linear(node):
        input_shape = node.args[0].shape
        return all(s == 1 for s in input_shape[:-1])

    if is_matmul(node):
        input_shape = node.args[0].shape
        if is_bmm(node):
            return input_shape[-2] == 1
        else:
            return all(s == 1 for s in input_shape[:-1])

    return False


def is_pooling(node: Node) -> bool:
    return node.target in [
        # Core Aten IR
        aten._adaptive_avg_pool2d,
        aten._adaptive_avg_pool3d,
        aten.adaptive_avg_pool1d,
        aten.avg_pool1d,
        aten.avg_pool2d,
        aten.avg_pool3d,
        aten.max_pool2d_with_indices,
        aten.max_pool3d_with_indices,
        # export_for_training IR
        aten.adaptive_avg_pool2d.default,
        aten.avg_pool2d.default,
        aten.max_pool2d.default,
        # NHWC variants (after the data-layout transform)
        torch.ops.quantized_ops.max_pool2d.default,
        torch.ops.quantized_ops.avg_pool2d.default,
        torch.ops.quantized_ops.adaptive_avg_pool2d.default,
        torch.ops.quantized_ops._adaptive_avg_pool2d.default,
    ]


def is_reshape_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
    ]


def is_prunable_op(node: Node) -> bool:
    """Operations that can be safely deleted from fx.Graph."""
    if node.target == torch.ops.aten.alias.default:
        return True

    # A slice from 0 to the end of the input tensor
    if node.target == torch.ops.aten.slice.Tensor:
        dim = get_arg_value(node, 1, "dim", 0)
        start = get_arg_value(node, 2, "start")
        end = get_arg_value(node, 3, "end")
        step = get_arg_value(node, 4, "step", 1)
        if start is not None and start != 0 or step != 1:
            return False
        if end is not None and hasattr(node.args[0], "shape"):
            return end >= node.args[0].shape[dim]
        return (start is None and end is None) or end == INT64_MAX

    if node.target == torch.ops.aten.expand.default:
        return all(x == 1 or x == -1 for x in node.args[1])

    # Dropout is the identity when the probability is zero or at inference.
    if node.target == torch.ops.aten.dropout.default:
        return (
            get_arg_value(node, 1, "p") == 0.0
            or not get_arg_value(node, 2, "train")
        )

    # A same-dtype ``to.dtype`` is a pure pass-through.
    if node.target == torch.ops.aten.to.dtype:
        dtype = get_arg_value(node, 1, "dtype")
        inp = node.args[0]
        val = getattr(inp, "value", inp.meta.get("val"))
        return isinstance(val, torch.Tensor) and dtype == val.dtype

    return False


def is_nop(node: Node) -> bool:
    """
    The following operations do not require any computation nor handling
    on the memory placement side. Generate a NOP instruction for these ops
    to keep the compute graph intact.
    """
    if is_prunable_op(node):
        return True

    # A select operation that selects the entire tensor
    if node.target == torch.ops.aten.select.int:
        shape = getattr(node.args[0], "shape", None)
        return shape is not None and shape[node.args[1]] == 1

    return node.target in [
        torch.ops.aten.as_strided.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
    ]


def is_aliasing_op(node: Node) -> bool:
    """``node`` is a second name for its input's bytes under a different shape
    — a ``view`` / ``reshape`` / ``squeeze`` / ``unsqueeze`` / size-1 ``select``
    that regroups or drops dims, computing nothing.  Such a node sitting
    *between* two fused compute stages breaks the single-iteration-space
    assumption and is relocated to the fused module's boundary by the
    iteration-space normalizer (see ``normalize.py``).  A nop that also
    preserves the shape (same in/out) can stay inside the fused chain.
    """
    if not is_nop(node):
        return False
    inp = node.args[0]
    return (
        hasattr(node, "shape")
        and hasattr(inp, "shape")
        and tuple(node.shape) != tuple(inp.shape)
    )


# --------------------------------------------------------------------------
# Operand transforms
# --------------------------------------------------------------------------


_BROADCAST_OPS = (
    torch.ops.aten.expand.default,
    torch.ops.aten.repeat.default,
)


def _tensor_value(node):
    """``node``'s traced tensor, or ``None`` — a multi-output op holds a tuple,
    which none of the addressing below can read."""
    if not isinstance(node, Node):
        return None
    value = getattr(node, "value", None)
    return value if isinstance(value, torch.Tensor) else None


def is_relayout_op(node: Node) -> bool:
    """A node that only re-addresses a single tensor: it shares elements or
    moves them, but computes none, so an index map replays through it exactly.
    """
    return (
        node.op == "call_function"
        and (
            is_nop(node) or is_reshape_op(node) or node.target in _BROADCAST_OPS
        )
        and len(node.all_input_nodes) == 1
    )


def _repeat_through(source: Node, ops, out: torch.Tensor):
    """Do ``ops``, applied to ``source``, amount to nothing more than repeating
    one of its dims?  If so, by how much: ``(1, 4, 1)`` says dim 1 was repeated
    4x and nothing else changed.  ``None`` if they do anything else.

    Proved by running the ops rather than by recognizing them, so it does not
    matter which ops they are.  Number the elements of ``source`` 0, 1, 2, ...
    and push *that* through ``ops``; the result says, for every position of
    ``out``, which element of ``source`` it reads.  Compare it against
    ``repeat_interleave``, which is what a pure repeat reads.  Equal everywhere
    => the ops only share elements along ``dim``, and the consumer can read
    ``index // factor`` of ``source`` instead of a materialized copy.

    Example: ``source`` ``[[0, 1, 2], [3, 4, 5]]`` through GQA's
    ``unsqueeze -> expand -> reshape`` gives ``[[0, 1, 2], [0, 1, 2],
    [3, 4, 5], [3, 4, 5]]`` — element for element what a repeat of 2 on dim 0
    gives, so: ``(2, 1)``.

    The shape checks up front are what make ``dim`` and ``factor`` well defined
    (with two dims growing there is no single repeat to compare against), and
    they reject most candidates before building a tensor.
    """
    value = _tensor_value(source)
    if value is None:
        return None
    shape = tuple(value.shape)
    out_shape = tuple(out.shape)
    if len(shape) != len(out_shape):
        return None

    grown = [d for d, (a, b) in enumerate(zip(shape, out_shape)) if a != b]
    if len(grown) != 1:
        return None
    dim = grown[0]
    factor, remainder = divmod(out_shape[dim], shape[dim])
    if remainder or factor < 2:
        return None

    seed = torch.arange(math.prod(shape)).reshape(shape)
    index = seed
    for n in ops:
        index = n.target(
            *(index if a is n.args[0] else a for a in n.args), **n.kwargs
        )
    if not torch.equal(index, seed.repeat_interleave(factor, dim=dim)):
        return None

    repeat = [1] * len(shape)
    repeat[dim] = factor
    return tuple(repeat)


def repeat_of(node: Node):
    """Is ``node`` a smaller tensor with one dim repeated?  Returns that tensor,
    the ops that repeat it, and by how much — ``(source, ops, repeat)`` — or
    ``None``.

    Grouped-query attention is the case that matters.  8 KV heads reach the
    attention matmul as 32, spelled ``unsqueeze -> expand -> reshape``, and
    those ops copy every head 4 times into DRAM.  Recognized, they need not run:
    the GEMM reads head ``h // 4`` of the 8-head tensor (``_InputSpec.repeat``)
    and the copies never happen.

    Walks up the relayout ops above ``node`` and asks ``_repeat_through`` where
    the repeat starts, trying the *deepest* candidate first.  Deeper is better —
    more ops fold away — but the chain does not always reach all the way down:
    in prefill a head transpose sits under the broadcast, and a transpose is not
    a repeat.  So it settles on the deepest source the ops above it are still a
    pure repeat of.  Those fold into the tile address; whatever is below keeps
    its own buffer.
    """
    out = _tensor_value(node)
    if out is None:
        return None

    chain = []
    src = node
    while is_relayout_op(src) and len(src.users) == 1:
        chain.append(src)
        src = src.args[0]
    if not chain:
        return None
    chain.reverse()

    for i, source in enumerate([src, *chain[:-1]]):
        ops = chain[i:]
        repeat = _repeat_through(source, ops, out)
        if repeat is not None:
            return source, ops, repeat
    return None


def swaps_last_two_dims(node: Node) -> bool:
    """Whether ``node`` transposes its input's last two dims — the ``Kᵀ`` an
    attention ``Q @ Kᵀ`` leaves on its weight."""
    if node.op != "call_function":
        return False
    src = node.args[0]
    if (value := _tensor_value(src)) is None:
        return False
    ndim = value.ndim
    if node.target is torch.ops.aten.transpose.int:
        return {a % ndim for a in node.args[1:3]} == {ndim - 2, ndim - 1}
    if node.target is torch.ops.aten.permute.default:
        swapped = list(range(ndim))
        swapped[-2], swapped[-1] = swapped[-1], swapped[-2]
        return list(node.args[1]) == swapped
    return False


def is_mha_qkv_permute(node):
    """
    Check if the node is a permutation used in multi-head attention (MHA)
    operations. It has characteristics that last dimension is a power of 2 and
    the permuted dimensions are the middle two dimensions (2 and 3) of a 4D
    tensor.
    """
    # Don't support head dimension not being a power of 2
    if (
        not hasattr(node, "shape")
        or len(node.shape) != 4
        or not math.log2(node.shape[-1]).is_integer()
    ):
        return False

    if node.target == torch.ops.aten.permute.default:
        dims = node.args[1]
        return len(dims) == 4 and dims == [0, 2, 1, 3]

    if node.target == torch.ops.aten.transpose.int:
        dims = {x if x >= 0 else x + 4 for x in node.args[1:]}
        return node.value.ndim == 4 and dims == {1, 2}

    return False


def trailing_mha_perm(fused_ops):
    """The MHA qkv permute at the end of ``fused_ops`` (peeled of a trailing
    microscaling ``quantize_mx``), or ``None`` if the tail is not such a
    relayout."""
    if not fused_ops:
        return None
    perm = fused_ops[-1]
    if perm.target is torch.ops.quantized_ops.quantize_mx.default:
        perm = perm.args[0]
    return perm if is_mha_qkv_permute(perm) else None


def weight_is_ck(node: Node) -> bool:
    """Whether ``node``'s right operand is physically stored ``[contraction,
    out]``.

    ``is_matmul`` gives the op's native layout (matmul CK, linear KC) and
    ``meta['transposed']`` says the layout transform flipped it, so the two
    compose by XOR.  The twins the flip retargets to keep answering
    ``is_linear`` / ``is_matmul``, which makes the meta the only signal.
    """
    return is_matmul(node) != bool(node.meta.get("transposed", False))


def weight_transforms(node: Node):
    """The transforms fused onto a GEMM weight, and the operand beneath them.

    An attention ``Q @ Kᵀ`` fuses ``K.transpose(-2, -1)`` onto the weight, and
    GQA fuses the repeat that turns 8 KV heads into 32.  Neither is emitted:
    ``transposed`` folds into the DMA (the fetch swaps its last two dims and
    ``async_copy`` ``.mT``s the tile into the bank), ``repeat`` into the block
    index (``grid_index // repeat[d]``, so four query heads share one KV tile).

    A ``dequantize`` -- a KIVI KV cache, packed in DRAM -- does *not* fold into
    the addressing, because it computes: it comes back for the builder to run on
    the fetched tile, which is what lets the cache stay packed all the way into
    the bank.

    Only transforms over an external operand — a placeholder, i.e. something
    the fused submodule is handed rather than computes — count; anything else
    comes back unchanged.

    Returns:
        ``(node, transposed, repeat, dequant)``: the operand the transforms
        read, followed by the transforms themselves.
    """
    inner = node
    dequant = None
    if inner.target is torch.ops.quantized_ops.dequantize.default:
        dequant = inner
        inner = inner.args[0]

    # A ``Kᵀ`` sits under the decode, never over it (``_insert_transpose_op``
    # hoists it there).  The decode is none the wiser: the tile the fetch
    # transposes into the bank is the one it was written against.
    transposed = swaps_last_two_dims(inner)
    if transposed:
        inner = inner.args[0]

    if inner.op == "placeholder":
        return inner, transposed, None, dequant

    found = repeat_of(inner)
    if found is not None and found[0].op == "placeholder":
        source, _, repeat = found
        return source, transposed, repeat, dequant

    return node, False, None, None


# --------------------------------------------------------------------------
# Argument and dtype access
# --------------------------------------------------------------------------


def get_arg_value(
    node: torch.fx.Node,
    arg_number: int,
    kwarg_name: Optional[str] = None,
    default=None,
) -> Any:
    return (
        node.args[arg_number]
        if len(node.args) > arg_number
        else node.kwargs.get(kwarg_name, default)  # type: ignore[arg-type]
    )


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")


def dtype_byte_size(dtype):
    """Bytes occupied by one element of ``dtype`` (fractional for sub-byte).

    Reads the width out of the dtype's *name*, so it serves both a
    ``torch.dtype`` and this project's spec-string dtypes (``int4``,
    ``fp8_e4m3``, ``posit8_1``).

    Args:
        dtype: A ``torch.dtype`` or a dtype name.

    Returns:
        Element size in bytes; below one for sub-byte formats.

    Raises:
        ValueError: If no bit width can be read from the name.
    """
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)(_.*)?$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_width = int(bit_search.groups()[0])
    return bit_width / 8.0


# --------------------------------------------------------------------------
# Shape and tiling geometry
# --------------------------------------------------------------------------


def normalize_shape(node, shape, bound=None):
    node_to_key = get_node_to_key_map(node, bound)
    shape = {n: shape[k] for n, k in node_to_key.items() if k in shape}
    return shape


def get_node_to_key_map(node, bound=None):
    """Each operand FX node -> the role it plays for ``node``.

    An anchor inside a fused submodule names its operands by the submodule's
    placeholders; ``bound`` (``bound_operands`` of the outer call) maps those
    back to the outer nodes the shape maps are keyed by.  A bare op's
    operands are already top-level, so no map is needed.
    """
    bound = bound or {}
    args_and_kwargs = normalize_function(
        node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    )
    node_to_key = {
        bound.get(n, n): k
        for k, n in args_and_kwargs.kwargs.items()
        if isinstance(n, Node)
    }
    node_to_key[node] = "output"
    return node_to_key


def reshape_preserves_full_blocks(
    input_shape: tuple[int, ...],
    input_axis: int,
    output_shape: tuple[int, ...],
    output_axis: int,
    group_size: int,
) -> bool:
    if math.prod(input_shape) != math.prod(output_shape):
        return False

    if group_size == 1:
        return True

    input_minor = math.prod(input_shape[input_axis + 1 :])
    output_minor = math.prod(output_shape[output_axis + 1 :])

    return (
        input_minor == output_minor
        and input_shape[input_axis] % group_size == 0
        and output_shape[output_axis] % group_size == 0
    )


def compute_tiled_shape(shape, divisor):
    ndim = len(shape)
    m = len(divisor)

    # Align divisor to shape dimensions
    if m < ndim:
        divisor = (1,) * (ndim - m) + divisor
    elif m > ndim:
        divisor = divisor[-ndim:]

    return tuple(s // d if s > 1 else s for s, d in zip(shape, divisor))


def compute_output_tiled_shapes(node, tiling, override_shapes=None):
    """
    Computes tiled shape for an output node

    Args:
        node: The output node containing value and shape.
        tiling: The tiling divisor/size configuration.
        override_shapes: Optional shapes to use instead of node's value shapes.
    """
    if isinstance(node.value, torch.Tensor):
        return compute_tiled_shape(override_shapes or node.shape, tiling)
    elif isinstance(node.value, (tuple, list)):
        shapes = []
        has_sparse_outputs = len(node.value) > 2

        for i, tensor in enumerate(node.value):
            old_shape = override_shapes[i] if override_shapes else tensor.shape
            if has_sparse_outputs and i < 3:
                if i == 2:
                    old_shape = old_shape[:-1] + (old_shape[-1] - 1,)
                output_shape = old_shape + (1,)
                s = compute_tiled_shape(output_shape, tiling)[-2]
                if i == 2:
                    s = s + 1
                shapes.append(old_shape[:-1] + (s,))
            else:
                shapes.append(compute_tiled_shape(old_shape, tiling))
        return tuple(shapes)

    return None


def tensor_alloc_bytes(numel, dtype, bank_width, vector_lanes=None):
    """Bytes one on-chip tensor of ``numel`` ``dtype`` elements occupies.

    The store path writes whole beats of ``vector_lanes`` values as whole
    ``bank_width``-byte words, and masks neither end: a tensor is charged the
    beats it takes to cover ``numel``, not the bytes its elements occupy, and
    a beat whose payload is not a whole number of words (a sub-byte dtype)
    also writes past its own end.  Both overshoots are reserved here --
    otherwise they land on the next buffer -- and the total is aligned to
    ``bank_width``.

    Charging beats rather than elements only matters when ``numel`` is not a
    whole number of them: a tile always is, but a CSR row-pointer or nonzero
    capacity need not be, and one of those is charged a whole beat for its
    handful of entries.  The tile search and the memory planner both size
    through here, so a tile the search accepts is a tile the planner can
    place.

    Args:
        numel: Element count of the tensor (or of one bank of a banked one).
        dtype: Its logical dtype (``meta['dtype']`` when quantized).
        bank_width: Allocation alignment and store-word width, bytes;
            ``None`` disables the alignment and the tail slack.
        vector_lanes: Store-beat width, elements; ``None`` charges the
            payload alone (a DRAM tensor, written byte-exact by the DMA).
    """

    def _align_size(size):
        if bank_width is None:
            return size
        return (size + bank_width - 1) // bank_width * bank_width

    size = math.ceil(numel * dtype_byte_size(dtype))
    if vector_lanes is not None and bank_width is not None:
        beat = math.ceil(vector_lanes * dtype_byte_size(dtype))
        beats = math.ceil(numel / vector_lanes)
        size = (beats - 1) * beat + _align_size(beat)
    return int(_align_size(size))


# The batched reductions whose backend kernel keeps intermediates on chip:
# the ``quantized_ops`` twin that names them, the regions reduced to one
# element per row, and the ones as big as the tile itself.  A softmax holds a
# max and a sum, a layer_norm a mean and a variance plus the normalized tile.
_REDUCTION_SCRATCH = {
    aten.softmax.int: (
        torch.ops.quantized_ops.softmax.default,
        ("max", "sum"),
        (),
    ),
    aten.layer_norm.default: (
        torch.ops.quantized_ops.layer_norm.default,
        ("mean", "variance"),
        ("normalized",),
    ),
    torch.ops.quantized_ops.layer_norm.default: (
        torch.ops.quantized_ops.layer_norm.default,
        ("mean", "variance"),
        ("normalized",),
    ),
}


def reduction_op(node):
    """The ``quantized_ops`` op that names ``node``'s scratch, or ``None`` if
    ``node`` is not a batched reduction that keeps any."""
    entry = _REDUCTION_SCRATCH.get(node.target)
    return entry[0] if entry else None


def _reduced_dims(node, ndim: int) -> set:
    """The dims ``node`` reduces over, as non-negative indices."""
    if node.target is aten.softmax.int:
        return {get_arg_value(node, 1, "dim", -1) % ndim}
    normalized_shape = get_arg_value(node, 1, "normalized_shape") or ()
    return set(range(ndim - len(normalized_shape), ndim))


def relayout_view_shape(relayout, shape):
    """The shape a relayout chain gives an input of ``shape``, or ``None``
    when the chain moves elements rather than only renaming dims.

    Proved by replaying the chain on an index tensor rather than by
    recognizing ops, so a transpose of a size-1 dim reads as the identity it
    is.

    Args:
        relayout: The chain, walked input-ward -- the op nearest the value
            first, as ``stream_breaking_quantize`` returns it.
        shape: The shape the chain's input tensor has.

    Returns:
        The chain's output shape when it only renames dims, else ``None``.
    """
    index = torch.arange(math.prod(shape)).reshape(shape)
    for n in reversed(relayout):
        index = n.target(
            *(index if a is n.args[0] else a for a in n.args), **n.kwargs
        )
    if not torch.equal(index.reshape(-1), torch.arange(index.numel())):
        return None
    return tuple(index.shape)


def stream_breaking_quantize(gm):
    """The terminal ``quantize_mx`` of a fused tail that cannot ride the
    compute pass, or ``None``.

    A non-last scale axis always breaks the ride: the scale unit groups
    along the stream, so the tile must be materialized first.  The last
    axis may be spelled ``-1`` or ``ndim - 1`` — the data-layout pass
    writes the projected physical axis as a positive index.  A relayout
    that moves elements breaks too: the store folds a permute into the
    value's addressing but writes the scales in stream order, so the two
    would disagree.  One that only renames dims rides.  Shared by the
    bufferize splitter (which cuts the tail here) and the tile search's
    footprint model (which must charge the staged tile this forces).

    Args:
        gm: The fused tail or submodule ending in the quantize; a
            non-``GraphModule`` never breaks.

    Returns:
        ``None`` when the tail has no such quantize, else ``(quant,
        relayout, spine)``: the quantize node, the relayout chain feeding
        it (walked input-ward), and the node that chain reads.
    """
    if not isinstance(gm, GraphModule):
        return None
    ops = [n for n in gm.graph.nodes if n.op == "call_function"]
    quant = ops[-1] if ops else None
    if (
        quant is None
        or quant.target is not torch.ops.quantized_ops.quantize_mx.default
    ):
        return None

    relayout = []
    spine = quant.args[0]
    while (
        spine.op == "call_function"
        and (is_reshape_op(spine) or is_nop(spine))
        and len(spine.users) == 1
    ):
        relayout.append(spine)
        spine = spine.args[0]
    permuted = False
    if any(is_reshape_op(n) for n in relayout):
        # An unknown shape counts as reordering: the caller then stages the
        # tile, which is always safe.
        value = _tensor_value(spine)
        permuted = (
            value is None
            or relayout_view_shape(relayout, tuple(value.shape)) is None
        )

    val = getattr(quant.args[0], "value", None)
    if val is None:
        val = quant.args[0].meta.get("val")
    ndim = getattr(val, "ndim", 0)
    if not permuted and all(
        a == -1 or (ndim and a == ndim - 1)
        for a in get_arg_value(quant, 2, "axes")
    ):
        return None
    return quant, relayout, spine


def reduction_scratch(node, out_tile, vector_lanes):
    """The on-chip regions the backend keeps beside a batched reduction's own
    tile: a softmax's max and sum, a layer_norm's mean and variance plus its
    normalized output.

    No FX node names them -- the graph goes straight from input tile to output
    tile -- so they are reserved rather than computed.  ``node`` may be a bare
    op or a fused ``call_module``, whose whole submodule is scanned: a group's
    reduction is not always its anchor.  Fusion normalizes a group onto its
    anchor's shape, and a reduction *is* a strong anchor, so one tile of the
    output is the tile the reduction reduces.

    Args:
        node: The op being bufferized or tiled.
        out_tile: One tile of ``node``'s output, or one per output -- a
            multi-output op names its iteration space with the last (the
            shape-preserving one; a microscaling scale or a CSR index is
            derived from it).
        vector_lanes: Lanes the vector unit reduces into.  A statistic is one
            value per row, but the unit addresses a whole lane group at a
            time, so it is held duplicated across the lanes and the reduced
            dims collapse to ``vector_lanes`` rather than to 1.

    Returns:
        ``[(name, shape, dtype), ...]``, one per region, named as the op's
        keyword takes it: the tile with its reduced dims collapsed to a lane
        group for a statistic, the whole tile for layer_norm's normalized
        output, each in the *physical* dtype of the op's input tensor (never
        the logical dtype in ``meta``).
    """
    submod = node.meta.get("submodule") if node.op == "call_module" else None
    ops = submod.graph.nodes if submod is not None else [node]
    if not out_tile:  # no tile to reduce: a scalar, or a non-tensor result
        return []
    if isinstance(out_tile[0], (tuple, list)):
        out_tile = out_tile[-1]
    tile = tuple(out_tile)

    scratch = []
    for op in ops:
        entry = _REDUCTION_SCRATCH.get(op.target)
        if entry is None:
            continue
        _, rows, tiles = entry
        reduced = _reduced_dims(op, len(tile))
        innermost = max(reduced)
        row = tuple(
            (vector_lanes if i == innermost else 1) if i in reduced else s
            for i, s in enumerate(tile)
        )
        dtype = get_arg_value(op, 0, "input").value.dtype
        scratch += [(name, row, dtype) for name in rows]
        scratch += [(name, tile, dtype) for name in tiles]
    return scratch
