import logging
import math
from functools import partial
from typing import List, Tuple, Generator, Optional, Union

import torch

from ...node_info import get_arg_value, _pair
from ...node_info import (
    get_node_to_key_map,
    is_mha_qkv_permute,
    normalize_shape,
)
from ...node_info import (
    get_anchor_node,
    is_bmm,
    is_elementwise_op,
    is_fully_connected,
    is_matmul,
    is_pooling,
)
from .banking import BANK_GROUPS, require_allocation, scratchpad_bytes
from .cost import vector_tile_latency
from ....layout_ops import NHWC_OP_VARIANTS

logger = logging.getLogger(__name__)

__all__ = [
    "run_gemv_tiling",
    "pool_op_tiling",
    "vector_op_tiling",
]


def _prime_factors(n: int):
    f, p = [], 2
    while p * p <= n:
        while n % p == 0:
            f.append(p)
            n //= p
        p += 1 if p == 2 else 2  # 2,3,5,7,...
    if n > 1:
        f.append(n)
    return f


def construct_tiled_shape(full_shape, tiled_dim: int, dims):
    """
    Reconstruct full-rank tiled shape.

    Args:
      full_shape: tuple/list[int] original shape (len N)
      tiled_dim: int, flattened size of the compressed (tiled) dims
      dims: iterable[int], indices of dims that were flattened into tiled_dim

    Returns:
      Tuple[int] of length N
    """
    full_shape = tuple(full_shape)
    N = len(full_shape)
    if N == 0:
        raise ValueError("full_shape must have at least one dimension.")

    # Normalize & validate compressed dims
    comp = sorted(set(int(i) for i in dims))
    if not comp:
        raise ValueError("dims cannot be empty.")
    if any(i < 0 or i >= N for i in comp):
        raise IndexError(f"dims must be in [0, {N-1}]. Got {dims}.")

    # Distribute prime factors of R across compressed dims (greedy balance)
    tiled = {i: 1 for i in comp}
    for p in _prime_factors(tiled_dim):
        for i in reversed(comp):
            if full_shape[i] % p == 0:
                tiled[i] *= p
                break

    # Build final shape
    out = [tiled[i] if i in comp else full_shape[i] for i in range(N)]
    return tuple(out)


def get_valid_tiling(
    input_shape: Tuple[int, ...],
    multiple_of: Optional[Tuple[int, ...]] = None,
    order: Optional[Tuple[int, ...]] = None,
    last_dim: Optional[int] = None,
) -> Generator[Tuple[Tuple[int, ...], Tuple[int, ...]], None, None]:
    """Yield tile shapes from the full shape downwards.

    Reduces one dimension at a time to its next valid size, in ``order``,
    leaving the dimensions already reduced at their smallest.  A size is valid
    when it divides the dimension and is a multiple of ``multiple_of``.

    Args:
        input_shape: The shape being tiled.
        multiple_of: Required tile multiple per dim, right-aligned to the shape
            and padded with 1s (1 is free).
        order: Dim indices to reduce, first reduced first; default is left to
            right.
        last_dim: Dims from here onwards stay at full size.

    Yields:
        ``(tile_shape, tiling_factors)``, largest tile first.
    """
    ndim = len(input_shape)

    def resolve(i):
        return i + ndim if i < 0 else i

    fixed = (
        set(range(resolve(last_dim), ndim)) if last_dim is not None else set()
    )
    traversal = [resolve(i) for i in order] if order else list(range(ndim))

    multiples = list(multiple_of) if multiple_of else []
    multiples = [1] * (ndim - len(multiples)) + multiples

    # Valid tile sizes per dim, largest first.  A fixed dim has only one.
    sizes = {}
    for dim, extent in enumerate(input_shape):
        if dim in fixed:
            sizes[dim] = [extent]
            continue
        valid = [
            s
            for s in range(extent, 0, -1)
            if extent % s == 0 and s % multiples[dim] == 0
        ]
        if not valid:
            logger.warning(
                f"No valid tiling for dim {dim} (size={extent}, "
                f"multiple_of={multiples[dim]}); keeping full size."
            )
            valid = [extent]
        sizes[dim] = valid

    tile = [sizes[dim][0] for dim in range(ndim)]

    def current():
        return tuple(tile), tuple(e // t for e, t in zip(input_shape, tile))

    yield current()
    for dim in traversal:
        for size in sizes[dim][1:]:  # a fixed dim has nothing left to give
            tile[dim] = size
            yield current()


def _build_gemm_shape_map(node, tile_sizes, divisor=None):
    bs = node.kwargs.get("block_size", 1)

    x_tiled, c_tiled, k_tiled = tile_sizes
    c_scaled = c_tiled // bs

    input_shape = node.args[0].shape
    tiled_input_shape = construct_tiled_shape(
        input_shape, x_tiled, list(range(len(input_shape) - 1))
    )

    input_dims = tiled_input_shape[:-1]
    batch_dims = tiled_input_shape[:-2]

    is_mat = is_matmul(node)
    weight_transposed = is_mat ^ node.meta.get("transposed", False)

    if weight_transposed:
        weight_shape = (c_tiled, k_tiled)
        weight_scale_shape = (c_scaled, k_tiled)
    else:
        weight_shape = (k_tiled, c_tiled)
        weight_scale_shape = (k_tiled, c_scaled)

    if is_bmm(node):
        weight_shape = batch_dims + weight_shape
        weight_scale_shape = batch_dims + weight_scale_shape

    A_indptr = node.kwargs.get("A_indptr")
    if A_indptr is not None:
        value = A_indptr.value.reshape(-1)
        diffs = value[x_tiled::x_tiled] - value[:-x_tiled:x_tiled]

        # Round up to avoid underestimating nnz per tile
        if divisor is not None:
            ratio = divisor[0] * divisor[1]
        else:
            X, C = math.prod(input_shape[:-1]), input_shape[-1]
            ratio = (X / x_tiled) * (C / c_tiled)
        A_data = node.kwargs.get("A_data")
        nnz = max(int(A_data.value.numel() / ratio), diffs.max())

    return {
        "input": input_dims + (c_tiled,),
        "other" if is_mat else "weight": weight_shape,
        "bias": (k_tiled,),
        "input_scale": input_dims + (c_scaled,),
        "weight_scale": weight_scale_shape,
        "A_data": batch_dims + (nnz,) if A_indptr else None,
        "A_indices": batch_dims + (nnz,) if A_indptr else None,
        "A_indptr": batch_dims + (x_tiled + 1,),
        "output": input_dims + (k_tiled,),
    }


def _log_tiling_details(node, tiled_shapes, extra_sharing=0):
    def fmt(s):
        if s is None:
            return "?"
        return str(tuple(s)).replace(" ", "")

    shared = f" (sharing {extra_sharing})" if extra_sharing else ""
    logger.info(f"Selected tiling for {node}{shared}:")

    for n in node.all_input_nodes:
        if n in tiled_shapes and require_allocation(n):
            orig_shape = fmt(n.shape)
            tile_shape = fmt(tiled_shapes[n])
            logger.info(f"  In[{n}]: {orig_shape} -> {tile_shape}")

    orig_shape = fmt(node.shape)
    tile_shape = fmt(tiled_shapes[node])
    logger.info(f"  Out[{node}]: {orig_shape} -> {tile_shape}")


def _search_tiling(
    node,
    full_shape,
    shape_builder_fn,
    cache_size,
    num_banks,
    bank_width,
    order=None,
    last_dim=None,
    multiple_of=None,
    cost_fn=None,
):
    """
    Generic driver over the valid tilings, scoring each by the scratchpad it
    needs (``scratchpad_bytes``: one bank per operand group, the two smallest
    merged while they outnumber the banks).

    ``get_valid_tiling`` yields candidates largest -> smallest.  Without
    ``cost_fn`` the first tiling that fits in ``cache_size`` wins (the largest
    fitting tile).  With ``cost_fn`` -- ``cost_fn(node, tile_sizes,
    tiled_shapes, tiling) -> latency`` -- every fitting candidate is
    scored and the minimum-latency one is returned (DRAM-aware two-step search).
    """
    key_to_node = {v: k for k, v in get_node_to_key_map(node).items()}
    bank_size = None if num_banks is None else cache_size // num_banks

    # Every operand group costs a whole bank, so ``G`` groups floor the
    # footprint at ``G * bank_size``: with as many groups as banks nothing
    # fits at any tile size.  Retry with progressively more sharing and keep
    # the first (least-shared) tiling that maps.
    for extra_sharing in range(len(BANK_GROUPS)):
        best = None  # (score, tile_sizes, node_to_shape)
        for tile_sizes, tiling in get_valid_tiling(
            full_shape,
            multiple_of=multiple_of,
            order=order,
            last_dim=last_dim,
        ):
            logger.debug(f"Trying tiling {tiling} with tile sizes {tile_sizes}")

            tiled_shapes = shape_builder_fn(node, tile_sizes, tiling)

            total_size = scratchpad_bytes(
                key_to_node,
                node,
                tiled_shapes,
                bank_width,
                bank_size,
                num_banks,
                extra_sharing,
            )

            if total_size > cache_size:
                continue

            node_to_shape = normalize_shape(node, tiled_shapes)

            if cost_fn is None:
                _log_tiling_details(node, node_to_shape, extra_sharing)
                return tile_sizes

            score = cost_fn(node, tile_sizes, tiled_shapes, tiling)
            if best is None or score < best[0]:
                best = (score, tile_sizes, node_to_shape)

        if best is not None:
            _, tile_sizes, node_to_shape = best
            _log_tiling_details(node, node_to_shape, extra_sharing)
            return tile_sizes

    logger.warning(f"Failed to tile {node} with cache size {cache_size}.")
    return None


def mha_projection_head_dim(node) -> Optional[int]:
    """``head_dim`` of the MHA relayout a projection gemm feeds, else ``None``.

    A projection's ``N`` is really ``(heads, head_dim)`` -- the reshape splits
    it and the permute makes the heads outer -- so its ``N`` tile has to hold
    whole heads.  A context matmul keeps its rank across the permute and carries
    no such constraint.  The tail is still flat here (fusion runs later), so
    walk the single-user chain to reach the permute.
    """
    ndim = len(node.shape)
    curr = node
    for _ in range(4):
        users = list(curr.users)
        if len(users) != 1:
            return None
        curr = users[0]
        if is_mha_qkv_permute(curr):
            return curr.shape[-1] if len(curr.shape) > ndim else None
    return None


def run_gemv_tiling(model, config):
    """
    Annotate the batch-1 fully-connected (matrix-vector) GEMMs with the
    ``l2_tiling`` factors that fit their operands in cache.

    Convs and matrix-matrix GEMMs are left alone: interstellar tiles those on
    demand during bufferization, off the same ``l2_tiling`` key.

    Args:
        model: A model object with a FX Graph containing GEMM nodes.
        config (AcceleratorConfig): The hardware description (vector lane count,
            scratchpad size, banking).
    """
    graph = model.graph

    for node in list(graph.nodes):
        if not is_fully_connected(node):
            continue

        input_shape = node.args[0].shape
        X = input_shape[-2] if is_bmm(node) else math.prod(input_shape[:-1])
        C = input_shape[-1]
        weight_shape = node.args[1].shape
        K = weight_shape[-1] if is_matmul(node) else weight_shape[0]

        # An N tile that cuts a head in half cannot be stored in the permuted
        # layout the bufferizer folds the relayout into.
        head_dim = mha_projection_head_dim(node)

        # A C tile short of a whole microscaling block leaves the scale tile
        # with no elements to expand against the input.
        block_size = node.kwargs.get("block_size") or 1

        logger.info(f"Running L2 tiling for GEMV: {node}")
        tile_sizes = _search_tiling(
            node=node,
            full_shape=(X, C, K),
            multiple_of=(
                block_size,
                math.lcm(config.vector_lanes, head_dim or 1),
            ),
            order=(2, 0, 1),
            shape_builder_fn=_build_gemm_shape_map,
            cache_size=config.scratchpad_size,
            num_banks=config.num_banks,
            bank_width=config.bank_width,
        )

        if tile_sizes is None:
            logger.warning(f"Failed to tile GEMV node: {node}")
            continue

        x_tiled, c_tiled, k_tiled = tile_sizes
        node.meta["l2_tiling"] = (X // x_tiled, K // k_tiled, C // c_tiled)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


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


def _build_vector_op_shape_map(node, tile_sizes, divisor):
    node_to_key = get_node_to_key_map(node)
    shapes_map = {}
    for n, k in node_to_key.items():
        if k == "output":
            shapes_map[k] = compute_output_tiled_shapes(node, divisor)
        elif require_allocation(n):
            shapes_map[k] = compute_tiled_shape(tuple(n.shape), divisor)
    return shapes_map


def _vector_op_tiling_limits(node, vector_unit_width):
    """``(last_dim, multiple_of)`` for a vector op's tile search, or ``None``
    when ``node`` is not an op this pass tiles.

    Args:
        node: The vector op whose dimensions are being constrained.
        vector_unit_width: Lanes the last dim has to fill.
    """
    if not is_elementwise_op(node) and node.target not in [
        torch.ops.aten.softmax.int,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.quantized_ops.layer_norm.default,
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.quantized_ops.quantize_mx_outlier.default,
    ]:
        return None

    # Certain dimensions cannot be tiled, e.g., transpose and reduction dims
    last_dim = -1
    multiple_of = (vector_unit_width,)
    if node.target == torch.ops.aten.softmax.int:
        last_dim = get_arg_value(node, 1, "dim", -1)
    elif node.target == torch.ops.aten.layer_norm.default:
        normalized_shape = get_arg_value(node, 1, "normalized_shape", None)
        last_dim = (
            -len(normalized_shape) if normalized_shape is not None else -1
        )
    elif node.target in [
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.quantized_ops.quantize_mx_outlier.default,
    ]:
        axes = get_arg_value(node, 2, "axes", None)
        block_size = get_arg_value(node, 3, "block_size", None)
        ndim = len(node.args[0].shape)

        # A quantization block must not straddle a tile boundary, so a tile on a
        # quantization axis holds a whole number of blocks; the last dim also
        # respects the hardware unroll.
        last_dim = None
        axes = set(a % ndim for a in (axes or ()))
        multiple_of = tuple(
            (
                math.lcm(block_size if i in axes else 1, vector_unit_width)
                if i == ndim - 1
                else block_size if i in axes else 1
            )
            for i in range(ndim)
        )
    elif node.target == torch.ops.aten.transpose.int:
        last_dim = min(*node.args[1:])
    elif node.target == torch.ops.aten.permute.default:
        last_dim = next((i for i, d in enumerate(node.args[1]) if i != d), None)

    return last_dim, multiple_of


def _output_shape(node):
    return (
        node.value.shape
        if isinstance(node.value, torch.Tensor)
        else node.value[-1].shape
    )


def vector_op_tiling(node, config):
    """Per-dim tile counts for a vector op, bare or fused.

    Called from the builders during bufferization, so it sees whatever fusion
    left behind however it ran.  A fused submodule is searched as a whole: its
    inputs key on their node names, so every operand the kernel loads is sized,
    not just the anchor's.  The anchor only supplies the dim constraints.

    Args:
        node: The op to tile -- a vector op, or a fused ``call_module`` around
            one.  Its output shape drives the grid.
        config (AcceleratorConfig): The hardware description.

    Returns:
        Tile counts over ``node``'s output, or ``None`` when the op is not one
        this tiles.

    Raises:
        RuntimeError: when no tiling of the op's operands fits the scratchpad.
    """
    anchor = get_anchor_node(node)
    if anchor is None:
        return None
    limits = _vector_op_tiling_limits(anchor, config.vector_lanes)
    if limits is None:
        return None
    last_dim, multiple_of = limits

    logger.info(f"Running L2 tiling for vector op: {node}")

    # With DRAM info, rank the fitting tiles by a pipeline latency model
    # instead of taking the largest that fits (see ``tiling_cost``).
    cost_fn = (
        partial(vector_tile_latency, config=config)
        if config.dram_bandwidth is not None
        else None
    )

    output_shape = _output_shape(node)
    tile_sizes = _search_tiling(
        node=node,
        full_shape=output_shape,
        multiple_of=multiple_of,
        last_dim=last_dim,
        shape_builder_fn=_build_vector_op_shape_map,
        cache_size=config.scratchpad_size,
        num_banks=config.num_banks,
        bank_width=config.bank_width,
        cost_fn=cost_fn,
    )
    if tile_sizes is None:
        raise RuntimeError(
            f"{node}: no tiling of its operands fits the scratchpad "
            f"(anchor {anchor}, {len(node.all_input_nodes)} inputs)"
        )
    return tuple(s // ts for s, ts in zip(output_shape, tile_sizes))


def _pool_input_extent(tile, stride, dilation, kernel_size):
    """Input extent covered by ``tile`` consecutive pooling outputs."""
    return (tile - 1) * stride + dilation * (kernel_size - 1) + 1


def _build_non_adaptive_pool_shape_map(node, tile_sizes, divisor=None):
    """
    Compute tiled input/output shapes for non-adaptive pooling ops.

    tile_sizes = (tile_N, tile_H, tile_W, tile_C), where H/W refer to the
    output spatial dimensions.  The corresponding input tile is derived from
    stride and dilation (padding does not change the input tile footprint).

    Handles both NHWC (quantized_ops, transposed) and NCHW (aten) layouts.
    The shape tuple ordering mirrors the node's actual tensor layout so that
    banking / scratchpad-size estimates are correct.

    Returns a dict with keys matching normalized op argument names:
        "input"   -> shape of the input tile
        "output" -> shape of the output tile
    """
    tile_N, tile_H, tile_W, tile_C = tile_sizes

    stride = _pair(get_arg_value(node, 2, "stride", 1))
    dilation = _pair(get_arg_value(node, 4, "dilation", 1))
    kernel_size = _pair(get_arg_value(node, 1, "kernel_size"))

    tile_H_in = _pool_input_extent(
        tile_H, stride[0], dilation[0], kernel_size[0]
    )
    tile_W_in = _pool_input_extent(
        tile_W, stride[1], dilation[1], kernel_size[1]
    )

    if node.target in NHWC_OP_VARIANTS.values():  # NHWC: (N, H, W, C)
        return {
            "input": (tile_N, tile_H_in, tile_W_in, tile_C),
            "output": (tile_N, tile_H, tile_W, tile_C),
        }
    else:  # NCHW: (N, C, H, W)
        return {
            "input": (tile_N, tile_C, tile_H_in, tile_W_in),
            "output": (tile_N, tile_C, tile_H, tile_W),
        }


def _build_adaptive_pool_shape_map(node, tile_sizes, divisor=None):
    """
    Compute tiled input/output shapes for adaptive pooling ops.

    tile_sizes = (tile_N, tile_C).  The full spatial extent of the input is
    always needed per tile because the adaptive window spans the whole input.

    Handles both NHWC (quantized_ops) and NCHW (aten) layouts.

    Returns a dict with keys matching normalized op argument names:
        "input"   -> shape of the input tile
        "output" -> shape of the output tile
    """
    tile_N, tile_C = tile_sizes
    if node.target in NHWC_OP_VARIANTS.values():  # NHWC: (N, H, W, C)
        H_in, W_in = node.args[0].shape[1], node.args[0].shape[2]
        H_out, W_out = node.shape[1], node.shape[2]
        return {
            "input": (tile_N, H_in, W_in, tile_C),
            "output": (tile_N, H_out, W_out, tile_C),
        }
    else:  # NCHW: (N, C, H, W)
        H_in, W_in = node.args[0].shape[2], node.args[0].shape[3]
        H_out, W_out = node.shape[2], node.shape[3]
        return {
            "input": (tile_N, tile_C, H_in, W_in),
            "output": (tile_N, tile_C, H_out, W_out),
        }


def pool_op_tiling(node, config):
    """Per-dim tile counts for a pooling op, bare or fused.

    Called from ``build_pool`` during bufferization, so it sees whatever fusion
    left behind.  A fused submodule is searched as a whole: its inputs key on
    their node names, so every operand the kernel loads is sized.

    Args:
        node: The op to tile -- a pooling op, or a fused ``call_module`` around
            one.
        config (AcceleratorConfig): The hardware description.

    Returns:
        Tile counts over ``(N, H_out, W_out, C)`` for a non-adaptive pool or
        ``(N, C)`` for an adaptive one, or ``None`` when the op is not a pool.

    Raises:
        RuntimeError: when no tiling of the op's operands fits the scratchpad.
    """
    anchor = get_anchor_node(node)
    if anchor is None or not is_pooling(anchor):
        return None

    vector_unit_width = config.vector_lanes

    # The NHWC twins keep the name of the aten op they replace (see layout_ops)
    # and differ only in namespace, so matching on the op name covers both.
    name = str(anchor.target)
    nhwc = anchor.target in NHWC_OP_VARIANTS.values()

    if name.endswith("max_pool2d.default"):
        if nhwc:
            N, H_out, W_out, C = anchor.shape
        else:
            N, C, H_out, W_out = anchor.shape
        logger.info(f"Running L2 tiling for non-adaptive pool op: {node}")
        full_shape = (N, H_out, W_out, C)
        multiple_of = (1, 1, 1, vector_unit_width)
        order = (3, 0, 1, 2)
        shape_builder_fn = _build_non_adaptive_pool_shape_map
    elif "adaptive" in name:
        full_shape = (anchor.shape[0], anchor.shape[-1 if nhwc else 1])
        logger.info(f"Running L2 tiling for adaptive pool op: {node}")
        multiple_of = (1, vector_unit_width)
        order = (1, 0)
        shape_builder_fn = _build_adaptive_pool_shape_map
    else:
        return None

    tile_sizes = _search_tiling(
        node=node,
        full_shape=full_shape,
        multiple_of=multiple_of,
        order=order,
        shape_builder_fn=shape_builder_fn,
        cache_size=config.scratchpad_size,
        num_banks=config.num_banks,
        bank_width=config.bank_width,
    )
    if tile_sizes is None:
        raise RuntimeError(
            f"{node}: no tiling of its operands fits the scratchpad "
            f"(anchor {anchor}, {len(node.all_input_nodes)} inputs)"
        )
    return tuple(s // ts for s, ts in zip(full_shape, tile_sizes))
