import logging
import math
from functools import partial
from typing import Generator, Optional, Tuple

import torch

from voyager_compiler.codegen.node_info import (
    _align_size,
    _pair,
    compute_output_tiled_shapes,
    compute_tiled_shape,
    dtype_byte_size,
    get_anchor_node,
    get_arg_value,
    is_bmm,
    is_elementwise_op,
    is_fully_connected,
    is_matmul,
    is_pooling,
    normalize_shape,
    quant_param_arg_nodes,
    reduction_scratch,
    require_allocation,
    trailing_mha_perm,
    weight_is_ck,
)
from voyager_compiler.codegen.transform.tiling.cost import (
    gemv_tile_latency,
    operand_roles,
    vector_tile_latency,
)
from voyager_compiler.ops.layout import NHWC_OP_VARIANTS

logger = logging.getLogger(__name__)

__all__ = [
    "gemv_op_tiling",
    "pool_op_tiling",
    "vector_op_tiling",
]

# How much longer than the best modeled runtime a tiling may take and still be
# chosen; the least-traffic one among those wins.  0.0 = only the fastest.
# Shared with the interstellar tiler, which selects the same way.
DEFAULT_RUNTIME_TOLERANCE = 0.01


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


# One bank per operand *group*, grouped by the read bandwidth each needs.  A
# GEMV streams its weight -- a fresh element per MAC, nothing reused -- so it
# takes a bank of its own.  The input and its scales are reused across
# ``vector_lanes`` outputs, so one fetch feeds a whole lane group and they need
# a fraction of the weight's bandwidth: nothing like a GEMM, where the input
# streams too.  The weight scales do matter, one per block, but the output is
# only written on the step that finishes a tile, so the two rarely contend.
GEMV_BANK_GROUPS = (
    ("input", "input_scale"),
    ("weight", "other"),
    ("output", "weight_scale", "bias"),
    ("A_data", "A_indices", "A_indptr"),
)


def _tensor_bytes(node, shape, bank_width=None, vector_lanes=None):
    """Bytes one tile of ``node`` occupies, aligned to ``bank_width``.

    Args:
        node: The operand the tile belongs to.
        shape: Its tiled shape -- one per output for a multi-output op.
        bank_width: Per-tensor alignment, bytes.
        vector_lanes: Sparse-output alignment, elements.
    """
    val = node.value
    if isinstance(val, torch.Tensor):
        dtype = node.meta.get("dtype") or val.dtype
        return _align_size(
            math.prod(shape) * dtype_byte_size(dtype), bank_width
        )

    if isinstance(val, (tuple, list)):
        numel = [math.prod(s) for s in shape]

        # Sparse outputs need to be aligned with fetch width
        if vector_lanes is not None:
            numel = [_align_size(s, vector_lanes) for s in numel]

        dtypes = node.meta.get("dtype") or [None for _ in val]
        sizes = [
            _align_size(n * dtype_byte_size(dt or t.dtype), bank_width)
            for t, n, dt in zip(val, numel, dtypes)
        ]

        return sum(sizes)

    logger.warning(f"Node {node} has a non-tensor output")
    return None


def scratchpad_bytes(
    node,
    tiled_shapes,
    bank_width,
    bank_size,
    num_banks,
    extra_sharing=0,
    vector_lanes=None,
):
    """Scratchpad bytes one candidate tile occupies.

    Every group takes a whole bank, since a bank cannot be split, and the two
    smallest merge while the groups outnumber the banks.
    ``GEMV_BANK_GROUPS`` names one op's roles, so an operand it misses -- a
    ``where`` mask, what a fused tail brings of its own -- takes a bank of its
    own.  A batched reduction also holds intermediates the graph never names
    (``reduction_scratch``), which the footprint has to carry too.

    Args:
        node: The op being tiled; its own entry is the output.
        tiled_shapes: Operand FX node -> tiled shape.
        bank_width: Per-tensor alignment, bytes.
        bank_size: Bytes per bank; ``None`` is unbanked, so just sum.
        num_banks: Banks available to merge down to.
        extra_sharing: Merge this many groups further.  ``G`` groups floor the
            footprint at ``G * bank_size``, so at ``G == num_banks`` no tile
            fits however small and the caller has to raise it.
        vector_lanes: Sparse-output alignment, elements.

    Returns:
        Bytes the tile needs with every group rounded to whole banks.
    """
    # ``groups`` is the bank layout, one entry per bank: the groups that match,
    # then any operand they do not name on its own.  A ``where`` lays out as
    # ("input",) | ("other", ...) | ("output", ...) | ("condition",).
    roles = operand_roles(node)
    groups = [
        [n for n in tiled_shapes if roles.get(n) in group]
        for group in GEMV_BANK_GROUPS
    ]
    grouped = {n for group in groups for n in group}
    groups += [[n] for n in tiled_shapes if n not in grouped]

    sizes = []
    for group in groups:
        total = 0
        for n in group:
            shape = tiled_shapes[n]
            if shape is None or not require_allocation(n):
                continue
            total += _tensor_bytes(n, shape, bank_width, vector_lanes)
        if total:
            sizes.append(total)

    # No DMA moves the reserved regions, so they share one bank rather than
    # taking one apiece.
    reserved = sum(
        _align_size(math.prod(shape) * dtype_byte_size(dtype), bank_width)
        for shape, dtype in reduction_scratch(node, tiled_shapes[node])
    )
    if reserved:
        sizes.append(reserved)

    if not sizes:
        return 0
    if not bank_size:
        return sum(sizes)

    target = len(sizes)
    if num_banks:
        target = num_banks
    target = max(1, target - extra_sharing)
    while len(sizes) > target:
        sizes.sort()
        sizes = [sizes[0] + sizes[1]] + sizes[2:]

    return sum(math.ceil(s / bank_size) * bank_size for s in sizes)


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
    tolerance=0.0,
):
    """
    Generic driver over the valid tilings, scoring each by the scratchpad it
    needs (``scratchpad_bytes``: one bank per operand group, the two smallest
    merged while they outnumber the banks).

    ``get_valid_tiling`` yields candidates largest -> smallest.  Without
    ``cost_fn`` the first tiling that fits in ``cache_size`` wins (the largest
    fitting tile).  With ``cost_fn`` -- ``cost_fn(node, tile_sizes,
    tiled_shapes, tiling) -> (latency, dram_bytes)`` -- every fitting candidate
    is scored and the one moving the fewest bytes wins among those within
    ``(1 + tolerance)`` of the best latency.  Latency alone is not enough: a
    compute-bound op is equally fast however its operands are diced, so the
    residue the model leaves behind would buy any amount of traffic for a
    rounding error.  This is how the interstellar tiler picks a mapping too
    (``mapping_point_generator``, with energy in place of bytes).
    """
    bank_size = None if num_banks is None else cache_size // num_banks

    # Every operand group costs a whole bank, so ``G`` groups floor the
    # footprint at ``G * bank_size``: with as many groups as banks nothing
    # fits at any tile size.  Retry with progressively more sharing and keep
    # the first (least-shared) tiling that maps.
    for extra_sharing in range(len(GEMV_BANK_GROUPS)):
        scored = []  # (latency, dram_bytes, tile_sizes)
        for tile_sizes, tiling in get_valid_tiling(
            full_shape,
            multiple_of=multiple_of,
            order=order,
            last_dim=last_dim,
        ):
            tiled_shapes = shape_builder_fn(node, tile_sizes, tiling)

            total_size = scratchpad_bytes(
                node,
                tiled_shapes,
                bank_width,
                bank_size,
                num_banks,
                extra_sharing,
            )

            if total_size > cache_size:
                continue

            if cost_fn is None:
                return tile_sizes

            latency, traffic = cost_fn(node, tile_sizes, tiled_shapes, tiling)
            scored.append((latency, traffic, tile_sizes))

        if scored:
            budget = min(s[0] for s in scored) * (1.0 + tolerance)
            return min(
                (s for s in scored if s[0] <= budget),
                key=lambda s: (s[1], s[0]),
            )[2]

    logger.debug(f"Failed to tile {node} with cache size {cache_size}.")
    return None


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

    if weight_is_ck(node):
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


def _operand_placeholders(root):
    """External placeholder operands feeding ``root``'s subtree (``root``
    inclusive), tracing through pre-processing ops (``dequantize`` / reshape)
    and skipping each op's quantization codebook / qmap args.
    """
    leaves, stack, visited = [], [root], set()
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        if n.op == "placeholder":
            leaves.append(n)
            continue
        codebooks = quant_param_arg_nodes(n)
        for inp in n.all_input_nodes:
            if inp not in codebooks:
                stack.append(inp)
    return leaves


def _build_gemv_shape_map(node, tile_sizes, tiling):
    """``_build_gemm_shape_map`` for the whole kernel a GEMV builds, keyed by
    the FX node each tile belongs to.

    The anchor's own operands are sized by the role they play there, resolved
    to the nodes the kernel really loads.  A fused kernel loads more than them:
    the tail streams a residual or a mask of its own, which has no role and is
    diced by the output block, the way the builder dices it.  The output is
    whatever fusion left of it -- a ``quantize_mx`` tail makes it a pair, an
    MHA relayout re-cuts ``N`` into heads -- diced by the same blocks.
    """
    anchor = get_anchor_node(node)
    tiles = _build_gemm_shape_map(anchor, tile_sizes, tiling)
    out_tile = tiles.pop("output")
    shapes = normalize_shape(anchor, tiles)
    divisor = tuple(max(1, s // t) for s, t in zip(anchor.shape, out_tile))

    if node is not anchor:
        # An operand reaching the anchor through a GQA expand or a dequantize
        # takes its role on that node, not on the placeholder the kernel loads,
        # so trace the anchor's own back and skip them by position.
        own = set(_operand_placeholders(anchor))
        placeholders = [
            p
            for p in node.meta["submodule"].graph.nodes
            if p.op == "placeholder"
        ]
        for n, p in zip(node.all_input_nodes, placeholders):
            if p in own or n in shapes or not require_allocation(n):
                continue
            shapes[n] = compute_tiled_shape(tuple(n.shape), divisor)

    if tuple(_output_shape(node)) == tuple(anchor.shape):
        shapes[node] = compute_output_tiled_shapes(node, divisor)
    else:
        # An MHA relayout re-cut ``N`` into ``(heads, head_dim)``, so
        # ``divisor`` -- indexed against the anchor's dims -- no longer lines
        # up.  The block count survives, and these shapes are only ever read
        # for their product.
        blocks = math.prod(divisor)
        values = node.value
        values = values if isinstance(values, (list, tuple)) else [values]
        out = [((math.prod(v.shape) + blocks - 1) // blocks,) for v in values]
        shapes[node] = out if len(out) > 1 else out[0]

    return shapes


def gemv_op_tiling(node, config):
    """Per-dim tile counts for a matrix-vector GEMM, bare or fused.

    Called from ``get_tiling`` during bufferization, so it sees the layout the
    transform settled on and the operands fusion left on the kernel.
    Interstellar maps a systolic array and skips a batch-1 GEMM -- that one runs
    on the vector unit -- so this is the whole tile search for one.

    Args:
        node: The op to tile -- a fully-connected GEMM, or a fused
            ``call_module`` around one.
        config (AcceleratorConfig): The hardware description.

    Returns:
        ``(n_m, n_n, n_k)`` -- the M, output and reduction tile counts -- or
        ``None`` when the op is not a GEMV.

    Raises:
        RuntimeError: when no tiling of its operands fits the scratchpad.
    """
    anchor = get_anchor_node(node)
    if anchor is None or not is_fully_connected(anchor):
        return None

    input_shape = anchor.args[0].shape
    X = input_shape[-2] if is_bmm(anchor) else math.prod(input_shape[:-1])
    C = input_shape[-1]
    weight_shape = anchor.args[1].shape
    K = weight_shape[-1] if weight_is_ck(anchor) else weight_shape[-2]

    # An N tile that cuts a head in half cannot be stored in the permuted
    # layout the bufferizer folds the relayout into.  Fusion has already run,
    # so the permute is the tail of the submodule rather than a user.
    head_dim = None
    if (submod := node.meta.get("submodule")) is not None:
        perm = trailing_mha_perm(
            [n for n in submod.graph.nodes if n.op == "call_function"]
        )
        if perm is not None and perm.value.ndim > anchor.value.ndim:
            head_dim = perm.value.shape[-1]

    # A C tile short of a whole microscaling block leaves the scale tile
    # with no elements to expand against the input.
    block_size = anchor.kwargs.get("block_size") or 1

    logger.info(f"Running L2 tiling for GEMV: {node}")

    search = partial(
        _search_tiling,
        node=node,
        full_shape=(X, C, K),
        multiple_of=(block_size, math.lcm(config.vector_lanes, head_dim or 1)),
        shape_builder_fn=_build_gemv_shape_map,
        cache_size=config.scratchpad_size,
        num_banks=config.num_banks,
        bank_width=config.bank_width,
        cost_fn=(
            partial(gemv_tile_latency, config=config)
            if config.dram_bandwidth is not None
            else None
        ),
        tolerance=DEFAULT_RUNTIME_TOLERANCE,
    )

    # C whole leaves the input vector loop-invariant, so its guarded load
    # fetches it once however many output tiles there are: tile K alone if any
    # such tile fits.  Failing that, split C as far as it takes to buy the
    # largest K tile, since the input is then re-read once per output tile.
    # ``get_valid_tiling`` reduces in ``order``, leaving what it already
    # reduced at its smallest, so C whole is only ever reachable from the first
    # pass -- the cost function ranks within one, it cannot cross them.
    tile_sizes = search(order=(2,)) or search(order=(1, 2))
    if tile_sizes is None:
        raise RuntimeError(
            f"{node}: no tiling of its operands fits the scratchpad "
            f"(anchor {anchor}, {len(node.all_input_nodes)} inputs)"
        )

    x_tiled, c_tiled, k_tiled = tile_sizes
    return X // x_tiled, K // k_tiled, C // c_tiled


def _build_vector_op_shape_map(node, tile_sizes, divisor):
    shapes_map = {
        n: compute_tiled_shape(tuple(n.shape), divisor)
        for n in node.all_input_nodes
        if require_allocation(n)
    }
    shapes_map[node] = compute_output_tiled_shapes(node, divisor)
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


def _pool_shapes(node, anchor, in_tile, out_tile):
    """The pool's halo and output tiles keyed by the FX node each belongs to,
    plus whatever a fused tail loads of its own -- diced by the output block,
    the way the builder dices it.  ``anchor`` resolves the halo to the node the
    kernel really loads, which a fused ``call_module`` cannot name itself."""
    shapes = normalize_shape(anchor, {"input": in_tile})
    divisor = tuple(max(1, s // t) for s, t in zip(anchor.shape, out_tile))
    if node is not anchor:
        for n in node.all_input_nodes:
            if n not in shapes and require_allocation(n):
                shapes[n] = compute_tiled_shape(tuple(n.shape), divisor)
    shapes[node] = compute_output_tiled_shapes(node, divisor)
    return shapes


def _build_non_adaptive_pool_shape_map(node, tile_sizes, divisor=None):
    """
    Compute tiled input/output shapes for non-adaptive pooling ops.

    tile_sizes = (tile_N, tile_H, tile_W, tile_C), where H/W refer to the
    output spatial dimensions.  The corresponding input tile is derived from
    stride and dilation (padding does not change the input tile footprint).

    Handles both NHWC (quantized_ops, transposed) and NCHW (aten) layouts.
    The shape tuple ordering mirrors the node's actual tensor layout so that
    banking / scratchpad-size estimates are correct.

    Returns the input and output tiles keyed by the FX node each belongs to.
    The geometry comes off the anchor, so a fused ``call_module`` reads its
    pool's stride and kernel rather than its own operand list.
    """
    anchor = get_anchor_node(node)
    tile_N, tile_H, tile_W, tile_C = tile_sizes

    stride = _pair(get_arg_value(anchor, 2, "stride", 1))
    dilation = _pair(get_arg_value(anchor, 4, "dilation", 1))
    kernel_size = _pair(get_arg_value(anchor, 1, "kernel_size"))

    tile_H_in = _pool_input_extent(
        tile_H, stride[0], dilation[0], kernel_size[0]
    )
    tile_W_in = _pool_input_extent(
        tile_W, stride[1], dilation[1], kernel_size[1]
    )

    if anchor.target in NHWC_OP_VARIANTS.values():  # NHWC: (N, H, W, C)
        in_tile = (tile_N, tile_H_in, tile_W_in, tile_C)
        out_tile = (tile_N, tile_H, tile_W, tile_C)
    else:  # NCHW: (N, C, H, W)
        in_tile = (tile_N, tile_C, tile_H_in, tile_W_in)
        out_tile = (tile_N, tile_C, tile_H, tile_W)

    return _pool_shapes(node, anchor, in_tile, out_tile)


def _build_adaptive_pool_shape_map(node, tile_sizes, divisor=None):
    """
    Compute tiled input/output shapes for adaptive pooling ops.

    tile_sizes = (tile_N, tile_C).  The full spatial extent of the input is
    always needed per tile because the adaptive window spans the whole input.

    Handles both NHWC (quantized_ops) and NCHW (aten) layouts.

    Returns the input and output tiles keyed by the FX node each belongs to.
    The spatial extents come off the anchor, so a fused ``call_module`` reads
    its pool's window rather than whatever fusion left on its own output.
    """
    anchor = get_anchor_node(node)
    tile_N, tile_C = tile_sizes
    if anchor.target in NHWC_OP_VARIANTS.values():  # NHWC: (N, H, W, C)
        H_in, W_in = anchor.args[0].shape[1], anchor.args[0].shape[2]
        H_out, W_out = anchor.shape[1], anchor.shape[2]
        in_tile = (tile_N, H_in, W_in, tile_C)
        out_tile = (tile_N, H_out, W_out, tile_C)
    else:  # NCHW: (N, C, H, W)
        H_in, W_in = anchor.args[0].shape[2], anchor.args[0].shape[3]
        H_out, W_out = anchor.shape[2], anchor.shape[3]
        in_tile = (tile_N, tile_C, H_in, W_in)
        out_tile = (tile_N, tile_C, H_out, W_out)

    return _pool_shapes(node, anchor, in_tile, out_tile)


def pool_op_tiling(node, config):
    """Per-dim tile counts for a pooling op, bare or fused.

    Called from ``build_pool`` during bufferization, so it sees whatever fusion
    left behind.  A fused submodule is searched as a whole: the pool's halo is
    resolved through the anchor and the tail's own operands are diced by the
    output block, so every operand the kernel loads is sized.

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
