import logging
import math
from functools import partial
from typing import List, Tuple, Generator, Optional, Union

import torch

from ...node_info import get_arg_value, _pair
from ...node_info import (
    get_node_bytes,
    get_node_to_key_map,
    is_mha_qkv_permute,
    normalize_shape,
)
from ...node_info import (
    is_bmm,
    is_elementwise_op,
    is_fully_connected,
    is_linear,
    is_matmul,
    is_pooling,
)
from .banking import BANK_GROUPS, require_allocation, scratchpad_bytes
from .cost import vector_tile_latency
from ....layout_ops import NHWC_OP_VARIANTS

logger = logging.getLogger(__name__)

__all__ = [
    "run_matrix_op_l2_tiling",
    "run_pool_op_l2_tiling",
    "run_vector_op_l2_tiling",
    "run_vector_op_node_l2_tiling",
]

DEFAULT_CACHE_SIZE = 8 * 1024 * 1024  # 8 MiB


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
    min_sizes: Optional[Union[List[int], Tuple[int, ...]]] = None,
    multiple_of: Optional[Union[List[int], Tuple[int, ...]]] = None,
    order: Optional[Union[List[int], Tuple[int, ...]]] = None,
    fixed_dims: Optional[Union[List[int], Tuple[int, ...]]] = None,
    last_dim: Optional[int] = None,
    reverse: bool = False,
    round_robin: bool = False,
) -> Generator[Tuple[Tuple[int, ...], Tuple[int, ...]], None, None]:
    """
    Yields tile shapes by progressively reducing dimensions in a specified order.

    Args:
        input_shape: The original shape (e.g., (1024, 1024)).
        min_sizes: Minimum size for each dimension. If the list is shorter than
                   input_shape, it is padded with 1s on the left.
        multiple_of: Required multiple for each dimension's tile size. If the list
                     is shorter than input_shape, it is padded with 1s on the left
                     (1 means no constraint). E.g., multiple_of=(1, 16) requires the
                     last dimension's tile to be a multiple of 16.
        order: Explicit order of dimension indices to reduce.
        fixed_dims: Indices of dims that should remain at full size.
        last_dim: Convenience arg to fix dimensions starting from this index.
        reverse: If True, reverses the traversal order (ignored if `order` is provided).
        round_robin: If True, cycles through dimensions reducing them one step at a time.
                     If False, fully reduces one dimension before moving to the next.

    Yields:
        (current_shape, tiling_factors)
    """
    ndim = len(input_shape)

    # --- 1. Normalize Inputs ---

    # helper: resolving negative indices to positive
    def resolve_idx(i):
        return i + ndim if i < 0 else i

    # Set up fixed dimensions set
    fixed_indices = set()
    if fixed_dims:
        fixed_indices.update(resolve_idx(d) for d in fixed_dims)
    if last_dim is not None:
        start = resolve_idx(last_dim)
        fixed_indices.update(range(start, ndim))

    # Set up traversal order
    if order:
        traversal_order = [resolve_idx(i) for i in order]
    else:
        traversal_order = list(range(ndim))
        if reverse:
            traversal_order.reverse()

    # Align min_sizes to input_shape length (pad left with 1s)
    targets = list(min_sizes) if min_sizes else []
    if len(targets) < ndim:
        targets = [1] * (ndim - len(targets)) + targets

    # Align multiple_of to input_shape length (pad left with 1s — 1 means no constraint)
    multiples = list(multiple_of) if multiple_of else []
    if len(multiples) < ndim:
        multiples = [1] * (ndim - len(multiples)) + multiples

    # --- 2. Pre-calculate Valid Factors ---

    # We calculate all valid tiling sizes for every dimension upfront.
    # A factor is valid if it divides the dimension, >= min_size, and is a
    # multiple of the required multiple (if specified).
    # Example: input 128, multiple_of=16 -> [128, 64, 32, 16]
    dim_factors = {}
    for i in range(ndim):
        limit = max(1, targets[i])  # Ensure min_size is at least 1
        size = input_shape[i]
        limit = min(limit, size)  # Cap limit to size

        if i in fixed_indices:
            factors = [size]
        else:
            mult = multiples[i]
            # Generate factors in descending order, respecting divisibility and multiple_of
            factors = [
                f
                for f in range(size, limit - 1, -1)
                if size % f == 0 and f % mult == 0
            ]
            if not factors:
                logger.warning(
                    f"No valid tiling found for dim {i} (size={size}, min={limit}, "
                    f"multiple_of={mult}); keeping full size."
                )
                factors = [size]

        dim_factors[i] = factors

    # --- 3. Traversal Logic ---

    # Current state: indices pointing to the current factor used for each dimension
    # initialized to 0 (which corresponds to the full input size)
    current_factor_indices = {i: 0 for i in range(ndim)}

    def get_current_state():
        """Constructs the shape and tiling tuple based on current indices."""
        shape = tuple(
            dim_factors[i][current_factor_indices[i]] for i in range(ndim)
        )
        tiling = tuple(input_shape[i] // shape[i] for i in range(ndim))
        return shape, tiling

    # Yield the initial full shape
    yield get_current_state()

    if not round_robin:
        # --- Sequential Mode ---
        # Reduce Dim A fully, then move to Dim B, etc.
        for dim_idx in traversal_order:
            if dim_idx in fixed_indices:
                continue

            factors = dim_factors[dim_idx]
            # Iterate through the remaining factors for this dimension
            for i in range(1, len(factors)):
                current_factor_indices[dim_idx] = i
                yield get_current_state()

    else:
        # --- Round Robin Mode ---
        # Reduce Dim A (step 1), yield. Reduce Dim B (step 1), yield. Repeat.
        active_dims = [d for d in traversal_order if d not in fixed_indices]

        while True:
            progress_made = False

            for dim_idx in active_dims:
                current_idx = current_factor_indices[dim_idx]
                max_idx = len(dim_factors[dim_idx]) - 1

                # If this dimension can be reduced further
                if current_idx < max_idx:
                    # Move one step down in size
                    current_factor_indices[dim_idx] += 1
                    progress_made = True
                    yield get_current_state()

            # If we went through all dims and none could change, we are done
            if not progress_made:
                break


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


def _merge_tiling(a, b):
    if b is None:
        return a

    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + a
    b = (1,) * (n - len(b)) + b

    return tuple(ai * bi for ai, bi in zip(a, b))


def _search_tiling(
    node,
    full_shape,
    min_sizes,
    shape_func,
    cache_size,
    bank_width,
    bank_size,
    num_banks=None,
    order=None,
    last_dim=None,
    fixed_dims=None,
    base_tiling=None,
    multiple_of=None,
    extra_size_fn=None,
    cost_fn=None,
):
    """
    Generic driver over the valid tilings, scoring each by the scratchpad it
    needs (``scratchpad_bytes``: one bank per operand group, the two smallest
    merged while they outnumber the banks).

    ``get_valid_tiling`` yields candidates largest -> smallest.  Without
    ``cost_fn`` the first tiling that fits in ``cache_size`` wins (the largest
    fitting tile).  With ``cost_fn`` -- ``cost_fn(node, tile_sizes,
    tiled_shapes, global_tiling) -> latency`` -- every fitting candidate is
    scored and the minimum-latency one is returned (DRAM-aware two-step search).
    """
    key_to_node = {v: k for k, v in get_node_to_key_map(node).items()}

    # Every operand group costs a whole bank, so ``G`` groups floor the
    # footprint at ``G * bank_size``: with as many groups as banks nothing
    # fits at any tile size.  Retry with progressively more sharing and keep
    # the first (least-shared) tiling that maps.
    for extra_sharing in range(len(BANK_GROUPS)):
        best = None  # (score, tile_sizes, node_to_shape)
        for tile_sizes, tiling in get_valid_tiling(
            full_shape,
            min_sizes=min_sizes,
            multiple_of=multiple_of,
            order=order,
            last_dim=last_dim,
            fixed_dims=fixed_dims,
        ):
            global_tiling = _merge_tiling(tiling, base_tiling)

            logger.debug(
                f"Trying tiling {global_tiling} with tile sizes {tile_sizes}"
            )

            tiled_shapes = shape_func(node, tile_sizes, global_tiling)

            total_size = scratchpad_bytes(
                key_to_node,
                node,
                tiled_shapes,
                bank_width,
                bank_size,
                num_banks,
                extra_sharing,
            )

            if extra_size_fn is not None:
                total_size += extra_size_fn(node, tile_sizes, tiled_shapes)

            if total_size > cache_size:
                continue

            node_to_shape = normalize_shape(node, tiled_shapes)

            if cost_fn is None:
                _log_tiling_details(node, node_to_shape, extra_sharing)
                return tile_sizes

            score = cost_fn(node, tile_sizes, tiled_shapes, global_tiling)
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


def search_gemm_tiling(
    node,
    pe_array_size,
    cache_size,
    bank_width,
    bank_size,
    num_banks=None,
    k_multiple=1,
):
    input_shape = node.args[0].shape
    X = input_shape[-2] if is_bmm(node) else math.prod(input_shape[:-1])
    C = input_shape[-1]

    is_mat = is_matmul(node)
    weight_shape = node.args[1].shape
    K = weight_shape[-1] if is_mat else weight_shape[0]

    x_min_size = min(sum(pe_array_size), X)

    num_c_tile = 1
    if bank_size is not None:
        input_bytes = get_node_bytes(node.args[0])
        c_max_size = bank_size / input_bytes / x_min_size
        for (c,), (num_c_tile,) in get_valid_tiling(
            (C,), min_sizes=(pe_array_size[0],)
        ):
            if c <= c_max_size:
                break
        else:
            logger.warning(
                f"Cannot find valid C tiling for {node} that fits bank size {bank_size}."
            )

    full_shape = (X, C // num_c_tile, K)
    min_sizes = (x_min_size, pe_array_size[0], pe_array_size[1])
    order = (2, 0, 1)

    logger.info(f"Running L2 tiling for matrix op: {node}")

    common_args = dict(
        node=node,
        full_shape=full_shape,
        min_sizes=min_sizes,
        multiple_of=(
            pe_array_size[0],
            math.lcm(pe_array_size[1], k_multiple),
        ),
        order=order,
        shape_func=_build_gemm_shape_map,
        cache_size=cache_size,
        bank_width=bank_width,
        bank_size=bank_size,
        num_banks=num_banks,
        base_tiling=(1, num_c_tile, 1),
    )

    def _gemm_residual_size(node, tile_sizes, tiled_shapes):
        """Extra L2 cost of the accumulator buffer when C is tiled."""
        _, c_tiled, _ = tile_sizes
        if c_tiled < C:
            return math.prod(tiled_shapes["output"]) * get_node_bytes(node)
        return 0

    # Tiling for non-first sub-GEMMs (budget for the accumulator)
    tile_sizes = _search_tiling(
        **common_args, extra_size_fn=_gemm_residual_size
    )

    if tile_sizes is None:
        return None

    c_tiled = tile_sizes[1]

    if c_tiled < C and c_tiled != C // num_c_tile:
        # Tiling for the first sub-GEMM (no accumulator buffer)
        search_args = {
            **common_args,
            "full_shape": (full_shape[0], c_tiled, full_shape[2]),
            "base_tiling": (1, C // c_tiled, 1),
            "fixed_dims": (1,),  # Fix C dim to ensure same C tile size
        }
        tile_sizes = _search_tiling(**search_args)

    return tile_sizes


def run_matrix_op_l2_tiling(model, config):
    """
    Annotate the batch-1 fully-connected (matrix-vector) GEMMs with the
    ``l2_tiling`` factors that fit their operands in cache.

    Convs and matrix-matrix GEMMs are left alone: interstellar tiles those on
    demand during bufferization, off the same ``l2_tiling`` key.

    Args:
        model: A model object with a FX Graph containing GEMM nodes.
        config (AcceleratorConfig): The hardware description (PE array size,
            scratchpad size, banking).
    """
    graph = model.graph

    pe_array_size = config.pe_array_size
    cache_size = config.scratchpad_size
    if cache_size is None:
        cache_size = DEFAULT_CACHE_SIZE
    num_banks = config.num_banks
    bank_size = None if num_banks is None else cache_size // num_banks

    for node in list(graph.nodes):
        if not (is_linear(node) or is_matmul(node)):
            continue
        if not is_fully_connected(node):
            continue

        # An N tile that cuts a head in half cannot be stored in the permuted
        # layout the bufferizer folds the relayout into.
        head_dim = mha_projection_head_dim(node)
        tile_sizes = search_gemm_tiling(
            node,
            pe_array_size,
            cache_size,
            None,
            bank_size,
            num_banks,
            k_multiple=head_dim or 1,
        )

        if tile_sizes is None:
            logger.warning(f"Failed to tile GEMM node: {node}")
            continue

        x_tiled, c_tiled, k_tiled = tile_sizes
        in_shape = node.args[0].shape
        X = in_shape[-2] if is_bmm(node) else math.prod(in_shape[:-1])
        C = in_shape[-1]
        w_shape = node.args[1].shape
        K = w_shape[-1] if is_matmul(node) else w_shape[0]
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


def run_vector_op_node_l2_tiling(node, config):
    vector_unit_width = config.vector_lanes
    cache_size = config.scratchpad_size
    if cache_size is None:
        cache_size = DEFAULT_CACHE_SIZE
    num_banks = config.num_banks
    bank_width = config.bank_width

    if not is_elementwise_op(node) and node.target not in [
        torch.ops.aten.softmax.int,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.quantized_ops.layer_norm.default,
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.quantized_ops.quantize_mx_outlier.default,
    ]:
        return

    # Certain dimensions cannot be tiled, e.g., transpose and reduction dims
    last_dim = -1
    min_sizes = (vector_unit_width,)
    multiple_of = None
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
        min_sizes = tuple(
            (
                max(block_size, vector_unit_width)
                if i == ndim - 1
                else block_size if i in axes else 1
            )
            for i in range(ndim)
        )
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

    output_shape = (
        node.value.shape
        if isinstance(node.value, torch.Tensor)
        else node.value[-1].shape
    )

    logger.info(f"Running L2 tiling for vector op: {node}")

    # With DRAM info, rank the fitting tiles by a pipeline latency model
    # instead of taking the largest that fits (see ``tiling_cost``).
    cost_fn = (
        partial(vector_tile_latency, config=config)
        if config.dram_bandwidth is not None
        else None
    )

    tile_sizes = _search_tiling(
        node=node,
        full_shape=output_shape,
        min_sizes=min_sizes,
        multiple_of=multiple_of,
        last_dim=last_dim,
        shape_func=_build_vector_op_shape_map,
        cache_size=cache_size,
        bank_width=bank_width,
        bank_size=None if num_banks is None else cache_size // num_banks,
        num_banks=num_banks,
        cost_fn=cost_fn,
    )

    if tile_sizes is not None:
        node.meta["l2_tiling"] = tuple(
            s // ts for s, ts in zip(output_shape, tile_sizes)
        )


def run_vector_op_l2_tiling(model, config):
    """
    Perform tiling on vector operations to fit intermediate data into cache.

    Args:
        model: A model object with a FX Graph containing vector operation nodes.
        config (AcceleratorConfig): The hardware description.  When it carries a
            DRAM bandwidth, tiles are chosen by the latency model (``tiling_cost``)
            rather than by largest-that-fits.
    """
    graph = model.graph

    for node in list(graph.nodes):
        run_vector_op_node_l2_tiling(node, config)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


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


def run_pool_op_l2_tiling(model, config):
    """
    Perform tiling on pooling operations to fit intermediate data into Scratchpad.

    Dispatches to the appropriate tiling strategy based on whether the op is
    adaptive (tiles N and C) or non-adaptive (tiles N, H, W, and C).

    Args:
        model: A model object with a FX Graph containing pooling nodes.
        config (AcceleratorConfig): The hardware description (vector lane count,
            scratchpad size, banking).
    """
    graph = model.graph

    vector_unit_width = config.vector_lanes
    cache_size = config.scratchpad_size
    if cache_size is None:
        cache_size = DEFAULT_CACHE_SIZE
    num_banks = config.num_banks
    bank_width = config.bank_width
    bank_size = None if num_banks is None else cache_size // num_banks

    for node in list(graph.nodes):
        if not is_pooling(node):
            continue

        # The NHWC twins keep the name of the aten op they replace (see
        # layout_ops) and differ only in namespace, so matching on the op
        # name covers both layouts.
        name = str(node.target)

        if name.endswith("max_pool2d.default"):
            if node.target in NHWC_OP_VARIANTS.values():
                N, H_out, W_out, C = node.shape
            else:
                N, C, H_out, W_out = node.shape
            logger.info(f"Running L2 tiling for non-adaptive pool op: {node}")
            full_shape = (N, H_out, W_out, C)
            tile_sizes = _search_tiling(
                node=node,
                full_shape=full_shape,
                min_sizes=(1, 1, 1, vector_unit_width),
                order=(3, 0, 1, 2),
                shape_func=_build_non_adaptive_pool_shape_map,
                cache_size=cache_size,
                bank_width=bank_width,
                bank_size=bank_size,
                num_banks=num_banks,
            )
            if tile_sizes is not None:
                node.meta["l2_tiling"] = tuple(
                    s // ts for s, ts in zip(full_shape, tile_sizes)
                )

        elif "adaptive" in name:
            N = node.shape[0]
            C = (
                node.shape[-1]
                if node.target in NHWC_OP_VARIANTS.values()
                else node.shape[1]
            )
            logger.info(f"Running L2 tiling for adaptive pool op: {node}")
            tile_sizes = _search_tiling(
                node=node,
                full_shape=(N, C),
                min_sizes=(1, vector_unit_width),
                order=(1, 0),
                shape_func=_build_adaptive_pool_shape_map,
                cache_size=cache_size,
                bank_width=bank_width,
                bank_size=bank_size,
                num_banks=num_banks,
            )
            if tile_sizes is not None:
                node.meta["l2_tiling"] = tuple(
                    s // ts for s, ts in zip((N, C), tile_sizes)
                )

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model
