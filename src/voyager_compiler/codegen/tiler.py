import logging
import math
from typing import Optional

from ..pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)


def get_dtype_width(dtype) -> int:
    """Element width in bits, derived from the canonical ``dtype_byte_size``
    so the dtype-name parsing lives in exactly one place."""
    return round(dtype_byte_size(dtype) * 8)


def _node_dtype_bits(node, default: Optional[int] = None) -> int:
    """
    Element width in bits for an FX node's tensor, read from the graph.

    Prefers node.meta["dtype"] (the compiler's tracked storage dtype, e.g. an
    NF4 weight), falling back to the runtime tensor dtype node.value.dtype.  A
    multi-output node (meta dtype / value is a list, e.g. quantize_mx) uses its
    primary (last) output.
    """
    if node is not None:
        dtype = node.meta.get("dtype")
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[-1]
        if dtype is None:
            val = getattr(node, "value", None)
            if isinstance(val, (list, tuple)):
                val = val[-1] if val else None
            dtype = getattr(val, "dtype", None)
        if dtype is not None:
            return round(dtype_byte_size(dtype) * 8)
    if default is None:
        raise ValueError(f"node {node} has no dtype to size the operand")
    return default


class RuntimeCalculator:
    def __init__(
        self,
        input_dtype_width: int,
        output_dtype_width: int,
        double_buffered_accum_buffer: bool,
        has_high_precision_vector_input: bool = False,
        has_sparse_op: bool = False,
    ):
        self.input_dtype_width = input_dtype_width
        self.output_dtype_width = output_dtype_width
        self.double_buffered_accum_buffer = double_buffered_accum_buffer
        self.has_sparse_op = has_sparse_op
        self.has_high_precision_vector_input = has_high_precision_vector_input
        if self.has_sparse_op:
            print(
                "Using sparse runtime calculator, "
                "which doubles the weight loading time"
            )

    def calculate_runtime(self, architecture, layer, mapping):
        import interstellar

        le = interstellar.le

        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        partitionings = mapping.loop_partitionings

        # IC is unrolled vertically; replication not handled
        sa_weight_loading_time = partitionings[le.IC][0] + 2

        # index of first loop that isn't OX or OY
        first_non_ox_oy_index = 6
        for i in range(le.NUM):
            if i == le.OX or i == le.OY:
                continue
            if orders[i][1] < first_non_ox_oy_index:
                first_non_ox_oy_index = orders[i][1]

        # weight reuse tile: all loops before first non-(OX/OY) loop
        weight_reuse_tile_size = 1
        for i in range(le.NUM):
            if orders[i][1] < first_non_ox_oy_index:
                weight_reuse_tile_size *= blockings[i][1]

        weight_reuse_tile_time = max(
            sa_weight_loading_time, weight_reuse_tile_size
        )

        num_remaining_l1_tiles = 1
        for i in range(le.NUM):
            if orders[i][1] >= first_non_ox_oy_index:
                num_remaining_l1_tiles *= blockings[i][1]
        # include the reduction loop at the L2 level
        num_remaining_l1_tiles *= blockings[le.IC][2]

        computation_l1_time = weight_reuse_tile_time * num_remaining_l1_tiles

        input_buffer_loading_size = 1
        for loop in [le.IC, le.OY, le.OX]:
            input_buffer_loading_size *= blockings[loop][1]
        input_buffer_loading_time = input_buffer_loading_size

        weight_buffer_loading_size = 1
        for loop in [le.IC, le.OC, le.FY, le.FX]:
            weight_buffer_loading_size *= blockings[loop][1]
        # Include the unrolled reduction loop
        weight_buffer_loading_size *= partitionings[le.IC][0]
        # Assume that each value in the weight buffer is loaded in one cycle
        weight_buffer_loading_time = weight_buffer_loading_size
        # If there is a fused SpMM operation, weight loading time is longer
        if self.has_sparse_op:
            weight_buffer_loading_time *= 2

        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        vector_unit_time = output_size

        requires_high_precision = (
            self.output_dtype_width > self.input_dtype_width
            or self.has_high_precision_vector_input
        )

        if requires_high_precision:
            vector_unit_time *= 2

        using_double_buffer_accum_buffer = (
            self.double_buffered_accum_buffer and requires_high_precision
        )

        if not using_double_buffer_accum_buffer:
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
            )
        else:
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
                vector_unit_time,
            )

        l2_blocks = 1
        for i in range(le.NUM):
            if i == le.IC:
                continue
            l2_blocks *= blockings[i][2]

        if self.double_buffered_accum_buffer:
            total_time = (
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                + vector_unit_time
            )
        else:
            if requires_high_precision:
                extra_vector_unit_time = output_size
            else:
                extra_vector_unit_time = 0
            total_time = (
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                + extra_vector_unit_time
            )

        return total_time


def build_architecture_and_schedule(
    ic_dim: int,
    oc_dim: int,
    l2_cache_size: int,
    input_buffer_size: int,
    weight_buffer_size: int,
    accum_buffer_size: int,
):
    import interstellar

    # L0/L1 are slot arrays: one fixed-width slot per element (the max dtype in a
    # mixed-precision design; narrower dtypes are padded into a full slot), so
    # their capacities are element / slot counts and the fit check is
    # dtype-independent.  L2 is a flat byte pool where sub-byte operands pack.
    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                input_buffer_size * ic_dim,
                accum_buffer_size * oc_dim,
                weight_buffer_size * oc_dim,
            ],
            [l2_cache_size],
        ],
        buf_access_cost_list=[[1, 1, 1], [10, 10, 10], [100]],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0]],
        para_count_list=[ic_dim * oc_dim, 1, 1],
        mac_capacity=0,
        partition_mode=[0, 0, 0],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0]],
        invalid_underutilized=False,
    )

    schedule_constraint = {
        "schedule_hint": {
            "IC": {
                "level0": {
                    "order": 1,
                    "partitioning_size": ic_dim,
                },
                "level1": {"order": -1},
                "level2": {"order": 0},
            },
            "OC": {
                "level0": {
                    "order": 0,
                    "partitioning_size": oc_dim,
                }
            },
            "FX": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
            },
            "FY": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
            },
        }
    }
    schedule_data = interstellar.extract_input.extract_schedule_info(
        schedule_constraint, 3
    )
    schedule = interstellar.Schedule(
        schedule_data["schedule_hint"],
        schedule_data["partition_loops"],
    )
    return architecture, schedule


def _extract_layer_dims(node, key_to_shape, output_shape):
    """
    Build an interstellar.Layer from tiled shapes.
    Returns None to signal the layer should be skipped.
    key_to_shape: dict mapping kwarg key names to tiled shapes.
    output_shape: tiled shape of the output tensor.
    """
    import interstellar
    from .passes.tiling import _conv2d_layout
    from .passes.utils import get_arg_value, _pair
    from .mapping_utils import is_conv2d, is_matmul, is_depthwise_conv

    if is_depthwise_conv(node):
        return None

    input_shape = key_to_shape.get("input")
    if input_shape is None:
        return None

    # Skip fully-connected layers (no spatial tiling needed)
    if math.prod(input_shape[:-1]) == 1:
        return None

    if is_conv2d(node):
        weight_shape = key_to_shape.get("weight")
        if weight_shape is None or len(weight_shape) != 4:
            return None

        transposed = node.meta.get("transposed", False)
        (
            kH,
            kW,
            input_channels,
            output_channels,
        ) = _conv2d_layout(weight_shape, True, not transposed)
        _, height, width, _ = _conv2d_layout(
            output_shape, False, not transposed
        )

        # Skip 3-channel first layer (torchvision convention)
        if input_channels == 3:
            return None

        stride_val = _pair(get_arg_value(node, 3, "stride", 1))
        stride_h, stride_w = stride_val
    else:
        # linear or matmul
        weight_shape = key_to_shape.get("weight") or key_to_shape.get("other")
        if weight_shape is None or len(weight_shape) != 2:
            return None

        # mirrors _build_gemm_shape_map in tiling.py
        weight_transposed = is_matmul(node) ^ node.meta.get("transposed", False)
        if weight_transposed:
            input_channels, output_channels = weight_shape  # (IC, OC)
        else:
            output_channels, input_channels = weight_shape  # transposed

        kH, kW = 1, 1
        height = 1
        width = math.prod(output_shape[:-1])
        stride_h, stride_w = 1, 1

    input_node = node.args[0] if len(node.args) > 0 else None
    weight_node = node.args[1] if len(node.args) > 1 else None
    return interstellar.Layer(
        nifm=input_channels,
        nofm=output_channels,
        wofm=width,
        hofm=height,
        wfil=kW,
        hfil=kH,
        wstd=stride_w,
        hstd=stride_h,
    )


def run_interstellar_for_tiled_op(
    output_node,
    gemm_node,
    tiled_shapes,
    architecture,
    schedule,
    double_buffered_accum_buffer,
    tiling_cache,
    named_modules,
):
    import interstellar

    from .mapping import get_node_to_key_map

    node_to_key = get_node_to_key_map(gemm_node)
    key_to_shape = {
        v: tiled_shapes[k] for k, v in node_to_key.items() if k in tiled_shapes
    }

    output_shape = tiled_shapes.get(output_node)
    if output_shape is None:
        return
    # multi-output (e.g. GEMM + sparse outputs): use the activation shape
    if isinstance(output_shape, tuple) and isinstance(output_shape[0], tuple):
        output_shape = output_shape[-1]

    layer = _extract_layer_dims(gemm_node, key_to_shape, output_shape)
    if layer is None:
        logger.debug(
            f"Skipping interstellar tiling for {output_node.name} "
            f"(FC, 3-channel, or depthwise)"
        )
        return

    logger.info(
        f"[interstellar] {output_node.name}: "
        f"IC={layer.nifm} OC={layer.nofm} "
        f"H={layer.hofm} W={layer.wofm} "
        f"kH={layer.hfil} kW={layer.wfil} "
        f"stride=({layer.hstd},{layer.wstd})"
    )

    # dtype widths from the outer graph node (not submodule placeholders);
    # ``_node_dtype_bits`` reads meta['dtype'] / value.dtype (multi-output ->
    # primary/last).
    input_width = _node_dtype_bits(output_node.args[0])
    output_width = _node_dtype_bits(output_node)

    # Check for high-precision operands in fused post-GEMM vector ops
    has_high_prec_vector_input = False
    if output_node.op == "call_module":
        gm = named_modules[output_node.target]
        for n in gm.graph.nodes:
            if n.op != "placeholder" or gemm_node in n.users:
                continue

            n = n.meta.get("source_node", n)
            if _node_dtype_bits(n) > input_width:
                has_high_prec_vector_input = True
                break

    has_sparse_op = gemm_node.kwargs.get("A_indptr") is not None

    cache_key = (
        layer.nifm,
        layer.nofm,
        layer.wofm,
        layer.hofm,
        layer.wfil,
        layer.hfil,
        layer.wstd,
        layer.hstd,
        input_width,
        output_width,
        has_high_prec_vector_input,
        has_sparse_op,
    )

    if cache_key in tiling_cache:
        logger.info(f"[interstellar] {output_node.name}: reusing cached tiling")
        mapping, access_list = tiling_cache[cache_key]
    else:
        logger.info(f"[interstellar] {output_node.name}: running optimizer")
        rc = RuntimeCalculator(
            input_width,
            output_width,
            double_buffered_accum_buffer,
            has_high_prec_vector_input,
            has_sparse_op,
        )
        _, runtime, mapping, _ = interstellar.optimizer.opt_optimizer(
            architecture,
            layer,
            schedule,
            rc.calculate_runtime,
            True,
        )
        _, _, access_list = interstellar.cost_model.get_cost(
            architecture, mapping, layer
        )
        tiling_cache[cache_key] = (mapping, access_list)
        logger.info(f"[interstellar] {output_node.name}: runtime={runtime}")
        logger.info(interstellar.utils.format_tiling(mapping))

    output_node.meta["interstellar_tiling"] = (mapping, access_list)
    output_node.meta["interstellar_architecture"] = architecture
