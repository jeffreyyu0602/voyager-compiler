import math
import re


def get_dtype_width(dtype) -> int:
    s = str(dtype).split(".")[-1]
    bit_search = re.search(r"[^\d](\d+)(_.*)?$", s)
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    return int(bit_search.groups()[0])


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


def _extract_layer_dims(first_node, key_to_shape, output_shape):
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

    if is_depthwise_conv(first_node):
        return None

    input_shape = key_to_shape.get("input")
    if input_shape is None:
        return None

    # Skip fully-connected layers (no spatial tiling needed)
    if math.prod(input_shape[:-1]) == 1:
        return None

    if is_conv2d(first_node):
        weight_shape = key_to_shape.get("weight")
        if weight_shape is None or len(weight_shape) != 4:
            return None

        transposed = first_node.meta.get("transposed", False)
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

        stride_val = _pair(get_arg_value(first_node, 3, "stride", 1))
        stride_h, stride_w = stride_val
    else:
        # linear or matmul
        weight_shape = (
            key_to_shape.get("weight") or key_to_shape.get("other")
        )
        if weight_shape is None or len(weight_shape) != 2:
            return None

        # mirrors _build_gemm_shape_map in tiling.py
        weight_transposed = (
            is_matmul(first_node)
            ^ first_node.meta.get("transposed", False)
        )
        if weight_transposed:
            input_channels, output_channels = weight_shape  # (IC, OC)
        else:
            output_channels, input_channels = weight_shape  # transposed

        kH, kW = 1, 1
        height = 1
        width = math.prod(output_shape[:-1])
        stride_h, stride_w = 1, 1

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


class RuntimeCalculatorWithDRAM:
    """
    Runtime cost model for a 4-level memory hierarchy:
      L0 = PE registers, L1 = scratchpad, L2 = on-chip SRAM, L3 = DRAM.

    Mirrors RuntimeCalculator for L0-L2, then wraps the L2 timing in an outer
    L3 loop and adds DRAM transfer latency (input + weight tile loaded from DRAM
    to L2 before each L3 iteration).

    dram_bandwidth: DRAM bandwidth in input elements per cycle. All timing in
    this model is expressed in "input elements processed per cycle", so bandwidth
    must use the same unit. To convert from standard specs:

        dram_bandwidth = (bandwidth_bytes_per_sec) / (clock_hz * input_elem_bytes)

    where input_elem_bytes = input_dtype_width / 8. Example: 51.2 GB/s at 1 GHz
    with int8 input → 51.2e9 / (1e9 * 1) = 51 elements/cycle.

    Input and weight transfers are assumed sequential. If the memory controller
    issues both simultaneously, replace the sum with max(...) in calculate_runtime.

    double_buffered_l2: when True, DRAM I/O and compute overlap (ping-pong).
    The L3 loop cost becomes max(dram_loading_time, per_l3_compute_time) instead
    of their sum. The L2 capacity should be halved in build_architecture_and_schedule_with_dram
    when this is enabled, so that two L3 tiles fit on-chip simultaneously.
    """

    def __init__(
        self,
        input_dtype_width: int,
        weight_dtype_width: int,
        output_dtype_width: int,
        double_buffered_accum_buffer: bool,
        dram_bandwidth: int,
        double_buffered_l2: bool = False,
        has_sparse_op: bool = False,
        has_high_precision_vector_input: bool = False,
    ):
        self.input_dtype_width = input_dtype_width
        self.weight_dtype_width = weight_dtype_width
        self.output_dtype_width = output_dtype_width
        self.double_buffered_accum_buffer = double_buffered_accum_buffer
        self.dram_bandwidth = dram_bandwidth
        self.double_buffered_l2 = double_buffered_l2
        self.has_sparse_op = has_sparse_op
        self.has_high_precision_vector_input = has_high_precision_vector_input

    def calculate_runtime(self, architecture, layer, mapping):
        import interstellar
        le = interstellar.le

        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        partitionings = mapping.loop_partitionings

        # --- L1: weight-reuse tile timing (identical to RuntimeCalculator) ---
        sa_weight_loading_time = partitionings[le.IC][0] + 2

        first_non_ox_oy_index = 6
        for i in range(le.NUM):
            if i == le.OX or i == le.OY:
                continue
            if orders[i][1] < first_non_ox_oy_index:
                first_non_ox_oy_index = orders[i][1]

        weight_reuse_tile_size = 1
        for i in range(le.NUM):
            if orders[i][1] < first_non_ox_oy_index:
                weight_reuse_tile_size *= blockings[i][1]
        weight_reuse_tile_time = max(sa_weight_loading_time, weight_reuse_tile_size)

        num_remaining_l1_tiles = 1
        for i in range(le.NUM):
            if orders[i][1] >= first_non_ox_oy_index:
                num_remaining_l1_tiles *= blockings[i][1]
        num_remaining_l1_tiles *= blockings[le.IC][2]
        computation_l1_time = weight_reuse_tile_time * num_remaining_l1_tiles

        input_buffer_loading_size = 1
        for loop in [le.IC, le.OY, le.OX]:
            input_buffer_loading_size *= blockings[loop][1]
        input_buffer_loading_time = input_buffer_loading_size

        weight_buffer_loading_size = 1
        for loop in [le.IC, le.OC, le.FY, le.FX]:
            weight_buffer_loading_size *= blockings[loop][1]
        weight_buffer_loading_size *= partitionings[le.IC][0]
        weight_buffer_loading_time = (
            weight_buffer_loading_size * self.weight_dtype_width / self.input_dtype_width
        )
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

        # --- L2: outer spatial-tile loop ---
        l2_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l2_blocks *= blockings[i][2]

        if requires_high_precision and not self.double_buffered_accum_buffer:
            extra_vector_unit_time = output_size
        else:
            extra_vector_unit_time = 0

        if self.double_buffered_accum_buffer:
            per_l3_compute_time = (
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                + vector_unit_time
            )
        else:
            per_l3_compute_time = (
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                + extra_vector_unit_time
            )

        # --- L3 (DRAM): outer tile loop + transfer latency ---
        # IC is pinned to blocking_size=1 at L3, so l3_blocks is purely spatial.
        # Each L3 iteration loads inputs+weights from DRAM and writes outputs back.
        l3_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l3_blocks *= blockings[i][3]

        # DRAM transfer size for one L3 block (levels 0-2 only; [3] is the L3 iteration
        # count and belongs in l3_blocks, not in the per-iteration tile size).
        dram_input_size = (
            partitionings[le.IC][0]
            * blockings[le.IC][1] * blockings[le.IC][2]
            * blockings[le.OY][1] * blockings[le.OY][2]
            * blockings[le.OX][1] * blockings[le.OX][2]
        )
        dram_weight_size = (
            partitionings[le.IC][0]
            * blockings[le.IC][1] * blockings[le.IC][2]
            * partitionings[le.OC][0]
            * blockings[le.OC][1] * blockings[le.OC][2]
            * blockings[le.FY][1]
            * blockings[le.FX][1]
            * self.weight_dtype_width / self.input_dtype_width
        )
        dram_output_size = (
            partitionings[le.OC][0]
            * blockings[le.OC][1] * blockings[le.OC][2]
            * blockings[le.OY][1] * blockings[le.OY][2]
            * blockings[le.OX][1] * blockings[le.OX][2]
            * self.output_dtype_width / self.input_dtype_width
        )
        dram_loading_time = (
            dram_input_size + dram_weight_size + dram_output_size
        ) / self.dram_bandwidth

        if self.double_buffered_l2:
            # DRAM I/O and compute overlap: bottleneck is the slower of the two.
            total_time = l3_blocks * max(dram_loading_time, per_l3_compute_time)
        else:
            total_time = l3_blocks * (dram_loading_time + per_l3_compute_time)

        return total_time


def build_architecture_and_schedule_with_dram(
    ic_dim: int,
    oc_dim: int,
    l2_cache_size: int,
    input_buffer_size: int,
    weight_buffer_size: int,
    accum_buffer_size: int,
    dram_size: int,
    dram_access_cost: int = 1000,
    double_buffered_l2: bool = False,
):
    """
    Build a 4-level interstellar Resource and Schedule with DRAM as L3.

    Extends build_architecture_and_schedule with a 4th level:
      L0 = PE registers, L1 = L1 scratchpad, L2 = on-chip SRAM, L3 = DRAM.

    IC is pinned to blocking_size=1 at L3 so that all IC accumulation completes
    on-chip and partial sums never spill to DRAM.

    When double_buffered_l2=True, two L3 tiles must fit in L2 simultaneously
    (one being computed, one being loaded), so the effective L2 capacity is halved.
    """
    import interstellar

    effective_l2 = l2_cache_size // 2 if double_buffered_l2 else l2_cache_size

    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                input_buffer_size * ic_dim,
                accum_buffer_size * oc_dim,
                weight_buffer_size * oc_dim,
            ],
            [effective_l2],
            [dram_size],
        ],
        buf_access_cost_list=[[1, 1, 1], [10, 10, 10], [100], [dram_access_cost]],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
        para_count_list=[ic_dim * oc_dim, 1, 1, 1],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
        mac_capacity=0,
        partition_mode=[0, 0, 0, 0],
        invalid_underutilized=False,
    )

    schedule_constraint = {
        "schedule_hint": {
            "IC": {
                "level0": {"order": 1, "partitioning_size": ic_dim},
                "level1": {"order": -1},
                "level2": {"order": 0},
            },
            "OC": {
                "level0": {"order": 0, "partitioning_size": oc_dim},
            },
            "FX": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
            "FY": {
                "level0": {"blocking_size": 1, "partitioning_size": 1},
                "level2": {"blocking_size": 1, "partitioning_size": 1},
                "level3": {"blocking_size": 1, "partitioning_size": 1},
            },
        }
    }
    schedule_data = interstellar.extract_input.extract_schedule_info(
        schedule_constraint, 4
    )
    schedule = interstellar.Schedule(
        schedule_data["schedule_hint"],
        schedule_data["partition_loops"],
    )
    return architecture, schedule


def _extract_layer_from_node(node):
    """
    Build an interstellar Layer from a node's current (pre-tiling) shapes.

    Unlike _extract_layer_dims (which takes already-tiled shapes from the memory
    mapping phase), this function reads shapes directly from the FX node and is
    meant to be called before any tiling has occurred.

    Returns None for layers that should be skipped (depthwise, FC with batch=1,
    3-channel first conv, unsupported weight shapes).
    """
    import interstellar
    import math
    from .passes.tiling import _conv2d_layout
    from .passes.utils import get_arg_value, _pair
    from .mapping_utils import is_conv2d, is_matmul, is_depthwise_conv

    if is_depthwise_conv(node):
        return None

    if is_conv2d(node):
        if len(node.args) < 2:
            return None
        weight_shape = node.args[1].shape
        if len(weight_shape) != 4:
            return None

        transposed = node.meta.get("transposed", False)
        kH, kW, input_channels, output_channels = _conv2d_layout(
            weight_shape, True, not transposed
        )
        _, height, width, _ = _conv2d_layout(node.shape, False, not transposed)

        if input_channels == 3:
            return None

        stride_h, stride_w = _pair(get_arg_value(node, 3, "stride", 1))
    else:
        input_shape = node.args[0].shape
        if math.prod(input_shape[:-1]) == 1:
            return None

        if len(node.args) < 2:
            return None
        weight_shape = node.args[1].shape
        if len(weight_shape) != 2:
            return None

        weight_transposed = is_matmul(node) ^ node.meta.get("transposed", False)
        if weight_transposed:
            input_channels, output_channels = weight_shape
        else:
            output_channels, input_channels = weight_shape

        kH, kW = 1, 1
        height = 1
        width = math.prod(node.shape[:-1])
        stride_h, stride_w = 1, 1

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


def run_interstellar_dram(
    node,
    architecture,
    schedule,
    dram_bandwidth: int,
    input_dtype_width: int = 8,
    weight_dtype_width: int = 8,
    output_dtype_width: int = 8,
    double_buffered_accum_buffer: bool = False,
    double_buffered_l2: bool = False,
):
    """
    Run interstellar with the 4-level DRAM architecture for a single GEMM/conv node.

    Extracts layer dims from the node's current (pre-tiling) shapes, runs the
    optimizer, logs the resulting L2 and L3 tile sizes, and stores the mapping
    in node.meta["interstellar_dram_tiling"] for future use.

    Returns the best MappingPoint, or None if the node is skipped.
    """
    import interstellar
    import logging
    logger = logging.getLogger(__name__)

    layer = _extract_layer_from_node(node)
    if layer is None:
        return None

    logger.info(
        f"[interstellar DRAM] {node.name}: "
        f"IC={layer.nifm} OC={layer.nofm} "
        f"H={layer.hofm} W={layer.wofm} "
        f"kH={layer.hfil} kW={layer.wfil}"
    )

    rc = RuntimeCalculatorWithDRAM(
        input_dtype_width,
        weight_dtype_width,
        output_dtype_width,
        double_buffered_accum_buffer,
        dram_bandwidth,
        double_buffered_l2=double_buffered_l2,
    )

    _, runtime, mapping, _ = interstellar.optimizer.opt_optimizer(
        architecture,
        layer,
        schedule,
        rc.calculate_runtime,
        verbose=False,
    )

    le = interstellar.le
    b = mapping.loop_blockings
    logger.info(
        f"[interstellar DRAM] {node.name} L1 tiles: "
        f"IC={b[le.IC][1]} OC={b[le.OC][1]} "
        f"OX={b[le.OX][1]} OY={b[le.OY][1]} ON={b[le.ON][1]}"
    )
    logger.info(
        f"[interstellar DRAM] {node.name} L2 tiles: "
        f"IC={b[le.IC][2]} OC={b[le.OC][2]} "
        f"OX={b[le.OX][2]} OY={b[le.OY][2]} ON={b[le.ON][2]}"
    )
    logger.info(
        f"[interstellar DRAM] {node.name} L3 (DRAM) tiles: "
        f"IC={b[le.IC][3]} OC={b[le.OC][3]} "
        f"OX={b[le.OX][3]} OY={b[le.OY][3]} ON={b[le.ON][3]}"
    )
    logger.info(f"[interstellar DRAM] {node.name} estimated runtime: {runtime}")
    logger.info(interstellar.utils.format_tiling(mapping))

    node.meta["interstellar_dram_tiling"] = mapping
    return mapping
