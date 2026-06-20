"""
Cost model.
"""

from operator import mul
from operator import add
from functools import reduce
import copy
import math

from . import loop_enum as le


def get_comp_cost(layer):
    """
    Compute the total # of MAC computation, it is independent of other optimizations

    Also it is independent of input size and input/filter stride
    Total # of computation = OX*OY*IC*OC*ON*FX*FY
    """
    cost = (
        layer.wofm
        * layer.hofm
        * layer.nifm
        * layer.nofm
        * layer.nimg
        * layer.wfil
        * layer.hfil
    )
    return cost


def get_ideal_performance(layer, resource):
    """
    Compute the ideal runtime in cycles by assuming 100% PE array utilization
    Ideal # of cycles = Total # of MAC computation / Total # of PEs

    #LMEI Need to be modified if later when adding precision-scalable PE.
    # of functional PE will change depending on different precision modes.
    """
    total_comp = get_comp_cost(layer)
    number_pe = reduce(mul, resource.para_count_list, 1)
    runtime = math.ceil(total_comp * 1.0 / number_pe)

    return runtime


def get_layer_size(layer):
    """
    Get size of ifmap, ofmap, filter of the layer

    #LMEI ifmap_size should be able to calculate based on ofmap_size and input stride(IS) /filter stride(FS)
    IX = IS*(OX-1) + FS*(FX-1) + 1
    wifm = wistd*(wofm-1) + wfstd*(wfil-1) + 1
    """

    ifmap_size = layer.wifm * layer.hifm * layer.nifm * layer.nimg
    ofmap_size = layer.wofm * layer.hofm * layer.nofm * layer.nimg
    flmap_size = layer.wfil * layer.hfil * layer.nifm * layer.nofm

    return [ifmap_size, ofmap_size, flmap_size]


def get_hinted_para(level, hint):
    """
    Get the actual total spatial unrolling size from loop schedule
    """
    assert hint

    hinted_para = 1
    for loop in range(le.NUM):
        if loop in hint:
            hinted_loop_para = hint[loop][level][2]
            hinted_para *= hinted_loop_para

    return hinted_para


def valid_dataflow(resource, hint):
    """
    Check if the actual spatial unrolling size from loop schedule meets the HW utilization requirement
    by comparing it with real HW parallelism size * utilization threshold.
    """
    num_levels = resource.buffer_levels()

    for level in range(num_levels):
        if resource.paras[level].count != 1 and get_hinted_para(level, hint) < (
            resource.paras[level].count * resource.utilization_threshold
        ):
            return False

    return True


def get_if_access(resource, point, layer, mac_capacity=1):
    """
    Number of accesses to the input buffer at each memory level.

    Accesses at one level are decomposed by where each loop sits relative to
    that level:

        accesses = block_elements * level_iters * outer_iters * parallel_units

      block_elements : # distinct input elements in one block held BELOW this
                       level. Only the input's own loops add elements; the
                       OX/OY extent is the receptive field FX + (OX-1)*stride.
      level_iters    : # of those blocks requested by the loops AT this level.
      outer_iters    : # of times the block is reloaded, driven by ALL loops
                       ABOVE this level.
      parallel_units : spatial replication (PEs) at this level.
    """

    # Loops that contribute distinct input elements to a block. OC is excluded
    # (the input does not depend on output channel); FX/FY are excluded here
    # because their effect is folded into the OX/OY receptive-field extent below.
    relevant_loops = [le.OX, le.OY, le.IC, le.ON]

    num_levels = resource.buffer_levels()
    access_counts_per_level = []

    block_elements = 1

    for level in range(num_levels):
        # --- level_iters: loops AT this level that trigger an input access ---
        # The input's own loops [OX, OY, IC, ON] and the filter loops [FX, FY]
        # always count (each iteration reads a different input element). OC is
        # the only loop the input does not depend on, so it adds an access only
        # when it sits OUTSIDE the input loops (same input re-read per output
        # channel); when OC is inner the input stays resident, so skip it.
        innermost_input_loop_order = min(
            point.loop_orders[le.OX][level],
            point.loop_orders[le.OY][level],
            point.loop_orders[le.IC][level],
            point.loop_orders[le.ON][level],
            point.loop_orders[le.FX][level],
            point.loop_orders[le.FY][level],
        )

        level_iters = 1
        for i in range(le.NUM):
            if i == le.OC:
                if point.loop_orders[i][level] > innermost_input_loop_order:
                    level_iters *= point.loop_blockings[i][level]
            else:
                level_iters *= point.loop_blockings[i][level]

        # --- outer_iters: every loop above reloads the block once per iter ---
        outer_iters = 1
        for upper_level in range(level + 1, num_levels):
            for i in range(le.NUM):
                outer_iters *= point.loop_blockings[i][upper_level]

        access_counts_per_level.append(
            block_elements
            * level_iters
            * outer_iters
            * resource.paras[level].count
        )

        # --- block_elements: distinct input elements in the block below ---
        fy = point.loop_blockings[le.FY][level]
        fx = point.loop_blockings[le.FX][level]

        for i in relevant_loops:
            loop_blocking = point.loop_blockings[i][level]
            if i == le.OX:
                stride = 1 if level == 0 or fx == 1 else layer.wstd
                block_elements *= loop_blocking * stride + (fx - 1)
            elif i == le.OY:
                stride = 1 if level == 0 or fy == 1 else layer.hstd
                block_elements *= loop_blocking * stride + (fy - 1)
            else:
                block_elements *= loop_blocking
            block_elements *= point.loop_partitionings[i][level]

    return access_counts_per_level


def get_of_access(resource, point, layer, mac_capacity=1):
    """
    Number of accesses to the output buffer at each memory level.

    Same decomposition as get_if_access:
        accesses = block_elements * level_iters * outer_iters * parallel_units
    The output's own loops are [OX, OY, OC, ON]; [FX, FY, IC] are the reduction
    loops that re-touch the same output element to accumulate into it.
    """

    relevant_loops = [le.OX, le.OY, le.OC, le.ON]

    num_levels = resource.buffer_levels()
    access_counts_per_level = []

    block_elements = 1

    for level in range(num_levels):
        # --- level_iters: loops AT this level that trigger an output access ---
        # The output's own loops always count. A reduction loop counts only when
        # it sits OUTSIDE the output loops (output re-read/re-accumulated per
        # step); when it is inner the output stays resident (stationary).
        innermost_output_loop_order = min(
            point.loop_orders[le.OX][level],
            point.loop_orders[le.OY][level],
            point.loop_orders[le.OC][level],
            point.loop_orders[le.ON][level],
        )

        level_iters = 1
        for i in range(le.NUM):
            if i not in relevant_loops:
                if point.loop_orders[i][level] > innermost_output_loop_order:
                    level_iters *= point.loop_blockings[i][level]
            else:
                level_iters *= point.loop_blockings[i][level]

        # --- outer_iters: every loop above reloads the block once per iter ---
        outer_iters = 1
        for upper_level in range(level + 1, num_levels):
            for i in range(le.NUM):
                outer_iters *= point.loop_blockings[i][upper_level]

        access_counts_per_level.append(
            block_elements
            * level_iters
            * outer_iters
            * resource.paras[level].count
        )

        # --- block_elements: distinct output elements in the block below ---
        for i in relevant_loops:
            block_elements *= point.loop_blockings[i][level]
            block_elements *= point.loop_partitionings[i][level]

    return access_counts_per_level


def get_fl_access(resource, point, layer, mac_capacity=1):
    """
    Number of accesses to the weight (filter) buffer at each memory level.

    Same decomposition as get_if_access:
        accesses = block_elements * level_iters * outer_iters * parallel_units
    The weight's own loops are [FX, FY, IC, OC]; [OX, OY, ON] are irrelevant
    (the same weight is reused across output positions and batch).
    """

    relevant_loops = [le.FX, le.FY, le.IC, le.OC]

    num_levels = resource.buffer_levels()
    access_counts_per_level = []

    block_elements = 1

    for level in range(num_levels):
        # --- level_iters: loops AT this level that trigger a weight access ---
        # The weight's own loops always count. An irrelevant loop counts only
        # when it sits OUTSIDE the weight loops (weight re-read per output
        # position); when it is inner the weight stays resident (stationary).
        innermost_weight_loop_order = min(
            point.loop_orders[le.FX][level],
            point.loop_orders[le.FY][level],
            point.loop_orders[le.IC][level],
            point.loop_orders[le.OC][level],
        )

        level_iters = 1
        for i in range(le.NUM):
            if i not in relevant_loops:
                if point.loop_orders[i][level] > innermost_weight_loop_order:
                    level_iters *= point.loop_blockings[i][level]
            else:
                level_iters *= point.loop_blockings[i][level]

        # --- outer_iters: every loop above reloads the block once per iter ---
        outer_iters = 1
        for upper_level in range(level + 1, num_levels):
            for i in range(le.NUM):
                outer_iters *= point.loop_blockings[i][upper_level]

        access_counts_per_level.append(
            block_elements
            * level_iters
            * outer_iters
            * resource.paras[level].count
        )

        # --- block_elements: distinct weight elements in the block below ---
        for i in relevant_loops:
            block_elements *= point.loop_blockings[i][level]
            block_elements *= point.loop_partitionings[i][level]

    return access_counts_per_level


def get_if_size(blocking_accum_list, partitioning_accum_list, partitioning_list, layer):
    """
    Get size of if block at current level including both temporal and spatial loop part

    blocking     -> temporal loop part
    partitioning -> spatial  loop part

    #LMEI to support filter stride(FS) later
    right now, FS/wfstd = 1 in
    IX = IS*(OX-1) + FS*(FX-1) + 1 or
    wifm = wistd*(wofm-1) + wfstd*(wfil-1) + 1

    #LMEI (new HW template) no need for Input Duplication when OC partitions
     by letting one reg broadcast Input to a row of OC partitioned PE
     and remove inner PE ifamp register
    """

    fx_acc = blocking_accum_list[le.FX] * partitioning_accum_list[le.FX]
    fy_acc = blocking_accum_list[le.FY] * partitioning_accum_list[le.FY]
    ox_acc = blocking_accum_list[le.OX] * partitioning_accum_list[le.OX]
    oy_acc = blocking_accum_list[le.OY] * partitioning_accum_list[le.OY]
    width = fx_acc + (ox_acc - 1) * layer.wstd
    height = fy_acc + (oy_acc - 1) * layer.hstd

    return (
        width
        * height
        * blocking_accum_list[le.IC]
        * partitioning_accum_list[le.IC]
        * blocking_accum_list[le.ON]
        * partitioning_accum_list[le.ON]
        * partitioning_list[le.OC]
    )  # Duplication when OC partitions


def get_of_size(blocking_accum_list, partitioning_accum_list, partitioning_list):
    """
    Get size of of block at current level including both temporal and spatial loop part

    #LMEI (new HW template) no need for Output Duplication when IC, FX or FY partitions
     by letting output data from a row of IC, FX or FY partitioned PE add together
     and remove inner PE ofamp register
    """

    return (
        blocking_accum_list[le.OX]
        * partitioning_accum_list[le.OX]
        * blocking_accum_list[le.OY]
        * partitioning_accum_list[le.OY]
        * blocking_accum_list[le.OC]
        * partitioning_accum_list[le.OC]
        * blocking_accum_list[le.ON]
        * partitioning_accum_list[le.ON]
        * partitioning_list[le.IC]
        * partitioning_list[le.FX]
        * partitioning_list[le.FY]
    )  # Duplication when IC, FX or FY partitions


def get_fl_size(blocking_accum_list, partitioning_accum_list, partitioning_list):
    """
    Get size of fl block at current level

    #LMEI (new HW template) no need for Weight Duplication when OX, OY or ON partitions
     by letting one reg broadcast Weight to a row of OX, OY or ON partitioned PE
     and remove inner PE weight register
    """

    return (
        blocking_accum_list[le.FX]
        * partitioning_accum_list[le.FX]
        * blocking_accum_list[le.FY]
        * partitioning_accum_list[le.FY]
        * blocking_accum_list[le.IC]
        * partitioning_accum_list[le.IC]
        * blocking_accum_list[le.OC]
        * partitioning_accum_list[le.OC]
        * partitioning_list[le.OX]
        * partitioning_list[le.OY]
        * partitioning_list[le.ON]
    )  # Duplication when OX, OY or ON partitions


def get_if_bank_size(blocking_accum_list, layer):
    """
    Get size of if block at current level

    blocking -> temporal loop part

    #LMEI to support filter stride(FS) later
    right now, FS/wfstd = 1 in
    IX = IS*(OX-1) + FS*(FX-1) + 1 or
    wifm = wistd*(wofm-1) + wfstd*(wfil-1) + 1
    """

    fx_acc = blocking_accum_list[le.FX]
    fy_acc = blocking_accum_list[le.FY]
    ox_acc = blocking_accum_list[le.OX]
    oy_acc = blocking_accum_list[le.OY]
    width = fx_acc + (ox_acc - 1) * layer.wstd
    height = fy_acc + (oy_acc - 1) * layer.hstd

    return width * height * blocking_accum_list[le.IC] * blocking_accum_list[le.ON]


def get_of_bank_size(blocking_accum_list):
    """
    Get size of of block at current level

    blocking -> temporal loop part
    """

    return (
        blocking_accum_list[le.OX]
        * blocking_accum_list[le.OY]
        * blocking_accum_list[le.OC]
        * blocking_accum_list[le.ON]
    )


def get_fl_bank_size(blocking_accum_list):
    """
    Get size of fl block at current level

    blocking -> temporal loop part
    """

    return (
        blocking_accum_list[le.FX]
        * blocking_accum_list[le.FY]
        * blocking_accum_list[le.IC]
        * blocking_accum_list[le.OC]
    )


def get_array_access_and_cost(level, para, access_list, point):
    """
    Get the access at array level from the access at the
    lower level of memory hierarchy
    """

    para_mode = para.access_mode
    assert para_mode == 1 or para_mode == 2  # Don't get it

    array_dim = para.array_dim
    para_count = para.array_width
    para_cost = para.array_access_cost * 1.0
    nearest_pe_cost = para_cost

    [if_block_access, of_block_access, fl_block_access] = access_list
    partitions = list(zip(*point.loop_partitionings))[level]
    para_dim = point.para_loop_dim[level]

    partitions_nearest = [
        1,
    ] * le.NUM
    partitions_far = []
    across_block_cost = [0] * array_dim

    if para_mode == 1:
        for i in range(len(para_dim)):
            para_index = para_dim[i]
            partitions_far.append(
                [
                    1,
                ]
                * le.NUM
            )
            if len(para_index) == 1:
                partitions_nearest[para_index[0]] = partitions[para_index[0]]
            else:
                inner_loop, outer_loop = para_index
                partitions_nearest[inner_loop] = partitions[inner_loop]
                partitions_far[i][outer_loop] = partitions[outer_loop]
                across_block_cost[i] = para_cost * partitions[inner_loop]

        array_if_block_access_nearest = (
            if_block_access
            * partitions_nearest[le.FX]
            * partitions_nearest[le.FY]
            * partitions_nearest[le.OC]
        )
        array_of_block_access_nearest = (
            of_block_access
            * partitions_nearest[le.FX]
            * partitions_nearest[le.FY]
            * partitions_nearest[le.IC]
        )
        array_fl_block_access_nearest = (
            fl_block_access
            * partitions_nearest[le.OX]
            * partitions_nearest[le.OY]
            * partitions_nearest[le.ON]
        )

        array_access = [
            [
                array_if_block_access_nearest,
                array_of_block_access_nearest,
                array_fl_block_access_nearest,
            ]
        ]

        for i in range(array_dim):  # Don't get it
            if_partitions_far = (
                partitions_far[i][le.FX]
                * partitions_far[i][le.FY]
                * partitions_far[i][le.OC]
            )
            if_partitions_far = if_partitions_far if if_partitions_far != 1 else 0
            of_partitions_far = (
                partitions_far[i][le.FX]
                * partitions_far[i][le.FY]
                * partitions_far[i][le.IC]
            )
            of_partitions_far = of_partitions_far if of_partitions_far != 1 else 0
            fl_partitions_far = (
                partitions_far[i][le.OX]
                * partitions_far[i][le.OY]
                * partitions_far[i][le.ON]
            )
            fl_partitions_far = fl_partitions_far if fl_partitions_far != 1 else 0

            if_array_block_access = if_block_access * if_partitions_far
            of_array_block_access = of_block_access * of_partitions_far
            fl_array_block_access = fl_block_access * fl_partitions_far

            array_access.append(
                [if_array_block_access, of_array_block_access, fl_array_block_access]
            )

        return [array_access, [nearest_pe_cost] + across_block_cost]

    elif para_mode == 2:
        for i in range(len(para_dim)):
            para_index = para_dim[i]
            for j in para_index:
                partitions_nearest[j] = partitions[j]

        array_if_block_access_nearest = (
            if_block_access
            * partitions_nearest[le.FX]
            * partitions_nearest[le.FY]
            * partitions_nearest[le.OC]
        )
        array_of_block_access_nearest = (
            of_block_access
            * partitions_nearest[le.FX]
            * partitions_nearest[le.FY]
            * partitions_nearest[le.IC]
        )
        array_fl_block_access_nearest = (
            fl_block_access
            * partitions_nearest[le.OX]
            * partitions_nearest[le.OY]
            * partitions_nearest[le.ON]
        )

        array_access = [
            [
                array_if_block_access_nearest,
                array_of_block_access_nearest,
                array_fl_block_access_nearest,
            ]
        ]

        return [array_access, [nearest_pe_cost]]


def get_access(point, layer, resource):
    """
    Get the total access of each block at each level,
    return the list as
    [[if_block_access, of_block_access, fl_block_access], ...].

    Assume all the buffers are inclusive, so buffers in lower level
    appear in higher level as well.

    For the parallelism case assume read from next memory level,

    Support more access modes in parallelism case
    """
    # TODO support more customized memory
    # TODO more access at overlapped boundary

    num_levels = resource.buffer_levels()
    mac_capacity = resource.mac_capacity

    access_list = []

    if_accesses = get_if_access(resource, point, layer, mac_capacity)
    of_accesses = get_of_access(resource, point, layer, mac_capacity)
    fl_accesses = get_fl_access(resource, point, layer, mac_capacity)

    access_list = list(zip(if_accesses, of_accesses, fl_accesses))

    # para_mode = [e.access_mode for i, e in enumerate(resource.paras) if e.access_mode != 0]
    para_mode_level = [i for i, e in enumerate(resource.paras) if e.access_mode != 0]
    partitions = list(zip(*point.loop_partitionings))
    array_costs = []
    if para_mode_level:
        # access at array level
        # para_mode_level = [i for i, e in enumerate(resource.paras) if e.access_mode != 0]
        delta = 0
        for level in para_mode_level:
            if level + delta + 1 >= num_levels:
                next_level_access = [1, 1, 1]
            else:
                next_level_access = copy.copy(access_list[level + delta + 1])
                next_level_access[1] = (next_level_access[1] + 1) / 2
            array_access, array_cost = get_array_access_and_cost(
                level, resource.paras[level], next_level_access, point
            )
            array_costs.append(array_cost)
            access_list.insert(level + delta + 1, array_access)
            delta += 1

    return [access_list, array_costs]


def _output_dtype_bits(point, layer, level):
    """
    Width (in bits) of the output tensor stored at the given level.

    The output holds wide partial sums (psum_dtype_bits) while IC accumulation
    is still incomplete *above* this level, and the narrow final/quantized
    output (of_dtype_bits) once IC is fully reduced at or below this level.
    """
    num_levels = len(point.loop_blocking(le.IC))
    ic_above = 1
    for lvl in range(level + 1, num_levels):
        ic_above *= point.loop_blocking(le.IC)[lvl]
        ic_above *= point.loop_partitioning(le.IC)[lvl]
    return layer.psum_dtype_bits if ic_above > 1 else layer.of_dtype_bits


def get_bank_size(point, layer, level):

    blocking_accum_list = []
    for i in range(le.NUM):
        blocking_accum_list.append(reduce(mul, point.loop_blocking(i)[: level + 1], 1))

    if_bank_size = get_if_bank_size(blocking_accum_list, layer)
    of_bank_size = get_of_bank_size(blocking_accum_list)
    fl_bank_size = get_fl_bank_size(blocking_accum_list)

    if level <= 1:
        # L0/L1 are slot arrays: each element occupies a fixed-width slot (the max
        # dtype in a mixed-precision design; narrower dtypes are padded), so the
        # fit is checked in element counts, independent of the layer's dtype.
        return (if_bank_size, of_bank_size, fl_bank_size)

    # L2/L3 are flat byte pools where sub-byte operands pack -> compare in bytes.
    of_bits = _output_dtype_bits(point, layer, level)
    return (
        if_bank_size * layer.if_dtype_bits / 8.0,
        of_bank_size * of_bits / 8.0,
        fl_bank_size * layer.fl_dtype_bits / 8.0,
    )


def get_block_size(point, layer, level):
    """
    Calculate the size of ifmap, ofmap, filter at current level
    """

    blocking_accum_list = []
    partitioning_accum_list = []
    partitioning_reshape = list(zip(*point.loop_partitionings))
    partitioning_list = partitioning_reshape[level]
    for i in range(le.NUM):
        blocking_accum_list.append(reduce(mul, point.loop_blocking(i)[: level + 1], 1))
        partitioning_accum_list.append(
            reduce(mul, point.loop_partitioning(i)[: level + 1], 1)
        )  # FIXME inclusive mode also duplicates data

    if_block_size = get_if_size(
        blocking_accum_list, partitioning_accum_list, partitioning_list, layer
    )
    of_block_size = get_of_size(
        blocking_accum_list, partitioning_accum_list, partitioning_list
    )
    fl_block_size = get_fl_size(
        blocking_accum_list, partitioning_accum_list, partitioning_list
    )

    if level <= 1:
        # L0/L1 are slot arrays (padded to a fixed width); fit is in element
        # counts, independent of the layer's dtype.  L2/L3 (below) are byte pools.
        return (if_block_size, of_block_size, fl_block_size)

    of_bits = _output_dtype_bits(point, layer, level)
    return (
        if_block_size * layer.if_dtype_bits / 8.0,
        of_block_size * of_bits / 8.0,
        fl_block_size * layer.fl_dtype_bits / 8.0,
    )


def get_block_sizes(num_levels, point, layer):
    """
    Get size of ifmap, ofmap, filter
    """
    bank_list = []
    block_list = []
    for level in range(num_levels):
        block_list.append(get_block_size(point, layer, level))
        bank_list.append(get_bank_size(point, layer, level))

    return [bank_list, block_list]


def fit_in_level(cap, blocks, invalid_underutilized, level, memory_partitions):
    """
    Check if the current level mem size >= current level loop blocking size
    invalid_underutilized is used to exclude mapping points with too low memory utilization (< 50%)
    #LMEI can later put the memory utilization threshold as a user defined parameter
    """
    if type(cap) is list:
        # I/O/W example: [0,0,1] I is stored in memory 0,  O is stored in memory 0,  W is stored in memory 1
        # leave last empty

        # memory_partitions = [[0,1, 2],[0,0,1],[0,0,None]] #if 3 level do not contain weights [0, 0, None]

        # capacity =  [[2,2], [30000,30000], [1000000,1000000]]
        for i in range(len(cap)):
            indices = [
                index
                for index, partition in enumerate(memory_partitions[level])
                if partition == i
            ]
            size = sum([blocks[j] for j in indices])
            if size == 0:
                continue
            if (size > cap[i]) == True:
                return False  # it does not fit

            check_if_underutilized = 0

            # print level, i, invalid_underutilized, memory_partitions[level+1][i], size, cap[i]
            if invalid_underutilized:

                last_layer = []
                for mem in indices:
                    last_layer.append(memory_partitions[level + 1][mem])
                if None not in last_layer:
                    if (
                        (size <= cap[i]) and (2 * size <= cap[i])
                    ) == True:  # if double the size fit then there will be a better to block partition that will utilized all memory,
                        # print "NO level: ", level,"blocks: ",  blocks, "size: ", size, "cap: ", cap, "indices: ", indices, "last_layer", last_layer
                        check_if_underutilized += 1

                    else:
                        test = 1
                else:
                    # print "OK level: ", level,"blocks: ",  blocks, "size: ", size, "cap: ", cap, "indices: ", indices, "last_layer", last_layer
                    test = 2

            if check_if_underutilized == len(cap):
                return False

        return True

    else:
        total_size = sum(blocks)
        # for size,contain in zip(blocks, contains):
        #     if contain:
        #         total_size += size

        # total_capacity = 0
        # for size,contain in zip(cap, contains):
        #     if contain:
        #         total_capacity += size

        # total_size = sum(blocks)
        if invalid_underutilized:
            return (total_size <= cap) and (2 * total_size >= cap)
        else:
            return total_size <= cap


def valid_partition_number(resource, partitioning, level):
    max_parallelism = resource.parallelism(level).count
    actual_parallelism = reduce(mul, partitioning[level], 1)
    return actual_parallelism <= max_parallelism


def valid_partitioning_current_level(resource, point, layer, level, verbose=False):
    valid_size = fit_in_level(
        resource.buffer(level).capacity,
        get_bank_size(point, layer, level),
        resource.invalid_underutilized,
        level,
        resource.memory_partitions,
    )

    return valid_size


def valid_mapping_point_current_level(resource, point, layer, level, verbose=False):
    if resource.paras[level].count > 1:
        valid_size = fit_in_level(
            resource.buffer(level).capacity,
            get_bank_size(point, layer, level),
            resource.invalid_underutilized,
            level,
            resource.memory_partitions,
        )
    else:
        valid_size = fit_in_level(
            resource.buffer(level).capacity,
            get_block_size(point, layer, level),
            resource.invalid_underutilized,
            level,
            resource.memory_partitions,
        )

    partitioning = list(zip(*(point.loop_partitionings)))
    valid_para = valid_partition_number(resource, partitioning, level)

    if verbose == 3:
        print("Level ", level, ": Partitioned block size fit in bank: ", valid_size)
        print("Level ", level, ": Partition number is valid: ", valid_para)

    return valid_size and valid_para


def valid_partitioning(resource, point, layer, verbose=False):
    para_level = resource.para_index
    for level in para_level:
        if not valid_partitioning_current_level(resource, point, layer, level, verbose):
            return False
    return True


def valid_blocking_size_current_level(resource, point, layer, level, verbose=False):
    """
    Check if the blocking size of the current level fits in memory.
    """
    if level == resource.buffer_levels() - 1:
        return True

    if type(resource.buffer(level).capacity) is list:
        capacity = copy.deepcopy(resource.buffer(level).capacity)
        for i in range(len(capacity)):
            capacity[i] = capacity[i] * resource.paras[level].count
        return fit_in_level(
            capacity,
            get_block_size(point, layer, level),
            (resource.invalid_underutilized and (level not in resource.para_index)),
            level,
            resource.memory_partitions,
        )
    else:
        return fit_in_level(
            resource.buffer(level).capacity * resource.paras[level].count,
            get_block_size(point, layer, level),
            (resource.invalid_underutilized and (level not in resource.para_index)),
            level,
            resource.memory_partitions,
        )

        # get_block_size(point, layer, level), (level > min(resource.para_index)))


def valid_blocking_size(resource, point, layer, verbose=False):
    for level in range(resource.buffer_levels()):
        if not valid_blocking_size_current_level(
            resource, point, layer, level, verbose
        ):
            return False
    return True


def valid_mapping_point(resource, point, layer, verbose=False):
    for i in range(resource.buffer_levels()):
        if not valid_mapping_point_current_level(resource, point, layer, i, verbose):
            return False
    return True


def get_total_access_cost(resource, array_cost):
    total_access_cost = copy.deepcopy(resource.access_cost)

    if not resource.array_access_cost:
        return total_access_cost

    para_index = [i for i, e in enumerate(resource.paras) if e.access_mode != 0]
    addition_levels = len(para_index)

    delta = 1
    for i in range(addition_levels):
        index = para_index[i]
        total_access_cost.insert(index + delta, array_cost[i])
        delta += 1
    return total_access_cost


def get_array_level_cost(
    resource, point, layer_size, level, next_level_access, verbose=False
):
    """
    Given next_level_access (above-level memory access)
    calculate the current level (paralleled level) inter-PE data access
    thus calculate the current level (paralleled level) inter-PE communication energy
    i.e. the energy spent on interconnection

    Specific to Systolic Array template.

    level_access: [[close access for I/O/W],[far access on one dimension for I/O/W],[far access on another dimension]]
    close access means data are passing from one PE to its neighbour PE
    Far access means data need to jump from one PE to PEs far away from it.
    Far jump happens because of dataflow spatial replication (e.g. 2D array -> kinds of 3D array)
    """

    # TODO add support for other access_mode # don't get it
    # LMEI to distinguish O (partial sum) in buffer_access from A and W

    assert resource.paras[level].count and resource.paras[level].access_mode

    level_access, level_cost = get_array_access_and_cost(
        level, resource.paras[level], next_level_access, point
    )

    total_cost = 0
    for i in range(len(level_access)):
        total_cost += level_access[i] * level_cost[i]

    if verbose >= 3:
        print("Level ", level, " array level access: ", level_access)

    return total_cost


def get_array_and_curr_level_cost(resource, point, layer, level, verbose=False):
    """
    Get the energy from current level of memory access + inter-PE access
    """

    # LMEI to distinguish O (partial sum) in buffer_access from A and W

    layer_size = get_layer_size(layer)
    mac_capacity = resource.mac_capacity

    level_access = [
        get_if_access(level, point, layer, mac_capacity),
        get_of_access(level, point, layer, mac_capacity),
        get_fl_access(level, point, layer, mac_capacity),
    ]

    [if_access, of_access, fl_access] = level_access

    buffer_level_access = [if_access, of_access, fl_access]
    # level_cost = sum(total_buffer_access) * resource.access_cost[level]
    level_cost = 0
    for i in range(len(buffer_level_access)):
        index = resource.memory_partitions[level][i]
        if index is not None:
            level_cost += buffer_level_access[i] * resource.access_cost[level][index]
    # operand_costs = [access_cost * num_accesses for access_cost,num_accesses in zip(total_buffer_access,resource.access_cost[level]) ]
    # level_cost = sum(operand_costs)

    if verbose >= 3:
        print("Level ", level, " access: ", buffer_level_access)

    # level_cost += get_array_level_cost(
    #     resource, point, layer_size, level - 1, level_access, verbose
    # )

    return level_cost


def get_level_cost(resource, point, layer, level, verbose=False):
    """
    Get the energy from current level of memory access

    #LMEI to distinguish O (partial sum) in buffer_access from A and W
    """

    layer_size = get_layer_size(layer)
    mac_capacity = resource.mac_capacity

    if_accesses = get_if_access(resource, point, layer, mac_capacity)
    of_accesses = get_of_access(resource, point, layer, mac_capacity)
    fl_accesses = get_fl_access(resource, point, layer, mac_capacity)

    buffer_access = list(zip(if_accesses, of_accesses, fl_accesses))

    # Inputs, weights, and outputs may have different costs
    # level_cost = sum(buffer_access) * resource.access_cost[level]
    level_cost = 0
    for i in range(3):
        memory_partition = resource.memory_partitions[level][i]
        level_cost += (
            buffer_access[level][i] * resource.access_cost[level][memory_partition]
        )

    if verbose >= 3:
        print("Level", level, " access: ", level_access)
    return level_cost


def get_total_access(resource, point, layer, verbose=False):
    layer_size = get_layer_size(layer)

    access_list, array_cost = get_access(point, layer, resource)

    if verbose >= 3:
        print("access breakdown: ", access_list)

    total_level_access = []
    for i in range(len(access_list)):
        """List of total access of each buffer at level i"""
        if not isinstance(access_list[i][0], list):
            buffer_access = list(map(mul, access_list[i], layer_size))
            total_level_access.append(sum(buffer_access))
        else:
            for j in range(len(access_list[i])):
                buffer_access = list(map(mul, access_list[i][j], layer_size))
                total_level_access.append(sum(buffer_access))

    return total_level_access


def get_level_costs(resource, point, layer, verbose=False):
    num_levels = resource.buffer_levels()

    level_energy = []
    for level in range(num_levels):
        level_energy.append(get_level_cost(resource, point, layer, level))

    para_index = [i for i, e in enumerate(resource.paras) if e.access_mode != 0]

    delta = 1
    for index in para_index:
        array_energy = (
            get_array_and_curr_level_cost(resource, point, layer, index + 1)
            - level_energy[index + delta]
        )
        level_energy.insert(index + delta, array_energy)
        delta += 1

    return level_energy


# FIXME
def get_block_cost(resource, point, layer, verbose=False):
    """
    Get the cost of the given mapping point on given resource.

    If the point is not feasible on the resource, return inf.
    """
    # TODO include static energy
    num_levels = resource.buffer_levels()

    access_list, array_cost = get_access(point, layer, resource)
    layer_size = get_layer_size(layer)

    total_access_cost = get_total_access_cost(resource, array_cost)
    assert len(total_access_cost) == len(access_list)

    block_costs = [0.0, 0.0, 0.0]
    for i in range(len(total_access_cost)):
        buffer_access = [a * b for a, b in list(zip(access_list[i], layer_size))]
        block_cost = [x * total_access_cost[i] for x in buffer_access]
        block_costs = list(map(add, block_cost, block_costs))

    if verbose:
        print("access_list: ", access_list)
        bank_size_list, block_size_list = get_block_sizes(num_levels, point, layer)
        print("bank_size_list: ", bank_size_list)
        print("block_size_list: ", block_size_list)
        print("layer_size: ", layer_size)
        print("block costs: ", block_costs)

    return block_costs


def get_cost(resource, point, layer, verbose=False):
    """
    Get the cost of the given mapping point on given resource.

    If the point is not feasible on the resource, return inf.
    """
    # TODO include static energy
    # TODO support other access_mode
    num_levels = resource.buffer_levels()
    assert len(point.loop_blockings[0]) == num_levels, (
        "number of blockings does not match with number of memory "
        "levels: %d" % num_levels
    )

    access_list, array_cost = get_access(point, layer, resource)

    total_access_cost = get_total_access_cost(resource, array_cost)
    assert len(total_access_cost) == len(access_list)

    total_cost = 0.0
    for i in range(len(total_access_cost)):
        """List of total access of each buffer at level i"""
        if not isinstance(access_list[i][0], list):
            total_cost += sum(
                [access * total_access_cost[i][0] for access in access_list[i]]
            )
        else:
            for j in range(len(access_list[i])):
                total_cost += access_list[i][j] * total_access_cost[i][j]

    if verbose:
        # print("total_access_cost", total_access_cost)
        # print("access_list", access_list)

        # print("layer_size",layer_size)

        idx_adjust = 0
        if len(total_access_cost) > 4:
            idx_adjust = 1

        layer_access_cost = (
            total_access_cost[: 1 + idx_adjust] + total_access_cost[2 + idx_adjust :]
        )
        print(
            "16b_Access_Energy_[RegisterFile(s),Buffer,DRAM]_(pJ): \n\tifmap: {}\n\tofmap: {}\n\tfilter: {}".format(
                [item[0] for item in layer_access_cost],
                [item[1] for item in layer_access_cost],
                [item[2] for item in layer_access_cost],
            )
        )
        print(
            "PE_Access_Cost_(pJ): \n\tifmap: {}\n\tofmap: {}\n\tfilter: {}".format(
                total_access_cost[1 + idx_adjust][0],
                total_access_cost[1 + idx_adjust][1],
                total_access_cost[1 + idx_adjust][2],
            )
        )

        layer_num_access = access_list[: 1 + idx_adjust] + access_list[2 + idx_adjust :]
        print(
            "Tiles_Accessed_from_[RegisterFile(s),Buffer,DRAM]_in_Layer: \n\tifmap: {}\n\tofmap: {}\n\tfilter: {}".format(
                [item[0] for item in layer_num_access],
                [item[1] for item in layer_num_access],
                [item[2] for item in layer_num_access],
            )
        )
        print(
            "Tiles_Accessed_from_[RegisterFile(s),Buffer,DRAM]_PEs_in_Layer: \n\tifmap: {}\n\tofmap: {}\n\tfilter: {}".format(
                access_list[1 + idx_adjust][0],
                access_list[1 + idx_adjust][1],
                access_list[1 + idx_adjust][2],
            )
        )

        bank_size_list, block_size_list = get_block_sizes(num_levels, point, layer)

        # print("bank_size_list", bank_size_list)
        # print("block_size_list", block_size_list)

        print(
            "Memory_Bank_Size_List_When_Parallelized/Unrolled_[RegisterFile(s),Buffer,DRAM]_(bytes): \n\tifmap: {}\n\tofmap: {}\n\tfilter: {}".format(
                [item[0] for item in bank_size_list],
                [item[1] for item in bank_size_list],
                [item[2] for item in bank_size_list],
            )
        )
        print(
            "Memory_Block_Size_List_When_NOT_Parallelized/Unrolled_[RegisterFile(s),Buffer,DRAM]_(bytes): \n\tifmap: {}\n\tofmap: {}\n\tfilter: {}".format(
                [item[0] for item in block_size_list],
                [item[1] for item in block_size_list],
                [item[2] for item in block_size_list],
            )
        )
        print(
            "Layer_Size_(number_of_pixels): \n\tifmap: {}\n\tofmap: {}\n\tfilter: {}".format(
                layer_size[0], layer_size[1], layer_size[2]
            )
        )
        # print('total cost: ', total_cost)

    # return total_cost
    return total_cost, total_access_cost, access_list
