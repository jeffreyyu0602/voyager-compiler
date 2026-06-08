"""
Mapping point generator

Current implementation only supports the blocking factor and
parallelism to be factors of the layer size
"""

# from __future__ import division
import itertools
import copy
from operator import mul
import math
import pickle
from functools import reduce

from .mapping_point import MappingPoint
from .cache import Cache
from . import cost_model

from . import loop_enum as le
from . import buffer_enum as be
from . import utils


def get_hinted_para(layer, level, hint):
    assert hint

    hinted_para = 1
    for loop in range(le.NUM):
        if loop in hint:
            hinted_loop_para = hint[loop][level][2]
            hinted_para *= hinted_loop_para

    return hinted_para


def get_hinted_partitioning(level, hint):
    hinted_partitioning = []
    hinted_para_dim = []
    for loop in range(le.NUM):
        if loop in hint:
            hinted_partitioning.append(hint[loop][level][2])
            hinted_para_dim.append([loop])
        else:
            hinted_partitioning.append(1)

    return [[hinted_partitioning], [hinted_para_dim]]


def get_fixed_partitioning(num_levels, hint):
    """
    Get a prefixed partitioning from hint
    Helper function used for developping
    """
    # TODO remove this function

    if not hint:
        return [(1,) * num_levels] * le.NUM

    partitioning_list = []
    for loop in range(le.NUM):
        partitioning = [1] * num_levels
        if loop in hint:
            for i in range(num_levels):
                if hint[loop][i]:
                    partitioning[i] = hint[loop][i][2]
        partitioning_list.append(tuple(partitioning))
    return partitioning_list


def get_non_empty_loops(point, num_levels):
    """
    non_empty_loops is a list that contains #levels tuples,
    each tuple contains the loop whose size is not 1 at this level
    """
    blocking = list(zip(*(point.loop_blockings)))
    partitioning = list(zip(*(point.loop_partitionings)))

    non_empty_loops = []
    for i in range(num_levels):
        t0 = blocking[i]
        t1 = partitioning[i]
        non_empty_blocking = [i for i, e in enumerate(t0) if e != 1]
        non_empty_partitioning = [i for i, e in enumerate(t1) if e != 1]
        non_empty_loop = list(set().union(non_empty_blocking, non_empty_partitioning))
        non_empty_loops.append(non_empty_loop)
    return non_empty_loops


def get_loop_order(partial_order, non_empty_loops, level):
    order_curr_level = [le.NUM - 1] * le.NUM
    for i in range(len(non_empty_loops[level])):
        order_curr_level[non_empty_loops[level][i]] = partial_order[i]
    return order_curr_level


def opt_order_generator_function(point, num_loops, num_levels):
    """
    Smart loop_order_list generator.

    We need this because the general one can easily generate
    more than 10^10 loop orders

    We reduce the number of generated loop orders by only
    order the loops whose size at current level is not 1
    """
    non_empty_loops = get_non_empty_loops(point, num_levels)
    # print "non_empty_loops: ", non_empty_loops

    all_order_permutations = []
    for level in range(num_levels):
        one_level_permutations = []
        for order in itertools.permutations(range(len(non_empty_loops[level]))):
            one_level_permutations.append(get_loop_order(order, non_empty_loops, level))
        all_order_permutations.append(one_level_permutations)

    for loop_order in itertools.product(*all_order_permutations):
        yield list(zip(*loop_order))


def level_order_generator_function(point, num_loops, non_empty_loops, level):
    for order in itertools.permutations(range(len(non_empty_loops[level]))):
        yield get_loop_order(order, non_empty_loops, level)


def order_generator_function(num_loops, num_levels):
    """
    General loop_order_list generator.

    Arguments are number of loop types, number of buffer levels.
    """

    """Generator all possible loop orders in one buffer level"""
    one_level_permutations = []
    for order in itertools.permutations(range(num_loops)):
        one_level_permutations.append(order)

    all_order_permutations = []
    for level in range(num_levels):
        all_order_permutations.append(one_level_permutations)

    """Consider system with all buffer levels, generator all
       possible loop orders, then transform the data
       organization to match with loop_order_list"""
    for order in itertools.product(*all_order_permutations):
        yield list(zip(*order))


def factors(n):
    return set(
        reduce(
            list.__add__, ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0)
        )
    )


def bounded_factor(n, end):
    l = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0 and n // i <= end:
            l.__iadd__([i, n // i])
        elif n % i == 0:
            l.__iadd__([i])
    s = set(l)
    s.remove(1)
    return s


def recursive_tile(tile_permutations, curr_loop_tile, n, curr_level, num_level):
    if curr_level == num_level - 1:
        curr_loop_tile.append(n)
        tile_permutations.append(curr_loop_tile)
        return

    for i in factors(n):
        new_loop_tile = copy.copy(curr_loop_tile)
        new_loop_tile.append(i)
        recursive_tile(
            tile_permutations, new_loop_tile, n // i, curr_level + 1, num_level
        )


def loop_tile_with_para_hint(tile_permutations, loop_extent, num_level, loop_hint):
    para_hint = loop_hint[0][2]
    # TODO use faster way for this checking
    for i in factors(loop_extent):
        if i >= para_hint:
            recursive_tile(tile_permutations, [i], loop_extent / i, 1, num_level)


def loop_tile_with_hint(tile_permutations, loop_extent, num_level, loop_hint):
    # TODO support more than 1 level of para hint
    for level in range(num_level):
        if loop_hint[level] != None:
            loop_hint_level = level
            break

    # check if the hint specifies the blocking. if not, we assume blocking of 1
    blocking_hint = (
        1 if loop_hint[loop_hint_level][1] == None else loop_hint[loop_hint_level][1]
    )
    assert loop_hint[loop_hint_level][2]
    para_hint = loop_hint[loop_hint_level][2]
    # para_hint = 1 if loop_hint[loop_hint_level][2] == None else loop_hint[loop_hint_level][2]
    blocking_factor = blocking_hint * para_hint

    pre_tile_permutations = []
    if loop_hint_level == 0:
        pre_tile_permutations.append([])
    else:
        for sub_extent in factors(
            (loop_extent + blocking_factor - 1) // blocking_factor
        ):
            recursive_tile(pre_tile_permutations, [], sub_extent, 0, loop_hint_level)

    for pre_tile in pre_tile_permutations:
        # TODO support not fixed blocking hint
        if loop_hint[loop_hint_level][1]:
            pre_tile.append(blocking_factor)
            blocking_accum = reduce(mul, pre_tile, 1)
            recursive_tile(
                tile_permutations,
                pre_tile,
                (loop_extent + blocking_accum - 1) // blocking_accum,
                loop_hint_level + 1,
                num_level,
            )
        else:
            blocking_accum = reduce(mul, pre_tile, 1)
            for i in factors((loop_extent + blocking_accum - 1) // blocking_accum):
                if i >= para_hint:
                    new_pre_tile = copy.copy(pre_tile)
                    new_pre_tile.append(i)
                    new_blocking_accum = blocking_accum * i
                    recursive_tile(
                        tile_permutations,
                        new_pre_tile,
                        (loop_extent + new_blocking_accum - 1) // new_blocking_accum,
                        loop_hint_level + 1,
                        num_level,
                    )


def loop_tile(loop_extent, num_level, loop_hint=None):
    tile_permutations = []

    # check if the hint specifies the blocking at any level
    is_valid_hint = False
    if loop_hint is not None:
        for level in range(num_level):
            if loop_hint[level] is not None:
                is_valid_hint = True
                break

    if not is_valid_hint:
        recursive_tile(tile_permutations, [], loop_extent, 0, num_level)
    else:
        loop_tile_with_hint(tile_permutations, loop_extent, num_level, loop_hint)

    return tile_permutations


def opt_valid_blocking(blocking_cache, resource, layer, blocking, schedule=None):
    """
    Checks if a given blocking configuration is valid for a specific layer and resource setup.
    """
    num_levels = resource.buffer_levels()
    blocking_tuple = list(zip(*blocking))
    dummy_partitioning = [(1,) * num_levels] * le.NUM
    dummy_mapping_point = MappingPoint(None, list(blocking), dummy_partitioning)

    """
    Check if blocking fits the blocking constraints in the schedule hint
    """
    if schedule is not None and schedule.schedule_hint is not None:
        for loop_index in range(le.NUM):
            if loop_index in schedule.schedule_hint:
                for level in range(num_levels):
                    if schedule.schedule_hint[loop_index][level] is not None:
                        if (
                            schedule.schedule_hint[loop_index][level][1] is not None
                            and blocking_tuple[level][loop_index]
                            != schedule.schedule_hint[loop_index][level][1]
                        ):
                            return False

    """
   Use cache to compute valid of first level
   """
    level = 0
    value_in_cache = blocking_cache.read_cache(level, blocking_tuple[level])
    if value_in_cache == None:
        valid = cost_model.valid_blocking_size_current_level(
            resource, dummy_mapping_point, layer, level
        )
        blocking_cache.write_cache(level, blocking_tuple[level], valid)
    else:
        valid = value_in_cache
    if not valid:
        return False

    for level in range(1, num_levels):
        if not cost_model.valid_blocking_size_current_level(
            resource, dummy_mapping_point, layer, level
        ):
            return False
    return True


def blocking_generator_function(resource, layer, schedule=None, verbose=False):
    """
    Generate all possible loop tilings for each loop,
    store them in one list.
    """

    hint = schedule.schedule_hint if schedule != None else None

    num_levels = resource.buffer_levels()

    all_tile_permutations = []
    for i in range(le.NUM):
        loop_hint = hint[i] if hint and i in hint else None
        all_tile_permutations.append(loop_tile(layer.sizes[i], num_levels, loop_hint))

    """
    Generate all possible loop tilings for all loops,
    then transform the data organizations to match with loop_blocking_list
    Use cache to buffer the valid status of blocking for the first level
    """
    blocking_cache = Cache(1, 100)
    for tile in itertools.product(*all_tile_permutations):
        # TODO here the generated is a list of lists, not a list of tuples
        # if cost_model.valid_blocking_size(resource, dummy_mapping_point, layer):
        if opt_valid_blocking(blocking_cache, resource, layer, tile, schedule):
            yield list(tile)


def current_level_recursive_partition_blocking_with_hint(
    para_permutation,
    slb,
    slp,
    cur_loop,
    cur_factor,
    para_count,
    hint,
    level,
    para_loops,
):
    p = 1
    if cur_loop in hint:
        p = hint[cur_loop][level][2] if hint[cur_loop][level][2] else 1

    if cur_loop == le.NUM - 1:
        if cur_factor <= slb[le.NUM - 1]:
            slp.append(cur_factor)
            para_permutation.append(slp)
        return

    cur_loop_in_para_loops = False
    if para_loops != None:
        cur_loop_in_para_loops = cur_loop in para_loops

    if cur_loop_in_para_loops:
        for f in list(factors(cur_factor)):
            if f * p <= slb[cur_loop]:
                new_slp = copy.copy(slp)
                new_slp.append(f * p)  # TODO not exact divide case
                current_level_recursive_partition_blocking_with_hint(
                    para_permutation,
                    slb,
                    new_slp,
                    cur_loop + 1,
                    cur_factor / f,
                    para_count,
                    hint,
                    level,
                    para_loops,
                )
    else:
        new_slp = copy.copy(slp)
        new_slp.append(p)
        current_level_recursive_partition_blocking_with_hint(
            para_permutation,
            slb,
            new_slp,
            cur_loop + 1,
            cur_factor,
            para_count,
            hint,
            level,
            para_loops,
        )


def current_level_partition_blocking_1d_no_replication(
    loop_tiles, slb, para_count, layer
):
    para_permutation = []
    para_dim_permutation = []

    for l0 in range(le.NUM):
        for f0 in loop_tiles[l0]:
            slp = [
                1,
            ] * le.NUM
            slp[l0] = f0
            para_index = [l0]
            if f0 <= para_count:  # and 2*f0 > para_count:
                para_permutation.append(slp)
                para_dim_permutation.append([para_index])

    return [para_permutation, para_dim_permutation]


def current_level_partition_blocking_1d_replication(
    loop_tiles, slb, para_count, layer, u_threshold
):
    para_permutation = []
    para_dim_permutation = []

    for l0 in range(le.NUM):
        for f0 in loop_tiles[l0]:
            slp = [
                1,
            ] * le.NUM
            slp[l0] = f0
            para_index = [l0]
            if f0 <= para_count and f0 >= para_count * u_threshold:
                para_permutation.append(slp)
                para_dim_permutation.append([para_index])
            else:
                for l1 in range(le.NUM):
                    if l1 == l0:
                        continue
                    for f1 in loop_tiles[l1]:
                        if (
                            f1 * f0 >= para_count * u_threshold
                            and f1 * f0 <= para_count
                        ):
                            new_slp = copy.copy(slp)
                            new_slp[l1] = f1
                            para_permutation.append(new_slp)

                            new_para_index = copy.copy(para_index)
                            new_para_index.append(l1)
                            para_dim_permutation.append([new_para_index])

    return [para_permutation, para_dim_permutation]


def current_level_partition_blocking_1d(
    loop_tiles, slb, para_count, layer, u_threshold, replication
):
    if replication:
        return current_level_partition_blocking_1d_replication(
            loop_tiles, slb, para_count, layer, u_threshold
        )
    else:
        return current_level_partition_blocking_1d_no_replication(
            loop_tiles, slb, para_count, layer
        )


def current_level_partition_blocking_1d_with_hint(
    loop_tiles, slb, para_count, layer, level, cur_loop, schedule, u_threshold
):

    hint = schedule.schedule_hint
    partition_loops = schedule.partition_loops
    para_permutation = []
    para_dim_permutation = []
    cur_para_factor = hint[cur_loop][level][2]

    if cur_para_factor == para_count:
        slp = [
            1,
        ] * le.NUM
        slp[cur_loop] = cur_para_factor
        para_index = [cur_loop]
        para_permutation.append(slp)
        para_dim_permutation.append([para_index])

        return [para_permutation, para_dim_permutation]

    for l0 in partition_loops:
        if l0 == cur_loop:
            for f in loop_tiles[cur_loop]:
                if (
                    f * cur_para_factor >= para_count * u_threshold
                    and f * cur_para_factor <= para_count
                ):
                    slp = [
                        1,
                    ] * le.NUM
                    slp[cur_loop] = f * cur_para_factor
                    para_index = [cur_loop]
                    para_permutation.append(slp)
                    para_dim_permutation.append([para_index])
        else:
            for f in loop_tiles[l0]:
                if (
                    f * cur_para_factor >= para_count * u_threshold
                    and f * cur_para_factor <= para_count
                ):
                    slp = [
                        1,
                    ] * le.NUM
                    slp[cur_loop] = cur_para_factor
                    slp[l0] = f
                    para_index = [cur_loop, l0]
                    para_permutation.append(slp)
                    para_dim_permutation.append([para_index])
                    para_index = [l0, cur_loop]
                    para_permutation.append(slp)
                    para_dim_permutation.append([para_index])

    return [para_permutation, para_dim_permutation]


def para_index_generator_function(para_index_perm_1d):
    for e in itertools.combinations(para_index_perm_1d, 2):
        yield e


def para_index_generator_function_with_hint(para_index_perm):
    for e in itertools.product(*para_index_perm):
        yield e


def current_level_partition_blocking_2d_with_hint(
    loop_tiles, slb, para_count, layer, level, schedule, u_threshold
):
    para_permutation = []
    para_dim_permutation = []

    para_perm_1d0, para_index_perm_1d0 = current_level_partition_blocking_1d_with_hint(
        loop_tiles,
        slb,
        para_count,
        layer,
        level,
        schedule.hint_para_index[level][0],
        schedule,
        u_threshold,
    )
    para_perm_1d1, para_index_perm_1d1 = current_level_partition_blocking_1d_with_hint(
        loop_tiles,
        slb,
        para_count,
        layer,
        level,
        schedule.hint_para_index[level][1],
        schedule,
        u_threshold,
    )
    para_index_generator = para_index_generator_function_with_hint(
        [para_index_perm_1d0, para_index_perm_1d1]
    )

    for slps in itertools.product(*[para_perm_1d0, para_perm_1d1]):
        slp0, slp1 = slps
        para_index0, para_index1 = next(para_index_generator)
        if set(para_index0[0]).isdisjoint(set(para_index1[0])):
            combined_slp = [a * b for a, b in list(zip(slp0, slp1))]
            para_permutation.append(combined_slp)
            combined_dim = [para_index0[0], para_index1[0]]
            para_dim_permutation.append(combined_dim)

    return [para_permutation, para_dim_permutation]


def current_level_partition_blocking_2d(
    loop_tiles, slb, para_count, layer, u_threshold, replication
):
    para_permutation = []
    para_dim_permutation = []

    para_perm_1d, para_index_perm_1d = current_level_partition_blocking_1d(
        loop_tiles, slb, para_count, layer, u_threshold, replication
    )
    para_index_generator = para_index_generator_function(para_index_perm_1d)

    for slps in itertools.combinations(para_perm_1d, 2):
        slp0, slp1 = slps
        para_index0, para_index1 = next(para_index_generator)
        if set(para_index0[0]).isdisjoint(set(para_index1[0])):
            combined_slp = [a * b for a, b in list(zip(slp0, slp1))]
            para_permutation.append(combined_slp)
            combined_dim = [para_index0[0], para_index1[0]]
            para_dim_permutation.append(combined_dim)

    return [para_permutation, para_dim_permutation]


def current_level_partition_blocking(slb, para, layer, u_threshold, replication):
    para_count = para.array_width
    loop_tiles = []
    for l in range(le.NUM):
        loop_tiles.append(bounded_factor(slb[l], para_count))

        # print "loop tile ", loop_tiles
    if para.array_dim == 1:
        return current_level_partition_blocking_1d(
            loop_tiles, slb, para_count, layer, u_threshold, replication
        )
    else:
        return current_level_partition_blocking_2d(
            loop_tiles, slb, para_count, layer, u_threshold, replication
        )


def current_level_partition_blocking_with_hint(
    slb, para, layer, level, schedule, u_threshold
):
    para_count = para.array_width
    loop_tiles = []
    for l in range(le.NUM):
        loop_tiles.append(bounded_factor(slb[l], para_count))

        # print "loop tile ", loop_tiles
    if para.array_dim == 1:
        assert (
            len(schedule.hint_para_index[level]) <= 1
        ), "do not support unrolling more than 2 loops in the schedule hint"
        return current_level_partition_blocking_1d_with_hint(
            loop_tiles,
            slb,
            para_count,
            layer,
            level,
            schedule.hint_para_index[level][0],
            schedule,
            u_threshold,
        )
    else:
        assert (
            len(schedule.hint_para_index[level]) <= 2
        ), "do not support unrolling more than 2 loops in the schedule hint"
        return current_level_partition_blocking_2d_with_hint(
            loop_tiles, slb, para_count, layer, level, schedule, u_threshold
        )


def para_dim_generator_function(para_dim_permutations):
    for para_dim in itertools.product(*para_dim_permutations):
        yield para_dim


def parallel_blocking_generator_function(lp, resource, layer, schedule=None):
    num_level = resource.buffer_levels()

    para_permutations = []
    para_dim_permutations = []
    for level in range(num_level):
        if resource.paras[level].count == 1:
            para_permutations.append([[1] * le.NUM])
            para_dim_permutations.append([None])
        else:
            para = resource.paras[level]
            para_count = para.array_width
            if schedule == None:
                # current_level_recursive_partition_blocking(para_permutation, lp[level], [], 0, para.count, para.count, layer, under_utilized)
                para_permutation, para_dim_permutation = (
                    current_level_partition_blocking(
                        lp[level],
                        para,
                        layer,
                        resource.utilization_threshold,
                        resource.replication,
                    )
                )
                para_permutations.append(para_permutation)
                para_dim_permutations.append(para_dim_permutation)
            else:
                hinted_para = get_hinted_para(layer, level, schedule.schedule_hint)
                assert (
                    hinted_para <= para.count
                ), "total parallelism in schedule hint exceeds the maximum parallelism"
                if para.count >= hinted_para * 2:
                    new_para_count = para.count / hinted_para
                    para_permutation, para_dim_permutation = (
                        current_level_partition_blocking_with_hint(
                            lp[level],
                            para,
                            layer,
                            level,
                            schedule,
                            resource.utilization_threshold,
                        )
                    )
                    para_permutations.append(para_permutation)
                    para_dim_permutations.append(para_dim_permutation)
                else:
                    para_permutation, para_dim_permutation = get_hinted_partitioning(
                        level, schedule.schedule_hint
                    )
                    para_permutations.append(para_permutation)
                    para_dim_permutations.append(para_dim_permutation)

    # print para_permutations
    # print para_dim_permutations

    para_dim_generator = para_dim_generator_function(para_dim_permutations)
    for partition in itertools.product(*para_permutations):
        para_dim = next(para_dim_generator)
        # print partition, para_dim
        yield [partition, para_dim]


def blocking_partitioning_generator_function(resource, layer, schedule, verbose=False):
    """
    loop_blocking_list and loop_partitioning_list generator.

    loop_blocking: [[Total size (temporal+spatial) of Fx @ mem level 0,1,2],[Fy],[OX],[OY],[OC],[IC],[ON]]
    loop_blocking_reshape: [(All loops' total size (temporal+spatial) @ mem level 0),(@ level 1),(@ level 2)]

    partition: [[All loops' spatial unrolled size @ mem level 0],[@ level 1],[@ level 2]]
    para_dim: [[Spatial unrolled loop dimensions @ mem level 0],[@ level 1],[@ level 2]]

    partitioned_loop_blocking_reshape: [[All loops' temporal unrolled size @ mem level 0],[@ level 1],[@ level 2]]
    blocking_list: [[Temporal unrolled size of Fx @ mem level 0,1,2],[Fy],[OX],[OY],[OC],[IC],[ON]]
    partitioning_list: [[Spatial unrolled size of Fx @ mem level 0,1,2],[Fy],[OX],[OY],[OC],[IC],[ON]]
    """

    num_level = resource.buffer_levels()
    blocking_generator = blocking_generator_function(resource, layer, schedule, verbose)

    for loop_blocking in blocking_generator:
        if verbose == 3:
            print("loop_tilling: ", loop_blocking)

        loop_blocking_reshape = list(zip(*loop_blocking))
        pb_generator = parallel_blocking_generator_function(
            loop_blocking_reshape, resource, layer, schedule
        )

        for pi in pb_generator:
            partition, para_dim = pi
            partitioned_loop_blocking_reshape = []
            for level in range(num_level):
                partitioned_loop_blocking_reshape.append(
                    [
                        (x + y - 1) // y
                        for x, y in list(
                            zip(loop_blocking_reshape[level], partition[level])
                        )
                    ]
                )  # TODO check if using two maps with floordiv is faster
            blocking_list = list(zip(*partitioned_loop_blocking_reshape))
            partitioning_list = list(zip(*partition))

            if verbose == 3:
                print("loop_blocking: ", blocking_list)
                print("loop_partition: ", partitioning_list)
                print("para_dimension: ", para_dim)

            dummy_mapping_point = MappingPoint(
                None, blocking_list, partitioning_list, para_dim
            )
            if cost_model.valid_partitioning(
                resource, dummy_mapping_point, layer, verbose
            ):
                #                if cost_model.valid_mapping_point(resource, dummy_mapping_point, layer, verbose):
                if verbose == 3:
                    print("Valid")
                    print("")
                yield [blocking_list, partitioning_list, para_dim]
            #                else:
            #                   print "invalid"
            #                    print ""
            else:
                if verbose == 3:
                    print("invalid")
                    print("")


def opt_get_best_loop_order(resource, layer, point, schedule=None, verbose=False):
    """

    [HW template right now: systolic array]
    [SRAM only talks to the PEs on the edge, most PE will get data from its neighbour PE]

    When there is no partitioning (parallelism), the cost of the current level only depends on the current
    level loop orders, given the blocking factors. Thus we can leverage this to
    find the best loop order for each level individually.

    When there is partitioning (parallelism),
    the # of times that the paralleled level of memory passing data to its neighbour PE
    (corresponding to the energy spent on interconnection, array_cost)
    depends on the current level parallelism size & memory access from the above level memory

    The lowest level memory access (talk to MAC) only depends on the NN layer size

    level access: [input, weight, output] # of element
    level order: [fx, fy, ox, oy, oc, ic, on], '0' for innermost loop, '6' for outermost loop / non-existed loop

    """
    num_levels = resource.buffer_levels()
    best_loop_order = []
    blocking = point.loop_blockings
    partitioning = point.loop_partitionings
    para_dim = point.para_loop_dim

    non_empty_loops = get_non_empty_loops(point, num_levels)
    # print blocking, partitioning

    best_cost = 0
    para_level = 0
    for level in range(num_levels):
        smallest_cost = float("inf")
        # LMEI later, might can speed up the exhaustive order search by identifying symmetrical terms,
        #  e.g. ox and oy, fx and fx, to remove some similar orders

        for curr_level_order in level_order_generator_function(
            point, le.NUM, non_empty_loops, level
        ):
            dummy_loop_order = [[0] * le.NUM] * num_levels
            dummy_loop_order[level] = curr_level_order
            mapping_point = MappingPoint(
                list(zip(*dummy_loop_order)), blocking, partitioning, para_dim
            )

            # check mapping point fits the schedule hint constraint
            valid_mapping_point = True
            if schedule is not None and schedule.schedule_hint is not None:
                for loop_index in range(le.NUM):
                    # if there's a blocking of 1, we can ignore the loop order constraint
                    if blocking[loop_index][level] == 1:
                        continue

                    if loop_index in schedule.schedule_hint:
                        if (
                            schedule.schedule_hint[loop_index][level] is not None
                            and schedule.schedule_hint[loop_index][level][0] is not None
                        ):
                            loop_index_constraint = schedule.schedule_hint[loop_index][
                                level
                            ][0]

                            # for negative value, it means we should count from the outermost loop
                            # e.g. -1 means the outermost loop
                            if loop_index_constraint < 0:
                                # find the max loop index (that is not 6) in the current level order
                                max_loop_index = max(
                                    [i if i != 6 else -1 for i in curr_level_order]
                                )
                                loop_index_constraint = (
                                    max_loop_index + 1 + loop_index_constraint
                                )

                            # if the loop index is not the same as the constraint, then it's not a valid mapping point
                            if curr_level_order[loop_index] != loop_index_constraint:
                                valid_mapping_point = False
                                break

            if not valid_mapping_point:
                continue

            if (
                level <= 0
                or resource.paras[level - 1].count <= 1
                or resource.paras[level - 1].access_mode < 1
            ):  # don't get it
                curr_cost = cost_model.get_level_cost(
                    resource, mapping_point, layer, level, verbose
                )
            else:
                curr_cost = cost_model.get_array_and_curr_level_cost(
                    resource, mapping_point, layer, level, verbose
                )
            if curr_cost < smallest_cost:
                best_curr_level_order = curr_level_order
                smallest_cost = curr_cost

            if verbose >= 3:
                print(
                    "Level",
                    level,
                    "Current order:",
                    curr_level_order,
                    "     Best order:",
                    best_curr_level_order,
                )
                print(
                    "Level",
                    level,
                    "Current energy:",
                    "%20d" % curr_cost,
                    "     Best energy:",
                    "%20d" % smallest_cost,
                )
                print("")

            # LMEI later, instead of using mac_capacity, we could use 4-level memory model, treat mac_capacity
            #  as the innermost memory level for output.
            if resource.mac_capacity == 0 and level == 0:
                break  # Here the author thinks the loop order in innermost level doesn't matter, thus break

        best_loop_order.append(best_curr_level_order)
        best_cost += smallest_cost

    return best_cost, list(zip(*best_loop_order))


def opt_get_loop_order_generator(resource, layer, point, schedule=None, verbose=False):
    """
    Generator version of opt_get_best_loop_order.

    Yields each valid mapping point instead of calculating the cost and returning the best loop order.
    """
    num_levels = resource.buffer_levels()
    blocking = point.loop_blockings
    partitioning = point.loop_partitionings
    para_dim = point.para_loop_dim
    non_empty_loops = get_non_empty_loops(point, num_levels)

    # Iterate through all possible combinations of loop orders for all levels.
    level_orders = [
        list(level_order_generator_function(point, le.NUM, non_empty_loops, level))
        for level in range(num_levels)
    ]

    for level_order_combination in itertools.product(*level_orders):
        dummy_loop_order = list(level_order_combination)
        mapping_point = MappingPoint(
            list(zip(*dummy_loop_order)), blocking, partitioning, para_dim
        )

        # check mapping point fits the schedule hint constraint
        valid_mapping_point = True
        if schedule is not None and schedule.schedule_hint is not None:
            for level in range(num_levels):
                for loop_index in range(le.NUM):
                    if blocking[loop_index][level] == 1:
                        continue

                    if loop_index in schedule.schedule_hint:
                        if (
                            schedule.schedule_hint[loop_index][level] is not None
                            and schedule.schedule_hint[loop_index][level][0] is not None
                        ):
                            loop_index_constraint = schedule.schedule_hint[loop_index][
                                level
                            ][0]

                            if loop_index_constraint < 0:
                                max_loop_index = max(
                                    [
                                        i if i != 6 else -1
                                        for i in level_order_combination[level]
                                    ]
                                )
                                loop_index_constraint = (
                                    max_loop_index + 1 + loop_index_constraint
                                )

                            if (
                                level_order_combination[level][loop_index]
                                != loop_index_constraint
                            ):
                                valid_mapping_point = False
                                break
                if not valid_mapping_point:
                    break
        if not valid_mapping_point:
            continue
        yield mapping_point


def opt_mapping_point_generator_function(
    resource, layer, schedule=None, runtime_calc_func=None, verbose=False
):
    """
    Mapping point generator.

    Generates a new mapping point each iteration.
    """
    num_levels = resource.buffer_levels()
    parallel_levels = resource.para_index
    ideal_perf = cost_model.get_ideal_performance(layer, resource)
    blocking_partitioning_generator = blocking_partitioning_generator_function(
        resource, layer, schedule
    )

    # dummy_partitioning = [(1,) * num_levels] * le.NUM

    smallest_cost = float("inf")
    smallest_runtime = float("inf")
    best_mapping_point = None
    for blocking_partitioning in blocking_partitioning_generator:
        """
        dummy_mapping_point is used to validate the current blocking_partitioning,
        and abandon the ones that exceed the buffer size at any level.
        Since this validation does not depend on loop_orders, we perform the validation
        at this early stage, so that we can avoid generating all the loop orders for
        an invalid blocking_partitioning
        """
        if verbose >= 2:
            print("Find best order for schedule: ", blocking_partitioning)
        [blocking, partitioning, para_dim] = blocking_partitioning
        dummy_mapping_point = MappingPoint(None, blocking, partitioning, para_dim)
        # print "blocking_partitioning: ", blocking_partitioning
        for mapping_point in opt_get_loop_order_generator(
            resource, layer, dummy_mapping_point, schedule, verbose
        ):
            if runtime_calc_func:
                runtime = runtime_calc_func(
                    resource,
                    layer,
                    mapping_point,
                )
            else:
                runtime = float("inf")

            cost = 0
            for level in range(num_levels):
                cost += cost_model.get_level_cost(
                    resource,
                    mapping_point,
                    layer,
                    level,
                    verbose,
                )

            # optimize for runtime first, then for energy
            if runtime < smallest_runtime or (
                runtime == smallest_runtime and cost < smallest_cost
            ):
                smallest_runtime = runtime
                smallest_cost = cost
                best_mapping_point = mapping_point
                unrolled_loops, utilized = partitioned_loop_string(
                    partitioning, parallel_levels, para_dim
                )
                utilization = get_utilization(utilized, resource)
                perf = ideal_perf / utilization

                if verbose >= 2:
                    print("best loop order: ", best_mapping_point.loop_orders)
                    print("Update smallest cost: ", smallest_cost)
                    print(
                        "Update best schedule: ",
                        utils.print_loop_nest(best_mapping_point),
                    )
    assert best_mapping_point, "No valid mapping point found."
    return smallest_cost, smallest_runtime, perf, best_mapping_point


def mapping_point_generator_function(resource, layer, schedule=None, verbose=False):
    """
    Mapping point generator.

    Generates a new mapping point each iteration.
    """

    num_levels = resource.buffer_levels()

    blocking_partitioning_generator = blocking_partitioning_generator_function(
        resource, layer, schedule
    )

    for blocking_partitioning in blocking_partitioning_generator:
        """
        dummy_mapping_point is used to validate the current blocking_partitioning,
        and abandon the ones that exceed the buffer size at any level.
        Since this validation does not depend on loop_orders, we perform the validation
        at this early stage, so that we can avoid generating all the loop orders for
        an invalid blocking_partitioning
        """
        [blocking, partitioning] = blocking_partitioning
        dummy_mapping_point = MappingPoint(None, blocking, partitioning)
        # print "blocking_partitioning: ", blocking_partitioning
        if cost_model.valid_mapping_point(
            resource, dummy_mapping_point, layer, verbose
        ):
            # opt_order_generator_function(dummy_mapping_point, le.NUM, num_levels)
            order_generator = opt_order_generator_function(
                dummy_mapping_point, le.NUM, num_levels
            )
            for loop_order in order_generator:
                mapping_point = MappingPoint(loop_order, blocking, partitioning)
                yield mapping_point


def partitioned_loop_string(partitioning, parallel_levels, para_dim):
    # TODO check for multi-level parallel case
    res = ""

    utilized = 1
    partitioning_reshape = list(zip(*partitioning))
    for level in parallel_levels:
        for para_idx in para_dim[level]:
            res += "("
            for loop in para_idx:
                e = partitioning_reshape[level][loop]
                utilized *= e
                res += str(loop)
            res += ")"
    return [res, utilized]


def get_utilization(utilized, resource):
    # utilized = 1
    # for i in range(len(partitioning)):
    #    utilized *= reduce(mul, partitioning[i], 1)

    total = resource.total_parallelism()

    return utilized * 1.0 / total


def dataflow_exploration(resource, layer, file_name, verbose=False):
    """
    Dataflow exploration.

    Generates a table, with unrolled loops being keys, the best energy (and utilization)
    being the values.
    """

    dataflow_tb = {}
    num_levels = resource.buffer_levels()
    parallel_levels = resource.para_index

    blocking_partitioning_generator = blocking_partitioning_generator_function(
        resource, layer, None
    )

    # dummy_partitioning = [(1,) * num_levels] * le.NUM

    smallest_cost = float("inf")
    # best_mapping_point = None
    for blocking_partitioning in blocking_partitioning_generator:
        """
        dummy_mapping_point is used to validate the current blocking_partitioning,
        and abandon the ones that exceed the buffer size at any level.
        Since this validation does not depend on loop_orders, we perform the validation
        at this early stage, so that we can avoid generating all the loop orders for
        an invalid blocking_partitioning
        """
        if verbose >= 2:
            print("Find best order for schedule: ", blocking_partitioning)
        [blocking, partitioning, para_dim] = blocking_partitioning
        dummy_mapping_point = MappingPoint(None, blocking, partitioning, para_dim)
        # print "partitioning: ", partitioning
        unrolled_loops, utilized = partitioned_loop_string(
            partitioning, parallel_levels, para_dim
        )
        utilization = get_utilization(utilized, resource)
        if resource.replication and utilization < resource.utilization_threshold:
            continue
        cost, loop_order = opt_get_best_loop_order(
            resource, layer, dummy_mapping_point, verbose
        )
        if unrolled_loops not in dataflow_tb or dataflow_tb[unrolled_loops][0] > cost:
            best_mapping_point = MappingPoint(
                loop_order, blocking, partitioning, para_dim
            )
            dataflow_tb[unrolled_loops] = (
                cost,
                utilization,
                best_mapping_point,
            )  # TODO utilization
            if verbose:
                print(
                    "unrolled loops: ",
                    unrolled_loops,
                    " with utilization ",
                    utilization,
                )
                # print "best loop order: ", best_mapping_point.loop_orders
                print("blocking: ", blocking)
                print("partitioning: ", partitioning)
                print("Update smallest cost: ", dataflow_tb[unrolled_loops][0])
                # print "Update best shedule: ", utils.print_loop_nest(best_mapping_point)
    # assert best_mapping_point, "No valid mapping point found."
    pickle_file_name = file_name + ".pickle"
    pickle.dump(dataflow_tb, open(pickle_file_name, "wb"))
    return dataflow_tb
