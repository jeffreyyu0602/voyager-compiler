from . import loop_enum as le
from . import buffer_enum as be


def get_loop_nest(point):
    loop_orders = list(zip(*point.loop_orders))
    loop_blockings = list(zip(*point.loop_blockings))
    loop_partitionings = list(zip(*point.loop_partitionings))
    para_dims = point.para_loop_dim
    num_level = len(loop_orders)
    order_lists = []
    for level in range(num_level):
        order_list = [None] * le.NUM
        for order in range(le.NUM):
            if (
                loop_blockings[level][order] != 1
                or loop_partitionings[level][order] != 1
            ):
                order_list[loop_orders[level][order]] = (
                    le.table[order],
                    loop_blockings[level][order],
                    loop_partitionings[level][order],
                )

        order_lists.append(order_list)

    return order_lists, para_dims


def format_tiling(point) -> str:
    order_lists, _ = get_loop_nest(point)

    bottom_up_prints = []

    for level in order_lists:
        for loops in level:
            if loops is not None:
                if loops[2] == 1:
                    bottom_up_prints.append(
                        f"for {loops[0]} in range({int(loops[1])}):"
                    )
                else:
                    bottom_up_prints.append(
                        f"parallel_for {loops[0]}"
                        f" in range({int(loops[2])}):"
                    )
            else:
                bottom_up_prints.append("")
                break

    lines = []
    space_count = 0
    for i in range(len(bottom_up_prints) - 1, -1, -1):
        if bottom_up_prints[i] == "":
            lines.append("")
        else:
            lines.append(("  " * space_count) + bottom_up_prints[i])
            space_count += 2
    return "\n".join(lines)


def print_tiling(point):
    print(format_tiling(point))


def print_loop_nest(point):
    order_lists, para_dims = get_loop_nest(point)

    print(order_lists, para_dims)


def print_best_schedule(point):
    loop_orders = list(zip(*point.loop_orders))
    loop_blockings = list(zip(*point.loop_blockings))
    loop_partitionings = list(zip(*point.loop_partitionings))
    para_dims = point.para_loop_dim
    num_level = len(loop_orders)
    order_lists = []
    for level in range(num_level):
        print("\tLevel_Number: {}".format(level))
        order_list = [None] * le.NUM
        for order in range(le.NUM):
            if (
                loop_blockings[level][order] != 1
                or loop_partitionings[level][order] != 1
            ):
                order_list[loop_orders[level][order]] = (
                    le.table[order],
                    loop_blockings[level][order],
                    loop_partitionings[level][order],
                )
                print(
                    "\t\tLoop_Name: {}, Loop_Bound: {}, Unrolling: {}".format(
                        le.table[order],
                        loop_blockings[level][order],
                        loop_partitionings[level][order],
                    )
                )

        order_lists.append(order_list)

    # print(order_lists)
