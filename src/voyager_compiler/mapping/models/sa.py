# Model systolic-array service without depending on the tiler command line
import re

import interstellar


# Return a compiler dtype's scalar width
def get_dtype_width(dtype: str):
    bit_search = re.search(r"[^\d](\d+)(_.*)?$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    return int(bit_search.groups()[0])


# Estimate systolic-array tile overlap for one compiler operation
class RuntimeCalculator:
    # Bind operation metadata and accumulation banking
    def __init__(self, operation, double_buffered_accum_buffer):
        self.operation = operation
        self.double_buffered_accum_buffer = double_buffered_accum_buffer

    # Estimate compute, operand-loading, and output-service cycles
    def calculate_runtime(self, architecture, layer, mapping):
        # assume IC is unrolled vertically
        # does not handle replication
        sa_weight_loading_time = mapping.loop_partitionings[interstellar.le.IC][0] + 2

        # index of first loop that isn't OX or OY
        first_non_ox_oy_index = 6
        for i in range(interstellar.le.NUM):
            if i == interstellar.le.OX or i == interstellar.le.OY:
                continue
            if mapping.loop_orders[i][1] < first_non_ox_oy_index:
                first_non_ox_oy_index = mapping.loop_orders[i][1]

        # calculate how big a weight reuse tile is
        # weights are reused in the OX and OY loops, so a weight reuse tile is product of
        # all loops from the first loop to the first non-(OX or OY) loop
        weight_reuse_tile_size = 1
        for i in range(interstellar.le.NUM):
            if mapping.loop_orders[i][1] < first_non_ox_oy_index:
                weight_reuse_tile_size *= mapping.loop_blockings[i][1]

        # calculate how long it takes to compute a weight_reuse_tile
        # this is the max of time to load the weights into the systolic array and weight reuse tile size
        weight_reuse_tile_time = max(sa_weight_loading_time, weight_reuse_tile_size)

        # calculate number of remaining L1 tiles
        num_remaining_l1_tiles = 1
        for i in range(interstellar.le.NUM):
            if mapping.loop_orders[i][1] >= first_non_ox_oy_index:
                num_remaining_l1_tiles *= mapping.loop_blockings[i][1]
        # include the reduction loop at the L2 level
        num_remaining_l1_tiles *= mapping.loop_blockings[interstellar.le.IC][2]

        # calculate time for computation at the L1 level
        computation_l1_time = weight_reuse_tile_time * num_remaining_l1_tiles

        input_relevant_loops = [
            interstellar.le.IC,
            interstellar.le.OY,
            interstellar.le.OX,
        ]
        input_buffer_loading_size = 1
        for loop in input_relevant_loops:
            input_buffer_loading_size *= mapping.loop_blockings[loop][1]
        # currently assume that each value in the input buffer is loaded in one cycle
        input_buffer_loading_time = input_buffer_loading_size

        if self.operation.WhichOneof("op_type") == "op":
            matrix_op = self.operation.op
        else:
            matrix_op = self.operation.fused_op.op_list[0]
        input_dtype = matrix_op.kwargs["input"].tensor.dtype
        input_dtype_width = get_dtype_width(input_dtype)

        if self.operation.HasField("output"):
            output_dtype = self.operation.output.dtype
        else:
            output_dtype = self.operation.outputs.tensors[1].dtype
        output_dtype_width = get_dtype_width(output_dtype)

        weight_relevant_loops = [
            interstellar.le.IC,
            interstellar.le.OC,
            interstellar.le.FY,
            interstellar.le.FX,
        ]
        weight_buffer_loading_size = 1
        for loop in weight_relevant_loops:
            weight_buffer_loading_size *= mapping.loop_blockings[loop][1]
        # include the unrolled reduction loop
        weight_buffer_loading_size *= mapping.loop_partitionings[interstellar.le.IC][0]
        # currently assume that each value in the weight buffer is loaded in one cycle
        weight_buffer_loading_time = weight_buffer_loading_size

        # calculate time for vector unit to process values from the accumulation buffer
        # for now, let's assume that all vector unit operations flow through the vector unit pipeline once
        # the only thing that matters then is if the other operand or the outputs are in high precision, which will 2x the time
        output_relevant_loops = [
            interstellar.le.OC,
            interstellar.le.OY,
            interstellar.le.OX,
        ]
        output_size = 1
        for loop in output_relevant_loops:
            output_size *= mapping.loop_blockings[loop][1]
        vector_unit_time = output_size

        requires_high_precision = False
        if output_dtype_width > input_dtype_width:
            requires_high_precision = True
        elif self.operation.WhichOneof("op_type") == "fused_op":
            # check if any of the operands in fused ops are in high precision
            for vector_op in self.operation.fused_op.op_list[1:]:
                for arg in vector_op.kwargs.values():
                    if not arg.HasField("tensor") or not arg.tensor.HasField("memory"):
                        continue

                    dtype_width = get_dtype_width(arg.tensor.dtype)
                    if dtype_width > input_dtype_width:
                        requires_high_precision = True
                        break

        if requires_high_precision:
            vector_unit_time *= 2

        using_double_buffer_accum_buffer = (
            self.double_buffered_accum_buffer and requires_high_precision
        )

        if not using_double_buffer_accum_buffer:
            # if we are not using double buffered accumulation buffer, the vector unit time should not factor in to the computation of l1_time
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
        for i in range(interstellar.le.NUM):
            # don't count the reduction loop here, since it's counted towards the number of L1 tiles
            if i == interstellar.le.IC:
                continue
            l2_blocks *= mapping.loop_blockings[i][2]

        if self.double_buffered_accum_buffer:
            total_time = (
                # initial buffer loading time
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                # time for last tile to be processed by vector unit
                + vector_unit_time
            )
        else:
            # if there's no double buffered accumulation buffer, we need to account for any extra time taken by the vector unit
            # this extra time is any stalling that occurs when the vector unit uses high precision
            if requires_high_precision:
                extra_vector_unit_time = output_size
            else:
                extra_vector_unit_time = 0

            total_time = (
                # initial buffer loading time
                max(input_buffer_loading_time, weight_buffer_loading_time)
                + l2_blocks * l1_time
                + extra_vector_unit_time
            )

        return total_time
