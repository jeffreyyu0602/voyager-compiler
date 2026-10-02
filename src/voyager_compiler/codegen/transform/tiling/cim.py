"""CIM schedule checks and weight reuse estimates for Interstellar.

The weight policy follows CIMWeightController::reader: innermost spatial
loops reuse one weight set, enclosing spatial loops repeat a stored weight
sequence, and a sequence that does not fit loads one set per descriptor. Counts
describe one L2 command (one step of the compiler's L3 grid).
"""

import math
from dataclasses import dataclass
from math import prod

import interstellar
from voyager_compiler.codegen.transform.tiling.input import input_buffer_usage
from voyager_compiler.codegen.transform.tiling.sa import RuntimeCalculator

le = interstellar.le
SPATIAL = (le.OX, le.OY)
WEIGHTS = (le.IC, le.OC, le.FX, le.FY)
REDUCTIONS = (le.IC, le.FX, le.FY)
OUTPUTS = (le.OC, le.OY, le.OX)


def build_architecture(config, dram_access_cost):
    """Keep the common four levels while freeing CIM's L1/L2 loop orders."""
    if (
        not config.usable_scratchpad_size
        or not config.num_banks
        or not config.bank_width
        or not config.dram_size
    ):
        raise ValueError(
            "CIM tiling requires scratchpad_size, num_banks, bank_width, "
            "and dram_size"
        )
    ic, oc = config.pe_array_size
    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            # Weight capacity is enforced by the residency policy, not by
            # rejecting a streamed sequence as an oversized SRAM tile.
            [
                config.input_buffer_size * ic,
                config.accum_buffer_size * oc,
                float("inf"),
            ],
            [config.usable_scratchpad_size],
            [config.dram_size * 1024**3],
        ],
        buf_access_cost_list=[
            [1, 1, 1], [10, 10, 10], [100], [dram_access_cost]
        ],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
        para_count_list=[ic * oc, 1, 1, 1],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
        mac_capacity=0,
        partition_mode=[0, 0, 0, 0],
        invalid_underutilized=False,
        bank_size_list=[None, None, config.bank_size, None],
    )
    hints = {}
    for loop in range(le.NUM):
        lanes = ic if loop == le.IC else oc if loop == le.OC else 1
        # L0 has fixed physical lanes and no temporal work. Keep filters
        # inside L1 and the L3 reduction innermost, as required by the current
        # bufferized convolution/GEMM grid. L1 and L2 orders are otherwise free.
        hints[loop] = [
            [
                1 if loop == le.IC else 0 if loop == le.OC else None,
                None if lanes > 1 else 1,
                lanes,
            ],
            [None, None, 1],
            [None, 1 if loop in (le.FX, le.FY) else None, 1],
            [
                0 if loop == le.IC else None,
                1 if loop in (le.FX, le.FY) else None,
                1,
            ],
        ]
    return architecture, interstellar.Schedule(hints)


@dataclass(frozen=True)
class WeightPolicy:
    """Weight loads, reuse, and descriptor counts for a schedule."""

    reader_l1: tuple
    reader_l2: tuple
    fits: bool
    sequence_sets: int
    compute_replays: int
    fetch_replays: int
    sequence_count: int
    macs_per_set_use: int

    @property
    def full_set_loads(self):
        return self.sequence_count * self.sequence_sets * self.fetch_replays

    @property
    def set_uses(self):
        return self.sequence_count * self.sequence_sets * self.compute_replays

    @property
    def descriptors(self):
        return self.sequence_count if self.fits else self.full_set_loads


def weight_policy(config, mapping):
    """Compute weight loads and reuse from the controller's loop order."""
    b, order = mapping.loop_blockings, mapping.loop_orders
    l1 = [row[1] for row in b]
    l2 = [row[2] for row in b]
    active = [w for w in WEIGHTS if l1[w] > 1]
    reuse = [
        s for s in SPATIAL if all(order[s][1] < order[w][1] for w in active)
    ]
    replay = [
        s for s in SPATIAL
        if s not in reuse and l1[s] > 1
        and all(order[w][1] < order[s][1] for w in active)
    ]
    reader_l1 = [1 if d in reuse + replay else l1[d] for d in range(le.NUM)]
    if prod(reader_l1) > config.cim_weight_sets:
        for d in replay:
            reader_l1[d] = l1[d]
        replay = []
    active = [w for w in (le.IC, le.OC, le.FY) if l2[w] > 1]
    outer_reuse = [
        s for s in SPATIAL if all(order[s][2] < order[w][2] for w in active)
    ]
    reader_l2 = [1 if d in outer_reuse else l2[d] for d in range(le.NUM)]
    replays = prod(l1[d] for d in replay) * prod(l2[d] for d in outer_reuse)
    sets = prod(reader_l1)
    fits = sets <= config.cim_weight_sets
    return WeightPolicy(
        tuple(reader_l1),
        tuple(reader_l2),
        fits,
        sets,
        replays,
        1 if fits else replays,
        prod(reader_l2),
        prod(l1[d] for d in reuse),
    )


@dataclass(frozen=True)
class Evaluation:
    """Candidate constraints and work counts for one hardware command."""

    reasons: tuple
    policy: WeightPolicy
    input_words: int
    accum_words: int
    local_contexts: int
    mac_requests: int
    output_vectors: int
    buffer_partial_updates: int
    input_fill_words: int
    weight_write_beats: int

    @property
    def legal(self):
        return not self.reasons


def evaluate(config, layer, mapping, *, banked_output=False):
    """Check a mapping against CIM buffer and controller limits.

    The search keeps complete filter loops at L1. Partial sums beyond the
    local register capacity use the accumulation SRAM.
    """
    b, p, order = (
        mapping.loop_blockings,
        mapping.loop_partitionings,
        mapping.loop_orders,
    )
    ic, oc = config.pe_array_size
    input_words, reasons = input_buffer_usage(
        mapping, (layer.hstd, layer.wstd), config.input_buffer_size
    )
    for d in range(le.NUM):
        expected = (ic if d == le.IC else oc if d == le.OC else 1, 1, 1, 1)
        if tuple(p[d]) != expected or b[d][0] != 1:
            reasons.append(
                "CIM requires fixed IC/OC lanes and unit L0 temporal factors"
            )
            break
    if any(n != 1 for n in b[le.ON]):
        reasons.append("batch must be handled outside the CIM command")
    if any(b[d][level] != 1 for d in (le.FX, le.FY) for level in (2, 3)):
        reasons.append("CIM mapping keeps complete filters at L1")
    if any(prod(b[d]) * prod(p[d]) != layer.sizes[d] for d in range(le.NUM)):
        reasons.append("mapping must cover the padded workload exactly")
    if any(b[d][level] > 1023 for d in range(le.NUM) for level in (1, 2)):
        reasons.append("temporal factor exceeds the 10-bit controller bound")
    macs = prod(b[d][level] for d in range(le.NUM) for level in (1, 2))
    if macs > 0xFFFFFFFF:
        reasons.append("operation count exceeds the 32-bit controller counter")

    def outer_context(d):
        return any(
            b[r][2] > 1 and order[d][2] < order[r][2]
            for r in (le.IC, le.FY)
        )

    if b[le.OC][2] > 1 and outer_context(le.OC):
        reasons.append("outer output-column partial contexts are unsupported")
    if config.double_buffered_accum_buffer and banked_output and any(
        b[d][2] > 1 and outer_context(d) for d in SPATIAL
    ):
        reasons.append(
            "outer partial contexts cannot use banked accumulation output"
        )
    accum_words = b[le.OC][1] * prod(
        b[d][1] * (b[d][2] if outer_context(d) else 1) for d in SPATIAL
    )
    if accum_words > config.accum_buffer_size:
        reasons.append("live partial outputs exceed accumulation capacity")
    policy = weight_policy(config, mapping)
    if policy.compute_replays > 65535:
        reasons.append("replay count exceeds the 16-bit descriptor field")

    # Count output coordinates enclosed by any non-unit reduction, matching
    # CIMProcessor::local_accum_layout. Remaining outputs use SRAM feedback.
    contexts = prod(
        b[d][level]
        for level in (1, 2) for d in OUTPUTS
        if any(
            b[r][outer] > 1 and (
                outer > level
                or (outer == level and order[d][level] < order[r][level])
            )
            for outer in (1, 2) for r in REDUCTIONS
        )
    )
    outputs = prod(b[d][level] for level in (1, 2) for d in OUTPUTS)
    reductions = prod(b[d][level] for level in (1, 2) for d in REDUCTIONS)
    spilled_outputs = outputs // contexts * max(
        0, contexts - config.cim_local_accum_contexts
    )
    # InputController loads an input tile, including its halo, on each L2
    # iteration, even when the weights remain loaded in the CIM array.
    input_fills = prod(b[d][2] for d in range(le.NUM))
    return Evaluation(
        tuple(reasons),
        policy,
        input_words,
        accum_words,
        contexts,
        macs,
        outputs,
        spilled_outputs * (reductions - 1),
        input_words * input_fills,
        policy.full_set_loads * ic
        * config.cim_output_axis_tiles // config.cim_b_port_tiles,
    )


def issue_window(config, input_width):
    """Minimum element issue spacing from CIMElement::issue_window()."""
    if config.cim_mode == 0:
        return (
            input_width + config.cim_base_a_width - 1
        ) // config.cim_base_a_width
    guard = max(1, (config.cim_macro_input_lanes - 1).bit_length())
    width = min(
        input_width, config.cim_base_c_width - config.cim_base_b_width - guard
    )
    interval = (
        (width + config.cim_base_a_width - 1) // config.cim_base_a_width
        * config.cim_base_a_width
    )
    return (input_width + width - 1) // width * interval


class CIMRuntimeCalculator(RuntimeCalculator):
    """Estimate CIM runtime with the shared SRAM bank and DRAM model.

    Weight programming and MAC issue work are summed conservatively; operand
    bank traffic can overlap that work. Partial sums beyond the local register
    capacity incur SRAM reads and writes. The estimate ranks mappings without
    modeling descriptor handshakes or weight prefetch within a command.
    """

    def __init__(self, *args, config, layer, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.layer = layer

    def evaluate(self, mapping):
        # A multi-beat output uses the banked path when the hardware provides
        # it, including when the matrix result feeds a fused vector tail.
        output_width = (
            self.accum_dtype_width
            if mapping.loop_blockings[le.IC][3] > 1
            else self.output_dtype_width
        )
        banked = (
            output_width * self.config.pe_array_size[1] > self.sram_bandwidth
        )
        return evaluate(
            self.config, self.layer, mapping, banked_output=banked
        )

    def calculate_runtime(self, architecture, layer, mapping):
        if not self.evaluate(mapping).legal:
            return math.inf
        return super().calculate_runtime(architecture, layer, mapping)

    def matrix_cycles(self, mapping, bank_groups):
        result = self.evaluate(mapping)
        if not result.legal:
            return math.inf
        ic, oc = self.config.pe_array_size
        policy = result.policy
        words = {
            "input": result.input_fill_words
            * self._bus_words(ic, self.input_dtype_width),
            "weight": policy.full_set_loads * ic
            * self._bus_words(oc, self.weight_dtype_width),
            "bias": self._bus_words(
                result.output_vectors * oc, self.bias_width
            ),
        }
        if mapping.loop_blockings[le.IC][3] > 1:
            words["scratch"] = 2 * self._bus_words(
                result.output_vectors * oc, self.accum_dtype_width
            )
        else:
            words.update(self._tail_words(mapping, 2))
        compute = (
            result.mac_requests
            * issue_window(self.config, self.input_dtype_width)
            + self.config.cim_mac_latency
            + result.weight_write_beats
            + 2 * result.buffer_partial_updates
        )
        return max(compute, self._bank_cycles(words, bank_groups))
