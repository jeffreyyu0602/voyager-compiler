"""CIM schedule checks and weight reuse estimates for Interstellar.

The weight policy follows CIMWeightController::reader: innermost spatial
loops reuse one weight set, enclosing spatial loops repeat a stored weight
sequence, and a sequence that does not fit loads one set per descriptor. Counts
describe one L2 command (one step of the compiler's L3 grid).
"""

from dataclasses import dataclass
from math import prod

import interstellar

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

    The search supports dense convolutions with stride 1 and complete filter
    loops at L1. Partial sums beyond the local register capacity use the
    accumulation SRAM.
    """
    b, p, order = (
        mapping.loop_blockings,
        mapping.loop_partitionings,
        mapping.loop_orders,
    )
    ic, oc = config.pe_array_size
    reasons = []
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
    if layer.wstd != 1 or layer.hstd != 1:
        reasons.append("CIM mapping requires convolution stride 1")
    if any(b[d][level] > 1023 for d in range(le.NUM) for level in (1, 2)):
        reasons.append("temporal factor exceeds the 10-bit controller bound")
    if b[le.FX][1] > 15 or b[le.FY][1] > 15:
        reasons.append("filter extent exceeds the 4-bit input-controller field")
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
    width = b[le.OX][1] + b[le.FX][1] - 1
    height = b[le.OY][1] + b[le.FY][1] - 1
    input_words = width * height * b[le.IC][1]
    if input_words > min(config.input_buffer_size, 65536):
        reasons.append(
            "input tile including halo exceeds one input-buffer bank"
        )
    if width > 512 or height > 1023 or max(b[le.OX][2], b[le.OY][2]) > 512:
        reasons.append(
            "input traversal exceeds the controller coordinate fields"
        )
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
