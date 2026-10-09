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
from voyager_compiler.codegen.transform.tiling.cim_timing import (
    TimingOptions,
    estimate_cycles,
)
from voyager_compiler.codegen.transform.tiling.input import input_buffer_usage
from voyager_compiler.codegen.transform.tiling.runtime import (
    BaseRuntimeCalculator,
)

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
        # L0 has fixed physical lanes and no temporal work. FX stays at L1;
        # FY may run at L1 or L2 inside one command. The bufferized grid keeps
        # filter loops below L3 and its channel reduction innermost.
        hints[loop] = [
            [
                1 if loop == le.IC else 0 if loop == le.OC else None,
                None if lanes > 1 else 1,
                lanes,
            ],
            [None, None, 1],
            [None, 1 if loop == le.FX else None, 1],
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
    local_reduction_steps: int = 1
    buffer_feedback_spacing: int = 0

    @property
    def legal(self):
        return not self.reasons


def accumulation_policy(config, mapping):
    """Local reduction lifetime and SRAM traffic, matching CIMProcessor.

    Registers cover the largest inner reduction with at most R live outputs.
    Its final partial sum spills once if an outer reduction remains. When no
    inner reduction fits, the existing first-R full-sum optimization remains.
    """
    b, order = mapping.loop_blockings, mapping.loop_orders
    loops = [
        (d, b[d][level])
        for level in (1, 2)
        for d in sorted(OUTPUTS + REDUCTIONS, key=lambda d: order[d][level])
    ]
    live, boundary = 1, -1
    for index, (dim, bound) in enumerate(loops):
        if dim in OUTPUTS:
            live *= bound
        if dim in REDUCTIONS and bound > 1 and live <= config.cim_local_accum_contexts:
            boundary = index
    outputs = prod(bound for dim, bound in loops if dim in OUTPUTS)
    reductions = prod(bound for dim, bound in loops if dim in REDUCTIONS)
    if boundary >= 0:
        contexts = prod(bound for dim, bound in loops[:boundary + 1] if dim in OUTPUTS)
        local_steps = prod(bound for dim, bound in loops[:boundary + 1] if dim in REDUCTIONS)
        updates = outputs * (reductions // local_steps - 1)
    else:
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
        local_steps = 1
        updates = outputs // contexts * max(0, contexts - config.cim_local_accum_contexts) * (reductions - 1)
    # A stored group's last contribution, rather than its first one, starts
    # the dependency interval before the next group's first SRAM read.
    period, local_span, spacing = 1, 0, 0
    for index, (dim, bound) in enumerate(loops):
        if dim in REDUCTIONS and bound > 1:
            if index > boundary:
                spacing = period - local_span
                break
            local_span += (bound - 1) * period
        period *= bound
    return contexts, local_steps, updates, spacing if updates else 0


def evaluate(config, layer, mapping, *, banked_output=False):
    """Check a mapping against CIM buffer and controller limits.

    FX stays at L1 and FY may run at L1 or L2. Partial sums beyond the local
    register capacity use the accumulation SRAM.
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
    if b[le.FX][2] != 1 or b[le.FX][3] != 1 or b[le.FY][3] != 1:
        reasons.append("CIM mapping keeps FX at L1 and FY at L1 or L2")
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

    contexts, local_steps, partial_updates, feedback_spacing = accumulation_policy(config, mapping)
    outputs = prod(b[d][level] for level in (1, 2) for d in OUTPUTS)
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
        partial_updates,
        input_words * input_fills,
        policy.full_set_loads * ic
        * config.cim_output_axis_tiles // config.cim_b_port_tiles,
        local_steps,
        feedback_spacing,
    )


# L1 order readers test only whether a loop is a reduction, spatial or OC
L1_CLASSES = {
    le.IC: "R", le.FX: "R", le.FY: "R", le.OX: "S", le.OY: "S",
    le.OC: "OC", le.ON: "ON",
}


def _merge_spatial_runs(tokens, innermost_only=False):
    """Turn ``(name, bound)`` tokens into ``(name, bounds)`` tokens.

    Each run of adjacent ``"S"`` tokens becomes one token with the sorted
    bounds of the run; with ``innermost_only``, only a run that starts at
    the innermost token.
    """
    merged = []
    for name, bound in tokens:
        if (
            name == "S"
            and merged
            and merged[-1][0] == "S"
            and not (innermost_only and len(merged) > 1)
        ):
            merged[-1] = ("S", merged[-1][1] + (bound,))
        else:
            merged.append((name, (bound,)))
    return tuple((name, tuple(sorted(bounds))) for name, bounds in merged)


def order_key(level, order, mapping, tail_specs=()):
    """Key under which the cost models cannot tell ``level``'s orders apart.

    ``order`` holds each loop's rank at ``level`` (0 = innermost). The key is
    the inner-to-outer sequence of the level's non-unit loops as ``(name,
    bound)`` pairs, with these exact equivalences:

    * L1: every reader of the L1 order (``weight_policy``, the contexts in
      ``evaluate``, ``estimate_cycles``, the bias and
      output timing, and the interstellar access counts) tests only whether
      a loop is a reduction (IC/FX/FY), spatial (OX/OY) or OC. Loops of one
      class with equal bounds can thus swap. The innermost run of spatial
      loops is reused inside every weight loop, and the readers use it only
      as the product of its bounds, so its order is free.
    * L2: OX and OY are read only as spatial loops, so they can swap when
      their bounds are equal. IC and OC are read by name.
    * L3: the matrix timing does not read the L3 order. The DRAM step counts
      (``BaseRuntimeCalculator._l3_loads``) and the access counts read only
      which loops are outside each operand's innermost loop. A split IC is
      innermost (the hint), and the input and weight loads span it, so the
      order is free when each fused tail operand spans every tiled output
      loop. Otherwise adjacent OX and OY can swap when each tail operand
      spans both or neither of them.

    L0 keeps every order. ``tail_specs`` are the fused tail's ``(dims,
    bits)`` operands.
    """
    b, p = mapping.loop_blockings, mapping.loop_partitionings
    loops = sorted(
        (d for d in range(le.NUM) if b[d][level] != 1 or p[d][level] != 1),
        key=lambda d: order[d],
    )
    if level == 1:
        return _merge_spatial_runs(
            [(L1_CLASSES[d], b[d][1]) for d in loops], innermost_only=True
        )
    if level == 2:
        return tuple(
            ("S" if d in SPATIAL else le.table[d], b[d][2]) for d in loops
        )
    if level == 3:
        if b[le.IC][3] > 1 and order[le.IC] != 0:
            # The free orders below need the split IC innermost, as the hint
            # requires. Without it, every L3 order keeps its own key.
            return tuple((le.table[d], b[d][3]) for d in loops)
        tiled = {d for d in (*OUTPUTS, le.ON) if b[d][3] > 1}
        if b[le.IC][3] > 1 and all(
            tiled <= set(dims) for dims, _ in tail_specs
        ):
            return ()
        if all((le.OX in dims) == (le.OY in dims) for dims, _ in tail_specs):
            return _merge_spatial_runs(
                [("S" if d in SPATIAL else le.table[d], b[d][3]) for d in loops]
            )
        return tuple((le.table[d], b[d][3]) for d in loops)
    return tuple(order)


def _blocking_key(mapping):
    """The blockings and partitionings of ``mapping``."""
    return (
        tuple(map(tuple, mapping.loop_blockings)),
        tuple(map(tuple, mapping.loop_partitionings)),
    )


def _matrix_key(mapping):
    """What ``evaluate`` and the matrix timing read from ``mapping``: the
    blockings, the partitionings and the loop orders below L3."""
    return (
        *_blocking_key(mapping),
        tuple(tuple(order[:3]) for order in mapping.loop_orders),
    )


class _LastResult:
    """The result for the last key, recomputed when the key changes.

    The search scores the loop orders of one blocking in a row and varies the
    L3 order fastest, so a result that reads only part of the mapping repeats
    across consecutive mappings.
    """

    def __init__(self):
        self.key, self.value = object(), None

    def get(self, key, compute, *args):
        if key != self.key:
            self.value = compute(*args)
            self.key = key
        return self.value


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


class CIMRuntimeCalculator(BaseRuntimeCalculator):
    """Estimate CIM runtime with the shared SRAM bank and DRAM model.

    Weight and input prefetch overlap MAC issue, subject to resident-set and
    buffer availability. Partial sums beyond the local register capacity
    incur SRAM reads and writes. Interacting operand and output waits use an
    additive bound, retaining queued output work. The estimate ranks mappings
    without modeling command serialization or unprofiled HLS pipeline stages.
    """

    def __init__(
        self, *args, config, layer, timing_options=TimingOptions(), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.layer = layer
        self.timing_options = timing_options
        self._input_cache = {}
        # The bank partition and the vector cycles read no loop order; the
        # evaluation and the matrix timing read no L3 order.
        self._partition = _LastResult()
        self._vector_cycles = _LastResult()
        self._evaluation = _LastResult()
        self._matrix_cycles = _LastResult()

    def uses_banked_output(self, mapping):
        # A multi-beat output uses the banked path when the hardware provides
        # it, including when the matrix result feeds a fused vector tail.
        width = (
            self.accum_dtype_width
            if mapping.loop_blockings[le.IC][3] > 1
            else self.output_dtype_width
        )
        if (
            self.has_tail
            and mapping.loop_blockings[le.IC][3] == 1
            and not self.single_k_tail_extra_pass
        ):
            # MatrixOps::should_use_direct_path also checks fused fetch ports.
            width = max([width, *(bits for _, bits in self.tail_specs)])
        return (
            self.config.double_buffered_accum_buffer
            and width * self.config.pe_array_size[1] > self.sram_bandwidth
        )

    def order_key(self, level, order, mapping):
        return order_key(level, order, mapping, self.tail_specs)

    def evaluate(self, mapping):
        return self._evaluation.get(
            _matrix_key(mapping), self._evaluate, mapping
        )

    def _evaluate(self, mapping):
        return evaluate(
            self.config, self.layer, mapping,
            banked_output=self.uses_banked_output(mapping),
        )

    def calculate_runtime(self, architecture, layer, mapping):
        if not self.evaluate(mapping).legal:
            return math.inf
        return super().calculate_runtime(architecture, layer, mapping)

    def timing(self, mapping, bank_groups=None):
        """Return command timing and traffic, or None for an illegal mapping."""
        result = self.evaluate(mapping)
        if not result.legal:
            return None
        return estimate_cycles(
            self,
            mapping,
            result,
            bank_groups,
            issue_window(self.config, self.input_dtype_width),
        )

    def matrix_cycles(self, mapping, bank_groups):
        return self._matrix_cycles.get(
            (_matrix_key(mapping), bank_groups),
            self._matrix_timing_cycles,
            mapping,
            bank_groups,
        )

    def _matrix_timing_cycles(self, mapping, bank_groups):
        timing = self.timing(mapping, bank_groups)
        if (
            timing is None
            or timing.readiness.get("accumulation_feedback_safe") is False
        ):
            return math.inf
        return timing.runtime_cycles

    def bank_partition(self, architecture, layer, mapping):
        return self._partition.get(
            (architecture, layer, layer.size_fn, _blocking_key(mapping)),
            super().bank_partition,
            architecture,
            layer,
            mapping,
        )

    def vector_cycles(self, mapping, bank_groups):
        return self._vector_cycles.get(
            (_blocking_key(mapping), bank_groups),
            super().vector_cycles,
            mapping,
            bank_groups,
        )
