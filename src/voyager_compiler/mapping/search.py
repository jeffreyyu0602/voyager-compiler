# Own Interstellar setup, candidate conversion, callbacks, and search statistics
from collections import Counter
from dataclasses import dataclass, replace
from math import prod
import interstellar
from interstellar import loop_enum as le
from interstellar.cost_model import get_access
from interstellar.evaluation import CandidateEvaluation
from .target import MappingTarget
from .schedule import LOOPS, Schedule, TemporalLevel
from .models import sa, cim
from .models.input import input_tile_shape
from .models.output import OutputOptions


# Bind generic enumeration to a target's constraints and evaluation callbacks
@dataclass
class SearchInputs:
    resource: object
    layer: object
    schedule: object
    candidate_evaluator: object
    capacity_filter: object = None


# Invoke Interstellar once and preserve its selected candidate evaluation
def search_mapping(inputs, *, verbose=0):
    from interstellar.optimizer import opt_optimizer
    if not callable(inputs.candidate_evaluator):
        raise ValueError("mapping search requires a complete backend evaluator")
    try:
        callbacks = {name: getattr(inputs, name)
                     for name in ("capacity_filter", "candidate_evaluator")
                     if getattr(inputs, name) is not None}
        _, _, mapping, _ = opt_optimizer(inputs.resource, inputs.layer, inputs.schedule,
                                         verbose=verbose, **callbacks)
    except AssertionError as error:
        if str(error) != "No valid mapping point found.":
            raise
        raise ValueError("no legal mapping satisfies the target constraints") from error
    return mapping


# Require complete integer loop matrices for L0, L1, and L2
def _matrix(mapping, name, minimum, maximum=None):
    rows = getattr(mapping, name, None)
    if rows is None or len(rows) != le.NUM or any(len(row) != 3 for row in rows):
        raise ValueError(f"{name} must contain seven loops with three levels")
    if any(type(value) is not int or value < minimum or
           (maximum is not None and value > maximum) for row in rows for value in row):
        raise ValueError(f"{name} contains an invalid integer")
    return rows


# Preserve inner-to-outer order and append omitted unit loops like Tiling.cc
def _level(orders, blocking, partitioning, level):
    ranks = [row[level] for row in orders]
    complete_order = len(set(ranks)) == le.NUM
    ordered = []
    for loop, rank in enumerate(ranks):
        if complete_order or rank != le.NUM - 1:
            ordered.append((rank, le.table[loop]))
        elif blocking[loop][level] != 1 or partitioning[loop][level] != 1:
            raise ValueError(f"L{level} omits non-unit {le.table[loop]}")
    ordered.sort()
    if [rank for rank, _ in ordered] != list(range(len(ordered))):
        raise ValueError(f"L{level} loop order must be contiguous and unique")
    return TemporalLevel.make(
        order=tuple(loop for _, loop in ordered if loop != "ON"),
        **{loop: blocking[le.loop_table[loop]][level] for loop in LOOPS},
    )


# Bind a complete candidate to fixed IC/OC lanes and the compiler's two levels
def schedule_from_mapping(target: MappingTarget, mapping, *,
                          write_output_to_accum_buffer: bool) -> Schedule:
    if target.backend == "sa":
        levels = [TemporalLevel.make(
            order=tuple(le.table[loop] for loop in sorted(range(le.NUM), key=lambda i: mapping.loop_orders[i][level]) if loop != le.ON),
            **{loop: mapping.loop_blockings[le.loop_table[loop]][level] for loop in LOOPS}) for level in range(3)]
        return Schedule(levels[1], levels[2], levels[0].bounds, write_output_to_accum_buffer,
                        tuple(tuple(mapping.loop_partitionings[le.loop_table[loop]][level] for loop in LOOPS) for level in range(3)))
    orders = _matrix(mapping, "loop_orders", 0, le.NUM - 1)
    blocking = _matrix(mapping, "loop_blockings", 1)
    partitioning = _matrix(mapping, "loop_partitionings", 1)
    if any(matrix[le.ON][level] != 1
           for matrix in (blocking, partitioning) for level in range(3)):
        raise ValueError("current CIM schedules do not represent batch ON")
    for loop in range(le.NUM):
        spatial = target.k if loop == le.IC else target.n if loop == le.OC else 1
        if tuple(partitioning[loop]) != (spatial, 1, 1):
            raise ValueError(f"{le.table[loop]} spatial factors must be {(spatial, 1, 1)}")
    if any(row[0] != 1 for row in blocking):
        raise ValueError("L0 temporal factors must be one; compiler tilings omit L0")
    if blocking[le.FX][2] != 1:
        raise ValueError("L2 FX is unsupported by the current compiler ABI")
    levels = [_level(orders, blocking, partitioning, level) for level in range(3)]
    return Schedule(levels[1], levels[2], levels[0].bounds, write_output_to_accum_buffer,
                    tuple(tuple(partitioning[le.loop_table[loop]][level] for loop in LOOPS) for level in range(3)))


# Preserve physical CIM lanes while exploring temporal factors and both loop orders
def cim_search_constraints(target):
    hints = {}
    for loop in interstellar.le.table.values():
        spatial = target.k if loop == "IC" else target.n if loop == "OC" else 1
        hints[interstellar.le.loop_table[loop]] = [
            [None, None if spatial > 1 else 1, spatial],
            [None, None, 1], [None, 1 if loop == "FX" else None, 1],
        ]
    return interstellar.Schedule(hints)


# Adapt both hardware models to the Interstellar callback interface
class CandidateEvaluator:
    # Bind one search to its workload, target, and vector timing assumptions
    def __init__(self, target, workload, model, banked_output, vector_timing):
        self.target, self.workload, self.model = target, workload, model
        self.banked_output, self.vector_timing = banked_output, vector_timing
        self.evaluated = self.legal = self.capacity_rejected = 0
        self.rejections = Counter()

    # Check exact input capacity and a minimum live-output footprint without orders
    def capacity_filter(self, resource, layer, point, stage):
        if stage not in ("blocking", "partitioned"):
            raise ValueError("unknown capacity stage: " + str(stage))
        for loop, factors in enumerate(point.loop_blockings):
            spatial = self.target.k if loop == interstellar.le.IC else self.target.n if loop == interstellar.le.OC else 1
            if factors[0] != (spatial if stage == "blocking" else 1):
                self.capacity_rejected += 1
                return False
        l1 = TemporalLevel.make(**{loop: point.loop_blockings[interstellar.le.loop_table[loop]][1]
                                   for loop in LOOPS})
        width, height = input_tile_shape(dict(zip(LOOPS, l1.bounds)), self.workload)
        fits = (width * height * l1.bound("IC") <= min(self.target.input_buffer_words, 65536)
                and prod(l1.bound(loop) for loop in ("OX", "OY", "OC")) <= self.target.accum_buffer_words)
        self.capacity_rejected += not fits
        return fits

    # Report every evaluated or capacity-rejected candidate without a shortlist
    def statistics(self):
        return dict(evaluated=self.evaluated, legal=self.legal, capacity_rejected=self.capacity_rejected,
                    rejections=dict(self.rejections))

    # Keep the backend result on the winning Interstellar point without rescoring
    def __call__(self, resource, layer, point):
        self.evaluated += 1
        schedule = schedule_from_mapping(self.target, point,
                                         write_output_to_accum_buffer=self.banked_output)
        result = self.model(schedule)
        if result.legal and self.target.backend == "sa":
            accesses, _ = get_access(point, layer, resource)
            result = replace(result, traffic=dict(accesses=accesses, unit="scalar element accesses",
                                                  operands=("input", "output", "weight")))
        result = replace(result, schedule=schedule)
        self.legal += result.legal
        self.rejections.update(result.reasons)
        timing = result.timing
        return CandidateEvaluation(legal=result.legal, reasons=result.reasons,
            runtime_cycles=result.runtime_cycles, traffic=result.traffic, metadata=result,
            ideal_cycles=timing.ideal_cycles if result.legal else None,
            spatial_utilization=timing.spatial_utilization if result.legal else None,
            useful_work_fraction=timing.useful_work_fraction if result.legal else None)


def sa_search_space(target):
    resource = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],  # PE register file
            [
                target.input_buffer_words * target.k,
                target.accum_buffer_words * target.n,
                target.weight_buffer_words * target.n,
            ],  # L1 buffer
            [12 * 1024 * 1024],  # L2 main memory
        ],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0]],
        buf_access_cost_list=None,
        buf_unit_static_cost_list=None,
        para_count_list=[target.k * target.n, 1, 1],
        mac_capacity=0,
        partition_mode=[0, 0, 0],
        invalid_underutilized=False,
    )

    # Restrict dataflow to what we can handle
    schedule_constraint = {
        "schedule_hint": {
            "IC": {
                "level0": {"order": 1, "partitioning_size": target.k},
                "level1": {"order": -1},
                "level2": {"order": 0},
            },
            "OC": {"level0": {"order": 0, "partitioning_size": target.n}},
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
    schedule = interstellar.extract_input.extract_schedule_info(schedule_constraint, 3)
    schedule = interstellar.Schedule(
        schedule["schedule_hint"], schedule["partition_loops"]
    )

    return resource, schedule


# Prepare one backend with common workload and schedule interfaces
def prepare_search(target, workload, *, vector_timing=None, write_output_to_accum_buffer=False, options=None):
    if target.backend == "cim":
        if workload.input_channels % target.k or workload.output_channels % target.n:
            raise ValueError("CIM search requires compiler-padded IC/OC channels matching the physical CIM lanes")
        if workload.output_x <= 0 or workload.output_y <= 0:
            raise ValueError("CIM search requires a positive output shape")
        options = options or cim.TimingOptions()
        if vector_timing is not None:
            write_output_to_accum_buffer = target.double_buffered_accum and vector_timing.port_cycles_per_vector > 1
            options = replace(options,
                              output_cycles_per_vector=max(options.output_cycles_per_vector, vector_timing.cycles_per_vector))
        model = cim.Evaluator(target, workload, options=options)
        resource = interstellar.Resource([[0]] * 3, None, None,
            [target.k * target.n, 1, 1], mac_capacity=0,
            partition_mode=[0, 0, 0], invalid_underutilized=False)
        constraints = cim_search_constraints(target)
        useful_fraction = 1.0
    else:
        model = sa.Evaluator(target, workload, vector_timing, options=options or OutputOptions())
        resource, constraints = sa_search_space(target)
        write_output_to_accum_buffer = target.double_buffered_accum and vector_timing.port_cycles_per_vector > 1
        useful_fraction = workload.useful_work_fraction
    layer = interstellar.Layer(workload.input_channels, workload.output_channels,
        workload.output_x, workload.output_y, workload.filter_x, workload.filter_y,
        wstd=workload.stride, hstd=workload.stride, useful_work_fraction=useful_fraction)
    evaluator = CandidateEvaluator(target, workload, model, write_output_to_accum_buffer, vector_timing)
    return SearchInputs(resource, layer, constraints, candidate_evaluator=evaluator,
                        capacity_filter=evaluator.capacity_filter if target.backend == "cim" else None)


# Search a standalone workload through the same callbacks as compiler mapping
def search(target, workload, *, vector_timing=None, write_output_to_accum_buffer=False, options=None):
    inputs = prepare_search(target, workload, vector_timing=vector_timing,
        write_output_to_accum_buffer=write_output_to_accum_buffer, options=options)
    return search_mapping(inputs), inputs.candidate_evaluator.statistics()
