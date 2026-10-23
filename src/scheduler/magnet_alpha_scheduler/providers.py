# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The provider set Alpha declares, and its rule-based GEMM engine.

The set is the whole of what the target chooses. Dispatch resolves the engine
within it without knowing which target asked. This port carries only Alpha's
deterministic rule-based encoder.

The near-memory provider is not another command-line choice. It sits in front
of the GEMM path because which unit owns an operation is a fact about the
machine, while both units' deterministic decisions still report the selected
``magnet_alpha_rule_based`` scheduling engine.
"""

import math
from dataclasses import replace

from scheduler.magnet_alpha_scheduler.evaluator import (
    MagnetAlphaRuntimeCalculator,
)
from scheduler.magnet_alpha_scheduler.nmp import schedule_nmp_operation
from scheduler.magnet_alpha_scheduler.scheduler import (
    MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    MagnetAlphaSchedulingError,
    single_call_gemm_schedule,
    single_call_gemm_schedule_result,
    project_gemm_layer,
)
from scheduler.types import SchedulerProviderSet

__all__ = [
    "MAGNET_ALPHA_RULE_BASED_SCHEDULER",
    "MAGNET_ALPHA_SCHEDULER_PROVIDERS",
    "alpha_operation_provider",
]


def _lanes(config) -> int:
    """The near-memory unit's channel parallelism, from the declared machine.

    Read from the config rather than imported as a constant: the unit that
    performs the work is named by ``compute_unit_bindings``, and a machine
    that binds a different one should be planned against that one.
    """
    unit = config.unit("nmp") if config is not None else None
    if unit is not None:
        return unit.unroll("CHANNEL") or unit.lanes
    from voyager_compiler.codegen.targets.magnet_alpha.hardware import (
        NMP_UNIT,
    )

    return NMP_UNIT.unroll("CHANNEL")


def alpha_operation_provider(workload, config):
    """Claim the near-memory unit's work, before any engine sees it."""
    return schedule_nmp_operation(workload, config, lanes=_lanes(config))


def _require_alpha_hardware(config) -> None:
    if config is None:
        raise TypeError("Alpha scheduling requires an accelerator config")
    missing = [role for role in ("matrix", "nmp") if config.unit(role) is None]
    if missing:
        raise TypeError(
            "Alpha scheduling requires matrix and NMP compute-unit bindings; "
            f"missing {missing}"
        )
    if config.local_memory is None or config.global_memory is None:
        raise TypeError("Alpha scheduling requires local and global memories")


def _gemm_facts(workload):
    if workload.gemm is None:
        raise TypeError("the Alpha GEMM provider needs a GEMM workload")
    input_operand = workload.require_operand("input")
    if input_operand.bits is None or workload.output.bits is None:
        raise ValueError(
            "the Alpha GEMM provider needs resolved input and output widths"
        )
    return workload.gemm, input_operand.bits, workload.output.bits


def _accumulator_is_double_buffered(config) -> bool:
    level = config.local_memory
    memory = (
        level.instance(level.output_memory)
        if level is not None and level.output_memory is not None
        else None
    )
    return bool(
        config.double_buffered_accum_buffer
        or (memory is not None and memory.double_buffered)
    )


def _route_cycles(config, transfers) -> float:
    """Apply the declared route roofline without adding target policy to it."""
    if not config.hierarchy_bandwidths:
        return 0.0
    cycles_by_group = {}
    for source, destination, role, bits in transfers:
        role_name = getattr(role, "value", role)
        matches = [
            route
            for route in config.hierarchy_bandwidths
            if route.source == source
            and route.destination == destination
            and any(
                getattr(candidate, "value", candidate) == role_name
                for candidate in route.roles
            )
        ]
        if len(matches) != 1:
            raise MagnetAlphaSchedulingError(
                f"expected one {role_name} route from {source} to "
                f"{destination}, found {len(matches)}"
            )
        route = matches[0]
        group = route.shared_group or f"route:{route.name}"
        cycles_by_group[group] = cycles_by_group.get(group, 0.0) + (
            float(bits) / route.bandwidth_bits_per_cycle
        )
    return max(cycles_by_group.values(), default=0.0)


def run_rule_based_gemm_scheduler(
    workload, config, provider_cache, options
):
    """Run Alpha's deterministic, single-call GEMM encoder.

    Provider state and options are accepted for the common dispatch contract;
    this engine is deterministic and needs neither.
    """
    del provider_cache, options
    _require_alpha_hardware(config)
    layer, input_bits, output_bits = _gemm_facts(workload)
    physical_layer = project_gemm_layer(layer, config, input_bits)
    result = single_call_gemm_schedule_result(layer, config, input_bits)
    calculator = MagnetAlphaRuntimeCalculator(
        input_bits,
        output_bits,
        _accumulator_is_double_buffered(config),
        workload.has_high_precision_vector_input,
        workload.has_sparse_op,
        vector_size=config.input_lanes(input_bits),
    )
    access_counts = calculator.calculate_access_counts(
        physical_layer, result.mapping
    )
    compute_cycles = calculator.calculate_runtime(
        physical_layer, result.mapping
    )
    local, global_accesses = access_counts[1], access_counts[2]
    transfer_cycles = _route_cycles(
        config,
        (
            (
                config.global_memory.name,
                config.local_memory.name,
                "activation",
                local.input * input_bits,
            ),
            (
                config.global_memory.name,
                config.local_memory.name,
                "weight",
                local.weight * input_bits,
            ),
            (
                config.local_memory.name,
                config.global_memory.name,
                "output",
                global_accesses.output * output_bits,
            ),
        ),
    )
    return replace(
        result,
        access_counts=access_counts,
        runtime=max(compute_cycles, float(math.ceil(transfer_cycles))),
    )


def alpha_gemm_tiling_candidate_fits(
    layer,
    full_layer,
    hardware_config,
    input_dtype_width,
    can_materialize_output_channel_tiles,
) -> bool:
    """Whether one graph tile is an encodable Alpha GEMM call."""
    if (
        layer.nofm != full_layer.nofm
        and not can_materialize_output_channel_tiles
    ):
        return False
    try:
        single_call_gemm_schedule(
            layer, hardware_config, input_dtype_width
        )
    except MagnetAlphaSchedulingError:
        return False
    return True


#: What Alpha declares. Unit ownership is independent of the selected GEMM
#: engine, so the NMP provider remains first refusal.
MAGNET_ALPHA_SCHEDULER_PROVIDERS = SchedulerProviderSet(
    default_backend=MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    gemm_providers={
        MAGNET_ALPHA_RULE_BASED_SCHEDULER: run_rule_based_gemm_scheduler
    },
    operation_provider=alpha_operation_provider,
    gemm_tiling_candidate_fits=alpha_gemm_tiling_candidate_fits,
)
