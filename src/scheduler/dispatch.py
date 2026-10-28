# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Deciding who schedules one operation.

There is exactly one rule here and it is worth stating on its own:

    the target's operation provider gets first refusal; only an operation it
    declines, and that has a canonical GEMM projection, reaches an engine.

Everything else in this module is bookkeeping around that.

The order matters because the two answers are not comparable. A GEMM engine
answers "how should this be blocked across the memory hierarchy". A target's
operation provider answers "this is mine" -- a near-memory unit's depthwise
convolution, its channel shuffle, its pooling -- and for that work there is no
blocking to decide: the unit reads the tensor and writes it back. Asking the
engine first would either fail (no mapping exists for an operation it cannot
model) or succeed with a mapping nothing will execute.

Note what is *not* here. Dispatch never sees a compilation target, and no
provider is chosen by target name. A target selects one immutable provider set
once, and this module resolves an engine *within* it. That keeps "which engine"
separable from "which machine": two targets can share an engine while
describing different hardware, and one target can carry several engines and
choose between them from the command line.
"""

import logging
from typing import MutableMapping, Optional

from scheduler.types import (
    LocalScheduleResult,
    OperationWorkload,
    SchedulerProviderSet,
)

logger = logging.getLogger(__name__)

__all__ = ["resolve_local_scheduler", "schedule_operation"]


def resolve_local_scheduler(
    backend: Optional[str],
    *,
    providers: SchedulerProviderSet,
) -> str:
    """The engine name to use, checked against what the target declared.

    Fail closed on an unregistered name. A target may carry multiple engines;
    asking for one it did not register is a typo or a lane pointed at the
    wrong target, and quietly using the default would produce a schedule
    nobody asked for and report the wrong backend on it.
    """
    if not isinstance(providers, SchedulerProviderSet):
        raise TypeError("providers must be a SchedulerProviderSet")
    selected = providers.default_backend if backend is None else backend
    if selected not in providers.gemm_providers:
        raise ValueError(
            f"scheduler {selected!r} is not registered in the selected "
            f"provider set; declared: {sorted(providers.gemm_providers)}"
        )
    return selected


def schedule_operation(
    workload: OperationWorkload,
    config,
    provider_cache=None,
    scheduler_options=None,
    result_cache: Optional[MutableMapping] = None,
    *,
    providers: SchedulerProviderSet,
    backend: Optional[str] = None,
) -> Optional[LocalScheduleResult]:
    """Make one scheduling decision, or return ``None`` if nobody claims it.

    ``None`` is a real answer and not a failure: an operation the host runs
    has no schedule, and inventing an empty one for it would make host work
    look like accelerator work to everything downstream.

    Args:
        workload: The operation, with its graph knowledge already frozen.
        config: The machine to plan against.
        provider_cache: Engine-owned state reused across operations -- a
            built architecture, a memoized search. Opaque here.
        scheduler_options: Passed to the engine unchanged.
        result_cache: Memoizes on ``(engine, workload)``. Sound because a
            workload is frozen and an engine is deterministic; worth having
            because a network repeats layer shapes and the search is the
            expensive part of a compile.
        providers: The target's provider set.
        backend: Which engine, within that set.
    """
    if not isinstance(workload, OperationWorkload):
        raise TypeError("scheduling takes an OperationWorkload")
    if not isinstance(providers, SchedulerProviderSet):
        raise TypeError("providers must be a SchedulerProviderSet")

    claim = providers.operation_provider
    if claim is not None:
        result = claim(workload, config)
        if result is not None:
            if not isinstance(result, LocalScheduleResult):
                raise TypeError(
                    f"the target's operation provider returned "
                    f"{type(result).__name__}, not a schedule result"
                )
            # Deliberately not coverage-checked against the GEMM projection.
            # A claimed operation is one the target's *own* unit runs, and
            # its mapping describes that unit's loops. A depthwise
            # convolution has a GEMM projection with a 3x3 filter and a
            # near-memory unit that does not block over one; measuring the
            # claim against the projection would reject the very answer the
            # provider exists to give.
            return result

    if workload.gemm is None:
        # Unclaimed and not a GEMM. There is no engine that could take it.
        return None

    selected = resolve_local_scheduler(backend, providers=providers)
    key = (selected, workload)
    if result_cache is not None and key in result_cache:
        return result_cache[key]

    result = providers.resolve_gemm(selected)(
        workload, config, provider_cache, scheduler_options
    )
    if result is not None:
        _check_gemm_coverage(result, workload, f"engine {selected!r}")
    if result_cache is not None:
        result_cache[key] = result
    return result


def _check_gemm_coverage(result, workload, who: str) -> None:
    """Reject an engine's mapping that does not cover the layer it answers.

    Cheap, and it catches a failure that is otherwise found much later: a
    mapping whose loop coverage does not reach the layer's extents has
    planned a different layer, and the first sign of it would be a converted
    call reading past its operand.

    Only for an engine's result. See ``schedule_operation`` for why a claimed
    operation is not measured this way.
    """
    if not isinstance(result, LocalScheduleResult):
        raise TypeError(f"{who} returned {type(result).__name__}, not a result")
    gemm = workload.gemm
    if gemm is None:
        return
    for loop, extent in enumerate(gemm.sizes):
        covered = result.mapping.loop_coverage(loop)
        if covered < extent:
            raise ValueError(
                f"{who} returned a mapping covering {covered} of loop "
                f"{loop}'s {extent} iterations"
            )
