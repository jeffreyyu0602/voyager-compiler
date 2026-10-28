# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Orchestrate Magnet Alpha's non-bufferized terminal codegen path.

This module owns the compile-shaped boundary: propagate final shapes, bind the
target scheduler's decisions, build the graph-wide storage plan, and only then
hand finalized state to the collateral serializer.  Keeping that order here
lets :mod:`.serialization` remain a projection rather than becoming a planner.
"""

import logging

from torch.fx import GraphModule

from voyager_compiler.codegen.execution_order import (
    END_TO_END_SCRATCHPAD_MAPPING,
    PER_OPERATION_SCRATCHPAD_MAPPING,
    require_execution_mode,
    validate_execution_order_mode,
)
from voyager_compiler.codegen.targets.magnet_alpha.allocation import (
    AlphaGlobalAllocator,
    build_alpha_allocation_requests,
    publish_alpha_bank_placements,
)
from voyager_compiler.codegen.targets.magnet_alpha.dram_allocation import (
    AlphaDramAllocator,
)
from voyager_compiler.codegen.targets.magnet_alpha.finalization import (
    build_alpha_finalized_serialization,
)
from voyager_compiler.codegen.targets.magnet_alpha.lowering import (
    apply_memory_rewrites,
)
from voyager_compiler.codegen.targets.magnet_alpha.scheduling import (
    bind_alpha_schedules,
)
from voyager_compiler.codegen.targets.magnet_alpha.serialization import (
    serialize_finalized_model,
    serialize_model,
)
from voyager_compiler.codegen.targets.magnet_alpha.storage import (
    EXECUTION_MODES,
    plan_alpha_storage,
)
from voyager_compiler.shape_prop import ShapeProp


LOGGER = logging.getLogger(__name__)


def gen_code(
    model: GraphModule,
    args,
    *,
    config=None,
    schedule=None,
    local_scheduler=None,
    scratchpad_mapping=None,
    execution_order=None,
    runtime_tolerance=None,
    tensor_dir=None,
):
    """Finalize and serialize Alpha's flat collateral.

    The signature is the target terminal-codegen contract used by
    :func:`voyager_compiler.compile`.  Scheduling and storage planning happen
    here, before serialization; the serializer only projects their finalized
    results.
    """
    del runtime_tolerance

    order_mode, mapping_mode = validate_execution_order_mode(
        execution_order,
        scratchpad_mapping,
    )
    require_execution_mode(order_mode, mapping_mode, EXECUTION_MODES)

    # The semantic graph must carry concrete shapes before a physical rewrite
    # can count channel blocks.  Re-propagate afterwards so every inserted
    # mover carries the same final value metadata as the owners around it.
    ShapeProp(model).propagate(*args)
    apply_memory_rewrites(model, config)

    # Target rewrites may have created nodes since the preceding propagation.
    # Every scheduler workload and serialized tensor below reads final shapes.
    ShapeProp(model).propagate(*args)

    if config is not None and schedule is not None:
        scheduled = bind_alpha_schedules(
            model,
            config,
            schedule,
            backend=local_scheduler,
        )
        LOGGER.info("bound local schedules to %d Alpha records", scheduled)
    elif local_scheduler is not None:
        raise ValueError(
            "an Alpha local scheduler was selected without a provider set"
        )

    planned = plan_alpha_storage(
        model,
        config=config,
        scratchpad_mapping=mapping_mode,
        execution_order=order_mode,
    )

    if mapping_mode == PER_OPERATION_SCRATCHPAD_MAPPING:
        return serialize_model(model, tensor_dir=tensor_dir)

    if mapping_mode != END_TO_END_SCRATCHPAD_MAPPING:
        raise AssertionError(f"unexpected Alpha mapping mode {mapping_mode!r}")

    prepared = build_alpha_allocation_requests(planned, config=config)
    scratchpad_placements = AlphaGlobalAllocator().allocate(
        prepared.allocation_requests
    )
    dram_placements = AlphaDramAllocator().allocate(prepared.plan)
    finalized = build_alpha_finalized_serialization(
        prepared,
        scratchpad_placements,
        dram_placements,
    )
    params = serialize_finalized_model(finalized, tensor_dir=tensor_dir)

    model.meta["storage_plan"] = prepared.plan
    model.meta["alpha_prepared_storage"] = prepared
    publish_alpha_bank_placements(
        model,
        prepared,
        scratchpad_placements,
    )
    return params


__all__ = ["gen_code"]
