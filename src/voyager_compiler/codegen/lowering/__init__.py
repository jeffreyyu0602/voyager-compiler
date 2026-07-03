"""
Bufferized FX lowering.

Rewrites tiled FX nodes into an explicit, executable *bufferized FX graph* that
contains tile loops (``torch.ops.higher_order.while_loop``) and explicit memory
primitives in the ``voyager`` torch.library namespace (``voyager.alloc``,
``voyager.async_copy``, ``voyager.insert``), then generates protobuf /
graphviz / text from that graph.

Modules: ``ops`` (voyager.* primitives), ``utils`` (shared builder helpers +
layout projections), ``pipeline`` (the unified Pallas-style software-pipelining
scheduler and its GEMM / conv2d / pointwise / pool builders),
``bufferization`` (the rewrite pass), ``codegen`` (loop-aware output).
"""

import torch
from torch.fx.node import has_side_effect

from . import ops  # noqa: F401  (registers the voyager.* torch.library ops)

# Mark the in-place / DMA primitives side-effecting so dead-code elimination
# never drops them.  The reduction kernel stores its result with a
# ``voyager.insert`` *inside* the finalize ``torch.cond`` (store / tail fires
# only on the last reduction step); ``insert`` mutates in place and the
# cond's output is unused, so default DCE would prune the whole cond and
# silently drop the store.  ``async_copy`` / ``async_wait`` are likewise guarded
# by output-unused conds (the DMA prefetch / store guards).  Marking ``cond``
# and these primitives side-effecting keeps every one of them live.
has_side_effect(torch.ops.higher_order.cond)
has_side_effect(torch.ops.voyager.insert.default)
has_side_effect(torch.ops.voyager.async_copy.default)
has_side_effect(torch.ops.voyager.async_wait.default)
from .bufferization import annotate_tensor_spaces, bufferize_graph  # noqa: F401
from .codegen import (  # noqa: F401
    gen_code_bufferized,
    gen_compute_graph_bufferized,
    print_bufferized_graph,
)
from .memory_planning import MemoryPlan, plan_memory  # noqa: F401
from .reporting import (  # noqa: F401
    ScheduleResult,
    compress_schedule,
    estimate_schedule,
    report,
    write_excel_report,
    write_perfetto,
)

__all__ = [
    "ops",
    "bufferize_graph",
    "annotate_tensor_spaces",
    "gen_code_bufferized",
    "gen_compute_graph_bufferized",
    "print_bufferized_graph",
    "plan_memory",
    "MemoryPlan",
    "ScheduleResult",
    "estimate_schedule",
    "compress_schedule",
    "write_excel_report",
    "write_perfetto",
    "report",
]
