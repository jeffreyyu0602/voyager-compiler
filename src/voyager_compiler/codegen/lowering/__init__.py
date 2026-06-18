"""
Bufferized FX lowering.

Rewrites tiled FX nodes into an explicit, executable *bufferized FX graph* that
contains tile loops (``torch.ops.higher_order.while_loop``) and explicit memory
primitives in the ``voyager`` torch.library namespace (``voyager.alloc``,
``voyager.zero_tile``, ``voyager.load_tile``, ``voyager.store_tile``), then
generates protobuf / graphviz / text from that graph.

Modules: ``ops`` (voyager.* primitives), ``common`` (shared builder helpers),
``gemm`` / ``pointwise`` / ``attention`` (loop builders), ``bufferization`` (the
rewrite pass), ``codegen`` (loop-aware output).
"""

from . import ops  # noqa: F401  (registers the voyager.* torch.library ops)
from .bufferization import annotate_tensor_spaces, bufferize_graph  # noqa: F401
from .codegen import (  # noqa: F401
    gen_code_bufferized,
    gen_compute_graph_bufferized,
    print_bufferized_graph,
)
from .memory_planning import MemoryPlan, plan_memory  # noqa: F401

__all__ = [
    "ops",
    "bufferize_graph",
    "annotate_tensor_spaces",
    "gen_code_bufferized",
    "gen_compute_graph_bufferized",
    "print_bufferized_graph",
    "plan_memory",
    "MemoryPlan",
]
