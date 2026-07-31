"""Choosing tile sizes: the analytic per-op tilers and the mapping search.

``search`` sizes a vector / GEMV op against its own DRAM cost model;
``tiler`` drives the ``interstellar`` loop-nest mapper for everything the
matrix unit runs.  Nothing here imports ``bufferize`` — the dependency runs
one way, so the builders can ask for a tiling without a cycle.
"""

from .banking import scratchpad_bytes
from .cost import gemv_tile_latency, vector_op_utilization, vector_tile_latency
from .search import (
    DEFAULT_RUNTIME_TOLERANCE,
    gemv_op_tiling,
    pool_op_tiling,
    vector_op_tiling,
)
from .tiler import (
    CONV_L3_ORDER,
    GEMM_L3_ORDER,
    TilerContext,
    build_interstellar_tiler,
    get_tiling,
    prefetch_tilings,
)

__all__ = [
    "CONV_L3_ORDER",
    "DEFAULT_RUNTIME_TOLERANCE",
    "GEMM_L3_ORDER",
    "TilerContext",
    "build_interstellar_tiler",
    "gemv_op_tiling",
    "gemv_tile_latency",
    "get_tiling",
    "pool_op_tiling",
    "prefetch_tilings",
    "scratchpad_bytes",
    "vector_op_tiling",
    "vector_op_utilization",
    "vector_tile_latency",
]
