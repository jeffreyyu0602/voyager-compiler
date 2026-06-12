"""
``voyager`` torch.library namespace: explicit bufferization / memory primitives.

These are intentionally distinct from ``quantized_ops`` so that compiler memory
semantics (logical storage objects, DMA tile loads/stores, accumulator init) are
not confused with model tensor semantics.  They are emitted by the bufferization
lowering pass and consumed by the loop-aware code generator.

Primitives
----------
``voyager.alloc(size, dtype)``       logical output / temporary storage (DRAM).
``voyager.zero_tile(size, dtype)``   zero-initialised accumulator tile.
``voyager.load_tile(...)``           DMA load of a tile into on-chip SRAM.
``voyager.store_tile(src, dest...)`` side-effecting DMA store of a tile.
``voyager.increment_indices(...)``   multi-dim tile-index counter (carry).
"""

from typing import List, Optional, Tuple

import torch
from torch.library import Library, impl

# "DEF" creates/owns the namespace.  Keep a module-level handle alive.
voyager_lib = Library("voyager", "DEF")


# ---------------------------------------------------------------------------
# voyager.alloc — logical storage object (output / temp buffer), lives in DRAM.
# ---------------------------------------------------------------------------
voyager_lib.define("alloc(SymInt[] size, ScalarType dtype) -> Tensor")


@impl(voyager_lib, "alloc", "CompositeExplicitAutograd")
def alloc(size: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(size, dtype=dtype)


@torch.library.register_fake("voyager::alloc")
def _alloc_fake(size: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(size, dtype=dtype)


# ---------------------------------------------------------------------------
# voyager.zero_tile — zero-initialised accumulator tile.
# ---------------------------------------------------------------------------
voyager_lib.define("zero_tile(SymInt[] size, ScalarType dtype) -> Tensor")


@impl(voyager_lib, "zero_tile", "CompositeExplicitAutograd")
def zero_tile(size: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(size, dtype=dtype)


@torch.library.register_fake("voyager::zero_tile")
def _zero_tile_fake(size: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(size, dtype=dtype)


# ---------------------------------------------------------------------------
# Per-dim block-index assembly shared by load_tile / store_tile.
# ---------------------------------------------------------------------------
def _full_indices(
    rank: int,
    indices: Tuple[int, ...],
    dims: Optional[Tuple[int, ...]],
    static_indices: Optional[Tuple[int, ...]],
) -> List[int]:
    """
    Assemble the per-dim block index of length ``rank``.

    ``indices`` supplies the block index for the *tiled* ``dims`` (one entry per
    ``dims`` entry); ``dims=None`` means ``indices`` already covers every dim.
    The remaining (untiled) dims take their block index from ``static_indices``
    (constant, default 0).  Keeping the dynamic ``indices`` separate from the
    constant fill lets each be a *homogeneous* list (all loop counters vs. all
    ints) — which the loop-aware code generator can serialize, unlike a single
    list that mixes counters and constants.
    """
    if dims is None:
        return list(indices)
    full = list(static_indices) if static_indices is not None else [0] * rank
    for value, d in zip(indices, dims):
        full[d] = value
    return full


# ---------------------------------------------------------------------------
# voyager.load_tile — emulate a DMA load of a tile from ``input``.
#
# ``indices`` is the per-dim block index: the tile start along dim d is
# ``indices[d] * tile_strides[d]`` and the tile spans ``tile_sizes[d]``.  When
# only some dims are tiled, pass those dims' block indices in ``indices`` and name
# them in ``dims`` (one per ``indices`` entry); the untiled dims take their block
# index from ``static_indices`` (constant, default 0).  ``dims=None`` (default)
# means ``indices`` covers every dim.  ``tile_strides`` defaults to ``tile_sizes``
# (contiguous tiling); a smaller step yields *overlapping* tiles (the conv input
# halo).  ``transposed`` applies ``.mT`` to the loaded tile.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "load_tile(Tensor input, SymInt[] indices, SymInt[] tile_sizes, "
    "SymInt[]? dims=None, SymInt[]? static_indices=None, "
    "SymInt[]? tile_strides=None, bool transposed=False) -> Tensor"
)


@impl(voyager_lib, "load_tile", "CompositeExplicitAutograd")
def load_tile(
    input: torch.Tensor,
    indices: Tuple[int, ...],
    tile_sizes: Tuple[int, ...],
    dims: Optional[Tuple[int, ...]] = None,
    static_indices: Optional[Tuple[int, ...]] = None,
    tile_strides: Optional[Tuple[int, ...]] = None,
    transposed: bool = False,
) -> torch.Tensor:
    if tile_strides is None:
        tile_strides = tile_sizes

    rank = input.dim()
    full = _full_indices(rank, indices, dims, static_indices)
    assert rank == len(tile_sizes) == len(tile_strides) == len(full)

    start = [full[d] * tile_strides[d] for d in range(rank)]
    slices = tuple(
        slice(start[d], start[d] + tile_sizes[d]) for d in range(rank)
    )
    return input[slices].mT if transposed else input[slices]


@torch.library.register_fake("voyager::load_tile")
def _load_tile_fake(
    input: torch.Tensor,
    indices: Tuple[int, ...],
    tile_sizes: Tuple[int, ...],
    dims: Optional[Tuple[int, ...]] = None,
    static_indices: Optional[Tuple[int, ...]] = None,
    tile_strides: Optional[Tuple[int, ...]] = None,
    transposed: bool = False,
) -> torch.Tensor:
    output = input.new_empty(tile_sizes)
    return output.mT if transposed else output


# ---------------------------------------------------------------------------
# voyager.store_tile — side-effecting DMA store of a tile into ``dest``.
# Returns ``dest`` (the stable buffer handle), mutated in place.  ``indices`` /
# ``dims`` / ``static_indices`` give the per-dim block index exactly as in
# ``load_tile``; the tile is written at ``index[d] * tile_sizes[d]``.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "store_tile(Tensor src, Tensor dest, SymInt[] indices, SymInt[] tile_sizes, "
    "SymInt[]? dims=None, SymInt[]? static_indices=None) -> Tensor"
)


@impl(voyager_lib, "store_tile", "CompositeExplicitAutograd")
def store_tile(
    src: torch.Tensor,
    dest: torch.Tensor,
    indices: Tuple[int, ...],
    tile_sizes: Tuple[int, ...],
    dims: Optional[Tuple[int, ...]] = None,
    static_indices: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    rank = dest.dim()
    full = _full_indices(rank, indices, dims, static_indices)
    assert rank == len(tile_sizes) == len(full)

    start = [full[d] * tile_sizes[d] for d in range(rank)]
    slices = tuple(
        slice(start[d], start[d] + tile_sizes[d]) for d in range(rank)
    )
    dest[slices] = src
    return dest


@torch.library.register_fake("voyager::store_tile")
def _store_tile_fake(
    src: torch.Tensor,
    dest: torch.Tensor,
    indices: Tuple[int, ...],
    tile_sizes: Tuple[int, ...],
    dims: Optional[Tuple[int, ...]] = None,
    static_indices: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    return torch.empty_like(dest)


# ---------------------------------------------------------------------------
# voyager.increment_indices — advance a nested tile-index counter by one step.
#
# Plain integer counters with innermost-first carry: the last index is bumped,
# and each overflow (``index[d] >= ends[d]``) carries +1 into ``index[d-1]`` and
# wraps ``index[d]``.  The outermost index (d=0) is never wrapped — the enclosing
# ``while_loop``'s bound on it terminates the loop.  This folds the per-dimension
# ``add / floordiv / mod`` carry arithmetic into a single node so the bufferized
# graph stays readable instead of sprouting many scalar ops per loop step.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "increment_indices(SymInt[] indices, SymInt[] ends) -> SymInt[]"
)


def _advance_indices(
    indices: Tuple[int, ...],
    ends: Tuple[int, ...],
) -> List[int]:
    out = list(indices)
    out[-1] = out[-1] + 1
    for d in range(len(out) - 1, 0, -1):
        out[d - 1] = out[d - 1] + out[d] // ends[d]
        out[d] = out[d] % ends[d]
    return out


@impl(voyager_lib, "increment_indices", "CompositeExplicitAutograd")
def increment_indices(
    indices: Tuple[int, ...],
    ends: Tuple[int, ...],
) -> List[int]:
    return _advance_indices(indices, ends)


@torch.library.register_fake("voyager::increment_indices")
def _increment_indices_fake(
    indices: Tuple[int, ...],
    ends: Tuple[int, ...],
) -> List[int]:
    return _advance_indices(indices, ends)
