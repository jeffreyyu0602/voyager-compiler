"""
``voyager`` torch.library namespace: explicit bufferization / memory primitives.

These are intentionally distinct from ``quantized_ops`` so that compiler memory
semantics (logical storage objects, DMA tile loads/stores, accumulator init) are
not confused with model tensor semantics.  They are emitted by the bufferization
lowering pass and consumed by the loop-aware code generator.

Primitives
----------
``voyager.alloc(size, dtype)``       logical output / temporary storage (DRAM).
``voyager.zeros(size, dtype)``       zero-initialised tile / bank (accumulator, semaphore).
``voyager.copy_tile(src, dst, ...)`` side-effecting tile DMA (load or store).
``voyager.async_copy(..., sem)``     guarded async tile DMA; signals semaphore ``sem``.
``voyager.async_wait(sem)``          waits on (consumes) a DMA semaphore.
``voyager.increment_indices(...)``   multi-dim tile-index counter (carry).
``voyager.delinearize_index(...)``   linear loop counter -> multi-dim index.
"""

from contextlib import contextmanager
from enum import IntEnum
from typing import List, Optional, Tuple

import torch
from torch.library import Library, impl

# "DEF" creates/owns the namespace.  Keep a module-level handle alive.
voyager_lib = Library("voyager", "DEF")


class MemoryLevel(IntEnum):
    """Memory-hierarchy levels, ordered by distance from the compute units
    (smaller = nearer/faster).  ``voyager.alloc(..., space=)`` records which
    level a bufferized storage object lives in (e.g. an ``SRAM``
    software-pipeline bank vs the ``DRAM`` output buffer); shared wherever a
    buffer's level is needed."""

    REGISTER = 0  # PE register file
    LOCAL = 1  # local (per-PE / line) buffers
    SRAM = 2  # on-chip scratchpad / SRAM
    DRAM = 3  # main system memory (e.g., DDR)


# ---------------------------------------------------------------------------
# voyager.alloc — logical storage object (output / staging buffer or on-chip
# tile).  ``space`` is a ``MemoryLevel`` (int): which level of the hierarchy
# it lives in — ``DRAM`` (default: output / staging) or ``SRAM`` (an on-chip
# scratchpad tile, e.g. a software-pipeline ping-pong bank filled by
# ``voyager.memcpy``).  Eager allocation is space-agnostic (a plain tensor);
# the level is metadata the planner / codegen read.
# ---------------------------------------------------------------------------
# ``space`` default 3 == ``MemoryLevel.DRAM``.
voyager_lib.define(
    "alloc(SymInt[] size, ScalarType dtype, int space=3) -> Tensor"
)


@impl(voyager_lib, "alloc", "CompositeExplicitAutograd")
def alloc(
    size: Tuple[int, ...], dtype: torch.dtype, space: int = MemoryLevel.DRAM
) -> torch.Tensor:
    return torch.empty(size, dtype=dtype)


@torch.library.register_fake("voyager::alloc")
def _alloc_fake(
    size: Tuple[int, ...], dtype: torch.dtype, space: int = MemoryLevel.DRAM
) -> torch.Tensor:
    return torch.empty(size, dtype=dtype)


# ---------------------------------------------------------------------------
# voyager.zeros — zero-initialised on-chip tile / bank (e.g. a reduction
# accumulator or a per-slot DMA-semaphore bank).  A ``voyager`` op (not
# ``aten.zeros``) so the bufferizer can tell a control/semaphore zero-init from
# a genuine ``aten.zeros`` compute op.
# ---------------------------------------------------------------------------
voyager_lib.define("zeros(SymInt[] size, ScalarType dtype) -> Tensor")


@impl(voyager_lib, "zeros", "CompositeExplicitAutograd")
def zeros(size: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(size, dtype=dtype)


@torch.library.register_fake("voyager::zeros")
def _zeros_fake(size: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(size, dtype=dtype)


# ---------------------------------------------------------------------------
# Per-dim block-index assembly for ``copy_tile``.
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
# voyager.copy_tile — the tile DMA: copy one tile between ``src`` and ``dst``,
# covering both a DRAM buffer -> SRAM tile load and an SRAM tile -> DRAM buffer
# store; the two are the same move in opposite directions.  The *buffer* operand
# (more than one tile of storage)
# is addressed at block ``indices`` (start ``indices[d] * strides[d]``, span
# ``sizes[d]``); the other operand is the whole ``sizes`` tile.  ``dims`` /
# ``static_indices`` give a partial block index; ``strides`` defaults to
# ``sizes`` (a smaller step yields overlapping tiles — the conv halo);
# ``transposed`` applies ``.mT``.  ``dst`` is mutated in place (``Tensor(a!)``),
# so a bufferized graph names its on-chip / output buffers explicitly
# (``voyager.alloc``) instead of threading tile tensors as SSA / loop-carried
# values.
#
# On a *load* (``src`` is the buffer), ``pad`` shifts the tile start by
# ``-pad[d]`` (the receptive-field offset of a halo with padding) and
# ``pad_value`` fills any region the shifted window pushes off an edge — so a
# pooling / conv boundary tile is handled in the load itself (uniform fetch
# params), instead of materializing a padded input buffer.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "copy_tile(Tensor src, Tensor(a!) dst, SymInt[] indices, SymInt[] sizes, "
    "SymInt[]? dims=None, SymInt[]? static_indices=None, "
    "SymInt[]? strides=None, bool transposed=False, "
    "SymInt[]? pad=None, float? pad_value=None) -> ()"
)


@impl(voyager_lib, "copy_tile", "CompositeExplicitAutograd")
def copy_tile(
    src: torch.Tensor,
    dst: torch.Tensor,
    indices: Tuple[int, ...],
    sizes: Tuple[int, ...],
    dims: Optional[Tuple[int, ...]] = None,
    static_indices: Optional[Tuple[int, ...]] = None,
    strides: Optional[Tuple[int, ...]] = None,
    transposed: bool = False,
    pad: Optional[Tuple[int, ...]] = None,
    pad_value: Optional[float] = None,
) -> None:
    if strides is None:
        strides = sizes
    # The buffer (the multi-tile operand) is sliced at the block index; the
    # other operand is exactly one ``sizes`` tile.  Identify it by shape, not
    # numel: a halo tile can be *larger* than a small whole input (so numel
    # mis-picks), but only the buffer differs from the tile shape.  ``src`` is
    # the buffer => a load (dst tile <- src block); else a store / whole-tile
    # copy (dst block <- src tile).  When both equal ``sizes`` (a whole,
    # untiled operand) it falls to the store branch, which copies the whole
    # tile.
    buf = src if tuple(src.shape) != tuple(sizes) else dst
    rank = buf.dim()
    full = _full_indices(rank, indices, dims, static_indices)
    # A halo with padding starts at ``full[d]*stride - pad[d]`` (default pad 0).
    off = pad if pad is not None else [0] * rank
    start = [full[d] * strides[d] - off[d] for d in range(rank)]
    if buf is src and pad_value is not None:
        # Padded halo load: ``start`` may run off either edge, so don't slice
        # directly (a negative Python index would wrap).  Fill the tile with
        # ``pad_value`` and overwrite the in-bounds intersection —
        # out-of-bounds rows/cols stay padded.
        dst.fill_(pad_value)
        src_sl, dst_sl = [], []
        for d in range(rank):
            lo, hi = start[d], start[d] + sizes[d]
            clo, chi = max(lo, 0), min(hi, src.shape[d])
            src_sl.append(slice(clo, chi))
            dst_sl.append(slice(clo - lo, chi - lo))
        dst[tuple(dst_sl)] = src[tuple(src_sl)]
        return
    sl = tuple(slice(start[d], start[d] + sizes[d]) for d in range(rank))
    if buf is src:
        region = src[sl]
        dst.copy_(region.mT if transposed else region)
    else:
        dst[sl] = src.mT if transposed else src


@torch.library.register_fake("voyager::copy_tile")
def _copy_tile_fake(
    src: torch.Tensor,
    dst: torch.Tensor,
    indices: Tuple[int, ...],
    sizes: Tuple[int, ...],
    dims: Optional[Tuple[int, ...]] = None,
    static_indices: Optional[Tuple[int, ...]] = None,
    strides: Optional[Tuple[int, ...]] = None,
    transposed: bool = False,
    pad: Optional[Tuple[int, ...]] = None,
    pad_value: Optional[float] = None,
) -> None:
    return None


# ---------------------------------------------------------------------------
# voyager.async_copy / voyager.async_wait — the asynchronous DMA pair.
#
# ``async_copy(..., semaphore)`` is ``copy_tile`` that "signals" a semaphore;
# ``async_wait(semaphore)`` "waits on" it.  Both take an int64 ``semaphore``
# scalar, **return nothing**, and declare it **mutable** (``Tensor(a!)`` /
# ``Tensor(b!)``).  The eager impls emulate a counting semaphore: ``async_copy``
# increments it (post / V), ``async_wait`` asserts ``> 0`` and decrements it
# (wait / P) — so a *correct* pipeline keeps every wait matched by a prior copy
# into the same slot (a stray wait trips the assert).  The transfer itself is
# synchronous (already finished), and the ``register_fake`` impls are no-ops
# (export traces through the fake, never the eager body).
#
# The ``(a!)`` / ``(b!)`` mutation does two jobs at the graph level:
#
#   * the shared-semaphore write-after-write (a copy writing ``sem[slot]`` then
#     a later wait writing the same ``sem[slot]``) is the copy→wait ordering
#     edge — the data dependency the old returned token used to carry through
#     the loop's token vectors;
#   * it makes both ops impure, so ``torch.export`` keeps them (a no-output,
#     non-mutating op is functionalized away *inside* the ``while_loop`` body
#     where the wait must live — verified empirically; the ``has_side_effect``
#     registrations in ``lowering/__init__.py`` are belt-and-suspenders).
# ---------------------------------------------------------------------------
voyager_lib.define(
    "async_copy(Tensor src, Tensor(a!) dst, SymInt[] indices, SymInt[] sizes, "
    "Tensor(b!) semaphore, SymInt[]? dims=None, SymInt[]? static_indices=None, "
    "SymInt[]? strides=None, bool transposed=False, "
    "SymInt[]? pad=None, float? pad_value=None) -> ()"
)


@impl(voyager_lib, "async_copy", "CompositeExplicitAutograd")
def async_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    indices: Tuple[int, ...],
    sizes: Tuple[int, ...],
    semaphore: torch.Tensor,
    dims: Optional[Tuple[int, ...]] = None,
    static_indices: Optional[Tuple[int, ...]] = None,
    strides: Optional[Tuple[int, ...]] = None,
    transposed: bool = False,
    pad: Optional[Tuple[int, ...]] = None,
    pad_value: Optional[float] = None,
) -> None:
    copy_tile(
        src,
        dst,
        indices,
        sizes,
        dims,
        static_indices,
        strides,
        transposed,
        pad,
        pad_value,
    )
    # Emulate a counting semaphore: the completed transfer signals its slot.
    semaphore.add_(1)
    return None


@torch.library.register_fake("voyager::async_copy")
def _async_copy_fake(
    src: torch.Tensor,
    dst: torch.Tensor,
    indices: Tuple[int, ...],
    sizes: Tuple[int, ...],
    semaphore: torch.Tensor,
    dims: Optional[Tuple[int, ...]] = None,
    static_indices: Optional[Tuple[int, ...]] = None,
    strides: Optional[Tuple[int, ...]] = None,
    transposed: bool = False,
    pad: Optional[Tuple[int, ...]] = None,
    pad_value: Optional[float] = None,
) -> None:
    return None


voyager_lib.define("async_wait(Tensor(a!) semaphore) -> ()")


# The counting-semaphore check in ``async_wait`` is a runtime *oracle*: it
# verifies that a real pipeline run keeps every wait matched by a prior copy
# into the same slot.  It is only meaningful when the whole loop executes in
# sequence.  Codegen's ShapeProp re-runs a single loop-body iteration in
# isolation — where a wait legitimately precedes its matching copy (which lives
# in an earlier iteration) — so it disables the oracle via ``oracle_disabled()``
# (it needs only shapes/values, not the balance invariant).
_oracle_enabled = True


@contextmanager
def oracle_disabled():
    """Suspend the ``async_wait`` counting-semaphore oracle in this scope."""
    global _oracle_enabled
    prev = _oracle_enabled
    _oracle_enabled = False
    try:
        yield
    finally:
        _oracle_enabled = prev


@impl(voyager_lib, "async_wait", "CompositeExplicitAutograd")
def async_wait(semaphore: torch.Tensor) -> None:
    # Emulate a counting semaphore: block until signaled, then consume.  The
    # transfer is synchronous (already done), so this only checks the invariant
    # that a matching ``async_copy`` ran first.  Skipped when the oracle is
    # disabled (codegen ShapeProp — see ``oracle_disabled``).
    if not _oracle_enabled:
        return None
    assert (
        semaphore.item() > 0
    ), "async_wait on a semaphore with no pending async_copy"
    semaphore.sub_(1)
    return None


@torch.library.register_fake("voyager::async_wait")
def _async_wait_fake(semaphore: torch.Tensor) -> None:
    return None


# ---------------------------------------------------------------------------
# voyager.increment_indices — advance a nested tile-index counter by one step.
#
# Plain integer counters with innermost-first carry: the last index is bumped,
# and each overflow (``index[d] >= ends[d]``) carries +1 into ``index[d-1]``
# and wraps ``index[d]``.  The outermost index (d=0) is never wrapped — the
# enclosing ``while_loop``'s bound on it terminates the loop.  This folds the
# per-dimension ``add / floordiv / mod`` carry arithmetic into a single node so
# the bufferized graph stays readable instead of sprouting many scalar ops per
# loop step.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "increment_indices(SymInt[] indices, SymInt[] ends) -> SymInt[]"
)


def _advance_indices(
    indices: Tuple[int, ...],
    ends: Tuple[int, ...],
) -> Tuple[int, ...]:
    out = list(indices)
    out[-1] = out[-1] + 1
    for d in range(len(out) - 1, 0, -1):
        out[d - 1] = out[d - 1] + out[d] // ends[d]
        out[d] = out[d] % ends[d]
    return tuple(out)


@impl(voyager_lib, "increment_indices", "CompositeExplicitAutograd")
def increment_indices(
    indices: Tuple[int, ...],
    ends: Tuple[int, ...],
) -> Tuple[int, ...]:
    return _advance_indices(indices, ends)


@torch.library.register_fake("voyager::increment_indices")
def _increment_indices_fake(
    indices: Tuple[int, ...],
    ends: Tuple[int, ...],
) -> Tuple[int, ...]:
    return _advance_indices(indices, ends)


# ---------------------------------------------------------------------------
# voyager.delinearize_index — recover a multi-dim tile index from a single
# linear index and a per-dimension ``basis`` (the tile counts), mirroring
# MLIR's ``affine.delinearize_index``.  A flattened (software-pipelined) loop
# carries just a *single* counter — a real ``for`` induction variable — and
# delinearizes it each iteration for addressing, instead of carrying the whole
# multi-dim index as loop state.  Row-major: ``linear = d0*(b1*…*bn) + … + dn``
# with ``0 <= di < basis[i]``.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "delinearize_index(SymInt linear, SymInt[] basis) -> SymInt[]"
)


def _delinearize(linear: int, basis: Tuple[int, ...]) -> Tuple[int, ...]:
    out = [0] * len(basis)
    for d in range(len(basis) - 1, -1, -1):
        out[d] = linear % basis[d]
        linear = linear // basis[d]
    return tuple(out)


@impl(voyager_lib, "delinearize_index", "CompositeExplicitAutograd")
def delinearize_index(linear: int, basis: Tuple[int, ...]) -> Tuple[int, ...]:
    return _delinearize(linear, basis)


@torch.library.register_fake("voyager::delinearize_index")
def _delinearize_index_fake(
    linear: int, basis: Tuple[int, ...]
) -> Tuple[int, ...]:
    return _delinearize(linear, basis)
