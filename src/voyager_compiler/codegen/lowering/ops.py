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
``voyager.subview(src, o, s, st)``   strided window onto a buffer (a bank slot).
``voyager.insert(src, dst, sem)``    destination-passing compute-result write.
``voyager.async_copy(..., sem)``     guarded async tile DMA; signals semaphore ``sem``.
``voyager.async_wait(sem)``          waits on (consumes) a DMA semaphore.
``voyager.increment_indices(...)``   multi-dim tile-index counter (carry).
``voyager.delinearize_index(...)``   linear loop counter -> multi-dim index.
"""

from contextlib import contextmanager
from enum import IntEnum
from typing import Optional, Tuple

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
    LOCAL = 1  # local buffers (input and weight buffers)
    SRAM = 2  # on-chip scratchpad / SRAM
    DRAM = 3  # main system memory (e.g., DDR)


# ---------------------------------------------------------------------------
# Software-pipeline banking.
#
# ``alloc`` / ``zeros`` take a ``banks`` count.  ``banks == 0`` is an unbanked
# storage object: the tensor has exactly ``size``.  ``banks >= 1`` prepends a
# bank dimension (``[banks, *size]``), so ``size`` stays the payload of *one*
# bank and ``buf[slot]`` reads/writes a slot.  The distinction matters to the
# code generator: a bank dimension is not a tensor dimension — it is serialized
# as ``TensorBox.bank_count`` and the ``select`` that picks a slot collapses
# into the operand's ``TensorBoxRef.bank``.  A single-slot bank (``banks == 1``)
# still keeps its dimension, so ``buf[0]`` addresses it uniformly.
# ---------------------------------------------------------------------------
UNBANKED = 0


def _banked_size(size, banks: int):
    return list(size) if banks == UNBANKED else [banks] + list(size)


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
    "alloc(SymInt[] size, ScalarType dtype, int space=3, int banks=0) "
    "-> Tensor"
)


@impl(voyager_lib, "alloc", "CompositeExplicitAutograd")
def alloc(
    size: Tuple[int, ...],
    dtype: torch.dtype,
    space: int = MemoryLevel.DRAM,
    banks: int = UNBANKED,
) -> torch.Tensor:
    size = _banked_size(size, banks)
    if dtype.is_floating_point:
        return torch.randn(size).to(dtype)
    return torch.zeros(size, dtype=dtype)


@torch.library.register_fake("voyager::alloc")
def _alloc_fake(
    size: Tuple[int, ...],
    dtype: torch.dtype,
    space: int = MemoryLevel.DRAM,
    banks: int = UNBANKED,
) -> torch.Tensor:
    return torch.empty(_banked_size(size, banks), dtype=dtype)


# ---------------------------------------------------------------------------
# voyager.zeros — zero-initialised on-chip tile / bank (e.g. a reduction
# accumulator or a per-slot DMA-semaphore bank).  A ``voyager`` op (not
# ``aten.zeros``) so the bufferizer can tell a control/semaphore zero-init from
# a genuine ``aten.zeros`` compute op.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "zeros(SymInt[] size, ScalarType dtype, int banks=0) -> Tensor"
)


@impl(voyager_lib, "zeros", "CompositeExplicitAutograd")
def zeros(
    size: Tuple[int, ...], dtype: torch.dtype, banks: int = UNBANKED
) -> torch.Tensor:
    return torch.zeros(_banked_size(size, banks), dtype=dtype)


@torch.library.register_fake("voyager::zeros")
def _zeros_fake(
    size: Tuple[int, ...], dtype: torch.dtype, banks: int = UNBANKED
) -> torch.Tensor:
    return torch.zeros(_banked_size(size, banks), dtype=dtype)


# ---------------------------------------------------------------------------
# voyager.subview — a strided window onto a buffer, after MLIR's memref.subview:
# ``offsets`` / ``sizes`` / ``strides`` all carry one entry per *source* dim, so
# the result has the source's rank.  It names no storage of its own; the code
# generator folds it into the reference to the buffer it views (a bank pick
# becomes ``TensorBoxRef.bank``).
#
# It must return a genuine *view*: a bank slot is a write destination
# (``insert(src, bank_slot)``), and the FX graph stays executable, so a copy
# would land the write in a throwaway tensor.  ``as_strided`` gives the view and
# keeps the result's shape static -- only its storage offset is dynamic, which is
# what lets the slot index (``step % num_banks``) stay a runtime value.
# ---------------------------------------------------------------------------
voyager_lib.define(
    "subview(Tensor(a) source, SymInt[] offsets, SymInt[] sizes, "
    "SymInt[] strides) -> Tensor(a)"
)


def _subview(
    source: torch.Tensor,
    offsets: Tuple[int, ...],
    sizes: Tuple[int, ...],
    strides: Tuple[int, ...],
) -> torch.Tensor:
    window = [source.stride(d) * s for d, s in enumerate(strides)]
    offset = source.storage_offset() + sum(
        o * source.stride(d) for d, o in enumerate(offsets)
    )
    return torch.as_strided(source, sizes, window, offset)


impl(voyager_lib, "subview", "CompositeExplicitAutograd")(_subview)
torch.library.register_fake("voyager::subview")(_subview)


voyager_lib.define(
    "insert(Tensor src, Tensor(a!) dst, Tensor(b!)? semaphore=None) -> ()"
)


@impl(voyager_lib, "insert", "CompositeExplicitAutograd")
def insert(
    src: torch.Tensor,
    dst: torch.Tensor,
    semaphore: Optional[torch.Tensor] = None,
) -> None:
    dst.copy_(src)
    if semaphore is not None:
        semaphore.add_(1)


@torch.library.register_fake("voyager::insert")
def _insert_fake(
    src: torch.Tensor,
    dst: torch.Tensor,
    semaphore: Optional[torch.Tensor] = None,
) -> None:
    return None


# ---------------------------------------------------------------------------
# voyager.async_copy / voyager.async_wait — the asynchronous DMA pair.
#
# ``async_copy(..., semaphore)`` copies a tile then "signals" a semaphore;
# ``async_wait(semaphore)`` "waits on" it. Both take an int64 ``semaphore``
# scalar, **return nothing**, and declare it **mutable** (``Tensor(a!)`` /
# ``Tensor(b!)``). The eager impls emulate a counting semaphore: ``async_copy``
# increments it (post / V), ``async_wait`` asserts ``> 0`` and decrements it
# (wait / P) — so a *correct* pipeline keeps every wait matched by a prior copy
# into the same slot (a stray wait trips the assert).
# ---------------------------------------------------------------------------
voyager_lib.define(
    "async_copy(Tensor src, Tensor(a!) dst, SymInt[] indices, SymInt[] sizes, "
    "Tensor(b!) semaphore, SymInt[]? dims=None, SymInt[]? strides=None, "
    "bool transposed=False, SymInt[]? pad=None, float? pad_value=None) -> ()"
)


@impl(voyager_lib, "async_copy", "CompositeExplicitAutograd")
def async_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    indices: Tuple[int, ...],
    sizes: Tuple[int, ...],
    semaphore: torch.Tensor,
    dims: Optional[Tuple[int, ...]] = None,
    strides: Optional[Tuple[int, ...]] = None,
    transposed: bool = False,
    pad: Optional[Tuple[int, ...]] = None,
    pad_value: Optional[float] = None,
) -> None:
    if strides is None:
        strides = sizes
    # The buffer (the multi-tile operand) is sliced at the block index; the
    # other operand is exactly one ``sizes`` tile. Identify it by shape, not
    # numel as a halo tile can be *larger* than a small whole input. A
    # transposed copy is always a load, so an untiled weight must not pick the
    # store branch.
    buf = src if transposed or tuple(src.shape) != tuple(sizes) else dst
    rank = buf.dim()
    # Assemble the per-dim block index: ``indices`` covers the tiled ``dims``
    # (or every dim when ``dims`` is None); untiled dims default to 0.
    if dims is None:
        full = list(indices)
    else:
        full = [0] * rank
        for value, d in zip(indices, dims):
            full[d] = value
    # A halo with padding starts at ``full[d]*stride - pad[d]`` (default pad 0).
    off = pad if pad is not None else [0] * rank
    start = [full[d] * strides[d] - off[d] for d in range(rank)]
    if buf is src and pad_value is not None:
        # Padded halo load: ``start`` may run off either edge, so don't slice
        # directly (a negative Python index would wrap). Fill the tile with
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
    else:
        sl = tuple(slice(start[d], start[d] + sizes[d]) for d in range(rank))
        if buf is src:
            region = src[sl]
            dst.copy_(region.mT if transposed else region)
        else:
            dst[sl] = src.mT if transposed else src
    # Emulate a counting semaphore: the completed transfer signals its slot.
    semaphore.add_(1)


@torch.library.register_fake("voyager::async_copy")
def _async_copy_fake(
    src: torch.Tensor,
    dst: torch.Tensor,
    indices: Tuple[int, ...],
    sizes: Tuple[int, ...],
    semaphore: torch.Tensor,
    dims: Optional[Tuple[int, ...]] = None,
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
