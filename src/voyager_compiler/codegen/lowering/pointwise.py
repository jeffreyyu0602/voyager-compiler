"""
Bufferization builders for pointwise / elementwise ops.

Builds an explicit, *fully bufferized* FX graph: every storage object — the DRAM
output buffer(s) and the on-chip SRAM tile banks — is named by ``voyager.alloc``, and
``voyager.copy_tile`` moves a tile between them (one DMA op; the same op loads
DRAM->SRAM and stores SRAM->DRAM).  No tile tensor flows as an SSA value or as
``while_loop`` carried state: the banks / output are *additional* (closed-over) inputs
the body mutates in place, so the loop carries only its integer index.  The compute op
itself (relu, layernorm, ...) is left as an ordinary node reading its input bank(s);
only its DMA operands are bufferized.

Two variants:
  * ``TiledPointwise``           — sequential loop over the tile grid: load each input
    tile into its bank, compute, store.  Carries the multi-dim index (advanced by
    ``increment_indices``), which codegen reconstructs as a nested ``for``-loop walk.
  * ``DoubleBufferedPointwise``  — software-pipelined with two SRAM banks per input,
    *unrolled by two* so the banks are referenced statically (``b0`` / ``b1``).  Because
    the prefetch crosses tile-grid dimension boundaries it can't be a nested loop, so it
    carries a **single linear counter** (a real ``for`` induction variable) and recovers
    the multi-dim tile index each iteration with ``voyager.delinearize_index`` (MLIR
    ``affine.delinearize_index``).  The prologue loads tile 0; the steady loop computes
    one bank while prefetching the next pair into the other; the trip count is peeled
    (``(total-1)//2``) so the body only ever prefetches in-bounds tiles, and a build-time
    -shaped epilogue drains the final pair / tile.  No clamp, no guard.

Trailing whole dims (``num_tiles == 1`` after the last tiled dim) are dropped from the
loop index; they're part of each tile, not loop levels.
"""

import math
from typing import Callable, List, Optional, Tuple

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler import export_model
from voyager_compiler.codegen.lowering.common import (
    _InputSpec,
    _OutputSpec,
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    voyager,
)
from voyager_compiler.codegen.lowering.ops import MemoryLevel, _delinearize

_SRAM = int(MemoryLevel.SRAM)


class _TiledPointwiseBase(torch.nn.Module):
    """
    Shared state + tile helpers for the pointwise builders.

    The grid is the **last** output's shape (the full, un-reduced one — ``quantize_mx``
    returns ``(scale, quantized)``) tiled at ``tile_sizes``; dims with ``tile_sizes[i] ==
    size`` are processed whole, so a batched reduction (layernorm / softmax / mx-quantize)
    leaves its reduction dim(s) whole and tiles the leading dims.

    The loop iterates dims ``0..last_tiled`` (``self.loop_ndim`` of them); any trailing
    whole dims are pinned to 0 in the block index (``_full_index``) and don't appear in
    the loop / trip.  ``kernel`` is a general ``Callable`` applied to the per-tile block
    index and the input *banks* — ``kernel(grid_index, *tiles)`` (a closure may re-insert
    scalar args, e.g. for ``quantize_mx``, or run a per-tile reduction keyed off
    ``grid_index``, e.g. conv's C-reduction); scalar / codebook operands are passed whole,
    the rest are loaded into SRAM banks.
    """

    def __init__(
        self,
        kernel: Callable,
        grid: Tuple[int, ...],
        input_specs: List[Optional[_InputSpec]],
        output_specs: List[_OutputSpec],
    ):
        super().__init__()
        self.kernel = kernel
        self.grid = tuple(grid)
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.total = math.prod(self.grid)
        self.ndim = len(self.grid)
        # Drop trailing whole dims: the loop covers dims 0..last_tiled.
        tiled = [d for d in range(self.ndim) if self.grid[d] > 1]
        self.loop_ndim = (tiled[-1] + 1) if tiled else 0
        self.loop_tiles = self.grid[:self.loop_ndim]

    # -- allocation ------------------------------------------------------------

    def _alloc_dram(self):
        """Allocate the DRAM output buffer(s)."""
        return [
            voyager.alloc(spec.shape, spec.dtype) for spec in self.output_specs
        ]

    def _alloc_sram(self, inputs):
        """Allocate one SRAM tile bank per *tiled* input (dtype from the input)."""
        return [
            voyager.alloc(list(spec.tile_sizes), inp.dtype, _SRAM)
            for inp, spec in self._tiled(inputs)
        ]

    # -- tile helpers ----------------------------------------------------------

    def _tiled(self, inputs):
        """The ``(input, spec)`` pairs that are tiled (loaded into banks) — i.e. the
        operands with a spec (a whole / scalar / codebook operand has ``None``)."""
        return [
            (inp, spec)
            for inp, spec in zip(inputs, self.input_specs)
            if spec is not None
        ]

    def _full_index(self, loop_indices):
        """Pad the ``loop_ndim`` loop indices back to a full ``ndim`` block index, the
        trailing (whole) dims pinned to 0."""
        return tuple(loop_indices) + (0,) * (self.ndim - self.loop_ndim)

    def _load(self, inputs, buffers, idx):
        """DMA each tiled input's tile at loop index ``idx`` into its SRAM buffer (side
        effect)."""
        grid_idx = self._full_index(idx)               # one block index per GRID dim
        for (inp, spec), buf in zip(self._tiled(inputs), buffers):
            # An operand dim is dynamic iff it maps to a tiled grid dim and isn't broadcast;
            # its index is the grid index there.  Whole / broadcast dims stay at block 0.
            dims, indices = [], []
            for d, grid_dim in enumerate(spec.index_map):
                if grid_dim < self.loop_ndim and not spec.is_broadcast[d]:
                    dims.append(d)
                    indices.append(grid_idx[grid_dim])
            # A halo spec (pooling / conv) overrides the contiguous defaults: ``strides`` is
            # the overlap step and ``pad`` / ``pad_value`` pad the boundary in-load.  All
            # ``None`` for a plain pointwise tile, so copy_tile keeps its contiguous defaults.
            voyager.copy_tile(
                inp,
                buf,
                indices,
                spec.tile_sizes,
                dims=dims if len(dims) < len(spec.index_map) else None,
                strides=spec.strides,
                pad=spec.pad,
                pad_value=spec.pad_value,
            )

    def _compute(self, inputs, buffers, idx):
        """Run ``kernel(grid_index, *tiles)`` — reading the input SRAM ``buffers`` for tiled
        operands and passing the whole (spec ``None``) scalar / codebook operands through.
        ``grid_index`` is the per-output-grid-dim block index (the loop counters at ``idx``
        padded with 0 for whole dims); a plain pointwise kernel ignores it."""
        grid_idx = self._full_index(idx)
        it = iter(buffers)
        args = [
            inp if spec is None else next(it)
            for inp, spec in zip(inputs, self.input_specs)
        ]
        return self.kernel(grid_idx, *args)

    def _store(self, outputs, buffers, idx):
        """DMA each output tile out to its (closed-over) DRAM buffer at loop index ``idx`` (side
        effect).  Mirrors ``_load``: an output dim takes a dynamic block index iff the grid dim
        it maps to is tiled (``grid[gd] > 1``); a whole grid dim is one tile, so it is omitted
        and ``copy_tile`` defaults it to block 0."""
        grid_idx = self._full_index(idx)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        for output, buf, spec in zip(outputs, buffers, self.output_specs):
            dims, indices = [], []
            for d, grid_dim in enumerate(spec.index_map):
                if self.grid[grid_dim] > 1:
                    dims.append(d)
                    indices.append(grid_idx[grid_dim])
            voyager.copy_tile(
                output,
                buf,
                indices,
                spec.tile_sizes,
                dims=dims if len(dims) < len(spec.index_map) else None,
            )

    def _result(self, outputs):
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    # -- shared sequential body (also the pipelined degenerate path) -----------

    def _emit_sequential(self, inputs, buffers):
        """One SRAM buffer per input; loop the grid loading, computing, and storing each
        tile into the DRAM output ``buffers``.  Carries the multi-dim index (advanced by
        ``increment_indices``); codegen rebuilds the nested ``for``-loop walk from the
        per-dim grid extents."""
        sram = self._alloc_sram(inputs)

        if self.loop_ndim == 0:  # nothing tiled — a single whole tile, no loop
            self._load(inputs, sram, ())
            self._store(self._compute(inputs, sram, ()), buffers, ())
        else:
            def cond_fn(*idx):
                return idx[0] < self.loop_tiles[0]

            def body_fn(*idx):
                self._load(inputs, sram, idx)
                self._store(self._compute(inputs, sram, idx), buffers, idx)
                return tuple(voyager.increment_indices(idx, self.loop_tiles))

            while_loop(cond_fn, body_fn, (0,) * self.loop_ndim)


class TiledPointwise(_TiledPointwiseBase):
    """Sequential bufferized pointwise: one ``while_loop`` over the tile grid; each
    iteration loads every input tile into its SRAM bank, computes, and stores the
    result.  Banks and output buffers are additional (closed-over) inputs written by the
    side-effecting ``copy_tile``; the loop carries only the tile index."""

    def forward(self, *inputs):
        buffers = self._alloc_dram()
        self._emit_sequential(inputs, buffers)
        return self._result(buffers)


class DoubleBufferedPointwise(_TiledPointwiseBase):
    """Software-pipelined pointwise, two SRAM banks per input, **unrolled by two** so the
    banks are static (``b0`` / ``b1``).  Carries a single linear counter ``i`` (a real
    ``for`` induction variable) and recovers each tile's multi-dim index with
    ``delinearize_index``.  Prologue loads tile 0 into ``b0``; iteration ``i`` covers the
    pair ``(2i, 2i+1)`` and prefetches the next pair's first tile ``2i+2`` into ``b0``;
    the trip ``(total-1)//2`` peels the tail so the prefetch is always in-bounds, and a
    build-time-shaped epilogue drains the final pair (even tile-count) or tile (odd)."""

    def forward(self, *inputs):
        buffers = self._alloc_dram()

        # Too few tiles to fill/drain the pipeline, or nothing tiled -> sequential.
        if self.total < 2:
            self._emit_sequential(inputs, buffers)
            return self._result(buffers)

        b0 = self._alloc_sram(inputs)
        b1 = self._alloc_sram(inputs)

        # Prologue: pre-load tile 0
        self._load(inputs, b0, [0] * self.loop_ndim)

        # The counter IS the linear index of the pair's first tile, so the loop steps by
        # 2 (one pair / iteration) and the body uses ``i, i+1, i+2`` directly.  It stops
        # at the peeled tail tile (total-2 even / total-1 odd), which the epilogue drains.
        tail = self.total - 2 if self.total % 2 == 0 else self.total - 1

        def cond_fn(i):
            return i < tail

        def body_fn(i):
            idx = voyager.delinearize_index(i, self.loop_tiles)
            idx1 = voyager.delinearize_index(i + 1, self.loop_tiles)
            idx2 = voyager.delinearize_index(i + 2, self.loop_tiles)
            self._load(inputs, b1, idx1)
            self._store(self._compute(inputs, b0, idx), buffers, idx)
            self._load(inputs, b0, idx2)
            self._store(self._compute(inputs, b1, idx1), buffers, idx1)
            return (i + 2,)

        while_loop(cond_fn, body_fn, (0,))

        # Epilogue: the tail tile (peeled off the loop, at linear index ``tail``); b0
        # already holds it from the loop's last prefetch.
        last0 = _delinearize(tail, self.loop_tiles)
        self._store(self._compute(inputs, b0, last0), buffers, last0)

        if self.total % 2 == 0:  # even: a partner tile remains
            last1 = _delinearize(self.total - 1, self.loop_tiles)
            # FIXME: should this load comes before the last compute?
            self._load(inputs, b1, last1)
            self._store(self._compute(inputs, b1, last1), buffers, last1)

        return self._result(buffers)


def build_pointwise_buffers(
    kernel: Callable,
    grid: Tuple[int, ...],
    in_specs: List[Optional[_InputSpec]],
    out_specs: List[_OutputSpec],
    inputs: Tuple[torch.Tensor, ...],
    *,
    pipelined: bool = False,
    kwargs: Optional[dict] = None,
) -> torch.fx.GraphModule:
    """
    Build the bufferized FX graph (a ``while_loop`` over ``voyager.*`` primitives) for a
    pointwise op — modelled on ``pl.pallas_call``.

    ``kernel(grid_index, *tiles)`` is the per-tile compute (a general ``Callable``); ``grid`` is
    the iteration space (tiles per dim); ``in_specs`` / ``out_specs`` are the per-operand
    BlockSpecs (``tile_sizes`` + ``index_map``, the output also carrying its buffer shape /
    dtype); ``inputs`` is the operand tuple (like ``torch.export``'s ``example_inputs``).
    ``pipelined`` selects the variant: ``False`` -> ``TiledPointwise`` (sequential), truthy ->
    ``DoubleBufferedPointwise`` (software-pipelined, two SRAM banks).
    """
    cls = DoubleBufferedPointwise if pipelined else TiledPointwise
    pattern = cls(kernel, grid, in_specs, out_specs)
    with _lenient_verifier():
        gm = export_model(pattern, inputs, kwargs=kwargs)
    gm = _finalize_exported_gm(gm)

    if pipelined and pattern.total >= 2:
        # Flattened double buffer: one ``for``-loop over a single linear counter that is
        # the pair's first tile index, stepping by 2 up to the peeled tail.  The multi-dim
        # index is recovered inside the body by ``delinearize_index`` (no carried
        # multi-index, no nested walk).
        tail = pattern.total - 2 if pattern.total % 2 == 0 else pattern.total - 1
        _tag_loop_extents(gm, [[(0, tail, 2)]])
    elif pattern.loop_tiles:
        # Sequential: codegen reconstructs the nested ``for``-loop walk from the per-dim
        # grid extents (the flat carried-index while_loop -> nested Loop protos).
        _tag_loop_extents(gm, [list(pattern.loop_tiles)])
    return gm
