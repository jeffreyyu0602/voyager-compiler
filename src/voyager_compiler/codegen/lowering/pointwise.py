"""
Bufferization builder for pointwise / elementwise ops.

Builds an explicit, executable *bufferized FX graph*: an output buffer allocated
with ``voyager.alloc`` and a single ``while_loop`` over the output-tile grid whose
body loads each input tile with ``voyager.load_tile``, applies the op, and writes the
result with ``voyager.store_tile``.

Self-contained: it follows the loop *pattern* of ``codegen/lowering/pointwise.py``
Broadcasting / scalar handling is
reimplemented locally.
"""

from typing import Callable, List, Optional, Tuple

import torch
from torch._higher_order_ops.while_loop import while_loop

from voyager_compiler import export_model
from voyager_compiler.codegen.lowering.common import (
    _InputSpec,
    _finalize_exported_gm,
    _lenient_verifier,
    _tag_loop_extents,
    voyager,
)


class TiledPointwise(torch.nn.Module):
    """
    Bufferized tiled N-D pointwise / batched-reduction op over a single ``while_loop``.

    The loop walks the grid of the **last** output (the full, un-reduced one) at
    ``tile_sizes``.  Dimensions whose ``tile_sizes[i] == size`` are processed whole, so a
    *batched reduction* — layernorm / softmax / mx-quantize over its last dim(s) — is
    expressed by leaving the reduction dim(s) whole and tiling the leading dims:
    ``target`` is applied to each whole tile, so the reduction is complete within it.

    The caller supplies the input ``_InputSpec``s and the per-output ``(full_shape,
    tile_shape, dtype)`` specs (each output's tile is the grid tile clamped to its own
    shape, so a reduced ``scale`` dim gets tile 1; codebook operands are marked whole).
    ``target`` is the callable applied to the loaded tiles — usually the op itself; for
    ops with scalar args interleaved between the tensor operands (quantize_mx) the caller
    passes a small closure that re-inserts the scalars.  Multiple outputs (quantize_mx ->
    scale + quantized) are each stored to their own buffer.
    """

    def __init__(
        self,
        target: Callable,
        tile_sizes: List[int],
        input_specs: List[_InputSpec],
        output_specs: List[Tuple[tuple, tuple, torch.dtype]],
    ):
        super().__init__()
        self.target = target
        # Per-operand ``_InputSpec`` and per-output ``(full_shape, tile_shape, dtype)``,
        # built by the caller (the bufferize pass marks codebook operands whole — see
        # ``_build_for_pointwise``).
        self.input_specs = input_specs
        self.output_specs = output_specs
        # Grid reference = the last output (the full one — quantize_mx returns
        # ``(scale, quantized)``); the loop walks its grid at ``tile_sizes``.
        grid_shape = tuple(output_specs[-1][0])
        self.ndim = len(grid_shape)
        self.num_tiles = tuple(s // t for s, t in zip(grid_shape, tile_sizes))

    def forward(self, *inputs):
        output_bufs = [
            voyager.alloc(shape, dtype) for shape, _, dtype in self.output_specs
        ]

        def cond_fn(*state):
            return state[0] < self.num_tiles[0]

        def body_fn(*state):
            indices = state[:self.ndim]
            bufs = state[self.ndim:]

            input_tiles = []
            for inp, spec in zip(inputs, self.input_specs):
                if spec.is_scalar:
                    input_tiles.append(inp)
                else:
                    # Static block index per input dim (0 for broadcast dims).
                    block = tuple(
                        0 if bcast else indices[i]
                        for i, bcast in zip(spec.idx_sel, spec.is_broadcast)
                    )
                    input_tiles.append(
                        voyager.load_tile(inp, block, spec.tile_sizes)
                    )

            output_tiles = self.target(*input_tiles)
            if not isinstance(output_tiles, (tuple, list)):
                output_tiles = (output_tiles,)
            # Each output stored at the shared loop block with its own tile shape (a
            # reduced dim has tile 1, so its offset there is 0).
            new_bufs = [
                voyager.store_tile(output_tile, output_buf, indices, ts)
                for output_tile, output_buf, (_s, ts, _d) in zip(
                    output_tiles, bufs, self.output_specs
                )
            ]

            # Advance the N-D tile grid (one fused increment op).
            new_idx = voyager.increment_indices(indices, self.num_tiles)
            return (*new_idx, *new_bufs)

        init_state = tuple(0 for _ in range(self.ndim)) + tuple(output_bufs)
        final = while_loop(cond_fn, body_fn, init_state)
        outputs = list(final[self.ndim:])
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def build_pointwise_buffers(
    target: Callable,
    tile_sizes: List[int],
    *inputs: torch.Tensor,
    input_specs: List[_InputSpec],
    output_specs: List[Tuple[tuple, tuple, torch.dtype]],
    kwargs: Optional[dict] = None,
) -> torch.fx.GraphModule:
    """
    Build the bufferized FX graph (single while_loop over voyager.* primitives) for a
    pointwise / batched-reduction op.  The loop walks the grid of the last (full) output.

    Args:
        target:       the callable applied to the loaded tiles — the op, or a closure
                      that re-inserts scalar args (see ``TiledPointwise``).
        tile_sizes:   grid tile (the main operand's tiling; reduction dims kept whole).
        inputs:       example input tensors (positional) — traced by export.
        input_specs:  per-operand load spec (codebook operands marked whole).
        output_specs: per-output ``(full_shape, tile_shape, dtype)`` sizing the buffers.
    """
    pattern = TiledPointwise(target, tile_sizes, input_specs, output_specs)
    with _lenient_verifier():
        gm = export_model(pattern, tuple(inputs), kwargs=kwargs)
    gm = _finalize_exported_gm(gm)

    grid_shape = tuple(output_specs[-1][0])
    num_tiles = [s // t for s, t in zip(grid_shape, tile_sizes)]
    _tag_loop_extents(gm, [num_tiles])
    return gm
