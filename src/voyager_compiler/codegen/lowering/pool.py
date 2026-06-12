"""
Bufferization builder for 2-D pooling ops (``max_pool2d`` / ``avg_pool2d``).

Modelled on ``TiledConv2d`` (``gemm.py``) but pooling is *per-channel*: output
channels equal input channels and there is no channel mixing, so there is **no
input-channel reduction loop** — and no weight / bias / scale.  A single outer
``while_loop`` walks the (N, C, oH, oW) output grid; each iteration loads the
receptive-field halo for one output tile and applies the pool op.

  input (N, C, H, W) -> output (N, C, oH, oW)

Spatial tiling reuses the conv halo trick: the input is padded once up front so
each tile pools with ``padding=0`` and its input region is a plain *strided halo
load* (overlap = the receptive field).  Max-pool pads with ``-inf`` (so a padded
boundary window's max ignores it); avg-pool pads with ``0``.

An optional ``tail_fn`` is applied to each output tile before the store (e.g. a
fused ``quantize`` after the pool).
"""

from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
from torch._higher_order_ops.while_loop import while_loop

import voyager_compiler.decomposed  # noqa: F401  registers quantized_ops
from voyager_compiler import export_model
from voyager_compiler.codegen.lowering.common import (
    _apply_tail,
    _finalize_exported_gm,
    _fuse_tail_in_body,
    _lenient_verifier,
    _tag_loop_extents,
    voyager,
)
from voyager_compiler.codegen.lowering.gemm import (
    _NHWC,
    _pad_spec,
    _phys_pos,
    _project,
    _unproject,
)
from voyager_compiler.codegen.passes.utils import _pair


class Pool2dAxes(NamedTuple):
    n: int  # batch
    c: int  # channels (input == output; no reduction)
    h: int  # output rows (oh as a size, oy as an index)
    w: int  # output cols (ow as a size, ox as an index)


class TiledPool2d(torch.nn.Module):
    """Bufferized tiled 2-D pooling — see the module docstring.

    Like ``TiledConv2d`` minus the input-channel reduction: a single outer
    ``while_loop`` over the (N, C, oH, oW) grid, each tile a strided halo load of
    the input followed by the pool op (with ``padding=0``).
    """

    def __init__(
        self,
        target: torch._ops.OpOverload,
        tile_sizes: Pool2dAxes,
        kernel_size,
        stride,
        padding=(0, 0),
        dilation=(1, 1),
        extra_args: Tuple = (),
        pad_value: float = 0.0,
        *,
        nhwc: bool = False,
        tail_fn: Optional[Callable[..., torch.Tensor]] = None,
        tail_input_specs: Optional[list] = None,
        output_specs: Optional[List[Tuple[tuple, tuple, torch.dtype]]] = None,
    ):
        super().__init__()
        self.target = target
        # A complete ``Pool2dAxes`` (one tile size per dim); the caller resolves any
        # whole-dimension default, as the operand shapes are known at export time.
        self.tile_sizes = tile_sizes
        self.kernel_size = _pair(kernel_size)
        # Pooling stride defaults to the kernel size (PyTorch convention).
        self.stride = _pair(stride) if stride else self.kernel_size
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        # The op's args *after* ``padding`` (which each tile forces to 0): e.g.
        # ``[dilation, ceil_mode]`` for max_pool2d, ``[ceil_mode, count_include_pad,
        # divisor_override]`` for avg_pool2d.  Replayed verbatim in ``_tile_op``.
        self.extra_args = tuple(extra_args)
        # ``-inf`` for max-pool (a padded boundary window's max then ignores it), ``0``
        # for avg-pool.
        self.pad_value = pad_value
        # One bit picks the layout (logical NCHW or transposed NHWC), as TiledConv2d;
        # loads/stores project the logical ``Pool2dAxes`` onto the physical order.
        self.input_dims = _NHWC if nhwc else None
        self.output_dims = _NHWC if nhwc else None
        self.tail_fn = tail_fn
        self.tail_input_specs = tail_input_specs
        self.output_specs = output_specs

    def _input_halo(self, tile_oh, tile_ow):
        """Input halo ``(size_h, size_w, step_h, step_w)`` for an output spatial tile:
        the receptive field of ``tile_o`` outputs, stepped by ``tile_o * stride`` along
        H/W (the same strided-overlap load as ``TiledConv2d``)."""
        kH, kW = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        ih = (tile_oh - 1) * sh + dh * (kH - 1) + 1
        iw = (tile_ow - 1) * sw + dw * (kW - 1) + 1
        return ih, iw, tile_oh * sh, tile_ow * sw

    def _tile_op(self, indices: Pool2dAxes, input):
        """Pool one output tile: a strided halo load of the (pre-padded) input, then
        the pool op with ``padding=0``."""
        tile = self.tile_sizes
        ih, iw, step_h, step_w = self._input_halo(tile.h, tile.w)
        # Input feature-map tile (logical NCHW, projected onto the physical order):
        # channels stepped by tile.c (no overlap), H/W a strided halo (overlap = the
        # receptive field).
        in_idx = (indices.n, indices.c, indices.h, indices.w)
        input_tile = voyager.load_tile(
            input,
            _project(in_idx, self.input_dims),
            _project((tile.n, tile.c, ih, iw), self.input_dims),
            tile_strides=_project((tile.n, tile.c, step_h, step_w), self.input_dims),
        )
        return self.target(
            input_tile,
            list(self.kernel_size),
            list(self.stride),
            [0, 0],            # padding already baked into the whole input
            *self.extra_args,
        )

    def _outer_loop_body(self, n, c, oy, ox, input, tail_operands):
        output_tile = self._tile_op(Pool2dAxes(n=n, c=c, h=oy, w=ox), input)
        # Fused pointwise tail over the output tile + its operands, at the output block
        # (logical (n, c, oy, ox) projected onto the output's physical order).
        return _apply_tail(
            self.tail_fn,
            output_tile,
            tail_operands,
            self.tail_input_specs,
            _project((n, c, oy, ox), self.output_dims),
        )

    def forward(
        self,
        input: torch.Tensor,
        tail_operands: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        N, C, H, W = _unproject(input.shape, self.input_dims)
        kH, kW = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        oH = (H + 2 * ph - dh * (kH - 1) - 1) // sh + 1
        oW = (W + 2 * pw - dw * (kW - 1) - 1) // sw + 1

        # Pad the whole input once (outside every loop) so each tile pools with
        # padding=0 and its input region is a plain strided halo load.  Pad the H/W
        # *physical* dims, with the pool's identity value (``-inf`` max / ``0`` avg).
        if ph or pw:
            pad = _pad_spec(self.input_dims, ph, pw)
            input = torch.nn.functional.pad(input, pad, value=self.pad_value)

        tile = self.tile_sizes
        num_n = N // tile.n
        num_c = C // tile.c
        num_oy = oH // tile.h
        num_ox = oW // tile.w
        n_dim = _phys_pos(0, self.input_dims)

        # One output buffer per tail output (resolved by build_pool2d_buffers / the
        # pass, like ``tile_sizes``).
        output_bufs = [
            voyager.alloc(shape, dtype)
            for shape, _t, dtype in self.output_specs
        ]

        def cond_fn(n, c, oy, ox, *bufs):
            return n < input.shape[n_dim] // tile.n

        def body_fn(n, c, oy, ox, *bufs):
            output_tiles = self._outer_loop_body(n, c, oy, ox, input, tail_operands)
            if not isinstance(output_tiles, (tuple, list)):
                output_tiles = (output_tiles,)
            # Store/index in the output's physical order (logical (N, C, oH, oW)).
            new_bufs = [
                voyager.store_tile(
                    output_tile,
                    output_buf,
                    _project((n, c, oy, ox), self.output_dims),
                    ts,
                )
                for output_tile, output_buf, (_s, ts, _d) in zip(
                    output_tiles, bufs, self.output_specs
                )
            ]
            # Advance the (n, c, oy, ox) tile grid (one fused increment op).
            n_next, c_next, oy_next, ox_next = voyager.increment_indices(
                (n, c, oy, ox), (num_n, num_c, num_oy, num_ox)
            )
            return (n_next, c_next, oy_next, ox_next, *new_bufs)

        final = while_loop(cond_fn, body_fn, (0, 0, 0, 0, *output_bufs))
        outputs = list(final[4:])
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def build_pool2d_buffers(
    target: torch._ops.OpOverload,
    tile_sizes: List[int],
    input: torch.Tensor,
    *,
    kernel_size,
    stride=(),
    padding=(0, 0),
    dilation=(1, 1),
    extra_args: Tuple = (),
    pad_value: float = 0.0,
    nhwc: bool = False,
    tail_fn: Optional[Callable[..., torch.Tensor]] = None,
    tail_operands: Optional[Tuple[torch.Tensor, ...]] = None,
    tail_input_specs: Optional[list] = None,
    output_specs: Optional[List[Tuple[tuple, tuple, torch.dtype]]] = None,
) -> torch.fx.GraphModule:
    """Build the bufferized ``while_loop`` nest for a 2-D pool op.

    ``tile_sizes`` is ``[tile_n, tile_c]`` (spatial kept whole) or
    ``[tile_n, tile_c, tile_oh, tile_ow]`` to also tile the output spatial grid.
    ``extra_args`` are the op's args after ``padding`` (``[dilation, ceil_mode]`` for
    max_pool2d; ``[ceil_mode, count_include_pad, divisor_override]`` for avg_pool2d);
    ``pad_value`` is the value the input is padded with (``-inf`` for max-pool).
    ``nhwc=True`` selects the transposed (NHWC) layout; ``tile_sizes`` stays logical.
    """
    if len(tile_sizes) == 4:
        tile_n, tile_c, tile_oh, tile_ow = tile_sizes
    else:
        tile_n, tile_c = tile_sizes
        tile_oh = tile_ow = None  # whole-spatial

    in_dims = _NHWC if nhwc else None
    N, C, H, W = _unproject(input.shape, in_dims)
    kH, kW = _pair(kernel_size)
    sh, sw = _pair(stride) if stride else (kH, kW)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    oH = (H + 2 * ph - dh * (kH - 1) - 1) // sh + 1
    oW = (W + 2 * pw - dw * (kW - 1) - 1) // sw + 1
    if tile_oh is None:
        tile_oh = oH
    if tile_ow is None:
        tile_ow = oW

    if output_specs is None:
        out_dims = _NHWC if nhwc else None
        output_specs = [(
            _project((N, C, oH, oW), out_dims),
            _project((tile_n, tile_c, tile_oh, tile_ow), out_dims),
            input.dtype,
        )]

    pattern = TiledPool2d(
        target,
        Pool2dAxes(n=tile_n, c=tile_c, h=tile_oh, w=tile_ow),
        kernel_size,
        (sh, sw),
        padding=padding,
        dilation=dilation,
        extra_args=extra_args,
        pad_value=pad_value,
        nhwc=nhwc,
        tail_fn=tail_fn,
        tail_input_specs=tail_input_specs,
        output_specs=output_specs,
    )

    export_kwargs = {}
    if tail_operands is not None:
        export_kwargs["tail_operands"] = tail_operands
    with _lenient_verifier():
        gm = export_model(pattern, (input,), kwargs=export_kwargs)
    gm = _finalize_exported_gm(gm)
    if tail_fn is not None:
        # Re-group the pool op + its tail into a nested call_module (L1 fusion).
        _fuse_tail_in_body(gm, target, tail_fn)

    # One loop level: the (n, c, oy, ox) output grid (no inner reduction).
    _tag_loop_extents(
        gm, [[N // tile_n, C // tile_c, oH // tile_oh, oW // tile_ow]]
    )
    return gm
