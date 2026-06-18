"""
Bufferization builder for GEMM-family ops (linear / matmul / bmm).

Builds an explicit, executable *bufferized FX graph* for a tiled GEMM: an output
buffer allocated with ``voyager.alloc``, an outer ``while_loop`` over the output-tile
grid, a ``voyager.zero_tile`` accumulator, an inner ``while_loop`` reduction with
``voyager.load_tile`` of the operand tiles plus the GEMM op, an optional fused
pointwise tail applied to the accumulator tile, and a final ``voyager.store_tile``.

Loop counters are plain integers and tiles are addressed statically, so the
graph carries no index tensors; the ``voyager.*`` memory primitives express the
DMA loads/stores.

Export note
-----------
``while_loop`` cond subgraphs must *read the operand tensors* (e.g. via
``input.shape``) so torch.export keeps them as live additional-inputs with a
populated ``val`` field; precomputing the loop bound outside the cond leaves the
operand placeholder without a ``val`` and trips the export verifier.  Both cond
functions below therefore compute their bound inline from ``input.shape``.
"""

import math
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
from voyager_compiler.codegen.passes.utils import _pair


# Per-dimension tile descriptor.  One value per tiled dimension — the *same*
# shape serves as both the tile sizes and the block index (a tile size and a
# block index for every dim), so a single type is used for both: ``tile`` holds
# the sizes, ``indices`` the block positions.  The loop counters are carried flat by
# ``while_loop`` and bundled into an ``indices`` only for readable field access in the
# bodies (``tile.c``, ``indices.h``) instead of positional unpacking.
class GemmAxes(NamedTuple):
    b: int  # batch
    x: int  # output rows (M)
    k: int  # output channels (N)
    c: int  # reduction (K)


class Conv2dAxes(NamedTuple):
    n: int  # batch
    h: int  # output rows (oh as a size, oy as an index)
    w: int  # output cols (ow as a size, ox as an index)
    k: int  # output channels
    c: int  # input channels (reduction)


# --- Double-buffered reduction --------------------------------------------------
# Two ways to sum a tiled C-reduction with the operand loads software-pipelined
# ahead of the matmuls (so a later async-DMA pass can overlap them).  Both take a
# ``load_fn(c) -> tuple[Tensor, ...]`` that DMA-loads block ``c``'s operand tiles
# (no compute) and a ``compute_fn(tiles) -> Tensor`` that runs the op on
# already-loaded tiles and returns the partial to accumulate; ``load_fn`` returns
# an all-Tensor tuple so its tiles can be carried as ``while_loop`` state.


def _double_buffered_reduce(psum, num_c, load_fn, compute_fn):
    """Rolled 2-deep double buffer (``num_c >= 3``).

    The prologue loads block 0, the steady ``while_loop`` runs ``c`` in
    ``1..num_c-1`` (prefetch block ``c`` while accumulating block ``c-1``), and the
    epilogue drains the last block.  The steady-state extent is therefore
    ``(1, num_c, 1)`` — see ``build_gemm_buffers``.
    """
    tiles0 = load_fn(0)

    def cond_fn(c, psum_buf, *carried):
        return c < num_c

    def body_fn(c, psum_buf, *carried):
        # Prefetch block c, then accumulate the previously-loaded block (c-1).
        nxt = load_fn(c)
        psum_buf = psum_buf + compute_fn(carried)
        return (c + 1, psum_buf, *nxt)

    result = while_loop(cond_fn, body_fn, (1, psum, *tiles0))
    return result[1] + compute_fn(tuple(result[2:]))


def _unrolled_reduce(psum, num_c, load_fn, compute_fn):
    """Fully-unrolled reduction (Python ``for``), used when there are too few blocks
    (``num_c < 3``) to fill/drain a rolled pipeline.  All blocks are loaded up front
    and then accumulated, so the loads precede the matmuls in the straight-line
    schedule (the same prefetch-ahead ordering, just unrolled rather than rolled)."""
    loaded = [load_fn(c) for c in range(num_c)]
    for tiles in loaded:
        psum = psum + compute_fn(tiles)
    return psum


class TiledGEMM(torch.nn.Module):
    """
    Bufferized tiled GEMM for linear / matmul / bmm.

    ``transpose_weight`` fuses a transpose on the *weight operand* — e.g. the Kᵀ
    in a Transformer attention QKᵀ — into the DMA tile load: when True the weight
    tile is loaded transposed (``load_tile(..., transposed=True)`` applies
    ``.mT``), so the DMA engine performs the transpose instead of it being a
    separate op.  It is only about that fused transpose, independent of the op
    kind and of bias.  ``batched_weight`` selects a per-batch (bmm) weight.

    Bias, when present, is appended as the op's bias argument (fused); see
    ``_tile_op``.  An optional ``tail_fn`` (a traceable callable) is applied to
    the accumulator tile after the reduction loop and before the store; it
    carries fused post-GEMM pointwise ops (e.g. relu / dequantize).
    """

    def __init__(
        self,
        target: torch._ops.OpOverload,
        tile_sizes: GemmAxes,
        *,
        block_size: Optional[int] = None,
        accumulate_fp32: bool = False,
        batched_weight: bool = False,
        transpose_weight: bool = False,
        weight_ck: bool = False,
        pipelined: bool = False,
        tail_fn: Optional[Callable[..., torch.Tensor]] = None,
        tail_input_specs: Optional[list] = None,
        output_specs: Optional[List[Tuple[tuple, tuple, torch.dtype]]] = None,
    ):
        super().__init__()
        self.target = target
        # One ``_InputSpec`` per fused-tail operand (built by the caller, which has the
        # whole submodule), or ``None`` => pass whole (codebook / scalar); else load each
        # tile per the spec.  See ``_apply_tail``.
        self.tail_input_specs = tail_input_specs
        # One ``(full_shape, tile_shape, dtype)`` per output buffer.  A fused
        # ``tail_fn`` may be multi-output (e.g. ``quantize_mx`` returns
        # ``(scale, quantized)``), so the caller passes the shape/tile/dtype of
        # each; ``None`` => the single GEMM output ``(B, X, K)`` derived in forward.
        self.output_specs = output_specs
        # A complete ``GemmAxes`` (one tile size per dimension); required, since
        # the operand shapes are known at export time and the caller resolves any
        # whole-dimension default.
        self.tile_sizes = tile_sizes
        self.block_size = block_size
        self.accumulate_fp32 = accumulate_fp32
        self.batched_weight = batched_weight
        self.transpose_weight = transpose_weight
        # When True the C-reduction is software-pipelined (double-buffered): operand
        # tile loads are issued one block ahead of the matmul that consumes them.
        # Rolled for >= 3 blocks, fully unrolled for fewer; see ``_reduce_pipelined``.
        self.pipelined = pipelined
        # ``weight_ck`` names the weight's physical layout directly (like conv's
        # ``nhwc``), avoiding the ambiguous word "transposed" — which means opposite
        # things for linear (natural K,C) and matmul (natural C,K).  ``True`` => the
        # weight (and its scale) is stored ``(..., C, K)``; ``False`` => ``(..., K, C)``
        # (the GemmAxes/load convention).  The C,K form is what ``transpose_linear_weights``
        # relayouts a linear to, and the natural form of a matmul operand.  It only
        # reorders the weight tile's index/sizes; unlike ``transpose_weight`` (a fused
        # ``.mT`` op) it is not an actual transpose.
        self.weight_ck = weight_ck
        self.tail_fn = tail_fn

    def _wkc(self, k_val, c_val):
        """Order a (K, C) pair for the weight's physical layout: ``(K, C)`` by
        default, ``(C, K)`` under ``weight_ck``."""
        return (c_val, k_val) if self.weight_ck else (k_val, c_val)

    def _load_tiles(
        self,
        indices: GemmAxes,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """DMA-load one (output, C-block) tile's operands (no compute).

        Returns ``(input_tile, weight_tile)``, plus the two scale tiles under
        microscaling.  The tuple is all-Tensor (no ``None``) so it can be carried
        as ``while_loop`` state by the double-buffered reduction.
        """
        tile = self.tile_sizes
        bs = self.block_size

        input_tile = voyager.load_tile(
            input, (indices.b, indices.x, indices.c), (tile.b, tile.x, tile.c)
        )

        if self.batched_weight:
            weight_tile = voyager.load_tile(
                weight,
                (indices.b, *self._wkc(indices.k, indices.c)),
                (tile.b, *self._wkc(tile.k, tile.c)),
                transposed=self.transpose_weight,
            )
        else:
            weight_tile = voyager.load_tile(
                weight,
                self._wkc(indices.k, indices.c),
                self._wkc(tile.k, tile.c),
                transposed=self.transpose_weight,
            )

        if input_scale is None or weight_scale is None:
            return (input_tile, weight_tile)

        input_scale_tile = voyager.load_tile(
            input_scale,
            (indices.b, indices.x, indices.c),
            (tile.b, tile.x, tile.c // bs),
        )
        if self.batched_weight:
            weight_scale_tile = voyager.load_tile(
                weight_scale,
                (indices.b, *self._wkc(indices.k, indices.c)),
                (tile.b, *self._wkc(tile.k, tile.c // bs)),
                transposed=self.transpose_weight,
            )
        else:
            weight_scale_tile = voyager.load_tile(
                weight_scale,
                self._wkc(indices.k, indices.c),
                self._wkc(tile.k, tile.c // bs),
                transposed=self.transpose_weight,
            )
        return (input_tile, weight_tile, input_scale_tile, weight_scale_tile)

    def _matmul_tiles(
        self,
        tiles: Tuple[torch.Tensor, ...],
        bias: Optional[torch.Tensor],
        input_code: Optional[torch.Tensor] = None,
        weight_code: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the GEMM op on already-loaded operand tiles; ``bias`` (if given) is
        fused.  ``tiles`` is the tuple returned by ``_load_tiles``."""
        # Bias is the 3rd positional arg; everything after it (scales / codes) is
        # keyword, so just append it when present.
        args = [tiles[0], tiles[1]]
        if bias is not None:
            args.append(bias)

        if len(tiles) > 2:
            return self.target(
                *args,
                input_scale=tiles[2],
                weight_scale=tiles[3],
                block_size=self.block_size,
                input_code=input_code,
                weight_code=weight_code,
            )
        return self.target(*args)

    def _tile_op(
        self,
        indices: GemmAxes,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        input_code: Optional[torch.Tensor] = None,
        weight_code: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multiply one (output, C-block) tile; ``bias`` (if given) is fused."""
        tiles = self._load_tiles(
            indices, input, weight, input_scale=input_scale, weight_scale=weight_scale
        )
        return self._matmul_tiles(
            tiles, bias, input_code=input_code, weight_code=weight_code
        )

    def _reduce_sequential(self, b, x, k, input, weight, psum, num_c, **kwargs):
        """Sequential C-reduction: load + matmul + accumulate one block per iteration
        (today's default; no prefetch)."""
        def cond_fn(c, psum_buf):
            return c < num_c

        def body_fn(c, psum_buf):
            # Partials are bias-free; bias is added once before the reduction.
            output_tile = self._tile_op(
                GemmAxes(b, x, k, c), input, weight, None, **kwargs
            )
            if self.accumulate_fp32 and output_tile.dtype != torch.float32:
                output_tile = output_tile.to(torch.float32)
            return (c + 1, psum_buf + output_tile)

        _, final_psum = while_loop(cond_fn, body_fn, (0, psum))
        return final_psum

    def _reduce_pipelined(self, b, x, k, input, weight, psum, num_c, **kwargs):
        """Double-buffered C-reduction: prefetch each block's operand tiles one step
        ahead of the matmul that consumes them.  Rolled for ``num_c >= 3``, fully
        unrolled for fewer; both keep the prefetch-ahead ordering."""
        input_scale = kwargs.get("input_scale")
        weight_scale = kwargs.get("weight_scale")
        input_code = kwargs.get("input_code")
        weight_code = kwargs.get("weight_code")

        def load_fn(c):
            return self._load_tiles(
                GemmAxes(b, x, k, c),
                input,
                weight,
                input_scale=input_scale,
                weight_scale=weight_scale,
            )

        def compute_fn(tiles):
            out = self._matmul_tiles(
                tiles, None, input_code=input_code, weight_code=weight_code
            )
            if self.accumulate_fp32 and out.dtype != torch.float32:
                out = out.to(torch.float32)
            return out

        reduce_fn = _double_buffered_reduce if num_c >= 3 else _unrolled_reduce
        return reduce_fn(psum, num_c, load_fn, compute_fn)

    def _outer_loop_body(
        self,
        b,
        x,
        k,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        tail_operands,
        **kwargs,
    ):
        tile = self.tile_sizes
        num_c = input.shape[-1] // tile.c

        bias_tile = None
        if bias is not None:
            bias_tile = voyager.load_tile(bias, (k,), (tile.k,))

        if num_c == 1:
            # No C tiling: one GEMM produces the whole output tile, so bias is
            # fused into the op (weight @ x + bias is a single hardware op).
            final_psum = self._tile_op(
                GemmAxes(b, x, k, 0), input, weight, bias_tile, **kwargs
            )
        else:
            acc_dtype = torch.float32 if self.accumulate_fp32 else input.dtype
            psum = voyager.zero_tile((tile.b, tile.x, tile.k), acc_dtype)

            if bias_tile is not None:
                psum = psum + bias_tile.to(acc_dtype)

            reduce = self._reduce_pipelined if self.pipelined else self._reduce_sequential
            final_psum = reduce(b, x, k, input, weight, psum, num_c, **kwargs)

        if self.accumulate_fp32 and final_psum.dtype != input.dtype:
            final_psum = final_psum.to(input.dtype)

        # Fused pointwise tail over the output tile plus its operands, loaded per
        # the precomputed specs at the (b, x, k) output block.  May return a tuple
        # when the tail is multi-output (e.g. quantize_mx).
        return _apply_tail(
            self.tail_fn,
            final_psum,
            tail_operands,
            self.tail_input_specs,
            (b, x, k),
        )

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tail_operands: Optional[Tuple[torch.Tensor, ...]] = None,
        **kwargs,
    ):
        assert input.ndim >= 3, (
            "TiledGEMM expects an (..., X, C) input (>=1 batch dim)"
        )
        *batch_dims, X, C = input.shape
        # Fold any extra leading batch dims (e.g. a Transformer's B and H) into a
        # single batch so the loop nest stays one (B, X, K) grid; the output is
        # reshaped back at the end.  Operands carrying the batch dims fold too — but
        # only the tiled (non-scalar) tail operands; codebooks are passed whole.
        nb = len(batch_dims)
        if nb > 1:
            input = input.reshape(-1, X, C)
            if self.batched_weight:
                weight = weight.reshape(-1, *weight.shape[nb:])
            if tail_operands is not None:
                specs = self.tail_input_specs
                new_operands = []
                for i, t in enumerate(tail_operands):
                    # A whole (scalar / codebook) operand has spec ``None`` when specs are
                    # given — pass it through; tiled operands (and the no-specs case at all)
                    # fold the batch dims.
                    if specs is not None and specs[i] is None:
                        new_operands.append(t)
                    else:
                        new_operands.append(t.reshape(-1, *t.shape[nb:]))
                tail_operands = tuple(new_operands)
            for key in ("input_scale", "weight_scale"):
                scale = kwargs.get(key)
                if scale is not None and (key == "input_scale" or self.batched_weight):
                    kwargs[key] = scale.reshape(-1, *scale.shape[nb:])

        B = input.shape[0]
        # K is the second-to-last weight dim normally, the last under the
        # transposed (..., C, K) layout.
        K = weight.shape[-1] if self.weight_ck else weight.shape[-2]

        tile = self.tile_sizes
        num_b = B // tile.b
        num_x = X // tile.x
        num_k = K // tile.k

        # One output buffer per tail output (resolved by build_gemm_buffers / the
        # bufferization pass, like ``tile_sizes``).
        output_bufs = [
            voyager.alloc(shape, dtype) for shape, _, dtype in self.output_specs
        ]

        # ``output_bufs`` are closed over (additional inputs); ``store_tile`` writes them
        # in place (a side effect), so the loop carries only the (b, x, k) tile index.
        def cond_fn(b, x, k):
            return b < num_b

        def body_fn(b, x, k):
            output_tiles = self._outer_loop_body(
                b,
                x,
                k,
                input=input,
                weight=weight,
                bias=bias,
                tail_operands=tail_operands,
                **kwargs,
            )
            if not isinstance(output_tiles, (tuple, list)):
                output_tiles = (output_tiles,)
            for output_tile, output_buf, (_s, ts, _d) in zip(
                output_tiles, output_bufs, self.output_specs
            ):
                voyager.store_tile(output_tile, output_buf, (b, x, k), ts)
            # Advance the (b, x, k) tile grid (one fused increment op).
            return voyager.increment_indices((b, x, k), (num_b, num_x, num_k))

        while_loop(cond_fn, body_fn, (0, 0, 0))
        outputs = list(output_bufs)

        if nb > 1:  # restore the folded batch dims
            outputs = [o.reshape(*batch_dims, *o.shape[1:]) for o in outputs]

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


# --- Conv layout -------------------------------------------------------------
# There are exactly two conv layouts, picked by one bit.  By default everything
# is logical NCHW (weight OIHW); the optional ``transpose_conv2d_inputs_and_weights``
# pass flips a conv to the systolic-array layout, where the *input* is NHWC and
# the weight and output layouts follow the input — weight becomes HWIO
# ``(kH, kW, C, K)`` and the output is NHWC.  ``TiledConv2d`` reasons in logical
# NCHW/OIHW terms (the ``Conv2dAxes`` fields) and projects onto each operand's
# physical order only at the load/store boundary; ``nhwc=True`` selects the
# transposed layout.
#
# Each constant is the NCHW->physical permutation (physical pos ``i`` holds
# logical axis ``perm[i]``); ``None`` everywhere is the logical NCHW / OIHW layout.
_NHWC = (0, 2, 3, 1)  # input / output feature maps under the transposed layout
_HWIO = (2, 3, 1, 0)  # weight (kH, kW, C, K) under the transposed layout


def _project(per_axis: Tuple, dims: Optional[Tuple[int, ...]]) -> Tuple:
    """Reorder a per-logical-axis tuple into a tensor's physical order.

    ``per_axis[a]`` is the value for logical axis ``a`` — feature maps use NCHW
    (N=0, C/K=1, H=2, W=3), weights use OIHW (K=0, C=1, kH=2, kW=3).  ``dims`` is
    the NCHW->physical permutation: physical position ``i`` holds logical axis
    ``dims[i]``.
    """
    if dims is None:
        return tuple(per_axis)
    return tuple(per_axis[a] for a in dims)


def _unproject(physical: Tuple, dims: Optional[Tuple[int, ...]]) -> Tuple:
    """Inverse of ``_project``: read a physical-order sequence (e.g. a ``shape``)
    back into logical NCHW / OIHW axis order."""
    if dims is None:
        return tuple(physical)
    return tuple(physical[dims.index(a)] for a in range(len(physical)))


def _phys_pos(axis: int, dims: Optional[Tuple[int, ...]]) -> int:
    """Physical position of logical ``axis`` under permutation ``dims``."""
    return axis if dims is None else dims.index(axis)


def _pad_spec(dims: Optional[Tuple[int, ...]], ph: int, pw: int) -> Tuple[int, ...]:
    """``F.pad`` argument that pads a feature map's H/W on their physical dims
    (so the one-shot input pad is correct under NHWC as well as NCHW)."""
    pairs = [(0, 0)] * 4
    pairs[_phys_pos(2, dims)] = (ph, ph)  # H
    pairs[_phys_pos(3, dims)] = (pw, pw)  # W
    # ``F.pad`` consumes pairs from the last dim backward.
    return tuple(x for pair in reversed(pairs) for x in pair)


class TiledConv2d(torch.nn.Module):
    """
    Bufferized tiled conv2d (NCHW).

    Outer ``while_loop`` over the 4-D output grid (N, K, oH, oW); inner
    ``while_loop`` reduction over input-channel blocks C (kernel dims kept whole).
    Because conv2d is linear over input channels, the per-C-block partial
    convolutions sum to the full result.

      input  (N, C, H, W)   weight (K, C, kH, kW)   output (N, K, oH, oW)

    Spatial tiling: the input is padded once up front so each tile's convolution
    runs with ``padding=0``; the input feature-map region feeding an output tile
    is then a *strided halo load* — tile size ``(tile_o - 1)*stride + dil*(k-1) +
    1`` stepped by ``tile_o * stride`` along H/W, so adjacent output tiles overlap
    by the receptive-field halo.  ``tile_oh`` / ``tile_ow`` default to the full
    output (whole-spatial, a single spatial tile).

    An optional ``tail_fn`` is applied to the accumulator tile before the store.
    """

    def __init__(
        self,
        target: torch._ops.OpOverload,
        tile_sizes: Conv2dAxes,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups: int = 1,
        *,
        nhwc: bool = False,
        block_size: Optional[int] = None,
        accumulate_fp32: bool = False,
        pipelined: bool = False,
        tail_fn: Optional[Callable[..., torch.Tensor]] = None,
        tail_input_specs: Optional[list] = None,
        output_specs: Optional[List[Tuple[tuple, tuple, torch.dtype]]] = None,
    ):
        super().__init__()
        self.target = target
        # A complete ``Conv2dAxes`` (one tile size per dimension); required, since
        # the operand shapes are known at export time and the caller resolves any
        # whole-dimension default.
        self.tile_sizes = tile_sizes
        # One ``_InputSpec`` per fused-tail operand (see ``_apply_tail``).
        self.tail_input_specs = tail_input_specs
        # One ``(full_shape, tile_shape, dtype)`` per output buffer (physical
        # layout); a multi-output fused tail (e.g. ``quantize_mx``) supplies one
        # per output.  ``None`` => the single conv output derived in forward.
        self.output_specs = output_specs
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        # Grouped / depthwise conv would reduce each output group over only its
        # own channel slice; the C-reduction here assumes a dense conv, so only
        # groups=1 is supported (callers must guard before bufferizing).
        assert groups == 1, "TiledConv2d supports only dense conv (groups=1)"
        self.groups = groups
        # One bit picks the layout (the only two that exist): logical NCHW/OIHW, or
        # the transposed NHWC input + HWIO weight + NHWC output (weight and output
        # follow the input).  Loads/stores project the logical ``Conv2dAxes`` onto
        # each operand's physical order via these permutations (``None`` = logical).
        self.input_dims = _NHWC if nhwc else None
        self.weight_dims = _HWIO if nhwc else None
        self.output_dims = _NHWC if nhwc else None
        self.block_size = block_size
        self.accumulate_fp32 = accumulate_fp32
        # See ``TiledGEMM.pipelined``: double-buffer the C-reduction (prefetch the
        # next channel block's operand tiles while the current block convolves).
        self.pipelined = pipelined
        self.tail_fn = tail_fn

    def _input_halo(self, kH, kW, tile_oh, tile_ow):
        """
        Input feature-map halo tile for an output spatial tile: the (size, step)
        along H/W.  Size is the receptive field of ``tile_o`` outputs; the step
        between adjacent output tiles is ``tile_o * stride`` (the overlap is the
        halo).  Derived from the conv params, so the input halo is not part of
        the (n, k, c, h, w) tile descriptor.
        """
        sh, sw = self.stride
        dh, dw = self.dilation
        ih = (tile_oh - 1) * sh + dh * (kH - 1) + 1
        iw = (tile_ow - 1) * sw + dw * (kW - 1) + 1
        return ih, iw, tile_oh * sh, tile_ow * sw

    def _load_tiles(
        self,
        indices: Conv2dAxes,
        input,
        weight,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """DMA-load one (output, C-block) tile's operands (no compute).

        Returns ``(input_tile, weight_tile)``, plus the two scale tiles under
        microscaling.  The tuple is all-Tensor (no ``None``) so it can be carried as
        ``while_loop`` state by the double-buffered reduction.
        """
        tile = self.tile_sizes
        bs = self.block_size
        _, _, kH, kW = _unproject(weight.shape, self.weight_dims)
        ih, iw, step_h, step_w = self._input_halo(kH, kW, tile.h, tile.w)

        # Input feature-map tile: a strided halo load (overlap = receptive field)
        # from the pre-padded input; channels stepped by tile.c (no overlap).
        # Indices/sizes are logical NCHW, projected onto the input's physical order.
        in_idx = (indices.n, indices.c, indices.h, indices.w)
        input_tile = voyager.load_tile(
            input,
            _project(in_idx, self.input_dims),
            _project((tile.n, tile.c, ih, iw), self.input_dims),
            tile_strides=_project((tile.n, tile.c, step_h, step_w), self.input_dims),
        )
        # Only K / C are tiled; the kernel dims (kH, kW) stay block 0.  Their
        # physical positions go in ``dims`` so the index stays all loop counters.
        w_dims = (_phys_pos(0, self.weight_dims), _phys_pos(1, self.weight_dims))
        weight_tile = voyager.load_tile(
            weight,
            (indices.k, indices.c),
            _project((tile.k, tile.c, kH, kW), self.weight_dims),
            dims=w_dims,
        )

        if input_scale is None or weight_scale is None:
            return (input_tile, weight_tile)

        # Microscaling: per-C-block scales, same spatial halo as the input.
        input_scale_tile = voyager.load_tile(
            input_scale,
            _project(in_idx, self.input_dims),
            _project((tile.n, tile.c // bs, ih, iw), self.input_dims),
            tile_strides=_project(
                (tile.n, tile.c // bs, step_h, step_w), self.input_dims
            ),
        )
        weight_scale_tile = voyager.load_tile(
            weight_scale,
            (indices.k, indices.c),
            _project((tile.k, tile.c // bs, kH, kW), self.weight_dims),
            dims=w_dims,
        )
        return (input_tile, weight_tile, input_scale_tile, weight_scale_tile)

    def _conv_tiles(
        self,
        tiles: Tuple[torch.Tensor, ...],
        bias,
        input_code: Optional[torch.Tensor] = None,
        weight_code: Optional[torch.Tensor] = None,
    ):
        """Run the conv op on already-loaded operand tiles; ``bias`` (if given) is
        fused.  ``tiles`` is the tuple returned by ``_load_tiles``."""
        if len(tiles) > 2:
            return self.target(
                tiles[0],
                tiles[1],
                bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
                input_scale=tiles[2],
                weight_scale=tiles[3],
                block_size=self.block_size,
                input_code=input_code,
                weight_code=weight_code,
            )
        return self.target(
            tiles[0],
            tiles[1],
            bias,
            self.stride,
            (0, 0),
            self.dilation,
            self.groups,
        )

    def _tile_op(
        self,
        indices: Conv2dAxes,
        input,
        weight,
        bias,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        input_code: Optional[torch.Tensor] = None,
        weight_code: Optional[torch.Tensor] = None,
    ):
        """Convolve one (output, C-block) tile; ``bias`` (if given) is fused."""
        tiles = self._load_tiles(
            indices, input, weight, input_scale=input_scale, weight_scale=weight_scale
        )
        return self._conv_tiles(
            tiles, bias, input_code=input_code, weight_code=weight_code
        )

    def _reduce_sequential(self, n, oy, ox, k, input, weight, psum, num_c, **kwargs):
        """Sequential C-reduction: load + conv + accumulate one channel block per
        iteration (today's default; no prefetch)."""
        def cond_fn(c, psum_buf):
            return c < num_c

        def body_fn(c, psum_buf):
            # Partials are bias-free; bias is added once after the reduction.
            output_tile = self._tile_op(
                Conv2dAxes(n=n, h=oy, w=ox, k=k, c=c), input, weight, None, **kwargs
            )
            if self.accumulate_fp32 and output_tile.dtype != torch.float32:
                output_tile = output_tile.to(torch.float32)
            return (c + 1, psum_buf + output_tile)

        _, final_psum = while_loop(cond_fn, body_fn, (0, psum))
        return final_psum

    def _reduce_pipelined(self, n, oy, ox, k, input, weight, psum, num_c, **kwargs):
        """Double-buffered C-reduction: prefetch each channel block's operand tiles
        one step ahead of the conv that consumes them.  Rolled for ``num_c >= 3``,
        fully unrolled for fewer."""
        input_scale = kwargs.get("input_scale")
        weight_scale = kwargs.get("weight_scale")
        input_code = kwargs.get("input_code")
        weight_code = kwargs.get("weight_code")

        def load_fn(c):
            return self._load_tiles(
                Conv2dAxes(n=n, h=oy, w=ox, k=k, c=c),
                input,
                weight,
                input_scale=input_scale,
                weight_scale=weight_scale,
            )

        def compute_fn(tiles):
            out = self._conv_tiles(
                tiles, None, input_code=input_code, weight_code=weight_code
            )
            if self.accumulate_fp32 and out.dtype != torch.float32:
                out = out.to(torch.float32)
            return out

        reduce_fn = _double_buffered_reduce if num_c >= 3 else _unrolled_reduce
        return reduce_fn(psum, num_c, load_fn, compute_fn)

    def _outer_loop_body(
        self,
        n,
        oy,
        ox,
        k,
        input,
        weight,
        bias,
        tail_operands,
        **kwargs,
    ):
        tile = self.tile_sizes
        c_dim = _phys_pos(1, self.input_dims)  # physical position of input C
        num_c = input.shape[c_dim] // tile.c

        bias_tile = None
        if bias is not None:
            bias_tile = voyager.load_tile(bias, (k,), (tile.k,))

        if num_c == 1:
            # No C tiling: one conv produces the whole output tile, so bias is
            # fused into the op (conv + bias run together in hardware, no add).
            final_psum = self._tile_op(
                Conv2dAxes(n=n, h=oy, w=ox, k=k, c=0),
                input,
                weight,
                bias_tile,
                **kwargs,
            )
        else:
            acc_dtype = torch.float32 if self.accumulate_fp32 else input.dtype
            psum = voyager.zero_tile(
                _project((tile.n, tile.k, tile.h, tile.w), self.output_dims),
                acc_dtype,
            )

            reduce = self._reduce_pipelined if self.pipelined else self._reduce_sequential
            final_psum = reduce(n, oy, ox, k, input, weight, psum, num_c, **kwargs)

            if bias_tile is not None:
                # Broadcast bias along the output channel (K's physical position).
                bias_shape = [1, 1, 1, 1]
                bias_shape[_phys_pos(1, self.output_dims)] = tile.k
                final_psum = final_psum + bias_tile.reshape(bias_shape)

        if self.accumulate_fp32 and final_psum.dtype != input.dtype:
            final_psum = final_psum.to(input.dtype)

        # Fused pointwise tail over the output tile plus its operands, loaded per
        # the precomputed specs at the output block (logical (n, k, oy, ox)
        # projected onto the output's physical order).  May return a tuple when the
        # tail is multi-output (e.g. quantize_mx).
        return _apply_tail(
            self.tail_fn,
            final_psum,
            tail_operands,
            self.tail_input_specs,
            _project((n, k, oy, ox), self.output_dims),
        )

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tail_operands: Optional[Tuple[torch.Tensor, ...]] = None,
        **kwargs,
    ):
        # Read shapes in logical NCHW / OIHW order regardless of physical layout.
        N, C, H, W = _unproject(input.shape, self.input_dims)
        K, _, kH, kW = _unproject(weight.shape, self.weight_dims)
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        oH = (H + 2 * ph - dh * (kH - 1) - 1) // sh + 1
        oW = (W + 2 * pw - dw * (kW - 1) - 1) // sw + 1

        # Pad the input (and any MX input_scale) once here, outside every loop,
        # so each tile convolution runs with padding=0 and the per-tile input
        # region is a plain strided halo load (the input halo size/step is then
        # derived per tile inside the reduction body).  Pad the H/W *physical*
        # dims so it is correct under NHWC as well.
        if ph or pw:
            pad = _pad_spec(self.input_dims, ph, pw)
            input = torch.nn.functional.pad(input, pad)
            input_scale = kwargs.get("input_scale")
            if input_scale is not None:
                kwargs["input_scale"] = torch.nn.functional.pad(input_scale, pad)

        tile = self.tile_sizes
        num_n = N // tile.n
        num_k = K // tile.k
        num_oy = oH // tile.h
        num_ox = oW // tile.w

        # One output buffer per tail output (resolved by build_conv2d_buffers / the
        # bufferization pass, like ``tile_sizes``).
        output_bufs = [
            voyager.alloc(shape, dtype) for shape, _t, dtype in self.output_specs
        ]

        # ``output_bufs`` are closed over (additional inputs); ``store_tile`` writes them
        # in place (a side effect), so the loop carries only the (n, oy, ox, k) index.
        def cond_fn(n, oy, ox, k):
            return n < num_n

        def body_fn(n, oy, ox, k):
            output_tiles = self._outer_loop_body(
                n,
                oy,
                ox,
                k,
                input,
                weight,
                bias,
                tail_operands,
                **kwargs,
            )
            if not isinstance(output_tiles, (tuple, list)):
                output_tiles = (output_tiles,)
            for output_tile, output_buf, (_s, ts, _d) in zip(
                output_tiles, output_bufs, self.output_specs
            ):
                voyager.store_tile(
                    output_tile,
                    output_buf,
                    _project((n, k, oy, ox), self.output_dims),
                    ts,
                )
            # Advance the (n, h, w, k) tile grid (one fused increment op).
            return voyager.increment_indices(
                (n, oy, ox, k), (num_n, num_oy, num_ox, num_k)
            )

        while_loop(cond_fn, body_fn, (0, 0, 0, 0))
        outputs = list(output_bufs)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def build_gemm_buffers(
    target: torch._ops.OpOverload,
    tile_sizes: List[int],
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    accumulate_fp32: bool = False,
    batched_weight: bool = False,
    transpose_weight: bool = False,
    weight_ck: bool = False,
    pipelined: bool = False,
    block_size: Optional[int] = None,
    tail_fn: Optional[Callable[..., torch.Tensor]] = None,
    tail_operands: Optional[Tuple[torch.Tensor, ...]] = None,
    tail_input_specs: Optional[list] = None,
    output_specs: Optional[List[Tuple[tuple, tuple, torch.dtype]]] = None,
    kwargs: Optional[dict] = None,
) -> torch.fx.GraphModule:
    """
    Build the bufferized FX graph (while_loop nest over voyager.* primitives) for a
    linear / matmul / bmm op.

    Args:
        target:      the GEMM OpOverload (e.g. aten.linear.default).
        tile_sizes:  [tile_b, tile_x, tile_c, tile_k].
        input/weight/bias: example CPU tensors.
        tail_fn:     optional fused pointwise tail.  Called as
                     ``tail_fn(acc_tile, *operand_tiles)`` where operand_tiles are
                     the per-tile loads of ``tail_operands`` (per ``tail_input_specs``).
        tail_operands: tail tensor operands (residual / mul / add operands and
                     quantization codebooks), in a single tuple.
        tail_input_specs: one ``_InputSpec`` per *tiled* tail operand, or ``None`` for a
                     whole (scalar / codebook) one.
        kwargs:      extra forward kwargs (e.g. MX input_scale/weight_scale/codes).

    Returns:
        An FX GraphModule containing the bufferized while_loop nest.  The exported
        placeholders are ``[input, weight, bias, *kwargs, *tail_operands]`` — the
        same order as a fused node's ``all_input_nodes`` — so the pass wires it up
        positionally with no reordering.
    """
    tile_b, tile_x, tile_c, tile_k = tile_sizes
    *batch, X, C = input.shape
    B = math.prod(batch)  # extra leading batch dims fold into one (see forward)
    K = weight.shape[-1] if weight_ck else weight.shape[-2]

    # Resolve the single-output default (one buffer = the GEMM result) when the
    # caller / pass does not supply per-output specs, like the whole-spatial tile
    # resolution below — the module's ``forward`` consumes ``output_specs`` directly.
    if output_specs is None:
        output_specs = [((B, X, K), (tile_b, tile_x, tile_k), input.dtype)]

    pattern = TiledGEMM(
        target,
        GemmAxes(tile_b, tile_x, tile_k, tile_c),
        block_size=block_size,
        accumulate_fp32=accumulate_fp32,
        batched_weight=batched_weight,
        transpose_weight=transpose_weight,
        weight_ck=weight_ck,
        pipelined=pipelined,
        tail_fn=tail_fn,
        tail_input_specs=tail_input_specs,
        output_specs=output_specs,
    )
    # ``tail_operands`` is the LAST input (after the MX scale/code kwargs), so the
    # exported placeholders match a fused node's ``all_input_nodes`` order.
    export_kwargs = dict(kwargs or {})
    if tail_operands is not None:
        export_kwargs["tail_operands"] = tail_operands
    with _lenient_verifier():
        gm = export_model(pattern, (input, weight, bias), kwargs=export_kwargs)
    gm = _finalize_exported_gm(gm)
    # Only num_c == 1 fuses the GEMM op into the tail (one accelerator pass); for a
    # tiled reduction the GEMM stays separate and just the tail fuses — true whether
    # the reduction is sequential or double-buffered (whose epilogue matmul lands in
    # the store body).  See ``_fuse_tail_in_body``.
    num_c = C // tile_c
    if tail_fn is not None:
        _fuse_tail_in_body(gm, target, tail_fn, fuse_ref_with_tail=(num_c == 1))

    # Pipelined (rolled) reduction primes block 0 in the prologue, so the steady
    # while_loop runs blocks 1..num_c-1; the unrolled path (num_c < 3) has no inner
    # loop, so the inner extent is ignored either way.
    inner = [(1, num_c, 1)] if (pipelined and num_c >= 3) else [num_c]
    _tag_loop_extents(
        gm,
        [
            [B // tile_b, X // tile_x, K // tile_k],  # outer (b, x, k) grid
            inner,                                    # inner reduction (c)
        ],
    )
    return gm


def build_conv2d_buffers(
    target: torch._ops.OpOverload,
    tile_sizes: List[int],
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
    nhwc: bool = False,
    block_size: Optional[int] = None,
    accumulate_fp32: bool = False,
    pipelined: bool = False,
    tail_fn: Optional[Callable[..., torch.Tensor]] = None,
    tail_operands: Optional[Tuple[torch.Tensor, ...]] = None,
    tail_input_specs: Optional[list] = None,
    output_specs: Optional[List[Tuple[tuple, tuple, torch.dtype]]] = None,
    kwargs: Optional[dict] = None,
) -> torch.fx.GraphModule:
    """
    Build the bufferized FX graph (while_loop nest over voyager.* primitives) for a
    conv2d op.  ``tile_sizes`` is ``[tile_n, tile_k, tile_c]`` (spatial kept whole)
    or ``[tile_n, tile_k, tile_c, tile_oh, tile_ow]`` to also tile the output
    spatial grid (the input is haloed/strided accordingly; see ``TiledConv2d``).

    ``tail_fn`` / ``tail_operands`` / ``tail_input_specs`` work as in
    ``build_gemm_buffers``: the tail is ``tail_fn(acc_tile, *operand_tiles)`` where
    each operand is loaded per its ``_InputSpec`` (a ResNet skip is tiled, a
    codebook is passed whole).  This expresses fused patterns such as
    ``relu(conv(x) + skip)``.

    For ``conv2d_mx``, pass ``block_size`` and the per-C-block scales/codes via
    ``kwargs`` (``input_scale`` / ``weight_scale`` / ``input_code`` /
    ``weight_code``); they are loaded per channel block inside the reduction loop.

    ``nhwc=True`` selects the transposed layout (NHWC input, HWIO weight, NHWC
    output); the example tensors are then physically laid out and ``tile_sizes``
    stays logical ``[tile_n, tile_k, tile_c, ...]``.
    """
    if len(tile_sizes) == 5:
        tile_n, tile_k, tile_c, tile_oh, tile_ow = tile_sizes
    else:
        tile_n, tile_k, tile_c = tile_sizes
        tile_oh = tile_ow = None  # whole-spatial

    # Example tensors may be physically permuted (NHWC / HWIO) when the layout
    # pass ran; read their shapes back in logical NCHW / OIHW order.
    in_dims = _NHWC if nhwc else None
    w_dims = _HWIO if nhwc else None
    N, C, H, W = _unproject(input.shape, in_dims)
    K, _, kH, kW = _unproject(weight.shape, w_dims)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    oH = (H + 2 * ph - dh * (kH - 1) - 1) // sh + 1
    oW = (W + 2 * pw - dw * (kW - 1) - 1) // sw + 1
    # Resolve whole-spatial to the full output so ``tile`` is complete.
    if tile_oh is None:
        tile_oh = oH
    if tile_ow is None:
        tile_ow = oW

    # Resolve the single-output default (one buffer = the conv result, in physical
    # layout) when the caller / pass does not supply per-output specs; the module's
    # ``forward`` consumes ``output_specs`` directly, like ``tile_sizes``.
    if output_specs is None:
        out_dims = _NHWC if nhwc else None
        output_specs = [(
            _project((N, K, oH, oW), out_dims),
            _project((tile_n, tile_k, tile_oh, tile_ow), out_dims),
            input.dtype,
        )]

    pattern = TiledConv2d(
        target,
        Conv2dAxes(n=tile_n, h=tile_oh, w=tile_ow, k=tile_k, c=tile_c),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        nhwc=nhwc,
        block_size=block_size,
        accumulate_fp32=accumulate_fp32,
        pipelined=pipelined,
        tail_fn=tail_fn,
        tail_input_specs=tail_input_specs,
        output_specs=output_specs,
    )

    # ``tail_operands`` is the LAST input (after the MX scale/code kwargs), so the
    # exported placeholders match a fused node's ``all_input_nodes`` order.
    export_kwargs = dict(kwargs or {})
    if tail_operands is not None:
        export_kwargs["tail_operands"] = tail_operands
    with _lenient_verifier():
        gm = export_model(pattern, (input, weight, bias), kwargs=export_kwargs)
    gm = _finalize_exported_gm(gm)
    # Only num_c == 1 fuses the conv op into the tail; a tiled reduction (sequential or
    # double-buffered) keeps the conv separate and fuses just the tail.
    num_c = C // tile_c
    if tail_fn is not None:
        _fuse_tail_in_body(gm, target, tail_fn, fuse_ref_with_tail=(num_c == 1))

    # See ``build_gemm_buffers``: the rolled pipeline runs channel blocks 1..num_c-1.
    inner = [(1, num_c, 1)] if (pipelined and num_c >= 3) else [num_c]
    _tag_loop_extents(
        gm,
        [
            # outer (n, h, w, k) grid
            [N // tile_n, oH // tile_oh, oW // tile_ow, K // tile_k],
            inner,  # inner reduction (c)
        ],
    )
    return gm
