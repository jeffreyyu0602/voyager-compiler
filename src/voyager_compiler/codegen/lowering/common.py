"""
Shared helpers for the bufferization builders (``gemm`` / ``pointwise`` /
``attention``).

Tile loads/stores go directly through ``voyager.load_tile`` /
``voyager.store_tile`` (addressed by integer ``indices`` block indices); the
helpers here are the remaining shared pieces: the fused pointwise tail, the
lenient export verifier, exported-graph cleanup, and tagging each
``while_loop`` with its per-dimension tile-grid extents (consumed by the
loop-aware code generator).
"""

import contextlib
import operator
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from voyager_compiler.codegen.lowering import (
    ops,
)  # noqa: F401  registers voyager.*

voyager = torch.ops.voyager


@dataclass
class _InputSpec:
    """Per-operand tile spec, computed once from the output + operand shapes.

    A whole (un-indexed) operand — a true scalar, or a quantization codebook /
    qmap — has **no** spec (``None`` in the spec list), not an _InputSpec: it is
    passed through, never tiled or DMA-loaded.  So an _InputSpec always
    describes a genuinely tiled operand.

    Fields:
      ``tile_sizes``   per-dim tile sizes for this operand.
      ``index_map``    output/grid dim each operand dim maps to.
      ``is_broadcast`` dim is size 1 in operand, >1 in output.
      ``strides``      step between tiles (``None`` => == ``tile_sizes``).
      ``pad``          per-dim low padding (start -= pad).
      ``pad_value``    out-of-bounds fill for a padded load.
    """

    tile_sizes: Tuple[int, ...]
    index_map: Tuple[int, ...]
    is_broadcast: Tuple[bool, ...]
    strides: Optional[Tuple[int, ...]] = None
    pad: Optional[Tuple[int, ...]] = None
    pad_value: Optional[float] = None


@dataclass
class _OutputSpec:
    """Per-output tile spec — the store-side mirror of ``_InputSpec``.

    Carries the DRAM buffer (``shape`` / ``dtype``) plus how the output tiles
    onto the grid (``tile_sizes`` + ``index_map``), exactly like an input.  The
    grid has no single ``tile_sizes``: it is reconstructed from the operand
    specs — the grid-defining output sets the tile for the dims it spans, and
    any extra (reduction) grid dim, e.g. conv's input channel C, takes its tile
    from the input that spans it.  An output whose ``index_map`` omits a grid
    dim does not see it in the store — that is how a reduction's C dim is
    dropped.

    Fields:
      ``shape``      the DRAM output buffer shape.
      ``tile_sizes`` per-dim output tile (the grid step for its dims).
      ``index_map``  grid dim each output dim maps to.
      ``dtype``      output dtype.
    """

    shape: Tuple[int, ...]
    tile_sizes: Tuple[int, ...]
    index_map: Tuple[int, ...]
    dtype: torch.dtype


def _compute_input_spec(
    output_shape: tuple, tile_sizes: tuple, input_shape: tuple
) -> Optional[_InputSpec]:
    """Derive the tile spec for one operand (NumPy/PyTorch right-align rules),
    or ``None`` for a 0-D / single-element operand that is passed whole (not
    tiled / loaded).
    """
    ndim_out = len(output_shape)

    # 0-D or single-element tensors are passed through (no spec).
    if len(input_shape) == 0 or (len(input_shape) == 1 and input_shape[0] == 1):
        return None

    ndim_in = len(input_shape)
    offset = ndim_out - ndim_in

    index_map, ts_for_inp, is_broadcast = [], [], []
    for d_in, (s_in, s_out, ts) in enumerate(
        zip(input_shape, output_shape[offset:], tile_sizes[offset:])
    ):
        index_map.append(offset + d_in)
        bcast = s_in == 1 and s_out > 1
        is_broadcast.append(bcast)
        ts_for_inp.append(1 if bcast else ts)

    return _InputSpec(tuple(ts_for_inp), tuple(index_map), tuple(is_broadcast))


@contextlib.contextmanager
def _lenient_verifier():
    """
    Tolerate a missing ``val`` on *unused* subgraph placeholders during export.

    A ``while_loop`` cond subgraph receives every operand as an additional-input
    even when it only reads the loop counter; torch.export does not populate a
    ``val`` for the operands the cond never touches, which the export verifier
    rejects.  Those placeholders are never executed, so it is safe to skip the
    check for them (genuine missing-val errors on used nodes still raise).
    """
    import torch._export.verifier as _verifier

    orig = _verifier._check_val

    def _check(node):
        try:
            orig(node)
        except Exception:
            if node.op == "placeholder" and not node.users:
                return
            raise

    _verifier._check_val = _check
    try:
        yield
    finally:
        _verifier._check_val = orig


def _apply_tail(tail_fn, acc, tail_operands, tail_input_specs, block):
    """
    Apply a fused pointwise tail to the accumulator tile.

    Each tail operand is loaded per its precomputed ``_InputSpec`` — built once
    by the bufferization pass, which has the whole submodule, rather than on
    the fly here.  A ``None`` spec means *pass the operand whole*: a true
    scalar, or a quantization codebook / qmap (only ever a quantization-op
    arg).  Otherwise the operand's tile is loaded, pinning broadcast dims
    (size 1 vs a >1 output dim) to block 0.  ``block`` is the output tile's
    block index.  The loaded tiles are passed positionally after the
    accumulator: ``tail_fn(acc, *operand_tiles)``.  ``tail_operands`` /
    ``tail_input_specs`` may be ``None`` (a unary tail such as relu / silu with
    no extra operands).
    """
    if tail_fn is None:
        return acc
    tiles = []
    for op, spec in zip(tail_operands or (), tail_input_specs or ()):
        if spec is None:  # whole (scalar / codebook) operand -> passed through
            tiles.append(op)
        else:
            blk = tuple(
                0 if bcast else block[i]
                for i, bcast in zip(spec.index_map, spec.is_broadcast)
            )
            tiles.append(voyager.load_tile(op, blk, spec.tile_sizes))
    return tail_fn(acc, *tiles)


def _split_block_indices(gm: torch.fx.GraphModule) -> None:
    """Post-export: split each ``voyager.load_tile`` / ``store_tile`` /
    ``copy_tile`` block index so its ``indices`` arg holds only loop-counter
    Node refs — any constant ``int`` entry (a whole / broadcast dim, or a fixed
    reduction tile) moves into ``static_indices``.  The proto codegen can't
    serialize a list mixing FX Nodes with ints; the tile op rebuilds the same
    full index from ``indices`` + ``dims`` + ``static_indices``, so this is
    numerically transparent.  Recurses into ``while_loop`` body subgraphs.

    Done here, not in the builder's ``forward``: at export-trace time loop
    counters and constants are *both* plain ints (so they can't be told apart,
    and a custom marker trips dynamo inside ``while_loop``); only after export
    are the counters FX Nodes and the constants int literals.  ``copy_tile``
    needs this on the double-buffered path: ``delinearize_index`` constant-folds
    a unit loop dim (``i % 1`` -> 0), so its ``indices`` mixes those literal 0s
    with the genuine counter getitems.
    """
    load = torch.ops.voyager.load_tile.default
    store = torch.ops.voyager.store_tile.default
    copy = torch.ops.voyager.copy_tile.default
    changed = False
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            sub = getattr(gm, str(node.target), None)
            if isinstance(sub, torch.fx.GraphModule):
                _split_block_indices(sub)
            continue
        if node.op != "call_function" or node.target not in (load, store, copy):
            continue
        # ``indices`` is arg 1 for load_tile, arg 2 for store_tile / copy_tile.
        ix = 1 if node.target is load else 2
        a = list(node.args)
        indices = a[ix]
        if not any(isinstance(i, int) for i in indices):
            continue  # all loop-counter nodes already
        rank = len(a[ix + 1])  # tile_sizes
        dims = a[ix + 2] if len(a) > ix + 2 else None
        static = a[ix + 3] if len(a) > ix + 3 else None
        pos = list(dims) if dims else list(range(len(indices)))
        new_static = list(static) if static else [0] * rank
        dyn, dyn_dims = [], []
        for idx, d in zip(indices, pos):
            if isinstance(idx, int):
                new_static[d] = idx
            else:
                dyn.append(idx)
                dyn_dims.append(d)
        while len(a) <= ix + 3:  # ensure the dims / static_indices slots exist
            a.append(None)
        a[ix], a[ix + 2], a[ix + 3] = dyn, dyn_dims, new_static
        node.args = tuple(a)
        changed = True
    if changed:
        gm.graph.lint()
        gm.recompile()


def _fuse_tail_in_body(
    gm: torch.fx.GraphModule, ref_target, tail_fn=None, fuse_ref_with_tail=True
) -> None:
    """Group a GEMM/conv op's post-op tail (inside the exported ``while_loop``
    body) into a nested ``call_module`` so codegen emits it as one protobuf
    ``fused_op``.

    Two cases, split by whether the reduction (C) dim is tiled:

    * **num_c == 1** (``fuse_ref_with_tail``) — the reference op sits directly
      in this body and produces the output tile, so the GEMM/conv *and* its
      tail fuse into one call_module (L1: one accelerator pass, no SRAM
      write-back).  The group is the ref plus every op reachable from it, which
      includes the GEMM's own output-dtype cast (part of the op, not a separate
      tail).
    * **num_c > 1** — the reference op's partials are summed with an
      accumulate-add and a bias/dtype epilogue the accelerator does in the GEMM
      pass, so the GEMM stays separate.  Only the **pointwise tail** fuses,
      located by matching ``tail_fn``'s op sequence against the body's trailing
      compute ops — so the bias-add + cast epilogue is excluded *exactly*, not
      by heuristics.  Needs ``tail_fn`` to be a ``GraphModule`` (the production
      / bufferize path); a bare-callable tail (unit-test builders) is left
      unfused.

    ``fuse_ref_with_tail`` is the caller's ``num_c == 1`` knowledge (the builder
    knows the tile count).  It is needed because a single ``ref_target`` beside
    the store does not uniquely mean num_c == 1: a double-buffered reduction
    leaves its *epilogue* matmul in this body too (one ref when rolled, several
    when unrolled).  Gating on the builder's flag keeps num_c > 1 tail-only
    whether sequential or pipelined.

    The ``load_tile`` / ``store_tile`` / loop-control nodes stay out of the
    group and become the call_module's args; multi-output tails
    (``quantize_mx``) keep their ``getitem`` users, rewired to the call_module.
    Recurses into nested bodies.
    """
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph

    store = voyager.store_tile.default
    skip = {
        voyager.load_tile.default,
        store,
        voyager.increment_indices.default,
        torch.ops.higher_order.while_loop,
        operator.getitem,
    }
    for node in list(gm.graph.nodes):
        if node.op == "get_attr":
            sub = getattr(gm, str(node.target), None)
            if isinstance(sub, torch.fx.GraphModule):
                _fuse_tail_in_body(sub, ref_target, tail_fn, fuse_ref_with_tail)

    if not any(
        n.op == "call_function" and n.target is store for n in gm.graph.nodes
    ):
        return

    # Only num_c == 1 (per the caller) fuses the ref op with the tail, and only
    # when it is the lone ref in this body (the output-producing op).  num_c > 1
    # — sequential (no ref here) or pipelined (the double-buffered epilogue
    # leaves one/several refs here) — falls to the tail-only branch.
    refs = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is ref_target
    ]
    ref = refs[0] if (fuse_ref_with_tail and len(refs) == 1) else None
    if ref is not None:
        # num_c == 1: GEMM + tail.  Group the ref and everything reachable
        # from it.
        reachable = {ref}
        group = [ref]
        for n in gm.graph.nodes:
            if n is ref or n.op != "call_function" or n.target in skip:
                continue
            if any(inp in reachable for inp in n.all_input_nodes):
                reachable.add(n)
                group.append(n)
        if len(group) < 2:
            return  # bare reference op, no tail to fuse
    else:
        # num_c > 1: fuse the pointwise tail only.  ``tail_fn``'s ops are
        # inlined as the body's trailing compute ops (after the bias-add + cast
        # epilogue), so the tail is exactly the last ``len(tail_targets)`` of
        # them.
        if not isinstance(tail_fn, torch.fx.GraphModule):
            return
        tail_targets = [
            n.target for n in tail_fn.graph.nodes if n.op == "call_function"
        ]
        body_compute = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target not in skip
        ]
        if not tail_targets or len(tail_targets) > len(body_compute):
            return
        group = body_compute[-len(tail_targets) :]
        if [n.target for n in group] != tail_targets:
            return  # pattern mismatch — leave unfused rather than guess

    _create_and_insert_subgraph(group, gm, dict(gm.named_modules()))
    gm.graph.lint()
    gm.recompile()


def _finalize_exported_gm(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Erase the unused tile-size int placeholders left by exporting with the tile
    sizes as positional args.  Loop counters are plain ints, so there are no
    lifted index-constant tensors to clean up.
    """
    for n in list(gm.graph.nodes):
        if n.op == "placeholder" and not n.users:
            gm.graph.erase_node(n)
    # Only ``aten`` ops are safe to drop as dead code.  The ``voyager.*``
    # primitives (notably the side-effecting ``store_tile``) and the
    # ``while_loop`` / ``cond`` HOPs carry side effects that DCE can't see — a
    # bufferized loop writes its output through ``store_tile`` into an
    # *additional-input* buffer, so the loop has no FX users and default DCE
    # would delete it.  Treating every non-``aten`` node as impure keeps the
    # loop machinery while still cleaning up dead ``aten`` compute.
    gm.graph.eliminate_dead_code(
        is_impure_node=lambda n: n.is_impure()
        or (
            n.op == "call_function"
            and getattr(n.target, "namespace", None) != "aten"
        )
    )
    gm.graph.lint()
    gm.recompile()
    # Route constant block-index entries into static_indices so codegen never
    # sees a mixed Node/int index list (recurses into the while_loop body
    # subgraphs).
    _split_block_indices(gm)
    return gm


def _tag_loop_extents(gm, extents_per_level, depth=0):
    """
    Tag the nested ``while_loop`` chain with the per-dimension tile-grid
    extents.

    ``extents_per_level[d]`` is the list of extents (each a bare ``end`` count
    or an ``(start, end, step)`` tuple) for the while_loop at nesting depth
    ``d``.  The builders flatten an N-D grid into one ``while_loop``, so a level
    with N extents is later emitted as N nested ``Loop`` protos.  Assumes one
    loop per nesting level (a linear chain), which holds for these builders.
    """
    if depth >= len(extents_per_level):
        return
    named = dict(gm.named_modules())
    for n in gm.graph.nodes:
        if (
            n.op == "call_function"
            and n.target is torch.ops.higher_order.while_loop
        ):
            n.meta["loop_extents"] = extents_per_level[depth]
            body = named.get(str(n.args[1].target))
            if isinstance(body, torch.fx.GraphModule):
                _tag_loop_extents(body, extents_per_level, depth + 1)
            return
