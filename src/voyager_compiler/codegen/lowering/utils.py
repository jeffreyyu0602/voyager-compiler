"""
Shared helpers for the bufferization builders (``gemm`` / ``pointwise`` /
``attention``).

Tile DMA goes through ``voyager.async_copy`` (the guarded prefetch / store) and
``voyager.copy_tile`` (the in-kernel whole-slot writes); the helpers here are
the remaining shared pieces: the fused pointwise tail, the lenient export
verifier, exported-graph cleanup, and tagging each ``while_loop`` with its
per-dimension tile-grid extents (consumed by the loop-aware code generator).
"""

import contextlib
import operator
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from voyager_compiler.codegen.lowering import ops

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
      ``index_map``    output/grid dim each operand dim maps to, or ``None`` for
                       an axis loaded whole / mapped to no grid dim (e.g. a conv
                       weight's kH/kW, never tiled).
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


@dataclass(frozen=True)
class _ScratchSpec:
    """A persistent on-chip SRAM scratch ref — a third class of SRAM object
    next to input / output banks.  Unlike them it is *local state*: allocated
    once and reused every grid step, not DMA-managed, not double-buffered, and
    not carried in the loop state.  It has only a ``shape`` + ``dtype`` (no
    ``index_map`` / DRAM shape / token / buffer count); the kernel mutates it
    in place (e.g. a tiled-reduction accumulator).  Mirrors Pallas's
    ``scratch_shapes`` appended after the normal input/output refs.
    """

    shape: Tuple[int, ...]
    dtype: torch.dtype


def _compute_input_spec(
    tiling: tuple, input_shape: tuple, grid_map: Optional[tuple] = None
) -> Optional[_InputSpec]:
    """Derive the tile spec for one operand (NumPy/PyTorch right-align rules),
    or ``None`` for a 0-D / single-element operand that is passed whole (not
    tiled / loaded).

    ``tiling`` is the per-output-dim tile-factor vector; the operand's tile is
    ``compute_tiled_shape(input_shape, tiling)`` — the same helper used
    everywhere else — so a block/reduction axis (input larger than its output,
    e.g. ``calculate_mx_qparam``) keeps its full per-tile block, and broadcast /
    unit dims (``s == 1``) stay at 1.

    ``grid_map`` maps each output dim to the grid dim it tiles along; ``None``
    means the grid *is* the output (pointwise / pool), so the operand's dim
    ``offset + d`` is its grid dim directly.  GEMM / conv pass their
    output->grid map (e.g. a transposed-conv NHWC permutation), since their grid
    carries a reduction dim the output doesn't span.
    """
    from voyager_compiler.codegen.passes.tiling import compute_tiled_shape

    # 0-D or single-element tensors are passed through (no spec).
    if len(input_shape) == 0 or (len(input_shape) == 1 and input_shape[0] == 1):
        return None

    ndim_in = len(input_shape)
    offset = len(tiling) - ndim_in

    ts_for_inp = compute_tiled_shape(input_shape, tiling)
    index_map = tuple(
        (offset + d) if grid_map is None else grid_map[offset + d]
        for d in range(ndim_in)
    )
    is_broadcast = tuple(s == 1 for s in input_shape)

    return _InputSpec(ts_for_inp, index_map, is_broadcast)


def _out_of_place(target):
    """The out-of-place aten overload of an in-place one (``add_.Tensor`` ->
    ``add.Tensor``), or ``target`` unchanged if it is not a convertible
    in-place op.
    """
    if not isinstance(target, torch._ops.OpOverload):
        return target
    name = target._schema.name  # e.g. "aten::add_"
    if "::" not in name or not name.endswith("_"):
        return target
    ns, op = name.split("::")
    packet = getattr(getattr(torch.ops, ns, None), op[:-1], None)
    if packet is None:
        return target
    return getattr(packet, target._overloadname, target)


def _build_fused_gm(submod, anchor_node, fused_ops, fused_inputs):
    """The post-anchor fused pointwise ops as a ``GraphModule``
    ``[acc, *fused_inputs] -> submodule output(s)``.

    ``anchor_node``'s output is rewired to the ``acc`` placeholder;
    ``fused_inputs`` are the submodule placeholders the ``fused_ops`` consume
    (e.g. a residual / scale).  Export inlines this, so the fused ops become
    standalone nodes in the loop body, applied to the anchor's result tile.
    """
    g = torch.fx.Graph()
    remap = {anchor_node: g.placeholder("acc")}
    for n in fused_inputs:
        remap[n] = g.placeholder(n.name)
    inputs = set(remap.values())  # the [acc, *fused_inputs] placeholders
    for n in fused_ops:
        new = g.node_copy(n, lambda x: remap[x])
        # An in-place fused op touching a fused input can mutate a while_loop
        # input (a residual is passed whole when the node is untiled), which
        # HOPs reject — so de-inplace it (add_ -> add); same result, no
        # mutation.
        if any(a in inputs for a in new.all_input_nodes):
            new.target = _out_of_place(n.target)
        remap[n] = new
    out = next(n for n in submod.graph.nodes if n.op == "output").args[0]
    if isinstance(out, (list, tuple)):
        g.output(tuple(remap[o] for o in out))
    else:
        g.output(remap[out])
    g.lint()
    return torch.fx.GraphModule(torch.nn.Module(), g)


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


def _fuse_tail_in_body(
    gm: torch.fx.GraphModule,
    target,
    fuse_anchor_with_tail=True,
) -> None:
    """Group a GEMM/conv anchor + its fused post-op pointwise ops (inside the
    exported ``while_loop`` body) into a nested ``call_module``, so the proto
    codegen emits them as one ``fused_op`` (L1: one accelerator pass).

    The group is found by *forward reachability* from the anchor (like
    mapping.py's ``find_sequential_nodes_``), not by op target: from the lone
    anchor walk *down* through ``.users``, collecting the compute cone, and stop
    at the output store (``copy_tile``) — excluding the write-out wrappers (a
    ``clone`` / multi-output ``getitem`` that only feed that store).  The tail's
    operand loads (residual / scale ``select`` reads) are *inputs* to the tail
    ops, not descendants of the anchor, so they fall outside the cone and become
    the call_module's args; the per-slot DMA / loop machinery is likewise never
    reached.  A multi-output op (``quantize_mx``) ends the cone with its
    ``getitem`` users left outside, rewired to the call_module.  Recurses into
    nested ``while_loop`` / ``cond`` bodies.

    Two modes, by ``fuse_anchor_with_tail``:

      * ``True`` (num_k == 1): the anchor sits directly in this body and
        produces the output tile, so the cone is anchor + bias + fused ops.

      * ``False`` (num_k > 1): the reduction splits the anchor and tail across
        separate ``torch.cond`` branches — the anchor accumulates into a scratch
        ref, and the tail runs in a *finalize* cond's true branch.  Group only
        that tail branch's compute (``_fuse_tail_only``), leaving the GEMM/conv
        anchor out.
    """
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph

    for node in list(gm.graph.nodes):
        if node.op == "get_attr":
            sub = getattr(gm, str(node.target), None)
            if isinstance(sub, torch.fx.GraphModule):
                _fuse_tail_in_body(sub, target, fuse_anchor_with_tail)

    if not fuse_anchor_with_tail:
        _fuse_tail_only(gm, target)
        return

    # The anchor appears only in the output-producing body, exactly once for
    # num_k == 1.  (A multi-anchor / no-anchor body is num_k > 1 or a non-output
    # region — out of scope for fusion.)
    anchors = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is target
    ]
    if len(anchors) != 1:
        return
    anchor = anchors[0]

    copy_tile = voyager.copy_tile.default
    clone = torch.ops.aten.clone.default

    def _is_writeout_wrapper(n) -> bool:
        # A ``clone`` / multi-output ``getitem`` that only feeds the output
        # store — the boundary between real compute and the bank write.
        return (
            n.target in (clone, operator.getitem)
            and bool(n.users)
            and all(u.target is copy_tile for u in n.users)
        )

    # Forward cone from the anchor, halting at the store and its write-out
    # wrappers — exactly the anchor + bias + fused compute ops.
    group, stack = {anchor}, [anchor]
    while stack:
        for u in stack.pop().users:
            if (
                u in group
                or u.op != "call_function"
                or u.target is copy_tile
                or _is_writeout_wrapper(u)
            ):
                continue
            group.add(u)
            stack.append(u)

    if len(group) < 2:
        return  # bare anchor — nothing fused into this body

    _create_and_insert_subgraph(list(group), gm, dict(gm.named_modules()))
    gm.graph.lint()
    gm.recompile()


def _subgraph_has(gm: torch.fx.GraphModule, target) -> bool:
    """True if ``gm`` — or any nested cond / while_loop body — has a
    ``call_function`` with ``target``.  The search recurses because a conv
    bias-gate ``torch.cond`` nests the anchor a level inside the accumulate
    branch."""
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is target:
            return True
        if n.op == "get_attr":
            sub = getattr(gm, str(n.target), None)
            if isinstance(sub, torch.fx.GraphModule) and _subgraph_has(
                sub, target
            ):
                return True
    return False


def _fuse_tail_only(gm: torch.fx.GraphModule, target) -> None:
    """num_k > 1: group the fused tail into a nested ``call_module``, leaving
    the GEMM/conv anchor separate.  The reduction splits the work across
    ``torch.cond``s: an *accumulate* cond (whose branch holds the anchor) sums
    the op into a scratch ref, then a *finalize* cond maps the completed scratch
    through the tail (``fused_gm``) into the output.

    The finalize cond is found by the **scratch data dependency**: the
    accumulate cond's result is ``copy_tile``'d into the scratch ref, and the
    finalize cond is the other cond that reads that same scratch as an operand.
    Its true branch is just ``fused_gm`` + NOP wrappers (the dense-view / cast /
    multi-output ``getitem`` that match the skip branch's metadata), so the
    group is its non-NOP ops; a single-op tail needs no fusing."""
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph
    from voyager_compiler.codegen.mapping_utils import is_nop

    cond = torch.ops.higher_order.cond
    copy_tile = voyager.copy_tile.default

    def true_branch(c):
        a = c.args[1]
        if isinstance(a, torch.fx.Node) and a.op == "get_attr":
            b = getattr(gm, str(a.target), None)
            return b if isinstance(b, torch.fx.GraphModule) else None
        return None

    # 1. The accumulate cond — its true branch holds the anchor.
    anchor_cond = None
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target is not cond:
            continue
        branch = true_branch(n)
        if branch is not None and _subgraph_has(branch, target):
            anchor_cond = n
            break
    if anchor_cond is None:
        return

    # 1a. Fuse the accumulate branch (anchor + cast + the ``+ scratch`` add) into
    #     one ``fused_op``.  The cond is ``torch.cond(coord == 0, init,
    #     accumulate)``: the true branch (``args[1]``) is ``init`` — just the
    #     anchor, whose result is stored directly (through at most a nop cast) —
    #     so only the false branch (``args[2]``, ``accumulate``) has the add.
    #     There the bare anchor result feeds the add, not a store, so the
    #     destination-passing check would reject it; folding it into the fused op
    #     makes the op's *result* the value that is ``copy_tile``'d into scratch.
    accum_branch = getattr(gm, str(anchor_cond.args[2].target), None)
    if isinstance(accum_branch, torch.fx.GraphModule):
        _fuse_tail_in_body(accum_branch, target)

    # 2. The scratch ref = the dest of the ``copy_tile`` fed by the accumulate
    #    cond's result (through the cond's ``getitem`` unpacking).
    scratch = None
    for u in anchor_cond.users:
        chain = list(u.users) if u.target is operator.getitem else [u]
        copy_n = next((n for n in chain if n.target is copy_tile), None)
        if copy_n is not None:
            scratch = copy_n.args[1]
            break
    if scratch is None:
        return

    # 3. The finalize cond reads that scratch, so it is among the scratch's
    #    users — the *other* cond there (the accumulate cond also reads it).
    finalize_cond = next(
        (u for u in scratch.users if u.target is cond and u is not anchor_cond),
        None,
    )
    if finalize_cond is None:
        return

    # 4. The tail branch is ``fused_gm`` + the finalize wrappers (a multi-output
    #    ``getitem``, a dense-view ``as_strided``, a dtype ``to`` — added so the
    #    finalize and skip branches share output metadata).  Group ``fused_gm``:
    #    the branch's non-NOP ops.  ``getitem`` and ``to`` aren't caught by
    #    ``is_nop`` (getitem never is; ``to`` only when the dtype is unchanged —
    #    under fp32 accumulation it is a real cast), so exclude them explicitly.
    branch = true_branch(finalize_cond)
    ops = [
        n
        for n in branch.graph.nodes
        if (
            n.op == "call_function"
            and n.target is not operator.getitem
            and n.target is not torch.ops.aten.to.dtype
            and n.target is not copy_tile
            and not is_nop(n)
        )
    ]
    if len(ops) < 2:
        return  # a lone op — nothing to fuse

    _create_and_insert_subgraph(ops, branch, dict(branch.named_modules()))
    branch.graph.lint()
    branch.recompile()


def _strip_assert_scalars(gm: torch.fx.GraphModule) -> None:
    """Drop export's deferred ``aten._assert_scalar`` range-checks — and the
    ``getitem`` / ``ge`` / ``le`` nodes that feed only them — recursing into
    ``while_loop`` / ``cond`` body subgraphs.

    A guarded ``torch.cond`` (the async-DMA wait guards) returns an ``int``
    (1 / 0) to stay alive; export materializes that as an unbacked symint and
    emits ``u >= 0`` / ``u <= 1`` deferred assertions on it.  They are provably
    true no-ops and only clutter codegen.  The cond itself stays (it carries the
    wait's side effect); only its dead int-output scaffolding is removed.
    """
    assert_op = torch.ops.aten._assert_scalar.default
    for n in list(gm.graph.nodes):
        if n.op == "call_function" and n.target is assert_op:
            gm.graph.erase_node(n)
    # Reverse sweep: the bool (``ge`` / ``le``) feeds only the now-gone assert,
    # then the cond's ``getitem`` feeds only those bools — erase once userless.
    sweep = {operator.ge, operator.le, operator.getitem}
    changed = True
    while changed:
        changed = False
        for n in reversed(list(gm.graph.nodes)):
            if n.op == "call_function" and n.target in sweep and not n.users:
                gm.graph.erase_node(n)
                changed = True
    for n in gm.graph.nodes:
        if n.op == "get_attr":
            sub = getattr(gm, str(n.target), None)
            if isinstance(sub, torch.fx.GraphModule):
                _strip_assert_scalars(sub)
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
    # primitives (notably the side-effecting ``async_copy`` / ``copy_tile``) and
    # the ``while_loop`` / ``cond`` HOPs carry side effects that DCE can't see — a
    # bufferized loop writes its output through ``copy_tile`` / ``async_copy``
    # into an *additional-input* buffer, so the loop has no FX users and default
    # DCE would delete it.  Treating every non-``aten`` node as impure keeps the
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
    # Drop export's deferred ``_assert_scalar`` range-checks on the guarded-wait
    # conds' int outputs (no-ops for codegen; the impure-aware DCE keeps them).
    _strip_assert_scalars(gm)
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


# ---------------------------------------------------------------------------
# Layout projection helpers (NCHW/OIHW logical <-> physical NHWC/HWIO).
#
# Specs are written in logical axis order — feature maps NCHW (N=0, C/K=1,
# H=2, W=3), weights OIHW (K=0, C=1, kH=2, kW=3) — and projected onto each
# operand's physical order only at the load/store boundary.  ``dims`` is the
# NCHW->physical permutation (physical position ``i`` holds logical axis
# ``dims[i]``); ``None`` is the logical NCHW / OIHW layout.
# ---------------------------------------------------------------------------
# input / output feature maps under the transposed (NHWC) layout
_NHWC = (0, 2, 3, 1)
# weight (kH, kW, C, K) under the transposed (HWIO) layout
_HWIO = (2, 3, 1, 0)


def _project(per_axis: Tuple, dims: Optional[Tuple[int, ...]]) -> Tuple:
    """Reorder a per-logical-axis tuple into a tensor's physical order.

    ``per_axis[a]`` is the value for logical axis ``a``.  ``dims`` is the
    NCHW->physical permutation: physical position ``i`` holds logical axis
    ``dims[i]``.
    """
    if dims is None:
        return tuple(per_axis)
    return tuple(per_axis[a] for a in dims)


def _unproject(physical: Tuple, dims: Optional[Tuple[int, ...]]) -> Tuple:
    """Inverse of ``_project``: read a physical-order sequence (e.g. a
    ``shape``) back into logical NCHW / OIHW axis order.
    """
    if dims is None:
        return tuple(physical)
    return tuple(physical[dims.index(a)] for a in range(len(physical)))
