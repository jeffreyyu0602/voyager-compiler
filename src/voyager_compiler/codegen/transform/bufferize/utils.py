"""
Shared helpers for the bufferization builders (``gemm`` / ``pointwise`` /
``attention``).

Tile DMA goes through ``voyager.async_copy`` (the guarded prefetch / store) and
``voyager.insert`` (the in-kernel whole-slot writes); the helpers here are
the remaining shared pieces: the fused pointwise tail, the lenient export
verifier, exported-graph cleanup, and tagging each ``while_loop`` with its
per-dimension tile-grid extents (consumed by the loop-aware code generator).
"""

import contextlib
import operator
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import is_nop, quant_param_arg_nodes

# Registers voyager.* before ``torch.ops.voyager`` is bound below.
from voyager_compiler.codegen.transform.bufferize import ops  # noqa: F401

voyager = torch.ops.voyager
_WHILE_LOOP = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond
_COMMIT = torch.ops.higher_order.commit


def effect_cond(
    pred: torch.Tensor | torch.SymBool | bool,
    action: Callable[[], None],
) -> None:
    """Run the side-effecting ``action()`` when ``pred`` holds, else nothing —
    a ``torch.cond`` guard whose two branches return the *same* constant.

    Returning different ints from the branches (the ``do`` returns 1 / ``skip``
    returns 0 idiom) makes ``torch.cond`` mint an unbacked symint for the
    discarded result; inside a ``()``-returning ``voyager.commit`` that dangling
    symbol trips export's ``PendingUnbackedSymbolNotFound``.  Matched constants
    mint none.  ``action``'s own return is ignored; ``has_side_effect(cond)``
    keeps the guard.
    """

    def true_branch():
        action()
        return 0

    def false_branch():
        return 0

    torch.cond(pred, true_branch, false_branch)


def _subgraph(gm: GraphModule, target) -> Optional[GraphModule]:
    """The GraphModule attribute named ``target`` on ``gm`` (a loop body/cond
    or fused submodule), or None if ``target`` is not a submodule."""
    try:
        sub = gm.get_submodule(str(target))
    except AttributeError:
        return None
    return sub if isinstance(sub, GraphModule) else None


def _passed_whole(node: Node, codebooks: set) -> bool:
    """An operand handed to the op whole rather than tiled into a bank, and so
    exempt from the Scratchpad rule: a codebook / qmap (a *param*, which carries
    no space at all), or a scalar (read from DRAM).  ``_build_for_untiled``
    loads neither."""
    if node in codebooks:
        return True
    val = node.meta.get("val", getattr(node, "value", None))
    return isinstance(val, torch.Tensor) and val.numel() == 1


def _collect_codebook_nodes(gm: GraphModule) -> set:
    """Every codebook / qmap node in ``gm`` — recursing into ``while_loop``
    bodies, ``cond`` branches, and fused ``call_module`` submodules.

    Codebook-ness flows bottom-up: an op flags its codebook args
    (``quant_param_arg_nodes``) and an operand bound to a codebook sub-graph
    placeholder is itself a codebook — so a top-level ``code`` / ``qmap``
    get_attr threaded through the loops is caught.
    """
    codebooks = set()

    def _thread(operands, sub):
        if sub is None:
            return
        sub_phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
        sub_cb = _collect_codebook_nodes(sub)
        codebooks.update(sub_cb)
        # A sub placeholder can only be a codebook at sub's own level, so
        # membership in the whole sub set is membership among its placeholders.
        for operand, ph in zip(operands, sub_phs):
            if isinstance(operand, Node) and ph in sub_cb:
                codebooks.add(operand)

    for n in gm.graph.nodes:
        if n.op == "call_function":
            codebooks.update(quant_param_arg_nodes(n))
            if n.target is _WHILE_LOOP:
                operands = list(n.args[2])
                if len(n.args) > 3:
                    operands += list(n.args[3])
                _thread(operands, _subgraph(gm, n.args[1].target))
            elif n.target is _COND:
                # Both cond branches share the operand list (args[3]); thread
                # codebook-ness into each, exactly like a while_loop body — a
                # tail op (e.g. quantize_mx) inside a finalize cond consumes its
                # codebook through a branch placeholder.
                operands = list(n.args[3]) if len(n.args) > 3 else []
                _thread(operands, _subgraph(gm, n.args[1].target))
                _thread(operands, _subgraph(gm, n.args[2].target))
            elif n.target is _COMMIT:
                _thread(list(n.args[1:]), _subgraph(gm, n.args[0].target))
        elif n.op == "call_module":
            _thread(list(n.args), _subgraph(gm, n.target))

    return codebooks


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
      ``repeat``       grid steps this operand's dim holds still for: its block
                       index is ``grid_index // repeat[d]``, so consecutive grid
                       points read the *same* tile.  A GQA operand repeats over
                       the query heads (8 KV heads feeding 32 query heads =>
                       ``repeat`` 4 on the head dim), which is a broadcast that
                       ``is_broadcast`` cannot say: the dim is 8, not 1.
                       ``None`` => all ones.
      ``strides``      step between tiles (``None`` => == ``tile_sizes``).
      ``pad``          per-dim low padding (start -= pad).
      ``pad_value``    out-of-bounds fill for a padded load.
      ``transposed``   load swaps the tile's last two dims (a fused weight ᵀ).
      ``num_banks``    per-operand software-pipeline depth (SRAM bank slots /
                       prefetch distance ``num_banks - 1``); ``None`` inherits
                       the scheduler's default.  Lets a reused / low-reuse
                       operand pick a shallower or deeper pipeline than its
                       peers.
      ``first_use_at_exit``
                       guarded operand first read at its sweep's **last** step,
                       so the load-wait is deferred to block-exit (overlaps the
                       sweep with a single buffer).  Builder-set only where
                       exit-only use is provable; default ``False`` =
                       wait-at-entry.
    """

    tile_sizes: Tuple[int, ...]
    index_map: Tuple[int, ...]
    is_broadcast: Tuple[bool, ...]
    repeat: Optional[Tuple[int, ...]] = None
    strides: Optional[Tuple[int, ...]] = None
    pad: Optional[Tuple[int, ...]] = None
    pad_value: Optional[float] = None
    transposed: bool = False
    num_banks: Optional[int] = None
    first_use_at_exit: bool = False


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
      ``num_banks``  per-output software-pipeline depth (``None`` inherits the
                     scheduler default); see ``_InputSpec.num_banks``.
      ``first_use_at_exit``
                     guarded output first written at its sweep's **last** step
                     (e.g. a reduction's ``post_process``), so the reuse-drain
                     is deferred to block-exit (lets the store overlap the next
                     tile's sweep with a single buffer).  Builder-set; default
                     ``False`` = drain-at-entry.  See ``_InputSpec``.
    """

    shape: Tuple[int, ...]
    tile_sizes: Tuple[int, ...]
    index_map: Tuple[int, ...]
    dtype: torch.dtype
    num_banks: Optional[int] = None
    first_use_at_exit: bool = False


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
    ``compute_tiled_shape(input_shape, tiling)``.  ``grid_map`` maps each output
    dim to the grid dim it tiles along; ``None`` means the grid *is* the output
    (pointwise / pool), so the operand's dim ``offset + d`` is its grid dim
    directly.  GEMM / conv pass their output->grid map (e.g. a transposed-conv
    NHWC permutation), since their grid carries a reduction dim the output
    doesn't span.
    """
    from voyager_compiler.codegen.node_info import compute_tiled_shape

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


def _is_writeout_wrapper(n: Node) -> bool:
    """A ``clone`` / multi-output ``getitem`` that only feeds output stores --
    the boundary between the compute cone and the bank write.  It is peeled off
    a store's source and left outside the fused module."""
    return (
        n.op == "call_function"
        and n.target in (torch.ops.aten.clone.default, operator.getitem)
        and bool(n.users)
        and all(u.target is voyager.insert.default for u in n.users)
    )


def _on_datapath(n: Node) -> bool:
    """A node the fused kernel computes over -- compute, cast, or relayout.

    A kernel reads tiles from SRAM and ends each result in a ``voyager.insert``,
    so the buffer primitives are the boundary: everything outside the two
    compute namespaces stops the walk (``voyager.*``, the HOPs, ``getitem``,
    loop-counter arithmetic), as does everything that is not a
    ``call_function`` (placeholders, ``get_attr``).  Target-based, not
    value-based -- the freshly exported body carries no shapes yet.
    """
    if n.op != "call_function":
        return False
    return getattr(n.target, "namespace", None) in ("aten", "quantized_ops")


def _store_cone(seed: Node) -> set:
    """The compute cone feeding one store: ``seed`` (the op whose result is
    ``insert``ed) plus its datapath ancestors, up to the load boundary.

    Grown *upward only* -- a cone is the ancestors of its stored tile.  The tile
    loads (placeholders, ``subview``s) end each branch, and a different cone's
    store, which this one reads back through a buffer rather than an SSA edge,
    is never crossed -- so the cones a reduction splits across bodies come out
    disjoint without the walk knowing they exist.
    """
    cone, stack = set(), [seed]
    while stack:
        n = stack.pop()
        if n in cone or not _on_datapath(n):
            continue
        cone.add(n)
        stack.extend(n.all_input_nodes)
    return cone


def fuse_store_cones(gm: GraphModule) -> None:
    """Fuse every cluster of compute that ends at a store into its own nested
    ``call_module``, recursing through the loop / cond / commit bodies.

    A bufferized kernel is a set of such clusters -- a GEMM tile, a fused tail,
    a reduction's per-round accumulate -- each ending in one ``voyager.insert``.
    The store is the anchor: seed from every ``insert`` and grow the datapath
    cone above it (:func:`_store_cone`).  Nothing here decides whether a cone is
    a GEMM, a tail, or an accumulate, nor whether the reduction is single-tile,
    software-pipelined, or async-flattened -- the buffer stores that separate
    those cases are exactly the cone boundaries, so one rule spans them all.
    """
    from voyager_compiler.codegen.subgraph import create_and_insert_subgraph

    for node in list(gm.graph.nodes):
        if node.op == "get_attr":
            sub = getattr(gm, str(node.target), None)
            if isinstance(sub, GraphModule):
                fuse_store_cones(sub)

    # A cone's result leaves the body one of two ways: an in-kernel ``insert``
    # (a tile store) or the body's ``output`` -- a cond branch / loop body whose
    # returned value the parent stores.  Seed from both: a reduction's
    # ``accumulate`` branch only *returns* ``gemm + scratch`` with no insert of
    # its own, yet its ``gemm``->``add`` must still fuse into one op.  A returned
    # buffer (a carried alloc, a sub-loop getitem) is not on the datapath and is
    # skipped.  A multi-output op (quantize_mx) is stored by several inserts that
    # all peel to the same seed, so it stays a single cone, its getitem users
    # left outside.
    seeds = []

    def _add_seed(src):
        while _is_writeout_wrapper(src):
            src = src.args[0]
        if _on_datapath(src) and src not in seeds:
            seeds.append(src)

    for n in gm.graph.nodes:
        if n.target is voyager.insert.default:
            _add_seed(n.args[0])
        elif n.op == "output":
            for o in n.all_input_nodes:
                _add_seed(o)

    consumed = set()
    for seed in seeds:
        if seed in consumed:
            continue
        cone = _store_cone(seed)
        # A view / same-dtype cast is no work -- the code generator drops it --
        # so a lone real op stores through its own insert and needs no wrapping.
        # Skip a cone overlapping one already fused (a shared operand's compute),
        # which would double-extract it.
        real = [n for n in cone if not is_nop(n)]
        if len(real) < 2 or cone & consumed:
            continue
        create_and_insert_subgraph(list(cone), gm)
        consumed |= cone

    gm.graph.lint()
    gm.recompile()


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
    # primitives (notably the side-effecting ``async_copy`` / ``insert``) and
    # the ``while_loop`` / ``cond`` HOPs carry side effects that DCE can't see — a
    # bufferized loop writes its output through ``insert`` / ``async_copy``
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
