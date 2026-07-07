"""
Bufferization pass: rewrite tiled FX nodes into an explicit bufferized FX graph.

For each *tiled* GEMM / conv2d / pointwise / pool ``call_function``, the pass
builds a ``while_loop`` nest over ``voyager.*`` primitives (via the ``pipeline``
software-pipelining builders) and splices it into the graph in place of the
node.  Untiled nodes (operands/outputs fit on-chip) are bufferized trivially;
fused ``call_module`` submodules dispatch on their anchor op and apply the
post-op tail in the kernel.

Runs after operator fusion and before memory allocation.
"""

import logging
import math
import operator
from typing import Dict, Optional

import torch
import torch.fx as fx
from torch.fx import GraphModule, Node

from ...pt2e_utils import update_submod_user_meta
from ..mapping import get_anchor_node, replace_node_with_graph_module
from ..mapping_utils import (
    quant_table_arg_nodes,
    is_conv2d,
    is_elementwise_op,
    is_gemm_op,
    is_indexing_or_concatenation_op,
    is_memory_op,
    is_nop,
    is_pooling,
    is_reshape_op,
    is_shape_changing_nop,
)
from .ops import MemoryLevel, oracle_disabled

logger = logging.getLogger(__name__)

voyager = torch.ops.voyager


# ---------------------------------------------------------------------------
# Memory-location annotation (self-contained; does NOT reuse _should_use_dram)
# ---------------------------------------------------------------------------

_VOYAGER_ALLOC = torch.ops.voyager.alloc.default  # DRAM output buffer
_VOYAGER_ZEROS = torch.ops.voyager.zeros.default  # accumulator / semaphore bank
_VOYAGER_INCR = torch.ops.voyager.increment_indices.default
_VOYAGER_DELIN = torch.ops.voyager.delinearize_index.default
_VOYAGER_ASYNC = (
    torch.ops.voyager.async_copy.default
)  # guarded DRAM<->Scratchpad DMA
_WHILE_LOOP = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond


def _subgraph(gm: GraphModule, target) -> Optional[GraphModule]:
    """The GraphModule attribute named ``target`` on ``gm`` (a loop body/cond
    or fused submodule), or None if ``target`` is not a submodule."""
    try:
        sub = gm.get_submodule(str(target))
    except AttributeError:
        return None
    return sub if isinstance(sub, GraphModule) else None


def _produces_tensor(node: Node) -> bool:
    """Whether ``node`` yields a tensor (or a tuple of tensors) — as opposed to
    an index / counter / SymInt from loop-counter arithmetic, which carries no
    memory space.
    """
    val = node.meta.get("val", getattr(node, "value", None))
    if isinstance(val, torch.Tensor):
        return True
    if isinstance(val, (tuple, list)):
        return any(isinstance(e, torch.Tensor) for e in val)
    return False


def _collect_codebook_nodes(gm: GraphModule, result: set) -> set:
    """Add every codebook / qmap node (recursing into ``while_loop`` bodies and
    fused ``call_module`` submodules) to ``result``; return this graph's
    placeholders that are codebooks, so a caller can flag the operands feeding
    them as codebooks too.

    Codebook-ness flows bottom-up: an op flags its codebook args
    (``quant_table_arg_nodes``) and an operand bound to a codebook sub-graph
    placeholder is itself a codebook — so a top-level ``code`` / ``qmap``
    get_attr threaded through the loops is caught.
    """
    local = set()

    def _thread(operands, sub):
        if sub is None:
            return
        sub_phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
        sub_cb = _collect_codebook_nodes(sub, result)
        for operand, ph in zip(operands, sub_phs):
            if isinstance(operand, Node) and ph in sub_cb:
                local.add(operand)

    for n in gm.graph.nodes:
        if n.op == "call_function":
            local |= quant_table_arg_nodes(n)
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
        elif n.op == "call_module":
            _thread(list(n.args), _subgraph(gm, n.target))

    result |= local
    return {p for p in gm.graph.nodes if p.op == "placeholder" and p in local}


_VOYAGER_WAIT = torch.ops.voyager.async_wait.default
_VOYAGER_INSERT = torch.ops.voyager.insert.default

# Memory / control-flow / semaphore ops — *not* tile compute, so they are exempt
# from the destination-passing check below: the buffer producers (``alloc`` /
# ``zeros``) and the ``insert`` store, the async DMA + its semaphore
# ``async_wait``, the index / counter ops (``increment_indices`` /
# ``delinearize_index``), the bank- and semaphore-read ``select``, and the
# multi-output / cond-unpack ``getitem``.  (These become registers; not
# annotated yet.)  ``voyager.zeros`` (the accumulator / per-slot semaphore bank
# init) is exempt — but a genuine ``aten.zeros`` *is* a compute op and is not.
_NON_COMPUTE = {
    _VOYAGER_ALLOC,
    _VOYAGER_ZEROS,
    _VOYAGER_ASYNC,
    _VOYAGER_WAIT,
    _VOYAGER_INSERT,
    _VOYAGER_INCR,
    _VOYAGER_DELIN,
    torch.ops.aten.select.int,
    operator.getitem,
}


def annotate_tensor_spaces(gm: GraphModule) -> None:
    """Alloc-only memory model: mark the *buffers* and validate that every tile
    computation lands in one.

      * **Buffers** carry ``node.meta['space']``: a ``voyager.alloc`` is
        Scratchpad (``MemoryLevel.SRAM``) or DRAM (its level arg); a top-level
        input placeholder and a weight ``get_attr`` are DRAM.  Codebooks /
        qmaps are *params* (passed to the accelerator, not memory) — unmarked.
      * **Compute** is *not* given a space.  Instead the destination-passing
        invariant is checked: every tile-compute ``call_function`` /
        ``call_module`` result must be written to a buffer via ``insert``.
        Control logic (``_NON_COMPUTE`` + NOPs) is exempt.

    Recurses into ``while_loop`` and ``cond`` bodies, checking every op inside;
    when an op's result is the body's output, the check threads to the users of
    the loop / cond node in the parent graph.  A violation raises.
    """
    codebooks: set = set()
    _collect_codebook_nodes(gm, codebooks)
    _annotate_and_validate(gm, codebooks, parent_hop=None, parent_ctx=None)


def _is_compute(node: Node) -> bool:
    """A tile-compute op subject to the destination-passing rule: it yields a
    tensor and is not a NOP / addressing op.  A fused ``call_module`` is one L1
    compute op."""
    if not _produces_tensor(node):
        return False
    if node.op == "call_module":
        return True
    if node.op != "call_function":
        return False
    if is_nop(node) or is_indexing_or_concatenation_op(node):
        return False
    return node.target not in _NON_COMPUTE


def _result_handles(node: Node, gm: GraphModule, parent_hop: Node) -> list:
    """For ``node`` returned as ``gm``'s output, the parent-graph ``getitem``
    handles of the loop / cond ``parent_hop`` at the matching output slot(s) —
    used to thread the store check across the body boundary."""
    out = next(x for x in gm.graph.nodes if x.op == "output")
    outs = out.args[0]
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    handles = []
    for i, o in enumerate(outs):
        if o is node:
            handles.append(
                next(
                    (
                        u
                        for u in parent_hop.users
                        if u.target is operator.getitem and u.args[1] == i
                    ),
                    None,
                )
            )
    return handles


def _stored(node: Node, ctx: tuple) -> bool:
    """True if every consumer of ``node``'s value — threading through NOP /
    ``getitem`` wrappers and the body->parent boundary — is an ``insert``."""
    gm, parent_hop, parent_ctx = ctx
    users = list(node.users)
    if not users:
        return False
    for u in users:
        if u.target is _VOYAGER_INSERT:
            continue
        if u.op == "output":
            if parent_hop is None:
                return False  # a top-level graph output, not a store
            for handle in _result_handles(node, gm, parent_hop):
                if handle is None or not _stored(handle, parent_ctx):
                    return False
            continue
        if (
            u.target is operator.getitem
            or u.target is torch.ops.aten.to.dtype
            or is_nop(u)
        ):
            # ``getitem`` / NOP wrappers, and a ``to.dtype`` cast (a cast in the
            # store path is fine — even a real dtype conversion), thread through.
            if not _stored(u, ctx):
                return False
            continue
        return False  # a real compute op consumes the tile -> not stored
    return True


def _annotate_and_validate(
    gm: GraphModule, codebooks: set, parent_hop, parent_ctx
) -> None:
    ctx = (gm, parent_hop, parent_ctx)
    for node in gm.graph.nodes:
        if node in codebooks:
            continue  # a param (codebook / qmap), not memory
        if node.op == "placeholder":
            if parent_hop is None and _produces_tensor(node):
                node.meta["space"] = "DRAM"  # a model input
        elif node.op == "get_attr":
            if _subgraph(gm, node.target) is None and _produces_tensor(node):
                node.meta["space"] = "DRAM"  # a weight param
        elif node.op == "call_function" and node.target is _VOYAGER_ALLOC:
            level = (
                node.args[2] if len(node.args) > 2 else int(MemoryLevel.DRAM)
            )
            node.meta["space"] = (
                "Scratchpad" if level == int(MemoryLevel.SRAM) else "DRAM"
            )
        elif node.op == "call_function" and node.target is _WHILE_LOOP:
            body = _subgraph(gm, node.args[1].target)
            if body is not None:
                _annotate_and_validate(body, codebooks, node, ctx)
        elif node.op == "call_function" and node.target is _COND:
            for graph_arg in (node.args[1], node.args[2]):
                branch = _subgraph(gm, graph_arg.target)
                if branch is not None:
                    _annotate_and_validate(branch, codebooks, node, ctx)
        elif is_memory_op(node) or is_reshape_op(node):
            node.meta["space"] = "DRAM"
        elif _is_compute(node):
            if not _stored(node, ctx):
                raise Exception(
                    f"destination-passing violation: result of {node.op} "
                    f"'{node.name}' ({node.target}) is not stored to a buffer "
                    f"via insert (it feeds a non-store consumer)"
                )


# ---------------------------------------------------------------------------
# Logical (quantized) dtype propagation
# ---------------------------------------------------------------------------


_SELECT = torch.ops.aten.select.int


def _dtype_of(n):
    return n.meta.get("dtype") if isinstance(n, Node) else None


def _set_dtype(n, dt) -> bool:
    """Assign logical dtype ``dt`` to ``n`` if it lacks one; True if changed."""
    if dt is None or not isinstance(n, Node) or n.meta.get("dtype") == dt:
        return False
    n.meta["dtype"] = dt
    return True


def _buffer_of(node):
    """Walk a copy destination back through ``select`` / NOP views to the
    underlying ``alloc`` (or placeholder bank) it writes, so a copy tags the
    buffer and not just the tile view."""
    seen = set()
    while isinstance(node, Node) and node not in seen:
        seen.add(node)
        if node.target is _VOYAGER_ALLOC or node.op == "placeholder":
            break
        if node.target is _SELECT or is_nop(node):
            node = node.args[0]
            continue
        break
    return node


def _outputs_of(gm: GraphModule) -> list:
    outs = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    return list(outs) if isinstance(outs, (list, tuple)) else [outs]


def _thread_hop(
    gm: GraphModule, hop: Node, operands, graph_args, rules
) -> None:
    """Recurse the forward pass into each loop body / cond branch: seed its
    placeholders from the bound ``operands``, propagate, then carry the dtypes it
    derived back out — onto the captured operands (SRAM banks written in place)
    and onto the HOP's ``getitem`` result handles (a cond branch's tensor
    output)."""
    handles = [
        u
        for u in hop.users
        if u.target is operator.getitem and isinstance(u.args[1], int)
    ]
    for ga in graph_args:
        sub = _subgraph(gm, ga.target)
        if sub is None:
            continue
        phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
        seed = {
            ph: _dtype_of(op)
            for op, ph in zip(operands, phs)
            if _dtype_of(op) is not None
        }
        propagate_logical_dtypes(sub, seed, rules)
        for op, ph in zip(operands, phs):
            d = _dtype_of(ph)
            _set_dtype(op, d)
            # Bridge a bank slot view (``select``) to its alloc, so an
            # in-loop-written output bank tags the bank, not just the slice.
            _set_dtype(_buffer_of(op), d)
        outs = _outputs_of(sub)
        for u in handles:
            if u.args[1] < len(outs):
                _set_dtype(u, _dtype_of(outs[u.args[1]]))


def _thread_call_module(gm: GraphModule, node: Node, rules) -> None:
    """Recurse into a fused ``call_module`` (operands → placeholders), propagate,
    and set the node's own dtype from the submodule's output(s)."""
    sub = _subgraph(gm, node.target)
    if sub is None:
        return
    phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
    seed = {
        ph: _dtype_of(a)
        for a, ph in zip(node.args, phs)
        if _dtype_of(a) is not None
    }
    propagate_logical_dtypes(sub, seed, rules)
    outs = _outputs_of(sub)
    if len(outs) > 1:
        dts = tuple(_dtype_of(o) for o in outs)
        if any(dts):
            _set_dtype(node, dts)
    else:
        _set_dtype(node, _dtype_of(outs[0]))


def propagate_logical_dtypes(
    gm: GraphModule,
    ph_dtypes: Optional[Dict[Node, str]] = None,
    compute_dtypes: Optional[Dict] = None,
) -> None:
    """Stamp each node's logical (quantized) ``meta['dtype']`` in one forward,
    program-order pass so codegen emits the quantized dtype, not the physical one.

    The rule is read straight from the original node / submodule the quantizer
    already annotated: ``compute_dtypes`` maps a compute op's target to its output
    dtype (an int GEMM's ``int24``, a ``quantize_mx`` ``(scale, value)`` tuple,
    …).  ``ph_dtypes`` seeds this graph's placeholders from the operands bound to
    them.  Then, in def-before-use order: a compute op takes its rule dtype, a
    pass-through (copy / select / getitem / NOP) inherits its input's dtype, and a
    ``while_loop`` / ``cond`` / fused ``call_module`` recurses inline.  Inputs
    always precede their uses, so a single pass settles the whole graph — no
    fixpoint or backward flow.
    """
    for ph, d in (ph_dtypes or {}).items():
        _set_dtype(ph, d)

    rules = compute_dtypes or {}
    for node in gm.graph.nodes:
        if node.op == "call_module":
            _thread_call_module(gm, node, rules)
            continue
        if node.op != "call_function":
            continue
        t = node.target
        if t is _WHILE_LOOP:
            extra = list(node.args[3]) if len(node.args) > 3 else []
            _thread_hop(
                gm, node, list(node.args[2]) + extra, (node.args[1],), rules
            )
        elif t is _COND:
            operands = list(node.args[3]) if len(node.args) > 3 else []
            _thread_hop(gm, node, operands, (node.args[1], node.args[2]), rules)
        elif rules.get(t) is not None:
            # A compute op: its output dtype comes from the original graph's rule.
            _set_dtype(node, rules[t])
        elif t is _VOYAGER_ASYNC or t is _VOYAGER_INSERT:
            # A copy: dst (and its buffer) inherits the src's dtype.
            d = _dtype_of(node.args[0])
            _set_dtype(node.args[1], d)
            _set_dtype(_buffer_of(node.args[1]), d)
        elif t is _SELECT:
            _set_dtype(node, _dtype_of(node.args[0]))  # bank slot
        elif t is operator.getitem:
            sdt = _dtype_of(node.args[0])
            i = node.args[1]
            if (
                isinstance(sdt, (list, tuple))
                and isinstance(i, int)
                and i < len(sdt)
            ):
                _set_dtype(node, sdt[i])
        elif is_nop(node):
            first = next((a for a in node.args if isinstance(a, Node)), None)
            _set_dtype(node, _dtype_of(first))


# ---------------------------------------------------------------------------
# Built-graph cache
#
# A built nest is a pure function of the node's *shapes* / structure — the
# weight / scale / codebook operands are placeholders wired at the splice site,
# never baked in (``replace_node_with_graph_module`` maps placeholders ->
# ``all_input_nodes``).  So identical layers (every Llama block is the same)
# build to identical nests; caching by a structural + shape/dtype signature
# skips the dominant ``export`` cost for every repeat.
#
# The signature is the canonical-form equivalent of an fx ``SubgraphMatcher``
# match (op / target / literals / topology) plus the shape / dtype / transposed
# metadata the matcher omits but the builder needs.
# ---------------------------------------------------------------------------


def _hashable(x):
    """Recursively coerce tiling / dtype metadata to a hashable form."""
    if isinstance(x, (list, tuple)):
        return tuple(_hashable(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in x.items()))
    return x


def _node_value(n):
    """``(shape, dtype)`` of the tensor ``n`` produced — nested for a tuple
    output, ``None`` for a non-tensor value.  Reads ``.value`` (else the
    exported ``meta['val']``), stamped during shape-prop / operator fusion."""
    val = getattr(n, "value", n.meta.get("val"))
    if isinstance(val, torch.Tensor):
        return (tuple(val.shape), str(val.dtype))
    if isinstance(val, (list, tuple)):
        return tuple((tuple(v.shape), str(v.dtype)) for v in val)
    return None


def _operand_key(n):
    """An argument node's *interface*: shape + physical / logical dtype.  Its
    own ``transposed`` / ``l2_tiling`` (how it is built) belong to where the
    node is keyed in its own right, not to each use of it as an operand."""
    return (_node_value(n), _hashable(n.meta.get("dtype")))


def _meta_key(n):
    """A node's own shape / dtype / build-affecting metadata signature: the
    operand interface plus the metadata that drives *this* node's build."""
    return (
        _operand_key(n),
        _hashable(n.meta.get("transposed")),  # layout flip
        _hashable(n.meta.get("l2_tiling")),  # explicit tile counts
    )


def _node_to_hashable(node, index_of=None):
    """Turn one FX node into a hashable signature: op, target, its own
    shape/dtype/meta, and its args/kwargs.  Each Node argument contributes its
    ``_meta_key`` (shape/dtype/meta must match) plus its position in the owning
    graph (``index_of``) to pin the wiring; literals contribute type + value
    (mirroring ``SubgraphMatcher._match_literals``)."""

    def canonicalize(a):
        if isinstance(a, Node):
            ref = index_of.get(a) if index_of is not None else None
            return ("node", ref, _operand_key(a))
        if isinstance(a, (list, tuple)):
            return ("seq", tuple(canonicalize(x) for x in a))
        return ("lit", type(a).__name__, a)

    # A placeholder / get_attr ``target`` is a per-instance *name* (unique per
    # layer), so it must not enter the key: placeholders are wildcards and
    # constants match by shape/dtype (their ``_meta_key``), as in
    # ``SubgraphMatcher``.  Only the call ops contribute their target.
    target = (
        None if node.op in ("placeholder", "get_attr") else str(node.target)
    )
    return (
        node.op,
        target,
        _meta_key(node),
        tuple(canonicalize(a) for a in node.args),
        tuple(sorted((k, canonicalize(v)) for k, v in node.kwargs.items())),
    )


def _gm_to_hashable(gm):
    """Hashable signature of a GraphModule: every node's ``_node_to_hashable``
    in topological order, with Node args referenced by index to pin the
    wiring."""
    index_of = {n: i for i, n in enumerate(gm.graph.nodes)}
    return tuple(_node_to_hashable(n, index_of) for n in gm.graph.nodes)


def _bufferize_key(node):
    """A structural + shape/dtype signature for ``node``'s bufferized nest:
    two nodes with the same key build to the same nest (only the operand
    *tensors* wired at the splice site differ).  ``None`` (uncacheable) for a
    non-tensor bare output or any value that fails to hash."""
    try:
        submod = node.meta.get("submodule")
        if isinstance(submod, GraphModule):
            key = ("fused", _gm_to_hashable(submod))
        elif _node_value(node) is None:
            return None
        else:
            key = ("bare", _node_to_hashable(node))
        hash(key)  # surface any unhashable literal as uncacheable
        return key
    except TypeError:
        return None


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------


def bufferize_graph(
    model: GraphModule, pipelined: bool = False, tiler=None
) -> GraphModule:
    """
    Rewrite tiled GEMM / pointwise nodes into bufferized while_loop nests.
    Returns the same (mutated) model.

    Assumes shapes are already populated (``node.value``); each node is built
    from its (and its inputs') shapes/dtypes and spliced in place.

    ``pipelined`` reuses the tiler's double-buffering decision (it already
    sizes tiles so two L2 tiles fit) to emit software-pipelined loop nests:
    GEMM/conv double-buffer their C-reduction; pointwise unrolls small grids
    ahead.
    """
    from .pipeline import (
        build_conv2d,
        build_gemm,
        build_pointwise,
        build_pool,
    )

    graph = model.graph
    # ``pipelined`` (the tiler's double-buffering decision) selects the input
    # software-pipeline depth: 2 banks (double buffer) vs 1 (single buffer).
    num_banks = 2 if pipelined else 1
    # Identical layers build identical nests; cache by structural signature so
    # ``export`` runs once per distinct shape, not once per node.
    build_cache = {}

    for node in list(graph.nodes):
        # Dispatch is driven by an *anchor* op: for a fused ``call_module`` it is
        # the submodule's matrix/pointwise anchor (the builders parse the rest of
        # the submodule and apply the fused tail in the kernel — num_k == 1 folds
        # the tail into ``compute``, num_k > 1 accumulates into a scratch ref and
        # applies it once on the last reduction step); for a bare
        # ``call_function`` it is the node itself.
        if node.op not in ("call_module", "call_function"):
            continue

        # A shape-preserving nop is a pure pass-through — rewire its users to
        # its input
        if is_nop(node) and not is_shape_changing_nop(node):
            inp = node.all_input_nodes[0]
            node.replace_all_uses_with(inp)
            update_submod_user_meta(model, inp)
            graph.erase_node(node)
            continue

        if (
            node.target == operator.getitem
            or is_nop(node)
            or is_memory_op(node)
            or is_indexing_or_concatenation_op(node)
            or is_reshape_op(node)
        ):
            continue

        anchor = get_anchor_node(node)

        key = _bufferize_key(node)
        cached = build_cache.get(key) if key is not None else None
        logger.debug(
            "[bufferize] %s anchor=%s %s",
            node.name,
            anchor.target,
            "HIT" if cached is not None else "MISS",
        )
        if cached is not None:
            sub_gm, n_out = cached
        else:
            # A fused submodule with no matrix/pool anchor runs whole through the
            # pointwise builder; a bare op only when it is elementwise / a
            # kept-reduction op (pool only when bare — a fused pool is pointwise).
            if is_conv2d(anchor):
                sub_gm = build_conv2d(node, num_banks=num_banks, tiler=tiler)
            elif is_gemm_op(anchor):
                sub_gm = build_gemm(node, num_banks=num_banks, tiler=tiler)
            elif is_pooling(anchor):
                sub_gm = build_pool(node, num_banks=num_banks)
            elif (
                is_elementwise_op(anchor)
                or anchor.target in _REDUCTION_POINTWISE_OPS
                or anchor.target in _RELAYOUT_POINTWISE_OPS
            ):
                sub_gm = build_pointwise(node, num_banks=num_banks)
            else:
                sub_gm = None

            if sub_gm is not None:
                val = node.value
                n_out = len(val) if isinstance(val, (list, tuple)) else 1
            elif not (
                (tiling := node.meta.get("l2_tiling")) is not None
                and math.prod(tiling) > 1
            ):
                # Untiled / interstellar-skipped op (FC batch-1, depthwise conv,
                # a shape-changing reshape/slice): bufferize whole (trip-1).
                built = _build_for_untiled(node)
                if built is None:
                    continue  # getitem / non-tensor: nothing to load/store
                sub_gm, n_out = built
            else:
                raise Exception(
                    f"bufferization: no builder for tiled node {node.op} "
                    f"'{node.name}' ({node.target})"
                )

            if key is not None:
                build_cache[key] = (sub_gm, n_out)

        # Stamp logical (quantized) dtypes on the nest, all read off the original
        # ``node`` (the quantizer already annotated it): seed the placeholders
        # from the input operands' dtypes, and the per-target output-dtype rule
        # from every compute op of the original submodule (a bare op = itself).
        # A single forward pass then spreads them through the nest.
        phs = [p for p in sub_gm.graph.nodes if p.op == "placeholder"]
        ph_seed = {
            ph: inp.meta.get("dtype")
            for ph, inp in zip(phs, node.all_input_nodes)
        }
        submod = node.meta.get("submodule")
        src_nodes = (
            submod.graph.nodes if isinstance(submod, GraphModule) else [node]
        )
        compute_seed = {
            n.target: n.meta["dtype"]
            for n in src_nodes
            if n.op == "call_function" and n.meta.get("dtype") is not None
        }
        propagate_logical_dtypes(sub_gm, ph_seed, compute_seed)

        with oracle_disabled():
            results = replace_node_with_graph_module(
                model, node, sub_gm, propagate=False
            )
        logger.debug("[bufferize] %s spliced (n_out=%d)", node.name, n_out)

        # Carry the original node's shape/value onto the nest output(s): later
        # builders read an operand's ``.value`` directly, and ``propagate=False``
        # skipped the per-node re-execution that used to set it.  Internal nest
        # nodes keep the exported ``meta['val']`` (enough for memory planning).
        out_vals = node.value if n_out > 1 else [node.value]
        out_shapes = node.shape if n_out > 1 else [node.shape]
        for r, v, s in zip(results, out_vals, out_shapes):
            r.value, r.shape = v, s

        if n_out == 1:
            if "dtype" in node.meta:
                results[0].meta["dtype"] = node.meta["dtype"]
            # Rewiring a node feeding a downstream fused call_module leaves that
            # submodule's placeholder name/source_node pointing at the now-dead
            # node; refresh it (as fuse_operator does) so codegen/print resolve
            # the live node.
            update_submod_user_meta(model, results[0])
        else:
            # Multi-output fused tail (e.g. quantize_mx): each getitem(idx) user
            # was rewired to the nest's idx-th output buffer; tag its dtype,
            # refresh downstream submodules, and drop the now-dead getitem.
            dtypes = node.meta.get("dtype")
            for user in list(node.users):
                assert (
                    user.target is operator.getitem
                ), f"multi-output fused node {node} has non-getitem user {user}"
                idx = user.args[1]
                res = results[idx]
                if (
                    isinstance(dtypes, (list, tuple))
                    and dtypes[idx] is not None
                ):
                    res.meta["dtype"] = dtypes[idx]
                update_submod_user_meta(model, res)
                graph.erase_node(user)
        graph.erase_node(node)

    graph.lint()
    model.recompile()
    annotate_tensor_spaces(model)
    return model


_MULTI_OUTPUT_POINTWISE = {
    torch.ops.quantized_ops.quantize_mx.default,
    torch.ops.quantized_ops.quantize_mx_outlier.default,
}
_REDUCTION_POINTWISE_OPS = _MULTI_OUTPUT_POINTWISE | {
    torch.ops.quantized_ops.quantize.default,
    torch.ops.quantized_ops.layer_norm.default,
    torch.ops.aten.layer_norm.default,
    torch.ops.aten.softmax.int,
}
# A standalone transpose / permute relayout — build_pointwise stores each tile
# identity and loads the input from the transposed source.  (Untiled ones fall
# to ``_build_for_untiled``; build_pointwise returns ``None`` when untiled.)
_RELAYOUT_POINTWISE_OPS = {
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
}


def _build_for_untiled(node: Node):
    """Bufferize an untiled op through the pipeline scheduler with *whole-tensor*
    tiles: a single-step ``PipelinedKernel`` (grid ``(1,)``) that DMA-loads each
    tiled input whole, runs the op, and stores each output whole — no tiling
    loop, since the operands and output(s) fit on-chip.  Codebooks and scalars
    (0-D / single-element) are passed whole, not loaded.

    Every tensor dim maps to the lone (extent-1) grid dim, so the spec is
    op-agnostic — it works for shape-changing ops (``reshape`` / ``slice``)
    where the elementwise-alignment spec of ``build_pointwise`` would not.

    Returns ``(sub_gm, n_outputs)``, or ``None`` for nodes with nothing to
    load/store (``getitem``, a non-tensor output).
    """
    from .pipeline import _map_kernel, build_pipelined_buffers
    from .utils import _InputSpec, _OutputSpec

    if node.target is operator.getitem:
        return None
    val = getattr(node, "value", None)
    if not isinstance(val, (torch.Tensor, list, tuple)):
        return None

    in_nodes = node.all_input_nodes
    inputs = [n.value.clone() for n in in_nodes]
    outputs = list(val) if isinstance(val, (list, tuple)) else [val]
    codebooks = quant_table_arg_nodes(node)

    # Resolve each arg *now* into a ``(value, is_index)`` template: an input Node
    # -> ``(tile_slot, True)`` (the loaded tile at that slot is substituted when
    # the body is traced — dynamo rejects FX-Node lookups there; mirrors
    # ``build_pointwise``), a constant -> ``(value, False)``; a list/tuple
    # recurses, so a Node nested in a list arg (``stack([t0, t1], dim)``) is
    # resolved instead of leaking an FX Node into the export.
    order = {n: i for i, n in enumerate(in_nodes)}

    def _resolve(a):
        if isinstance(a, fx.Node):
            return (order[a], True)
        if isinstance(a, (list, tuple)):
            return [_resolve(x) for x in a]
        return (a, False)

    arg_tmpl = [_resolve(a) for a in node.args]
    kw_tmpl = {k: _resolve(v) for k, v in node.kwargs.items()}
    op = node.target

    def _fill(t, tiles):
        if isinstance(t, list):
            return [_fill(x, tiles) for x in t]
        value, is_index = t
        return tiles[value] if is_index else value

    def compute(*tiles):
        args = [_fill(t, tiles) for t in arg_tmpl]
        kwargs = {k: _fill(t, tiles) for k, t in kw_tmpl.items()}
        return op(*args, **kwargs)

    grid = (1,)
    in_specs = []
    for n in in_nodes:
        if n in codebooks or n.value.ndim == 0 or list(n.value.shape) == [1]:
            in_specs.append(None)
        else:
            in_specs.append(
                _InputSpec(
                    tuple(n.value.shape),
                    (0,) * n.value.ndim,
                    (False,) * n.value.ndim,
                )
            )
    out_specs = [
        _OutputSpec(tuple(o.shape), tuple(o.shape), (0,) * o.ndim, o.dtype)
        for o in outputs
    ]
    kernel = _map_kernel(compute, len(outputs))
    gm = build_pipelined_buffers(
        kernel, grid, in_specs, out_specs, tuple(inputs), num_banks=1
    )
    return gm, len(outputs)
