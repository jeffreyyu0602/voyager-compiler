"""
Bufferization pass: rewrite tiled FX nodes into an explicit bufferized FX graph.

For each *tiled* GEMM / conv2d / pointwise / pool ``call_function``, the pass
builds a ``while_loop`` nest over ``voyager.*`` primitives (via the ``pipeline``
software-pipelining builders) and splices it into the graph in place of the
node.  Untiled nodes (operands/outputs fit on-chip) are bufferized trivially;
fused ``call_module`` submodules have no pipelined path yet (tail fusion is
being redesigned) and raise.

Runs after operator fusion and before memory allocation.
"""

import math
import operator
from typing import Dict, Optional

import torch
import torch.fx as fx
from torch.fx import GraphModule, Node

from ...pt2e_utils import update_submod_user_meta
from ..mapping import get_anchor_node, replace_node_with_graph_module
from ..mapping_utils import (
    is_conv2d,
    is_elementwise_op,
    is_gemm_op,
    is_nop,
    is_pooling,
)
from .ops import MemoryLevel

voyager = torch.ops.voyager


# ---------------------------------------------------------------------------
# Memory-location annotation (self-contained; does NOT reuse _should_use_dram)
# ---------------------------------------------------------------------------

_VOYAGER_LOAD = torch.ops.voyager.load_tile.default
_VOYAGER_STORE = torch.ops.voyager.store_tile.default
_VOYAGER_ALLOC = torch.ops.voyager.alloc.default  # DRAM output buffer
_VOYAGER_ZERO = torch.ops.voyager.zero_tile.default  # Scratchpad accumulator
_VOYAGER_INCR = torch.ops.voyager.increment_indices.default
_VOYAGER_DELIN = torch.ops.voyager.delinearize_index.default
_VOYAGER_ASYNC = (
    torch.ops.voyager.async_copy.default
)  # guarded DRAM<->Scratchpad DMA
_WHILE_LOOP = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond

# Whole-tensor preprocessing done in DRAM before tiling (the conv / pool halo
# pad); its padded output is then load_tiled, so it stays in DRAM rather than
# being a tile compute.
_DRAM_TRANSFORM = {
    torch.ops.aten.pad.default,
    torch.ops.aten.constant_pad_nd.default,
}


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


def _is_scalar_operand(node: Node) -> bool:
    """A 0-D or single-element operand — passed *whole* into an op (like a
    codebook), not tiled, so it may legitimately be a DRAM input to a compute
    op."""
    val = node.meta.get("val", getattr(node, "value", None))
    return isinstance(val, torch.Tensor) and (
        val.ndim == 0 or list(val.shape) == [1]
    )


def _collect_codebook_nodes(gm: GraphModule, result: set) -> set:
    """Add every codebook / qmap node (recursing into ``while_loop`` bodies and
    fused ``call_module`` submodules) to ``result``; return this graph's
    placeholders that are codebooks, so a caller can flag the operands feeding
    them as codebooks too.

    Codebook-ness flows bottom-up: an op flags its codebook args
    (``_codebook_arg_nodes``) and an operand bound to a codebook sub-graph
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
            local |= _codebook_arg_nodes(n)
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
_VOYAGER_COPY = torch.ops.voyager.copy_tile.default

# Control logic / addressing / DMA / store / buffer ops — *not* tile compute, so
# they are exempt from the destination-passing check below.  The prefetch
# semaphore (``where`` / ``eq`` / the DMA-guard's ``zeros`` dummy), the async
# DMA + its token wait, the index ops, the bank-read ``select``, the
# multi-output ``getitem``, the ``copy_tile`` store itself, and the ``alloc``
# buffer all fall here.  (These become registers; not annotated yet.)
_NON_COMPUTE = {
    _VOYAGER_ALLOC,
    _VOYAGER_ASYNC,
    _VOYAGER_WAIT,
    _VOYAGER_COPY,
    _VOYAGER_INCR,
    _VOYAGER_DELIN,
    torch.ops.aten.select.int,
    torch.ops.aten.where.self,
    torch.ops.aten.eq.Scalar,
    torch.ops.aten.eq.Tensor,
    torch.ops.aten.zeros.default,
    torch.ops.aten.arange.default,  # grid-index range (addressing)
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
        ``call_module`` result must be written to a buffer via ``copy_tile``.
        Control logic (``_NON_COMPUTE`` + NOPs) is exempt.

    Recurses into ``while_loop`` and ``cond`` bodies, checking every op inside;
    when an op's result is the body's output, the check threads to the users of
    the loop / cond node in the parent graph.  A violation raises.
    """
    codebooks: set = set()
    _collect_codebook_nodes(gm, codebooks)
    _annotate_and_validate(gm, codebooks, parent_hop=None, parent_ctx=None)


def _is_compute(node: Node, codebooks: set) -> bool:
    """A tile-compute op subject to the destination-passing rule: it yields a
    tensor and is not a codebook, a NOP, or a control / DMA / store / buffer
    op.  A fused ``call_module`` is one L1 compute op."""
    if node in codebooks or not _produces_tensor(node):
        return False
    if node.op == "call_module":
        return True
    if node.op != "call_function":
        return False
    return node.target not in _NON_COMPUTE and not is_nop(node)


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
    ``getitem`` wrappers and the body->parent boundary — is a ``copy_tile``."""
    gm, parent_hop, parent_ctx = ctx
    users = list(node.users)
    if not users:
        return False
    for u in users:
        if u.target is _VOYAGER_COPY:
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
        elif _is_compute(node, codebooks):
            if not _stored(node, ctx):
                raise Exception(
                    f"destination-passing violation: result of {node.op} "
                    f"'{node.name}' ({node.target}) is not stored to a buffer "
                    f"via copy_tile (it feeds a non-store consumer)"
                )


# ---------------------------------------------------------------------------
# Logical (quantized) dtype propagation
# ---------------------------------------------------------------------------


def propagate_logical_dtypes(
    gm: GraphModule, ph_dtypes: Optional[Dict[Node, str]] = None
) -> None:
    """Flow logical (quantized) dtypes from DRAM tensors onto the nest's tiles.

    The quantizer leaves ``meta['dtype']`` (a string like ``'nf4_6'`` /
    ``'fp8_e5m3'``) on input / weight / scale nodes, and the pass tags each
    output buffer (``voyager.alloc``) with the output's logical dtype.  A
    ``load_tile`` then inherits its source's dtype and the tile feeding a
    ``store_tile`` inherits the destination buffer's, so codegen emits the
    logical dtype rather than the physical storage dtype.  Threads through
    ``while_loop`` bodies via their carried / additional inputs.  Ops whose
    output is genuinely physical (the fp32 accumulator from a GEMM/conv) keep
    no logical dtype.
    """
    for ph, d in (ph_dtypes or {}).items():
        if isinstance(d, str):
            ph.meta.setdefault("dtype", d)

    def _dt(n):
        return n.meta.get("dtype") if isinstance(n, Node) else None

    while_loop = torch.ops.higher_order.while_loop
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is _VOYAGER_LOAD:
            if (d := _dt(node.args[0])) is not None:
                node.meta["dtype"] = d
        elif node.target is _VOYAGER_STORE:
            if (d := _dt(node.args[1])) is not None:  # destination buffer
                node.meta["dtype"] = d
                if isinstance(node.args[0], Node):
                    node.args[0].meta.setdefault("dtype", d)
        elif node.target is while_loop:
            body = getattr(gm, str(node.args[1].target), None)
            if isinstance(body, GraphModule):
                inputs = list(node.args[2])
                if len(node.args) > 3:
                    inputs += list(node.args[3])
                body_phs = [
                    n for n in body.graph.nodes if n.op == "placeholder"
                ]
                child = {ph: _dt(inp) for ph, inp in zip(body_phs, inputs)}
                propagate_logical_dtypes(body, child)
        elif node.target is operator.getitem:
            src = node.args[0]
            if isinstance(src, Node) and src.target is while_loop:
                carried = list(src.args[2])
                idx = node.args[1]
                if isinstance(idx, int) and idx < len(carried):
                    if (d := _dt(carried[idx])) is not None:
                        node.meta["dtype"] = d


# ---------------------------------------------------------------------------
# Tile-size derivation from node.meta
# ---------------------------------------------------------------------------


def _is_tiled(node: Node) -> bool:
    tiling = node.meta.get("l2_tiling")
    return tiling is not None and math.prod(tiling) > 1


_CODEBOOK_PARAMS = {
    "qmap",
    "scale_qmap",
    "output_code",
    "code",
    "input_qmap",
    "output_qmap",
    "input_code",
    "weight_code",
}


def _codebook_arg_nodes(node: Node) -> set:
    """Tensor args of ``node`` that are quantization codebooks / qmaps."""
    result = set()
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return result
    for i, arg in enumerate(schema.arguments):
        if arg.name not in _CODEBOOK_PARAMS:
            continue
        val = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
        if isinstance(val, Node):
            result.add(val)
    return result


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------


def bufferize_graph(model: GraphModule, pipelined: bool = False) -> GraphModule:
    """
    Rewrite tiled GEMM / pointwise nodes into bufferized while_loop nests.
    Returns the same (mutated) model.

    Assumes shapes are already populated (``node.value``).  Build specs are
    snapshotted up front (only shapes/dtypes are needed for export), so splicing
    one node does not invalidate the build of a later node whose inputs change.

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

    # Snapshot which nodes to bufferize (and their built nests) before mutating
    # the graph: each entry is ``(node, sub_gm, n_outputs)``.
    specs = []
    for node in list(graph.nodes):
        if node.op == "call_module":
            # Fused GEMM/conv + post-op pointwise ops.  Dispatch to the pipeline
            # builders, which parse the submodule's anchor + fused ops and apply
            # them in the kernel (operands threaded through ``in_specs``).  They
            # handle num_k == 1 (tail in ``compute``) and num_k > 1 (accumulate
            # into a scratch ref, tail once on the last reduction step); an
            # unsupported anchor returns ``None``.
            submod = getattr(model, str(node.target), None)
            if not isinstance(submod, GraphModule):
                continue
            node.meta.setdefault("submodule", submod)
            anchor_node = get_anchor_node(submod.graph.nodes)
            if anchor_node is None:
                continue
            if is_conv2d(anchor_node):
                sub_gm = build_conv2d(node, num_banks=num_banks)
            elif is_gemm_op(anchor_node):
                sub_gm = build_gemm(node, num_banks=num_banks)
            else:
                # A pure-pointwise fused submodule (e.g. relu(x + residual)):
                # run the whole submodule per output tile.
                sub_gm = build_pointwise(node, num_banks=num_banks)
            if sub_gm is None:
                raise NotImplementedError(
                    f"fused submodule {node.name!r}: no pipelined fusion path"
                )
            val = node.value
            n_out = len(val) if isinstance(val, (list, tuple)) else 1
            specs.append((node, sub_gm, n_out))
            continue
        if node.op != "call_function":
            continue
        if not _is_tiled(node):
            # Untiled op (operands/output fit on-chip): bufferize trivially —
            # load every input whole, run the op, store the output(s) whole (no
            # loop).
            built = _build_for_untiled(node)
            if built is not None:
                specs.append((node, *built))
            continue
        if is_conv2d(node):
            sub_gm, n_out = build_conv2d(node, num_banks=num_banks), 1
        elif is_gemm_op(node):
            sub_gm, n_out = build_gemm(node, num_banks=num_banks), 1
        elif is_pooling(node):
            sub_gm, n_out = build_pool(node, num_banks=num_banks), 1
        elif is_elementwise_op(node) or node.target in _REDUCTION_POINTWISE_OPS:
            sub_gm = build_pointwise(node, num_banks=num_banks)
            val = node.value
            n_out = len(val) if isinstance(val, (list, tuple)) else 1
        else:
            raise NotImplementedError(
                f"Unsupported tiled op for bufferization: {node.target}"
            )
        if sub_gm is not None:
            specs.append((node, sub_gm, n_out))

    for node, sub_gm, n_out in specs:
        results = replace_node_with_graph_module(model, node, sub_gm)

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
    # Flow logical (quantized) dtypes onto the new load/store tiles + buffers so
    # the emitted proto reports the quantized dtype, not the physical storage
    # one.
    propagate_logical_dtypes(model)
    annotate_tensor_spaces(model)
    return model


_MULTI_OUTPUT_POINTWISE = {
    torch.ops.quantized_ops.quantize_mx.default,
    torch.ops.quantized_ops.quantize_mx_outlier.default,
}
_REDUCTION_POINTWISE_OPS = _MULTI_OUTPUT_POINTWISE | {
    torch.ops.quantized_ops.quantize.default,
    torch.ops.aten.layer_norm.default,
    torch.ops.aten.softmax.int,
    torch.ops.aten._softmax.default,
}


def _build_for_untiled(node: Node):
    """Bufferize an untiled op trivially: load each tiled tensor input whole
    (codebooks and scalars passed through, not loaded), run the op, store each
    output whole — no loop, since the operands and output(s) fit on-chip.

    Returns ``(sub_gm, n_outputs)``, or ``None`` for nodes with nothing to
    load/store (``getitem``, a non-tensor output, or no tensor inputs).
    """
    if node.target is operator.getitem:
        return None
    val = getattr(node, "value", None)
    if not isinstance(val, (torch.Tensor, list, tuple)):
        return None

    g = fx.Graph()
    # Placeholders in ``all_input_nodes`` order, so the splice wires them
    # uniformly.  Codebook / qmap operands and scalars (0-D or single-element)
    # are passed *whole* — the op indexes / broadcasts them directly, they are
    # not tiled DMA loads — as in the tiled pointwise path.  Every other input
    # is DMA'd in whole via a single load_tile.
    codebooks = _codebook_arg_nodes(node)
    remap = {}
    for inp in node.all_input_nodes:
        shape = list(inp.value.shape)
        placeholder = g.placeholder(inp.name)
        if inp in codebooks or len(shape) == 0 or shape == [1]:
            remap[inp] = placeholder  # whole operand, not a tiled load
        else:
            remap[inp] = g.call_function(
                voyager.load_tile.default,
                (placeholder, [0] * len(shape), shape),
            )
    # The op itself, with its tensor args now the loaded tiles (scalar /
    # codebook args pass through unchanged).
    op = g.node_copy(node, lambda n: remap[n])

    multi = isinstance(val, (list, tuple))
    vals = list(val) if multi else [val]
    dtype_meta = node.meta.get("dtype")
    # ``dtype`` is per-output for a multi-output op, but absent (None) for a
    # non-quantized one — so key off whether it's actually a sequence, not off
    # ``multi`` (list(None)!).
    dtypes = (
        list(dtype_meta)
        if isinstance(dtype_meta, (list, tuple))
        else [dtype_meta]
    )
    outputs = []
    for idx, v in enumerate(vals):
        shape = list(v.shape)
        buf = g.call_function(voyager.alloc.default, (shape, v.dtype))
        if idx < len(dtypes) and isinstance(dtypes[idx], str):
            buf.meta["dtype"] = dtypes[idx]  # emit the dtypes (quantized) dtype
        src = g.call_function(operator.getitem, (op, idx)) if multi else op
        # ``store_tile`` writes ``buf`` in place (a side effect, returns
        # nothing); the op's output is the buffer itself, not the store.
        g.call_function(
            voyager.store_tile.default, (src, buf, [0] * len(shape), shape)
        )
        outputs.append(buf)
    g.output(tuple(outputs))
    g.lint()
    return fx.GraphModule(torch.nn.Module(), g), len(vals)
