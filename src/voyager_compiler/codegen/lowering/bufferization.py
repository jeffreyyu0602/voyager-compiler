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
        elif n.op == "call_module":
            _thread(list(n.args), _subgraph(gm, n.target))

    result |= local
    return {p for p in gm.graph.nodes if p.op == "placeholder" and p in local}


def annotate_tensor_spaces(gm: GraphModule) -> None:
    """
    Annotate ``node.meta['space']`` with each tensor's memory space and
    validate the bufferized memory model:

      * placeholder / get_attr param                  -> DRAM
      * ``voyager.alloc`` buffer                      -> DRAM
      * ``voyager.zero_tile`` accumulator             -> Scratchpad
      * ``load_tile`` (source DRAM)                   -> Scratchpad
      * ``store_tile`` (source Scratchpad, dest DRAM) -> DRAM
      * any other op / fused ``call_module`` (inputs Scratchpad) -> Scratchpad

    Spaces thread through ``while_loop`` / ``call_module`` boundaries: a body
    placeholder inherits the space of the operand it is bound to and a loop
    result inherits its carried value — so e.g. the carried ``zero_tile``
    accumulator stays Scratchpad inside the reduction loop.  Codebook / qmap
    operands are *unallocated*: left unmarked and skipped in the checks.
    Asserts each rule, so a violation raises ``AssertionError``.
    """
    codebooks: set = set()
    _collect_codebook_nodes(gm, codebooks)
    _annotate_spaces(gm, {}, codebooks)


def _annotate_spaces(gm: GraphModule, ph_space: dict, codebooks: set) -> list:
    """Annotate ``gm`` given ``{placeholder: space}`` from its caller; return
    the spaces of its output values (to thread a ``while_loop`` body's results
    back to the loop's getitems)."""
    space: dict = {}
    loop_results: dict = {}  # while_loop node -> [result spaces]
    out_spaces: list = []

    def sp(x):
        return space.get(x) if isinstance(x, Node) else None

    def scratchpad_inputs(node):
        for a in node.all_input_nodes:
            if a in codebooks or sp(a) is None or _is_scalar_operand(a):
                continue  # codebook / scalar (whole) or non-tensor
            assert (
                sp(a) == "Scratchpad"
            ), f"{node.target} input '{a}' must be Scratchpad, got {sp(a)}"

    for node in gm.graph.nodes:
        if node in codebooks:
            continue  # unallocated; not marked

        if node.op == "placeholder":
            if _produces_tensor(node):
                space[node] = ph_space.get(node, "DRAM")
        elif node.op == "get_attr":
            if _subgraph(gm, node.target) is None and _produces_tensor(node):
                space[node] = "DRAM"  # a param tensor (sub-graph attrs skipped)
        elif node.op == "output":
            outs = node.args[0]
            outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
            out_spaces = [sp(o) for o in outs]
        elif node.op == "call_module":
            # A fused op: a re-fused nest op reads tiles (-> Scratchpad); a
            # raw, non-bufferized fused op reads whole DRAM tensors (-> DRAM).
            # Infer from its operands (codebooks excluded); recurse only into
            # the tile-op form so its interior tiles are annotated without
            # tripping the check on a raw DRAM op.
            ins = {
                sp(a)
                for a in node.all_input_nodes
                if a not in codebooks and sp(a) is not None
            }
            tile_op = "Scratchpad" in ins
            space[node] = "Scratchpad" if tile_op else "DRAM"
            sub = _subgraph(gm, node.target)
            if sub is not None and tile_op:
                sub_phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
                bound = {
                    p: sp(o)
                    for p, o in zip(node.args, sub_phs)
                    if isinstance(o, Node) and sp(o) is not None
                }
                _annotate_spaces(sub, bound, codebooks)
        elif node.op == "call_function":
            t = node.target
            if t is _VOYAGER_ALLOC:
                # alloc(size, dtype, space): the MemoryLevel arg selects the
                # buffer's home — DRAM (default) or an on-chip SRAM tile bank
                # (-> Scratchpad).
                level = (
                    node.args[2]
                    if len(node.args) > 2
                    else int(MemoryLevel.DRAM)
                )
                space[node] = (
                    "Scratchpad" if level == int(MemoryLevel.SRAM) else "DRAM"
                )
            elif t is _VOYAGER_ZERO:
                space[node] = "Scratchpad"
            elif t is _VOYAGER_LOAD:
                assert sp(node.args[0]) == "DRAM", (
                    f"load_tile source '{node.args[0]}' must be DRAM, "
                    f"got {sp(node.args[0])}"
                )
                space[node] = "Scratchpad"
            elif t is _VOYAGER_STORE:
                assert sp(node.args[0]) == "Scratchpad", (
                    f"store_tile source '{node.args[0]}' must be Scratchpad, "
                    f"got {sp(node.args[0])}"
                )
                assert sp(node.args[1]) == "DRAM", (
                    f"store_tile dest '{node.args[1]}' must be DRAM, "
                    f"got {sp(node.args[1])}"
                )
                space[node] = "DRAM"
            elif t is _WHILE_LOOP:
                operands = list(node.args[2])
                if len(node.args) > 3:
                    operands += list(node.args[3])
                body = _subgraph(gm, node.args[1].target)
                if body is not None:
                    body_phs = [
                        p for p in body.graph.nodes if p.op == "placeholder"
                    ]
                    bound = {
                        p: sp(o)
                        for p, o in zip(body_phs, operands)
                        if sp(o) is not None
                    }
                    loop_results[node] = _annotate_spaces(
                        body, bound, codebooks
                    )
            elif t is _COND:
                # torch.cond: annotate both branch regions (the shared operands
                # bound to each branch's placeholders) and thread the branch
                # result spaces to the cond's getitems, exactly like a
                # while_loop.  Inputs may mix DRAM + Scratchpad (a guarded DMA
                # reads a DRAM buffer and writes a Scratchpad bank), so the
                # generic all-Scratchpad check must not apply here.
                operands = list(node.args[3]) if len(node.args) > 3 else []
                results = None
                for graph_arg in (node.args[1], node.args[2]):
                    branch = _subgraph(gm, graph_arg.target)
                    if branch is None:
                        continue
                    phs = [
                        p for p in branch.graph.nodes if p.op == "placeholder"
                    ]
                    bound = {
                        p: sp(o)
                        for p, o in zip(phs, operands)
                        if sp(o) is not None
                    }
                    res = _annotate_spaces(branch, bound, codebooks)
                    results = results if results is not None else res
                if results is not None:
                    loop_results[node] = results
            elif t is _VOYAGER_ASYNC:
                # Guarded async DMA: a DRAM<->Scratchpad tile move whose result
                # is an on-chip token (not a data tile).  Its operands are a
                # (DRAM, Scratchpad) pair, so the generic check doesn't apply;
                # mark the token Scratchpad (on-chip, like zero_tile).
                space[node] = "Scratchpad"
            elif t is operator.getitem:
                src, idx = node.args[0], node.args[1]
                if (
                    isinstance(src, Node)
                    and src.target in (_WHILE_LOOP, _COND)
                    and src in loop_results
                    and isinstance(idx, int)
                    and idx < len(loop_results[src])
                ):
                    if loop_results[src][idx] is not None:
                        space[node] = loop_results[src][idx]
                elif sp(src) is not None:  # getitem on a multi-output op tuple
                    space[node] = sp(src)
            elif t is _VOYAGER_INCR or t is _VOYAGER_DELIN:
                pass  # loop indices, not a tensor
            elif t in _DRAM_TRANSFORM:
                # whole-tensor preprocessing in DRAM (the conv/pool halo pad)
                # before tiling; space-preserving — the padded buffer is then
                # load_tiled.
                if sp(node.args[0]) is not None:
                    space[node] = sp(node.args[0])
            elif _produces_tensor(node):
                scratchpad_inputs(node)
                space[node] = "Scratchpad"
            # else: index arithmetic (operator.add on counters, ...) — not a
            # tensor

        if node in space:
            node.meta["space"] = space[node]

    return out_spaces


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
            # Fused GEMM/conv + post-op tail.  The pipelined builders fold only
            # a bias, not an arbitrary tail; tail fusion is being redesigned, so
            # a fused submodule has no bufferization path yet.
            submod = getattr(model, str(node.target), None)
            ref = (
                get_anchor_node(submod.graph.nodes)
                if isinstance(submod, GraphModule)
                else None
            )
            if ref is not None and (is_conv2d(ref) or is_gemm_op(ref)):
                raise NotImplementedError(
                    f"fused submodule {node.name!r}: pipelined tail fusion is "
                    "not yet wired (the old gemm.py fused path was retired)"
                )
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
