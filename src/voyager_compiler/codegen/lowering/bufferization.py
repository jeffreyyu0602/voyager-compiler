"""
Bufferization pass: rewrite tiled FX nodes into an explicit bufferized FX graph.

For each *tiled* node (a GEMM or elementwise ``call_function`` carrying tiling
metadata, or a fused ``call_module`` whose reference op is a GEMM), the pass
builds a ``while_loop`` nest over ``voyager.*`` primitives (via the builders in
``gemm`` / ``pointwise``) and splices it into the graph in place of the
node.  Nodes with no tiling (operands/outputs fit on-chip) are left unchanged.

Runs after operator fusion and before memory allocation.  Self-contained — uses
helpers under ``codegen/`` (mapping.py, mapping_utils.py) but nothing under
``codegen/lowering/``.
"""

import math
import operator
from typing import Dict, List, Optional

import torch
import torch.fx as fx
from torch.fx import GraphModule, Node

from ...pt2e_utils import update_submod_user_meta
from ..mapping import get_anchor_node, replace_node_with_graph_module
from ..mapping_utils import (
    is_bmm,
    is_conv2d,
    is_elementwise_op,
    is_gemm_op,
    is_matmul,
    is_pooling,
)
from ..passes.utils import _pair, get_arg_value
from ..shape_prop import ShapeProp
from .common import _InputSpec, _OutputSpec, _compute_input_spec
from .ops import MemoryLevel
from .gemm import (
    _HWIO,
    _NHWC,
    _phys_pos,
    _project,
    _unproject,
    build_conv2d_buffers,
    build_gemm_buffers,
)
from .pointwise import build_pointwise_buffers

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


def _name_nest_fused_op(gm: GraphModule, name: str) -> bool:
    """Rename the nest's re-fused ``call_module`` (recursing into bodies) to
    ``name``.

    ``get_submodule_name`` can only name it generically (``conv2d_mx_fused``):
    inside the nest the reference op's weight is a ``load_tile`` of a nest
    input, not the weight ``get_attr`` it keys off — the param lives in the main
    graph, bound to the nest only after splicing.  So carry the source fused
    node's already-param-based name instead.
    """
    for n in gm.graph.nodes:
        if n.op == "call_module":
            n.name = name
            return True
        if n.op == "get_attr":
            sub = getattr(gm, str(n.target), None)
            if isinstance(sub, GraphModule) and _name_nest_fused_op(sub, name):
                return True
    return False


# Quantization codebook / qmap parameters: these tensor operands are passed to
# the tail *whole* (never tiled), since they are only ever quantization-op args.
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
    graph = model.graph

    # Snapshot which nodes to bufferize (and their built nests) before mutating
    # the graph: each entry is ``(node, sub_gm, n_outputs)``.
    specs = []
    for node in list(graph.nodes):
        # Fused submodule (GEMM/conv + post-op tail): the reference op inside
        # carries the tiling.  Handled as one bufferized nest (tail inlined).
        if node.op == "call_module":
            built = _build_for_fused_submodule(model, node, pipelined=pipelined)
            if built is not None:
                _name_nest_fused_op(built[0], node.name)
                specs.append((node, *built))
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
            sub_gm = _build_for_conv2d(node, pipelined=pipelined)
        elif is_gemm_op(node):
            sub_gm = _build_for_gemm(node, pipelined=pipelined)
        elif is_pooling(node):
            # ``pipelined`` picks the schedule (double-buffer vs sequential);
            # the pool builder reuses the pointwise engine either way.  None
            # for adaptive / 3-D / with-indices.
            sub_gm = _build_for_pool2d(node, pipelined=pipelined)
        elif is_elementwise_op(node) or node.target in _REDUCTION_POINTWISE_OPS:
            built = _build_for_pointwise(
                node, pipelined=pipelined
            )  # (sub_gm, n_out)
            if built is not None:
                specs.append((node, *built))
            continue
        else:
            raise NotImplementedError(
                f"Unsupported tiled op for bufferization: {node.target}"
            )
        if sub_gm is not None:
            specs.append((node, sub_gm, 1))

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


def _mx_operands(node: Node) -> dict:
    """Example tensors for a GEMM/conv node's tensor kwargs (MX scales / codes),
    in ``node.kwargs`` order — which is the order they appear in
    ``all_input_nodes``, so the builder's placeholders line up.  (``value`` is
    set by ShapeProp.)"""
    return {
        k: v.value.clone()
        for k, v in node.kwargs.items()
        if isinstance(v, Node)
    }


def _gemm_tile_sizes(node: Node, input_ts, output_ts) -> Optional[List[int]]:
    """Return [tile_b, tile_x, tile_c, tile_k] from the input + output tiles,
    or None."""
    if input_ts is None or output_ts is None:
        return None
    tile_c = input_ts[-1]
    tile_k = output_ts[-1]
    if is_bmm(node):
        tile_b = output_ts[0]
        tile_x = output_ts[-2]
    else:
        tile_b = 1
        tile_x = math.prod(output_ts[:-1])
    return (int(tile_b), int(tile_x), int(tile_c), int(tile_k))


def _build_for_gemm(
    node: Node, pipelined: bool = False
) -> Optional[GraphModule]:
    shapes = node.meta.get("tiled_shapes", {})
    tile_sizes = _gemm_tile_sizes(node, shapes[node.args[0]], shapes[node])
    input_t = node.args[0].value.clone()
    weight_t = node.args[1].value.clone()
    if input_t.ndim != 3:
        # Builder expects 3-D (B, X, C) operands; skip otherwise (e.g. 2-D
        # linear).
        return None
    bias_n = node.args[2] if len(node.args) > 2 else None
    bias_t = bias_n.value.clone() if isinstance(bias_n, Node) else None
    return build_gemm_buffers(
        node.target,
        tile_sizes,
        input_t,
        weight_t,
        bias_t,
        accumulate_fp32=False,
        pipelined=pipelined,
        batched_weight=is_bmm(node) and weight_t.ndim == 3,
        # The weight is C-major ``(..., C, K)`` when natural for a matmul
        # (A @ B) or for a linear relayouted by transpose_linear_weights.
        # Natural layout is C-major for matmul, K-major for linear;
        # ``meta["transposed"]`` (set on the node by that pass) flips it.  So
        # C-major == is_matmul XOR transposed — which disambiguates e.g. a
        # transposed vs untransposed ``matmul_mx``.
        weight_ck=(is_matmul(node) != bool(node.meta.get("transposed", False))),
        block_size=node.kwargs.get("block_size"),
        kwargs=_mx_operands(node) or None,
    )


def _conv2d_tile_sizes(
    input_ts, output_ts, transposed: bool = False
) -> Optional[List[int]]:
    """Return logical [tile_n, tile_k, tile_c, tile_oh, tile_ow] from the input
    + output tiles, or None.

    The tiles are physically laid out (NHWC, input and output alike) when the
    layout pass ran, so un-project them to logical NCHW before reading.  The
    output spatial tile (oH, oW) is included so the builder tiles spatial when
    the L2 tiling does — otherwise a fused residual (loaded at the spatial tile)
    would not match the conv's whole-spatial accumulator."""
    if (
        input_ts is None
        or output_ts is None
        or len(input_ts) != 4
        or len(output_ts) != 4
    ):
        return None
    dims = _NHWC if transposed else None
    input_ts = _unproject(input_ts, dims)  # logical N, C, iH, iW
    output_ts = _unproject(output_ts, dims)  # logical N, K, oH, oW
    return [
        int(output_ts[0]),  # tile_n
        int(output_ts[1]),  # tile_k
        int(input_ts[1]),  # tile_c
        int(output_ts[2]),  # tile_oh
        int(output_ts[3]),  # tile_ow
    ]


def _build_for_conv2d(
    node: Node, pipelined: bool = False
) -> Optional[GraphModule]:
    # grouped / depthwise conv tiling is not supported
    if (get_arg_value(node, 6, "groups") or 1) != 1:
        return None
    # One bit picks the layout: the conv layout pass tags meta["transposed"].)
    shapes = node.meta.get("tiled_shapes", {})
    transposed = node.meta.get("transposed", False)
    tile_sizes = _conv2d_tile_sizes(
        shapes[node.args[0]], shapes[node], transposed
    )
    input_t = node.args[0].value.clone()
    weight_t = node.args[1].value.clone()
    bias_t = node.args[2].value.clone() if len(node.args) > 2 else None
    if input_t.ndim != 4:
        return None
    return build_conv2d_buffers(
        node.target,
        tile_sizes,
        input_t,
        weight_t,
        bias_t,
        stride=get_arg_value(node, 3, "stride") or 1,
        padding=get_arg_value(node, 4, "padding") or 0,
        dilation=get_arg_value(node, 5, "dilation") or 1,
        groups=get_arg_value(node, 6, "groups") or 1,
        nhwc=transposed,
        accumulate_fp32=False,
        pipelined=pipelined,
        block_size=node.kwargs.get("block_size"),
        kwargs=_mx_operands(node) or None,
    )


# Batched-reduction ops routed through the pointwise builder (the leading dims
# tile, the reduction dim(s) stay whole).  Only these two are multi-output
# (scale + quantized).
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


def _build_for_pointwise(node: Node, pipelined: bool = False):
    """Bufferize a pointwise / batched-reduction op as a single while_loop over
    the leading-dim tile grid (the reduction dim(s) kept whole).  Multi-output
    ops (``quantize_mx`` -> scale + quantized) store each output to its own
    buffer.

    Returns ``(sub_gm, n_outputs)`` or ``None``.
    """
    val = getattr(node, "value", None)
    if not isinstance(val, (torch.Tensor, list, tuple)):
        return None

    tiled_shapes = node.meta.get("tiled_shapes")
    if not tiled_shapes:
        return None

    outputs = list(val) if isinstance(val, (list, tuple)) else [val]
    if len(outputs) > 1 and node.target not in _MULTI_OUTPUT_POINTWISE:
        raise NotImplementedError(
            f"{node.target}: multi-output pointwise tiling is only supported "
            "for quantize_mx / quantize_mx_outlier"
        )

    in_nodes = node.all_input_nodes
    inputs = [n.value.clone() for n in in_nodes]

    # Resolve each op arg to a loaded-tile index (tensor operand) or a plain
    # constant *here*, not in the closure: the closure runs in the traced
    # while_loop body, where dynamo rejects FX-Node lookups and immutable_list
    # constants — so it only indexes ``tiles``.  ``_plain`` flattens FX's
    # immutable_list args (e.g. normalized_shape).
    order = {n: i for i, n in enumerate(in_nodes)}
    _plain = lambda a: list(a) if isinstance(a, list) else a
    arg_slots = [order[a] if isinstance(a, Node) else None for a in node.args]
    kw_slots = {
        k: order[v] if isinstance(v, Node) else None
        for k, v in node.kwargs.items()
    }
    op_args = [_plain(a) for a in node.args]
    op_kwargs = {k: _plain(v) for k, v in node.kwargs.items()}
    op = node.target

    # plain pointwise: the block index is unused
    def kernel(grid_index, *tiles):
        args = [
            tiles[i] if i is not None else a for i, a in zip(arg_slots, op_args)
        ]
        kwargs = {
            k: tiles[i] if i is not None else op_kwargs[k]
            for k, i in kw_slots.items()
        }
        return op(*args, **kwargs)

    # Grid tile = the (last, full) output's tile (leading dims tiled, reduction
    # dim(s) whole).  A multi-output node keys a per-output tuple of tiles; the
    # grid is the last.
    output_ts = tiled_shapes.get(node)
    if isinstance(val, (list, tuple)):
        output_ts = output_ts[-1]

    output_shape = tuple(outputs[-1].shape)
    grid = tuple(s // t for s, t in zip(output_shape, output_ts))
    codebooks = _codebook_arg_nodes(node)
    input_specs = [
        (
            _compute_input_spec(output_shape, output_ts, tuple(n.shape))
            if n not in codebooks
            else None
        )
        for n in in_nodes
    ]
    output_specs = [
        _OutputSpec(
            tuple(o.shape),
            tuple(min(t, s) for t, s in zip(output_ts, o.shape)),
            tuple(range(o.ndim)),
            o.dtype,
        )
        for o in outputs
    ]

    sub_gm = build_pointwise_buffers(
        kernel,
        grid,
        input_specs,
        output_specs,
        tuple(inputs),
        pipelined=pipelined,
    )
    return sub_gm, len(outputs)


# 2-D max / avg pool ops the pool builder handles (adaptive pools are *global*
# — output 1x1, no kernel/stride — so they're left untiled; ``is_pooling`` is
# broader).
_POOL2D_SUPPORTED = {
    torch.ops.aten.max_pool2d.default,
    torch.ops.quantized_ops.max_pool2d.default,  # NHWC (after data_layout)
    torch.ops.aten.avg_pool2d.default,
}


def _build_for_pool2d(
    node: Node, pipelined: bool = True
) -> Optional[GraphModule]:
    """Build the bufferized nest for a 2-D max/avg pool by reusing the
    **pointwise** engine (``build_pointwise_buffers``): pooling is a pointwise
    map over the (N, C, oH, oW) output grid whose only twist is the input tile —
    a strided receptive-field *halo* (overlap = the kernel footprint) with the
    boundary padding folded into the load (``copy_tile``'s ``pad`` /
    ``pad_value``), so the kernel pools each halo with ``padding=0`` (no
    materialized padded input).  ``pipelined`` picks the engine's variant:
    ``True`` (default) -> two SRAM banks + ``delinearize_index``; ``False`` ->
    one bank + ``increment_indices``.  Returns ``None`` for the pools it doesn't
    cover (adaptive / 3-D / with-indices).

    A fused post-pool tail (which the retired ``TiledPool2d`` builder had, but
    nothing used) would fold into the kernel closure — ``kernel(in_tile,
    *tail_tiles) = tail_fn(pool(in_tile), *tail_tiles)`` — with the tail
    operands as extra ``build_pointwise_buffers`` inputs; add it if a fused pool
    ever appears.
    """
    if node.target not in _POOL2D_SUPPORTED:
        return None
    nhwc = bool(node.meta.get("transposed", False))
    in_dims = _NHWC if nhwc else None
    input_t = node.args[0].value.clone()
    shapes = node.meta.get("tiled_shapes", {})
    output_ts = shapes.get(node, tuple(node.value.shape))
    tn, tc, toh, tow = _unproject(
        output_ts, in_dims
    )  # logical (N, C, oH, oW) tile

    kernel_size = get_arg_value(node, 1, "kernel_size")
    stride = get_arg_value(node, 2, "stride", [])
    padding = get_arg_value(node, 3, "padding", 0)
    if "max_pool" in str(node.target):
        dilation = get_arg_value(node, 4, "dilation", 1)
        ceil_mode = get_arg_value(node, 5, "ceil_mode", False)
        extra_args = (dilation, ceil_mode)
        # so a padded boundary window's max ignores it
        pad_value = float("-inf")
    else:  # avg_pool2d
        ceil_mode = get_arg_value(node, 4, "ceil_mode", False)
        count_include_pad = get_arg_value(node, 5, "count_include_pad") is None
        dilation = 1
        extra_args = (
            ceil_mode,
            count_include_pad,
            get_arg_value(node, 6, "divisor_override"),
        )
        pad_value = 0.0

    N, C, H, W = _unproject(input_t.shape, in_dims)
    kH, kW = _pair(kernel_size)
    sh, sw = _pair(stride) if stride else (kH, kW)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    oH = (H + 2 * ph - dh * (kH - 1) - 1) // sh + 1
    oW = (W + 2 * pw - dw * (kW - 1) - 1) // sw + 1

    # Input halo for one output tile: the receptive field, stepped by tile_o *
    # stride (overlap), with the boundary padding folded into the load (pad /
    # pad_value) — the conv/pool halo trick without materializing a padded
    # input.
    ih = (toh - 1) * sh + dh * (kH - 1) + 1
    iw = (tow - 1) * sw + dw * (kW - 1) + 1
    step_h, step_w = toh * sh, tow * sw

    # One input blockspec: identity ``index_map`` (input & grid share the
    # physical layout, NCHW or NHWC), a strided halo (overlap); only sizes /
    # strides / pad / shapes are projected onto the physical order.
    out_tile = _project((tn, tc, toh, tow), in_dims)
    out_shape = _project((N, C, oH, oW), in_dims)
    grid = tuple(s // t for s, t in zip(out_shape, out_tile))
    in_spec = _InputSpec(
        tile_sizes=_project((tn, tc, ih, iw), in_dims),
        index_map=(0, 1, 2, 3),
        is_broadcast=(False,) * 4,
        strides=_project((tn, tc, step_h, step_w), in_dims),
        pad=_project((0, 0, ph, pw), in_dims),
        pad_value=pad_value,
    )
    output_specs = [
        _OutputSpec(
            out_shape, out_tile, tuple(range(len(out_shape))), input_t.dtype
        )
    ]

    # Kernel: pool the loaded halo with padding=0 (already folded into the
    # load); the closure bakes the op's trailing args (``extra_args``).  No
    # reduction, so the block index is unused.  A fused tail would wrap this
    # (see above).
    def kernel_fn(grid_index, tile):
        return node.target(tile, [kH, kW], [sh, sw], [0, 0], *extra_args)

    return build_pointwise_buffers(
        kernel_fn,
        grid,
        [in_spec],
        output_specs,
        (input_t,),
        pipelined=pipelined,
    )


def build_conv2d_pointwise(
    target: torch._ops.OpOverload,
    tile_sizes: List[int],  # logical [tile_n, tile_k, tile_c, tile_oh, tile_ow]
    input: torch.Tensor,  # (N, C, iH, iW) physical (NHWC if nhwc)
    weight: torch.Tensor,  # (K, C, kH, kW) physical (HWIO if nhwc)
    bias: Optional[torch.Tensor] = None,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
    nhwc: bool = False,
    pipelined: bool = False,
) -> GraphModule:
    """Tiled conv2d via the pointwise engine (like ``_build_for_pool2d``),
    using the generalized ``kernel(grid_index, *tiles)``.

    Conv is a pointwise map over the (N, K, oH, oW) output grid; the input
    feature map is a strided receptive-field *halo* (pad-on-load,
    ``pad_value=0``) and the weight is tiled on K.  The input-channel C is the
    matmul-``k`` reduction:
      * ``num_c == 1``: the kernel convolves the whole-C halo + weight tile into
        the output tile (bias fused) and it's stored directly — reuses
        ``build_pointwise_buffers``.
      * ``num_c > 1``: the kernel reads ``(n,k,oy,ox)`` from ``grid_index`` and
        runs the C-reduction itself — a ``zero_tile`` accumulator + inner C-loop
        of ``copy_tile`` block-loads + conv + accumulate; returns ``acc +
        bias``.  (next)

    Layout: NCHW/OIHW, or NHWC input/output + HWIO weight (``nhwc=True``);
    ``tile_sizes`` is logical, projected per operand.  (MX scales / fused tail
    not yet wired.)
    """
    assert (
        groups == 1
    ), "build_conv2d_pointwise supports only dense conv (groups=1)"
    in_dims = _NHWC if nhwc else None
    w_dims = _HWIO if nhwc else None
    out_dims = _NHWC if nhwc else None
    tn, tk, tc, toh, tow = tile_sizes
    N, C, H, W = _unproject(input.shape, in_dims)
    K, _, kH, kW = _unproject(weight.shape, w_dims)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    oH = (H + 2 * ph - dh * (kH - 1) - 1) // sh + 1
    oW = (W + 2 * pw - dw * (kW - 1) - 1) // sw + 1
    num_c = C // tc

    # Input halo (receptive field of one output tile), stepped by tile_o *
    # stride; the boundary padding folds into the load (pad_value=0), exactly as
    # in the pool builder.
    ih = (toh - 1) * sh + dh * (kH - 1) + 1
    iw = (tow - 1) * sw + dw * (kW - 1) + 1
    step_h, step_w = toh * sh, tow * sw

    out_tile = _project((tn, tk, toh, tow), out_dims)
    out_shape = _project((N, K, oH, oW), out_dims)
    # output-grid rank; a "whole" operand dim maps to ndim (>= loop_ndim ->
    # static)
    ndim = 4

    # index_map[d] = the physical output-grid dim that physical operand dim d
    # maps to (N->N, H->oH, W->oW, K->K), with the reduced C and the kernel
    # kH/kW mapped to ``ndim`` (whole).
    def _imap(dims, to_grid):
        return tuple(to_grid(dims[d] if dims else d) for d in range(4))

    in_imap = _imap(
        in_dims, lambda a: ndim if a == 1 else _phys_pos(a, out_dims)
    )
    w_imap = _imap(w_dims, lambda a: _phys_pos(1, out_dims) if a == 0 else ndim)

    if num_c == 1:
        in_spec = _InputSpec(
            tile_sizes=_project((tn, C, ih, iw), in_dims),
            index_map=in_imap,
            is_broadcast=(False,) * 4,
            strides=_project((tn, C, step_h, step_w), in_dims),
            pad=_project((0, 0, ph, pw), in_dims),
            pad_value=0.0,
        )
        w_spec = _InputSpec(
            tile_sizes=_project((tk, C, kH, kW), w_dims),
            index_map=w_imap,
            is_broadcast=(False,) * 4,
        )
        inputs, specs = [input, weight], [in_spec, w_spec]
        if bias is not None:
            inputs.append(bias)
            specs.append(
                _InputSpec(
                    tile_sizes=(tk,),
                    index_map=(_phys_pos(1, out_dims),),
                    is_broadcast=(False,),
                )
            )

        # Whole-C conv on the loaded halo + weight tile (padding folded into
        # the load); the output channel K is the grid's K dim, so no reduction —
        # the block index is unused.
        def kernel(grid_index, in_tile, w_tile, *bias_tile):
            return target(
                in_tile,
                w_tile,
                bias_tile[0] if bias_tile else None,
                [sh, sw],
                [0, 0],
                [dh, dw],
                groups,
            )

        grid = tuple(s // t for s, t in zip(out_shape, out_tile))
        out_specs = [
            _OutputSpec(
                out_shape, out_tile, tuple(range(len(out_shape))), input.dtype
            )
        ]
        return build_pointwise_buffers(
            kernel,
            grid,
            specs,
            out_specs,
            tuple(inputs),
            pipelined=pipelined,
        )

    # num_c > 1: the C-reduction.  TODO — the engine drives a flattened
    # (N,K,oH,oW,C) grid; the input/weight C-block banks + an SRAM accumulator
    # are allocated once in ``forward``; the engine loads each C-block and the
    # kernel runs ONE reduction iteration (partial conv + accumulate into the
    # accumulator, zeroed when the C index is 0).  Needs the engine's reduction
    # generalization first.
    raise NotImplementedError(
        "build_conv2d_pointwise: num_c > 1 (engine-driven flattened "
        "C-reduction) — next step"
    )


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


# ---------------------------------------------------------------------------
# Fused submodule (GEMM/conv + post-op tail) bufferization
# ---------------------------------------------------------------------------


def _out_of_place(target):
    """The out-of-place aten overload of an in-place one (``add_.Tensor`` ->
    ``add.Tensor``), or None if ``target`` is not a convertible in-place op."""
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


def _build_tail_gm(
    submod: GraphModule,
    output_node: Node,
    tail_ops: List[Node],
    tail_inputs: List[Node],
) -> GraphModule:
    """Tail as a GraphModule ``[acc, *tail_inputs] -> submodule output``.

    ``output_node``'s output is rewired to the ``acc`` placeholder;
    ``tail_inputs`` are the submodule placeholders the ``tail_ops`` consume.
    Export inlines this, so the tail ops become standalone nodes in the loop
    body.
    """
    g = fx.Graph()
    remap = {output_node: g.placeholder("acc")}
    for n in tail_inputs:
        remap[n] = g.placeholder(n.name)
    inputs = set(remap.values())  # the [acc, *tail_inputs] placeholders
    for n in tail_ops:
        new = g.node_copy(n, lambda x: remap[x])
        # An in-place tail op touching a tail input can mutate a while_loop
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
    return fx.GraphModule(torch.nn.Module(), g)


def _build_for_fused_submodule(
    model: GraphModule, node: Node, pipelined: bool = False
):
    """Build the bufferized nest for a fused ``call_module`` (GEMM/conv + tail).

    Returns ``(sub_gm, n_outputs)`` or ``None`` if unsupported.  The nest's
    placeholders come out in the call_module's ``all_input_nodes`` order, so the
    caller wires them up with ``node.all_input_nodes`` (no reordering).
    """
    submod = getattr(model, node.target, None)
    if not isinstance(submod, GraphModule):
        return None
    ref = get_anchor_node(submod.graph.nodes)
    if ref is None:
        return None
    is_conv = is_conv2d(ref)
    if not is_conv and not is_gemm_op(ref):
        return None

    # ShapeProp the submodule so its inner nodes (reference + tail) carry
    # shapes; the main-graph ShapeProp does not populate submodule internals.
    # Feed the call_module's inputs (in all_input_nodes = submodule-placeholder
    # order).
    ShapeProp(submod).propagate(
        *(n.value.clone() for n in node.all_input_nodes)
    )

    # ``adjust_tiling`` ran before bufferization and node-keyed the SRAM-fit
    # tiling onto this call_module: inputs by their outer node, output(s) by
    # ``node``.  An *untiled* node (operands/output fit on-chip) has no
    # ``tiled_shapes`` entry — fall back to the full tensor shape so the builder
    # makes every loop trip-1 (the nest just wraps the whole op).
    shapes = node.meta.get("tiled_shapes") or {}
    in_node = node.all_input_nodes[0]
    input_ts = shapes.get(in_node, tuple(in_node.shape))
    output_ts = shapes.get(node, tuple(node.shape))
    if isinstance(node.value, (list, tuple)):
        output_ts = output_ts[-1]
    output_shape = tuple(ref.shape)

    # Walk the ops after the reference (the fused tail).  Record each op and a
    # load spec for every new placeholder it consumes (codebook whole, residual
    # tiled); collected in submodule order = the call_module's all_input_nodes
    # order.
    reachable = {ref}
    tail_ops: List[Node] = []
    tail_inputs: List[Node] = []
    tail_operands = []
    tail_specs: List[Optional[_InputSpec]] = []
    for sn in submod.graph.nodes:
        if sn is ref or sn.op != "call_function":
            continue
        if not any(inp in reachable for inp in sn.all_input_nodes):
            continue
        reachable.add(sn)
        tail_ops.append(sn)
        codebooks = _codebook_arg_nodes(sn)
        for inp in sn.all_input_nodes:
            if inp.op != "placeholder" or inp in tail_inputs:
                continue
            tail_inputs.append(inp)
            tail_operands.append(inp.value.clone())

            # Codebooks / qmaps and scalars (0-D or single-element) are passed
            # whole; every other operand — notably a residual — is load_tiled at
            # the output block.  Its tile comes from ``tiled_shapes`` (keyed by
            # the outer source node), or the full shape when the fused node is
            # untiled (trip-1) — so the residual is always loaded, never used
            # raw.
            src = inp.meta.get("source_node", inp)
            if inp in codebooks or inp.value.numel() == 1:
                tail_specs.append(None)  # whole operand -> no spec
            else:
                rt = shapes.get(src, tuple(inp.value.shape))
                off = len(output_shape) - len(rt)
                tail_specs.append(
                    _InputSpec(
                        tuple(rt),
                        tuple(range(off, len(output_shape))),
                        (False,) * len(rt),
                    )
                )

    tail_gm = _build_tail_gm(submod, ref, tail_ops, tail_inputs)

    # Reference operands -> example tensors (ShapeProp set their ``value``).
    input_t = ref.args[0].value.clone()
    weight_t = ref.args[1].value.clone()
    bias_t = ref.args[2].value.clone() if len(ref.args) > 2 else None
    mx_kwargs = _mx_operands(ref)
    block_size = ref.kwargs.get("block_size")
    transposed = bool(ref.meta.get("transposed", False))

    multi = isinstance(node.value, (list, tuple))
    vals = list(node.value) if multi else [node.value]
    full_shapes = [tuple(v.shape) for v in vals]
    keyed = shapes.get(node)  # None when the node is untiled
    if keyed is None:
        tile_shapes = full_shapes  # tile == full tensor (trip-1)
    else:
        tile_shapes = list(keyed) if multi else [keyed]
    dtypes = [v.dtype for v in vals]
    output_specs = list(zip(full_shapes, tile_shapes, dtypes))

    common = dict(
        pipelined=pipelined,
        tail_fn=tail_gm,
        tail_operands=tail_operands or None,
        tail_input_specs=tail_specs or None,
        output_specs=output_specs,
        kwargs=mx_kwargs or None,
    )

    if is_conv:
        if (get_arg_value(ref, 6, "groups") or 1) != 1 or input_t.ndim != 4:
            return None
        tile_sizes = _conv2d_tile_sizes(input_ts, output_ts, transposed)
        sub_gm = build_conv2d_buffers(
            ref.target,
            tile_sizes,
            input_t,
            weight_t,
            bias_t,
            stride=get_arg_value(ref, 3, "stride") or 1,
            padding=get_arg_value(ref, 4, "padding") or 0,
            dilation=get_arg_value(ref, 5, "dilation") or 1,
            groups=1,
            nhwc=transposed,
            block_size=block_size,
            accumulate_fp32=False,
            **common,
        )
    else:
        if input_t.ndim != 3:
            return None
        tile_sizes = _gemm_tile_sizes(ref, input_ts, output_ts)
        sub_gm = build_gemm_buffers(
            ref.target,
            tile_sizes,
            input_t,
            weight_t,
            bias_t,
            accumulate_fp32=False,
            batched_weight=is_bmm(ref) and weight_t.ndim == 3,
            weight_ck=(is_matmul(ref) != transposed),
            block_size=block_size,
            **common,
        )

    # Tag each output buffer with the dtypes (quantized) output dtype so
    # ``propagate_logical_dtypes`` can flow it onto the stored tiles; the
    # empties are created in output order, matching ``node.meta['dtype']``.
    dtype_meta = node.meta.get("dtype")
    dtypes = (
        list(dtype_meta)
        if isinstance(dtype_meta, (list, tuple))
        else [dtype_meta]
    )
    empties = [
        n
        for n in sub_gm.graph.nodes
        if n.op == "call_function" and n.target is voyager.alloc.default
    ]
    for em, ld in zip(empties, dtypes):
        if isinstance(ld, str):
            em.meta["dtype"] = ld

    return sub_gm, len(output_specs)
