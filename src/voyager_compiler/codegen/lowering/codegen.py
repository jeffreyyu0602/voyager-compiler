"""
Loop-aware output generation for bufferized FX graphs.

Three consumers of the bufferized FX dialect (``while_loop`` + ``voyager.*``
nodes), grouped here because they share the same traversal concerns:

  * ``gen_code_bufferized``          FX graph -> protobuf ``Model`` (Loop ops)
  * ``gen_compute_graph_bufferized`` FX graph -> graphviz SVG (loop clusters)
  * ``print_bufferized_graph``       FX graph -> indented text

Mirrors ``codegen/mapping.gen_code`` / ``gen_compute_graph`` but understands the
bufferized dialect.  Opt-in add-on; the default codegen path is untouched.
"""

import operator
import os
from typing import Dict, List, Optional

import graphviz
import torch
from torch.fx import GraphModule

from ..mapping_utils import (
    is_nop,
    map_node,
    set_output_field,
    set_tensor_field,
)
from ..param_pb2 import Model, Operation, Tensor
from ..shape_prop import ShapeProp

WHILE_LOOP = torch.ops.higher_order.while_loop
COND = torch.ops.higher_order.cond
INCREMENT_INDICES = torch.ops.voyager.increment_indices.default


def _target_name(target, short: bool = False) -> str:
    if isinstance(target, torch._ops.OpOverload):
        name = target._schema.name
        return name.split("::")[-1] if short else name
    return getattr(target, "__name__", str(target))


# ===========================================================================
# 1. Protobuf code generation
# ===========================================================================


def _is_lifted_index(node: torch.fx.Node) -> bool:
    return node.op == "get_attr" and str(node.target).startswith(
        "lifted_tensor"
    )


def _loop_input_value(n):
    """Resolve a while_loop carried/additional input to a runtime value.

    Loop inputs may be FX Nodes (use their propagated value) or captured scalar
    constants (e.g. kernel/spatial sizes), which are passed through literally.
    """
    return n.value if isinstance(n, torch.fx.Node) else n


def _norm_extent(e):
    """Normalise a ``loop_extents`` entry to ``(start, end, step)``.

    An entry is either a bare ``end`` count (start 0, step 1) or an explicit
    ``(start, end[, step])`` tuple.
    """
    if isinstance(e, (tuple, list)):
        start, end = int(e[0]), int(e[1])
        step = int(e[2]) if len(e) > 2 else 1
        return start, end, step
    return 0, int(e), 1


def _loop_extents(node) -> List[tuple]:
    """Per-dimension ``(start, end, step)`` for a flattened-grid while_loop.

    Builders tag each ``while_loop`` with ``node.meta['loop_extents']`` — the
    extents of the (possibly multi-dimensional) tile grid it iterates.  A loop
    with N extents is emitted as N nested ``Loop`` protos.
    """
    ext = node.meta.get("loop_extents")
    if ext:
        return [_norm_extent(e) for e in ext]
    # Legacy / untagged: a single loop with an unknown bound.
    return [(0, 0, 1)]


def _trip_str(node) -> str:
    """Per-dimension bounds annotation for the text / graphviz views."""
    ext = node.meta.get("loop_extents")
    if not ext:
        return ""
    dims = []
    for start, end, step in (_norm_extent(e) for e in ext):
        dims.append(
            str(end) if (start, step) == (0, 1) else f"{start}:{end}:{step}"
        )
    return f" trip=[{', '.join(dims)}]"


def _emit_body(
    gm: GraphModule, body_args, ops: List[Operation], output_dir
) -> None:
    """ShapeProp a (sub)graph with given input values and append Operations."""
    # Pass one value per placeholder, in order (tensors and scalar constants).
    ShapeProp(gm).propagate(*body_args)
    named = dict(gm.named_modules(remove_duplicate=False))
    for node in gm.graph.nodes:
        _emit_node(node, gm, named, ops, output_dir)


def _emit_fused_op(node, named, output_dir) -> Operation:
    """Emit a body ``call_module`` (a fused GEMM/conv + tail group) as a
    protobuf ``fused_op`` — an ``OpOverloadList`` of the submodule's compute
    ops, mirroring the default ``gen_code`` call_module path.  The body's
    ShapeProp gave the call_module node + its ``load_tile`` args their
    ``value``, so we ShapeProp the submodule with those to populate its inner
    nodes for ``map_node``."""
    sub = named[str(node.target)]
    ShapeProp(sub).propagate(
        *(
            a.value.clone() if isinstance(a, torch.fx.Node) else a
            for a in node.args
        )
    )
    op = Operation()
    op.fused_op.name = node.name
    for n in sub.graph.nodes:
        if (
            n.op == "call_function"
            and not n.meta.get("fused", False)
            and not is_nop(n)
        ):
            op.fused_op.op_list.append(map_node(n, output_dir))
    set_output_field(op, node, output_dir)
    return op


_LOAD_TILE = torch.ops.voyager.load_tile.default
_STORE_TILE = torch.ops.voyager.store_tile.default
_COPY_TILE = torch.ops.voyager.copy_tile.default


def _feeds_tile_index(node, _seen=None) -> bool:
    """True if ``node`` (transitively) computes a tile DMA block index — it
    reaches the ``indices`` argument of a ``load_tile`` / ``store_tile`` /
    ``copy_tile`` through a chain of scalar index arithmetic.  Such addressing
    (a pipelined prefetch's ``j+1``, or a ``delinearize_index(i)`` of the linear
    counter) is real computation, not loop control, so the whole cone must be
    emitted rather than dropped."""
    if _seen is None:
        _seen = set()
    for u in node.users:
        if u in _seen or u.op != "call_function":
            continue
        _seen.add(u)
        if u.target in (_LOAD_TILE, _STORE_TILE, _COPY_TILE):
            ix = (
                1 if u.target is _LOAD_TILE else 2
            )  # position of the ``indices`` arg
            if len(u.args) > ix and node in (u.args[ix] or ()):
                return True
        elif not isinstance(
            getattr(u, "value", None), (torch.Tensor, list, tuple)
        ):
            # scalar index arithmetic (add / floordiv / mod / getitem) —
            # recurse.
            if _feeds_tile_index(u, _seen):
                return True
    return False


def _emit_node(node, gm, named, ops: List[Operation], output_dir) -> None:
    if node.op == "call_module":
        ops.append(_emit_fused_op(node, named, output_dir))
        return
    if node.op != "call_function":
        return

    if node.target is WHILE_LOOP:
        ops.append(_emit_loop(node, gm, named, output_dir))
        return

    # getitem usually just unpacks loop results (no compute) — dropped, UNLESS
    # it extracts a tile-index component (a ``delinearize_index`` output) that a
    # tile DMA addresses by, which is genuine address-gen and must be emitted.
    if node.target is operator.getitem:
        if not _feeds_tile_index(node):
            return
    elif is_nop(node):
        return
    # increment_indices is loop-counter bookkeeping; the nested Loop's
    # start/end/step already encodes the iteration, so it is not a compute op.
    elif node.target is INCREMENT_INDICES:
        return
    # A side-effecting tile DMA (``copy_tile`` / ``store_tile``; returns
    # ``None`` since the buffer is a closed-over additional input mutated in
    # place) is a real instruction, always emitted despite its non-tensor value
    # — its output is the tile it writes (set from the dest in
    # ``set_output_field``).
    elif node.target not in (_COPY_TILE, _STORE_TILE):
        # A non-tensor call_function is usually integer loop-index carry
        # arithmetic (k+1, k % num_k, ...) — loop control made explicit by the
        # Loop structure, so not emitted.  The exception is a value that
        # *indexes* a tile DMA (a prefetch's ``j+1``, or a ``delinearize_index``
        # vector): genuine addressing — emit it so the DMA can reference it by
        # name (``map_node`` serialises it like any op).
        if not isinstance(
            getattr(node, "value", None), (torch.Tensor, list, tuple)
        ):
            if not _feeds_tile_index(node):
                return

    op = Operation()
    op.op.CopyFrom(map_node(node, output_dir))
    set_output_field(op, node, output_dir)
    ops.append(op)


def _emit_loop(node, parent_gm, parent_named, output_dir) -> Operation:
    body_gm = parent_named[str(node.args[1].target)]
    carried = list(node.args[2])
    extra = list(node.args[3]) if len(node.args) > 3 else []
    body_args = [_loop_input_value(n) for n in carried + extra]
    placeholders = [n for n in body_gm.graph.nodes if n.op == "placeholder"]

    body_ops: List[Operation] = []
    _emit_body(body_gm, body_args, body_ops, output_dir)

    # One Loop per grid dimension; the first len(extents) carried placeholders
    # are the per-dim loop indices (used by the body's static tile addressing).
    extents = _loop_extents(node)
    loops = []
    for i, (start, end, step) in enumerate(extents):
        lop = Operation()
        lop.loop.node = (
            placeholders[i].name
            if i < len(placeholders)
            else f"{node.name}_d{i}"
        )
        lop.loop.start = start
        lop.loop.end = end
        lop.loop.step = step
        loops.append(lop)

    # Nest innermost-first: protobuf ``repeated.append`` copies by value, so
    # each inner loop must be fully populated before it is appended into its
    # outer.
    loops[-1].loop.body.extend(body_ops)
    for outer, inner in zip(reversed(loops[:-1]), reversed(loops[1:])):
        outer.loop.body.append(inner)
    return loops[0]


def gen_code_bufferized(model: GraphModule, args, output_dir=None) -> Model:
    """
    Generate a protobuf ``Model`` from a bufferized FX graph, emitting ``Loop``
    operations for ``while_loop`` nodes (recursively) and ``voyager.*`` / aten
    ops as ordinary operations.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    ShapeProp(model).propagate(*args)
    named = dict(model.named_modules(remove_duplicate=False))
    model_params = Model()

    for node in model.graph.nodes:
        if node.op == "placeholder":
            tensor = Tensor()
            set_tensor_field(tensor, node, output_dir)
            model_params.inputs.append(tensor)
            continue
        if node.op == "get_attr":
            if _is_lifted_index(node):
                continue
            mod = getattr(model, str(node.target), None)
            if isinstance(mod, GraphModule):
                continue  # cond/body subgraphs are emitted inside their loop
            tensor = Tensor()
            set_tensor_field(tensor, node, output_dir)
            if "memory" in node.meta:
                model_params.parameters.append(tensor)
            continue

        _emit_node(node, model, named, model_params.ops, output_dir)

    return model_params


# ===========================================================================
# 2. Graphviz rendering (loops as clusters)
# ===========================================================================


def _label(node: torch.fx.Node) -> str:
    name = node.name
    val = getattr(node, "value", None)
    if isinstance(val, torch.Tensor):
        return f"{name}\\n{tuple(val.shape)}"
    if isinstance(val, (tuple, list)):
        shapes = ", ".join(
            str(tuple(t.shape)) for t in val if isinstance(t, torch.Tensor)
        )
        return f"{name}\\n{shapes}"
    return name


def _render_graph(gm, g, env: Dict, counter: list, scope: str = "") -> None:
    named = dict(gm.named_modules(remove_duplicate=False))

    def gid(node):
        return env.get(node, f"{scope}{node.name}")

    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        if node.op == "placeholder":
            if node not in env:
                nid = f"{scope}{node.name}"
                env[node] = nid
                g.node(nid, _label(node), shape="oval")
            continue
        if node.op == "get_attr":
            continue
        if node.op == "call_function" and node.target is WHILE_LOOP:
            _render_loop(node, gm, named, g, env, counter, scope)
            continue

        # Ordinary op node — namespace id by scope so names are unique across
        # loop bodies (FX node names repeat between subgraphs).
        nid = f"{scope}{node.name}"
        label = f"{{{_target_name(node.target, short=True)}\\n{_label(node)}}}"
        g.node(nid, label, shape="record")
        env[node] = nid
        for inp in node.all_input_nodes:
            g.edge(gid(inp), nid)


def _render_loop(
    node, parent_gm, parent_named, g, env, counter, scope=""
) -> None:
    body_gm = parent_named[str(node.args[1].target)]
    carried = list(node.args[2])
    extra = list(node.args[3]) if len(node.args) > 3 else []
    body_inputs = carried + extra

    body_args = [_loop_input_value(n) for n in body_inputs]
    try:
        ShapeProp(body_gm).propagate(*body_args)
    except Exception:
        pass

    placeholders = [n for n in body_gm.graph.nodes if n.op == "placeholder"]
    for ph, src in zip(placeholders, body_inputs):
        if isinstance(src, torch.fx.Node):
            env[ph] = env.get(src, src.name)

    counter[0] += 1
    cluster_name = f"cluster_{node.name}_{counter[0]}"
    body_scope = f"{cluster_name}/"
    title = f"loop {node.name}{_trip_str(node)}"
    with g.subgraph(name=cluster_name) as sub:
        sub.attr(label=title, style="rounded", color="blue")
        _render_graph(body_gm, sub, env, counter, scope=body_scope)

    out_node = next((n for n in body_gm.graph.nodes if n.op == "output"), None)
    if out_node is not None and isinstance(out_node.args[0], (list, tuple)):
        last = out_node.args[0][-1]
        if isinstance(last, torch.fx.Node):
            env[node] = env.get(last, last.name)


def gen_compute_graph_bufferized(
    model: GraphModule,
    output_file: str = "bufferized_graph",
    args: Optional[tuple] = None,
) -> None:
    """
    Render a bufferized FX graph to ``<output_file>.svg``; each ``while_loop``
    is a labelled cluster box containing its (possibly nested) body.
    """
    if args is not None:
        ShapeProp(model).propagate(*args)

    g = graphviz.Digraph()
    g.attr(compound="true")
    env: Dict[torch.fx.Node, str] = {}
    _render_graph(model, g, env, counter=[0])
    g.render(output_file, format="svg", cleanup=True)


# ===========================================================================
# 3. Indented text printer
# ===========================================================================

# Short names for the builtin torch dtypes used in the ``<shape x dtype>``
# annotation (custom quantized dtypes carry their own string, e.g. ``nf4_6``,
# used verbatim).
_DTYPE_SHORT = {
    torch.float32: "f32",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "bool",
}


def _type_str(node) -> str:
    """MLIR-like ``<DxDx...xdtype>`` (plus ``, space`` when known) for a node
    that produces a single tensor; ``""`` otherwise (multi-output / scalar /
    unknown).

    The dtype is the custom (quantized) ``meta['dtype']`` string (e.g.
    ``nf4_6``) used verbatim when set, else a short name for the builtin torch
    dtype (``f32`` / ``bf16`` / ...).  ``space`` is ``meta['space']`` (``DRAM``
    / ``Scratchpad``) from the bufferize pass.  Shapes come from the node's
    ShapeProp ``value`` or, inside loop bodies, the exported ``meta['val']``.
    """
    val = getattr(node, "value", None)
    if not isinstance(val, torch.Tensor):
        val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return ""
    custom = node.meta.get("dtype")
    dtype = (
        custom
        if isinstance(custom, str)
        else _DTYPE_SHORT.get(val.dtype, str(val.dtype).replace("torch.", ""))
    )
    dims = "x".join(str(int(d)) for d in val.shape)
    ty = f"{dims}x{dtype}" if dims else dtype  # 0-D scalar -> just the dtype
    space = node.meta.get("space")
    return f"<{ty}, {space}>" if space else f"<{ty}>"


def _fmt_arg(a):
    if isinstance(a, torch.fx.Node):
        # Annotate every tensor argument with its ``<shape×dtype, space>`` so a
        # copy_tile's direction is legible inline (e.g. ``copy_tile(src<…,DRAM>,
        # dst<…,Scratchpad>)`` = a load); scalar / index args (``_type_str``
        # empty) print as the bare name.
        return f"{a.name}{_type_str(a)}"
    if isinstance(a, (list, tuple)):
        return "[" + ", ".join(_fmt_arg(x) for x in a) + "]"
    return repr(a)


def _print_loop(
    node, gm, named, lines: List[str], indent: int, pad: str
) -> None:
    """
    Print a while_loop in scf.for-like form, binding each loop-body argument to
    the value it receives so the dataflow across the loop boundary is explicit:

        while_loop = loop trip=N carried(arg0_1 = 0, arg1_1 = empty) \
                     extra(arg2_1 = x) {
          <body>
        }
    """
    body_t = str(node.args[1].target)
    body_mod = named.get(body_t) or getattr(gm, body_t, None)
    carried = list(node.args[2]) if len(node.args) > 2 else []
    extra = list(node.args[3]) if len(node.args) > 3 else []
    phs = (
        [n for n in body_mod.graph.nodes if n.op == "placeholder"]
        if isinstance(body_mod, GraphModule)
        else []
    )

    def _bindings(ph_list, src_list):
        # Annotate only the bound placeholder (LHS); the source (RHS) is a node
        # already defined and annotated above, so printing it bare (just its
        # name) avoids the noise.
        return ", ".join(
            f"{ph.name}{_type_str(ph)} = {src}"
            for ph, src in zip(ph_list, src_list)
        )

    header = (
        f"{pad}{node.name} = loop{_trip_str(node)} "
        f"carried({_bindings(phs[:len(carried)], carried)})"
    )
    if extra:
        header += f" extra({_bindings(phs[len(carried):], extra)})"
    lines.append(header + " {")
    if isinstance(body_mod, GraphModule):
        _print_graph(body_mod, lines, indent + 1, skip_placeholders=True)
    lines.append(f"{pad}}}")


def _print_cond(
    node, gm, named, lines: List[str], indent: int, pad: str
) -> None:
    """Print a ``torch.cond`` in MLIR ``scf.if`` form — both branch regions
    descended into, operands bound to each branch's placeholders, the branch
    output rendered as ``yield``:

        cond = if pred (b_arg0 = src0, ...) {
          <true region>
          yield <true results>
        } else (b_arg0 = src0, ...) {
          <false region>
          yield <false results>
        }
    """
    pred = _fmt_arg(node.args[0])
    operands = list(node.args[3]) if len(node.args) > 3 else []

    def _branch(graph_arg):
        mod = named.get(str(graph_arg.target)) or getattr(
            gm, str(graph_arg.target), None
        )
        phs = (
            [n for n in mod.graph.nodes if n.op == "placeholder"]
            if isinstance(mod, GraphModule)
            else []
        )
        binds = ", ".join(
            f"{ph.name}{_type_str(ph)} = {src}"
            for ph, src in zip(phs, operands)
        )
        return mod, binds

    true_mod, true_binds = _branch(node.args[1])
    false_mod, false_binds = _branch(node.args[2])

    lines.append(
        f"{pad}{node.name}{_type_str(node)} = if {pred} ({true_binds}) {{"
    )
    if isinstance(true_mod, GraphModule):
        _print_graph(
            true_mod,
            lines,
            indent + 1,
            skip_placeholders=True,
            terminator="yield",
        )
    lines.append(f"{pad}}} else ({false_binds}) {{")
    if isinstance(false_mod, GraphModule):
        _print_graph(
            false_mod,
            lines,
            indent + 1,
            skip_placeholders=True,
            terminator="yield",
        )
    lines.append(f"{pad}}}")


def _print_graph(
    gm: GraphModule,
    lines: List[str],
    indent: int,
    skip_placeholders: bool = False,
    terminator: str = "return",
) -> None:
    pad = "  " * indent
    named = dict(gm.named_modules())

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # while_loop body placeholders are bound in the loop header
            # (scf.for-style iter_args), so they are skipped here.
            if not skip_placeholders:
                lines.append(f"{pad}arg {node.name}{_type_str(node)}")
            continue
        if node.op == "output":
            lines.append(f"{pad}{terminator} {_fmt_arg(node.args[0])}")
            continue
        if node.op == "get_attr":
            sub = getattr(gm, str(node.target), None)
            if isinstance(sub, GraphModule):
                continue  # printed at the while_loop use-site
            lines.append(
                f"{pad}{node.name}{_type_str(node)} = get_attr {node.target}"
            )
            continue

        if node.op == "call_function" and node.target is WHILE_LOOP:
            _print_loop(node, gm, named, lines, indent, pad)
            continue

        if node.op == "call_function" and node.target is COND:
            _print_cond(node, gm, named, lines, indent, pad)
            continue

        if node.op == "call_module":
            # A fused submodule: just a braced block labelled by the node name
            # (its placeholders reuse their source-node names, so the body ops
            # read directly against the outer nodes — no ``arg`` lines, and the
            # ``= fused <target>`` tag only repeats the node name).
            sub = named.get(str(node.target))
            lines.append(f"{pad}{node.name}{_type_str(node)} = fused {{")
            if isinstance(sub, GraphModule):
                _print_graph(sub, lines, indent + 1, skip_placeholders=True)
            lines.append(f"{pad}}}")
            continue

        parts = [_fmt_arg(a) for a in node.args]
        parts += [f"{k}={_fmt_arg(v)}" for k, v in node.kwargs.items()]
        lines.append(
            f"{pad}{node.name}{_type_str(node)} = "
            f"{_target_name(node.target)}({', '.join(parts)})"
        )


def print_bufferized_graph(model: GraphModule, to_string: bool = False):
    """
    Print (or return) an indented textual rendering of a bufferized FX graph,
    descending into ``while_loop`` bodies and fused ``call_module`` submodules.
    """
    lines: List[str] = ["graph {"]
    _print_graph(model, lines, 1)
    lines.append("}")
    text = "\n".join(lines)
    if to_string:
        return text
    print(text)

    for m in model.modules():
        if isinstance(m, torch.fx.GraphModule):
            m.graph.print_tabular()
    return text
