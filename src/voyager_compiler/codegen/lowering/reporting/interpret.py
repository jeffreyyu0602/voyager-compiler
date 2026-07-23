"""The recursive scheduler: walk a bufferized FX graph and emit timing.

A lightweight interpreter (modeled on ``shape_prop.ShapeProp``'s node loop, but
specialized) keeps an ``env`` of concrete **scalar** values and evaluates only
the scalar / control sublanguage — index math, predicates, counters — which is
all that is needed to thread the loop-carried state, resolve semaphore slots,
and pick ``torch.cond`` branches.  Tensor *shapes* come from the static
``meta['val']`` already on the nodes, so no tensors are ever materialized.

For each node it overlays timing on the ``ResourceState`` (see ``scheduler``):
compute on the systolic array, ``async_copy`` on the DRAM interface, and waits
as zero-time control. A DPS ``insert`` writes a compute result into its buffer
and is pure bookkeeping (skipped). An asynchronous compute is dispatched by
``voyager.commit``, which posts its done-semaphore when it finishes -- so
``async_wait`` can reconcile compute the same way it does a DMA.
"""

import math
import operator
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.fx import GraphModule, Node

from ...mapping_utils import is_compute_op, is_nop
from ...passes.utils import get_arg_value
from ..bufferization import _produces_tensor, _viewed_buffer
from ..codegen import COMMIT, COND, WHILE_LOOP, _loop_extents, _norm_extent
from .cost import _shape, _val, tile_bytes
from .model import ScheduleResult
from .scheduler import ResourceState
from ....hardware import AcceleratorConfig

_ALLOC = torch.ops.voyager.alloc.default
_ZEROS = torch.ops.voyager.zeros.default
_SUBVIEW = torch.ops.voyager.subview.default
_ASYNC_COPY = torch.ops.voyager.async_copy.default
_ASYNC_WAIT = torch.ops.voyager.async_wait.default
_INSERT = torch.ops.voyager.insert.default
_FILL = torch.ops.voyager.fill.default


@dataclass
class _Ctx:
    rs: ResourceState
    cost: AcceleratorConfig
    # placeholder -> the outer source node it is bound to (for semaphore-bank
    # rooting across loop / cond boundaries).
    bind: Dict[Node, Node]


def _resolve(a, env):
    """Resolve an FX arg to a concrete value: a scalar from ``env`` (falling
    back to the static trace value), a literal, or a resolved list / tuple."""
    if isinstance(a, Node):
        return env[a] if a in env else _val(a)
    if isinstance(a, (list, tuple)):
        return type(a)(_resolve(x, env) for x in a)
    return a


def _is_tensor(node) -> bool:
    return isinstance(_val(node), torch.Tensor)


def _should_eval(node: Node) -> bool:
    """A control node worth evaluating: it yields a scalar / index value (not a
    tensor and not a pure buffer op)."""
    return (
        node.op == "call_function"
        and node.target not in (_ALLOC, _ZEROS, _SUBVIEW)
        and not _is_tensor(node)
    )


def _eval(node: Node, env):
    args = [_resolve(a, env) for a in node.args]
    kwargs = {k: _resolve(v, env) for k, v in node.kwargs.items()}
    return node.target(*args, **kwargs)


def _defining(node, bind: Dict[Node, Node]):
    """Follow placeholder bindings (loop / cond boundaries) to the node that
    actually defines a value in the enclosing scope."""
    while isinstance(node, Node) and node in bind:
        node = bind[node]
    return node


def _root(node, bind: Dict[Node, Node]):
    """Follow placeholder bindings and tile ``subview`` / ``getitem`` indexing to
    the root buffer node (the top-level alloc / input that carries
    ``meta['space']``)."""
    while isinstance(node, Node):
        if node in bind:
            node = bind[node]
        elif node.op == "call_function" and node.target is _SUBVIEW:
            node = node.args[0]
        elif (
            node.op == "call_function"
            and node.target is operator.getitem
            and isinstance(_val(node), torch.Tensor)
        ):
            node = node.args[0]
        elif node.op == "call_function" and is_nop(node):
            node = node.args[0]
        else:
            break
    return node


def _dma_dir(node: Node, bind):
    """Identify an ``async_copy``'s DRAM buffer and direction: the operand
    rooted in DRAM is the buffer (``src`` DRAM => load / read; ``dst`` DRAM =>
    store / write).  Exactly one operand must be DRAM -- every copy is
    DRAM<->Scratchpad -- so an ambiguous pair means a malformed graph and
    raises.  Returns ``(root_buffer, sizes, is_load)`` -- the root carries the
    dtype for bytes."""
    src, dst = node.args[0], node.args[1]
    sizes = tuple(int(s) for s in node.args[3])
    src_root, dst_root = _root(src, bind), _root(dst, bind)

    def space(n):
        return n.meta.get("space") if isinstance(n, Node) else None

    ssp, dsp = space(src_root), space(dst_root)
    if ssp == "DRAM" and dsp != "DRAM":
        return src_root, sizes, True
    if dsp == "DRAM" and ssp != "DRAM":
        return dst_root, sizes, False
    raise ValueError(
        f"async_copy {node.name}: exactly one operand must be in DRAM "
        f"(src {src} -> root {src_root} in {ssp!r}; "
        f"dst {dst} -> root {dst_root} in {dsp!r})"
    )


def _is_cache_attr(node) -> bool:
    """A ``get_attr`` naming a KV-cache buffer (``key_cache_*`` /
    ``value_cache_*`` / a ``StaticCache`` ``*_keys`` / ``*_values``)."""
    if not isinstance(node, Node) or node.op != "get_attr":
        return False
    low = f"{node.name} {node.target}".lower()
    return "cache" in low and ("key" in low or "value" in low or "kv" in low)


# Compute ops a source trace still walks through: they re-encode a tensor
# rather than combine it with another.
_QUANTIZE_OPS = frozenset(
    {
        torch.ops.quantized_ops.quantize.default,
        torch.ops.quantized_ops.dequantize.default,
        torch.ops.quantized_ops.quantize_mx.default,
        torch.ops.quantized_ops.quantize_mx_outlier.default,
    }
)


def _trace_source(node, bind: Dict[Node, Node]) -> Node:
    """Recurse *upward* from a buffer to the ``get_attr`` or placeholder it
    originates from, stopping at the first op that computes on its input.
    Everything in between only moves data (``repeat_kv``'s expand / stack /
    permute / reshape, the ``index_copy_`` cache write) or re-encodes it
    (``_QUANTIZE_OPS``)."""
    seen = set()
    while isinstance(node, Node) and node not in seen:
        seen.add(node)
        if node in bind:
            node = bind[node]
            continue
        if node.op != "call_function" or not node.all_input_nodes:
            return node
        if is_compute_op(node) and node.target not in _QUANTIZE_OPS:
            return node
        node = node.all_input_nodes[0]
    return node


def _buf_category(buf, bind: Dict[Node, Node]) -> str:
    """Classify a DMA's DRAM root buffer by the tensor role it plays, so
    traffic can be split: a KV-cache buffer (``key_cache_*`` /
    ``value_cache_*``) is ``"kv"``; a real parameter (``get_attr``) is
    ``"weight"``; a graph input or an intermediate ``voyager.alloc`` is
    ``"activation"``.

    The buffer is often not the cache ``get_attr`` itself but an intermediate
    (e.g. the ``repeat_kv`` ``permute`` that produces the 32-head K), so trace
    *upward* to the originating ``get_attr`` before classifying."""
    if not isinstance(buf, Node):
        return "activation"
    src = _trace_source(buf, bind)
    if _is_cache_attr(src) or _is_cache_attr(buf):
        return "kv"
    if buf.op == "get_attr" or src.op == "get_attr":
        return "weight"
    return "activation"


def _is_dram_copy(node: Node) -> bool:
    """A tensor the accelerator does not compute but must still materialize in
    DRAM -- a ``pad`` / ``expand`` / ``cat`` / ``permute``.  These are exactly
    the nodes bufferization gives ``space='DRAM'`` *itself* (rather than
    inheriting it): an ``alloc`` names a buffer without filling it, and a view
    (``reshape`` / ``getitem`` / ``select``) borrows its source's space without
    moving a byte.
    """
    return (
        node.meta.get("space") == "DRAM"
        and node.target is not _ALLOC
        and _viewed_buffer(node) is None
    )


def _is_scratchpad_op(node: Node, bind: Dict[Node, Node]) -> bool:
    """On-chip work the datapath performs but ``is_compute_op`` does not name --
    a tile copy (``insert(x.clone(), dst)``) or a tile fill (the ``full_like`` /
    ``zeros_like`` that reset an accumulator).  Both sweep the tile through the
    vector unit, so both are costed like any vector op rather than as free
    control.  Only its *inputs* say where it lives: a value stored via
    ``insert`` carries no space of its own.
    """
    if node.target is _ALLOC or _viewed_buffer(node) is not None:
        return False
    ins = [i for i in node.all_input_nodes if _is_tensor(i)]
    return bool(ins) and all(
        _root(i, bind).meta.get("space") == "Scratchpad" for i in ins
    )


# An op that writes *into* an existing tensor rather than producing a new one.
# Arg 0 is its destination: written through, not read, and its output is that
# whole destination however little of it the op touches.  The value is the
# argument whose size is what the op really moves -- ``None`` when the op
# overwrites the whole destination, so its output already sizes it.  The
# KV-cache write ``index_copy_(cache, dim, pos, kv)`` is why this exists: sized
# by its output it would charge the whole cache to store a single token.
_INPLACE_SRC = {
    torch.ops.aten.index_copy_.default: 3,
    torch.ops.aten.copy_.default: None,
}


def _copy_traffic(node: Node, bind: Dict[Node, Node]):
    """``(reads, write)`` for a DRAM materialization -- every tensor input is
    read, the output is written.  ``cat`` / ``stack`` simply have several reads.

    A read is capped at the output size: a copy never reads more than it writes.
    ``pad`` / ``stack`` / ``permute`` do sweep their whole source, but one that
    keeps only part of it does not -- a ``slice`` reads just the slice, and an
    ``embedding`` just the rows it gathers.  Uncapped, a 512-token lookup would
    charge the entire embedding table (1 GB for 4 MB of rows).  The cap is a
    heuristic, not a law: it is exact for every op we lower today, but an op
    that genuinely re-reads its source would need its own rule.

    An in-place write (``_INPLACE_SRC``) does not read the destination it
    overwrites, and one that scatters is sized by the tensor it scatters rather
    than by its output.
    """
    src = _INPLACE_SRC.get(node.target)
    dest = node.args[0] if node.target in _INPLACE_SRC else None
    written = node.args[src] if src is not None else node
    out_bytes = tile_bytes(written, _shape(written))
    reads = []
    for inp in node.all_input_nodes:
        if not _is_tensor(inp) or inp is dest:
            continue
        root = _root(inp, bind)
        space = root.meta.get("space") if isinstance(root, Node) else None
        if space != "DRAM":
            raise ValueError(
                f"{node.name}: input {inp.name} roots in {space!r}, not DRAM "
                f"-- a Scratchpad source would be an async_copy"
            )
        n_bytes = min(tile_bytes(inp, _shape(inp)), out_bytes)
        reads.append((n_bytes, _buf_category(inp, bind)))
    write = (out_bytes, _buf_category(node, bind))
    return reads, write


def _sem_key(sem_arg, env, bind):
    """``(bank_id, slot)`` for an ``async_copy``/``async_wait`` semaphore arg.

    The arg is a ``subview(bank, [slot], [1], [1])`` behind the NOP that squeezes
    the bank dim off — possibly reached through a chain of placeholder bindings
    when the pick sits outside a ``cond`` that the DMA lives in.  ``slot`` is
    resolved against the shared ``env`` (the slot index, e.g.
    ``step % num_banks``, was computed in that outer scope), so an ``async_copy``
    into a slot and its matching ``async_wait`` hash equal.
    """
    node = _defining(sem_arg, bind)
    while (
        isinstance(node, Node) and node.op == "call_function" and is_nop(node)
    ):
        node = _defining(node.args[0], bind)
    if (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target is _SUBVIEW
    ):
        bank = _defining(node.args[0], bind)
        return (id(bank), _resolve(node.args[1][0], env))
    return (id(node), 0)


def _num_steps(node: Node) -> int:
    n = 1
    for start, end, step in (_norm_extent(e) for e in _loop_extents(node)):
        n *= max(0, math.ceil((end - start) / step))
    return n


def _placeholders(gm: GraphModule) -> List[Node]:
    return [n for n in gm.graph.nodes if n.op == "placeholder"]


def _walk(gm: GraphModule, env, ctx: _Ctx, path):
    """Schedule every node of ``gm`` in program order; return the resolved
    values of its ``output`` (for loop-carried threading)."""
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op == "output":
            return _resolve(node.args[0], env)

        t = node.target
        if node.op == "call_module":
            ctx.rs.compute(node, path)
        elif t is WHILE_LOOP:
            env[node] = _run_loop(node, gm, env, ctx, path)
        elif t is COND:
            env[node] = _run_cond(node, gm, env, ctx, path)
        elif t is COMMIT:
            env[node] = _run_commit(node, gm, env, ctx, path)
        elif t is _ASYNC_COPY:
            buf, sizes, is_load = _dma_dir(node, ctx.bind)
            n_bytes = tile_bytes(buf, sizes)
            key = _sem_key(node.args[4], env, ctx.bind)
            post_count = _resolve(get_arg_value(node, 10, "post_count", 1), env)
            ctx.rs.async_copy(
                node,
                n_bytes,
                is_load,
                key,
                path,
                _buf_category(buf, ctx.bind),
                post_count=post_count,
            )
        elif t is _FILL:
            # A credit-seeded output store-sem: seed each bank slot's FIFO so
            # the first commit that waits the slot free draws the credit.
            value = int(get_arg_value(node, 2, "value", 0) or 0)
            banks = int(get_arg_value(node, 3, "banks", 1) or 1)
            for slot in range(banks):
                ctx.rs.seed_semaphore((id(node), slot), value)
        elif t is _ASYNC_WAIT:
            ctx.rs.async_wait(node, _sem_key(node.args[0], env, ctx.bind), path)
        elif t is _INSERT:
            # Destination-passing write: pure bookkeeping, zero-time -- the
            # producing compute already carries its destination.  (An async
            # producer posts its completion via voyager.commit's ``post``, not
            # the insert.)
            pass
        elif _produces_tensor(node) and is_compute_op(node):
            ctx.rs.compute(node, path)
        elif _is_dram_copy(node):
            reads, write = _copy_traffic(node, ctx.bind)
            ctx.rs.dram_copy(node, reads, write, path)
        elif _is_scratchpad_op(node, ctx.bind):
            ctx.rs.compute(node, path)
        elif _should_eval(node):
            env[node] = _eval(node, env)
    return None


def _bind(phs, sources, env, ctx, carried_vals=None):
    """Bind a body / branch's placeholders into the shared ``env``: carried
    indices take threaded scalar values; the rest resolve from their source.
    Record each binding's source node so semaphore selects can be rooted across
    the loop / cond boundary."""
    for i, (ph, src) in enumerate(zip(phs, sources)):
        if carried_vals is not None and i < len(carried_vals):
            env[ph] = carried_vals[i]
        else:
            env[ph] = _resolve(src, env)
        if isinstance(src, Node):
            ctx.bind[ph] = src


def _run_loop(node: Node, gm: GraphModule, env, ctx: _Ctx, path):
    body = getattr(gm, str(node.args[1].target))
    carried = list(node.args[2])
    extra = list(node.args[3]) if len(node.args) > 3 else []
    phs = _placeholders(body)
    carried_vals = [_resolve(c, env) for c in carried]

    steps = _num_steps(node)
    out = carried_vals
    prev_loop = ctx.rs.cur_loop
    ctx.rs.cur_loop = id(node)
    ctx.rs.loop_names[id(node)] = node.name
    for step in range(steps):
        _bind(phs, carried + extra, env, ctx, carried_vals)
        out = _walk(body, env, ctx, tuple(path) + (step,))
        carried_vals = list(out) if isinstance(out, (list, tuple)) else [out]
    ctx.rs.cur_loop = prev_loop
    return out


def _run_cond(node: Node, gm: GraphModule, env, ctx: _Ctx, path):
    pred = _resolve(node.args[0], env)
    branch = node.args[1] if pred else node.args[2]
    branch_gm = getattr(gm, str(branch.target))
    operands = list(node.args[3]) if len(node.args) > 3 else []
    _bind(_placeholders(branch_gm), operands, env, ctx)
    return _walk(branch_gm, env, ctx, path)


def _run_commit(node: Node, gm: GraphModule, env, ctx: _Ctx, path):
    """A committed (async) region: it is parked until every semaphore in
    ``dependencies`` is posted, then its body walks off the program clock (its
    compute occupies the datapath without advancing the program clock) and
    ``post`` is signalled with the last body op's completion.  See
    ``ResourceState.register_commit``.  The dep / done-sem slots are resolved
    now (their indices were computed before the commit); the body walk is
    deferred into ``run``."""
    sub = getattr(gm, str(node.args[0].target))
    deps = [
        _sem_key(d, env, ctx.bind)
        for d in node.kwargs.get("dependencies") or ()
    ]
    post = node.kwargs.get("post")
    done_key = _sem_key(post, env, ctx.bind) if post is not None else None
    result = {}

    def run():
        _bind(_placeholders(sub), list(node.args[1:]), env, ctx)
        result["out"] = _walk(sub, env, ctx, path)
        if done_key is not None:
            ctx.rs.post_semaphore(done_key, ctx.rs.last_commit_node or node)

    ctx.rs.register_commit(deps, run)
    return result.get("out")


def estimate_schedule(model: GraphModule, config) -> ScheduleResult:
    """Walk a bufferized + memory-planned FX graph and return its schedule:
    per-node timing records, total latency, and DRAM read / write bytes.

    ``config`` is the ``AcceleratorConfig``; ``cost.py`` converts its physical
    units to cycles.  Shapes are read from the nodes' existing ``meta['val']`` /
    ``.value`` (set during bufferization), so the model needs no re-execution.
    """
    cost = config
    rs = ResourceState(cost)
    ctx = _Ctx(rs=rs, cost=cost, bind={})
    _walk(model, {}, ctx, ())
    rs.assert_commits_drained("<graph end>")

    return ScheduleResult(
        records=rs.records,
        ops=list(rs.ops.values()),
        total_latency=rs.now,
        dram_read_bytes=rs.read_bytes,
        dram_write_bytes=rs.write_bytes,
        cost=cost,
        dram_weight_bytes=rs.cat_bytes["weight"],
        dram_activation_bytes=rs.cat_bytes["activation"],
        dram_kv_bytes=rs.cat_bytes["kv"],
        loop_names=dict(rs.loop_names),
    )
