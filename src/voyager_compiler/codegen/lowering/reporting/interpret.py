"""The recursive scheduler: walk a bufferized FX graph and emit timing.

A lightweight interpreter (modeled on ``shape_prop.ShapeProp``'s node loop, but
specialized) keeps an ``env`` of concrete **scalar** values and evaluates only
the scalar / control sublanguage — index math, predicates, counters — which is
all that is needed to thread the loop-carried state, resolve semaphore slots,
and pick ``torch.cond`` branches.  Tensor *shapes* come from the static
``meta['val']`` already on the nodes, so no tensors are ever materialized.

For each node it overlays timing on the ``ResourceState`` (see ``scheduler``):
compute on the systolic array, ``async_copy`` on the DRAM interface, and waits
as zero-time control. DPS ``copy_tile`` (a compute result written into its
buffer) is bookkeeping, not a real op, so it is skipped entirely.
"""

import math
import operator
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.fx import GraphModule, Node

from ..codegen import _loop_extents, _norm_extent
from .classify import classify
from .cost import _val, op_info, tile_bytes
from .model import CostParams, ScheduleResult
from .scheduler import ResourceState

_ALLOC = torch.ops.voyager.alloc.default
_ZEROS = torch.ops.voyager.zeros.default
_SELECT = torch.ops.aten.select.int


@dataclass
class _Ctx:
    rs: ResourceState
    cost: CostParams
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
        and node.target not in (_ALLOC, _ZEROS, _SELECT)
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
    """Follow placeholder bindings and tile ``select`` / ``getitem`` indexing to
    the root buffer node (the top-level alloc / input that carries
    ``meta['space']``)."""
    while isinstance(node, Node):
        if node in bind:
            node = bind[node]
        elif node.op == "call_function" and node.target is _SELECT:
            node = node.args[0]
        elif (
            node.op == "call_function"
            and node.target is operator.getitem
            and isinstance(_val(node), torch.Tensor)
        ):
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
        f"(src space={ssp!r}, dst space={dsp!r})"
    )


def _sem_key(sem_arg, env, bind):
    """``(bank_id, slot)`` for an ``async_copy``/``async_wait`` semaphore arg.

    The arg is a ``select(bank, 0, slot)`` — possibly reached through a chain of
    placeholder bindings when the select sits outside a ``cond`` that the DMA
    lives in.  ``slot`` is resolved against the shared ``env`` (the slot index,
    e.g. ``step % num_banks``, was computed in that outer scope), so an
    ``async_copy`` into a slot and its matching ``async_wait`` hash equal.
    """
    node = _defining(sem_arg, bind)
    if (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target is _SELECT
    ):
        bank = _defining(node.args[0], bind)
        return (id(bank), _resolve(node.args[2], env))
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

        kind = classify(node)
        if kind == "while_loop":
            env[node] = _run_loop(node, gm, env, ctx, path)
        elif kind == "cond":
            env[node] = _run_cond(node, gm, env, ctx, path)
        elif kind == "compute":
            ctx.rs.compute(node, ctx.rs.get_op(node, op_info), path)
        elif kind == "async_copy":
            buf, sizes, is_load = _dma_dir(node, ctx.bind)
            n_bytes = tile_bytes(buf, sizes)
            key = _sem_key(node.args[4], env, ctx.bind)
            ctx.rs.async_copy(node, n_bytes, is_load, key, path)
        elif kind == "async_wait":
            ctx.rs.async_wait(node, _sem_key(node.args[0], env, ctx.bind), path)
        elif kind == "copy_tile":
            # Destination-passing bookkeeping (writes a compute result into its
            # buffer): zero-time, no resource, no traffic -> not a scheduled op.
            pass
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


def estimate_schedule(
    model: GraphModule,
    dram_bandwidth: float,
    dram_access_latency: float,
    frequency: float,
    unroll: tuple[int, int],
) -> ScheduleResult:
    """Walk a bufferized + memory-planned FX graph and return its schedule:
    per-node timing records, total latency, and DRAM read / write bytes.

    ``dram_bandwidth`` (GB/s), ``dram_access_latency`` (ns) and ``frequency``
    (GHz) are physical units; ``cost.py`` converts them to cycles.  Shapes are
    read from the nodes' existing ``meta['val']`` / ``.value`` (set during
    bufferization), so the model needs no re-execution.
    """
    cost = CostParams(
        dram_bandwidth=dram_bandwidth,
        dram_access_latency=dram_access_latency,
        frequency=frequency,
        unroll=tuple(unroll),
    )
    rs = ResourceState(cost)
    ctx = _Ctx(rs=rs, cost=cost, bind={})
    _walk(model, {}, ctx, ())

    return ScheduleResult(
        records=rs.records,
        ops=list(rs.ops.values()),
        total_latency=rs.now,
        dram_read_bytes=rs.read_bytes,
        dram_write_bytes=rs.write_bytes,
        cost=cost,
        loop_names=dict(rs.loop_names),
    )
