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

A ``while_loop`` is not walked iteration by iteration: once its steady
state repeats, the remaining periods are folded in one step (``_Fold``),
exactly, with the loop's ``Structure`` vouching for every skipped
iteration.
"""

import math
import operator
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import (
    QUANTIZE_FAMILY_OPS,
    get_arg_value,
    is_compute_op,
    is_nop,
)
from voyager_compiler.codegen.reporting.calibration import kernel_signatures
from voyager_compiler.codegen.reporting.cost import _shape, _val, tile_bytes
from voyager_compiler.codegen.reporting.model import (
    LoopSkip,
    LoopStats,
    ScheduleResult,
)
from voyager_compiler.codegen.reporting.periods import MAX_PERIOD, PeriodFinder
from voyager_compiler.codegen.reporting.busy import busy_unions
from voyager_compiler.codegen.reporting.scheduler import (
    T_LOOP,
    T_SKIP,
    Checkpoint,
    Recipe,
    ResourceState,
    advance_snapshot,
    snapshot_recipe,
)
from voyager_compiler.codegen.reporting.structure import (
    Structure,
    loop_structure,
)
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _produces_tensor,
    _viewed_buffer,
)
from voyager_compiler.codegen.transform.bufferize.utils import CSR_FILL_META
from voyager_compiler.codegen.transform.bufferize.emit import (
    COMMIT,
    COND,
    WHILE_LOOP,
    _loop_extents,
    _norm_extent,
)
from voyager_compiler.hardware_config import AcceleratorConfig
from voyager_compiler.shape_prop import run_op

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
    # placeholder -> the outer source node it is bound to (for semaphore-slot
    # rooting across loop / cond boundaries).
    bind: Dict[Node, Node]
    fold: bool = True  # fold each loop's steady state instead of walking it
    depth: int = 0  # control-flow nesting below the top-level graph


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
    """Re-run a control node on its resolved operands.  ``run_op`` supplies
    the stand-in for a scalar read off a fake buffer."""
    args = [_resolve(a, env) for a in node.args]
    kwargs = {k: _resolve(v, env) for k, v in node.kwargs.items()}
    return run_op(node.target, args, kwargs)


def _defining(node, bind: Dict[Node, Node]):
    """Follow placeholder bindings (loop / cond boundaries) to the node that
    actually defines a value in the enclosing scope."""
    while isinstance(node, Node) and node in bind:
        node = bind[node]
    return node


def _root(node, bind: Dict[Node, Node]):
    """Follow placeholder bindings and tile ``subview`` / ``getitem``
    indexing to the root buffer node (the top-level alloc / input that
    carries ``meta['space']``)."""
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


def _trace_source(node, bind: Dict[Node, Node]) -> Node:
    """Recurse *upward* from a buffer to the ``get_attr`` or placeholder it
    originates from, stopping at the first op that computes on its input.
    Everything in between only moves data (``repeat_kv``'s expand / stack /
    permute / reshape, the ``index_copy_`` cache write) or re-encodes it
    (``QUANTIZE_FAMILY_OPS``)."""
    seen = set()
    while isinstance(node, Node) and node not in seen:
        seen.add(node)
        if node in bind:
            node = bind[node]
            continue
        if node.op != "call_function" or not node.all_input_nodes:
            return node
        if is_compute_op(node) and node.target not in QUANTIZE_FAMILY_OPS:
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
    ``insert`` carries no space of its own.  A scalar read off a tile (the
    sparse nests' ``_local_scalar_dense`` pointer bookkeeping) has no tile to
    sweep: it is control math for the scalar evaluator, not datapath work.
    """
    if node.target is _ALLOC or _viewed_buffer(node) is not None:
        return False
    if not _produces_tensor(node):
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
    """``(slots_id, slot)`` for an ``async_copy``/``async_wait`` semaphore arg.

    The arg is a ``subview(slots, [slot], [1], [1])`` behind the NOP that
    squeezes the slot dim off — possibly reached through a chain of
    placeholder bindings when the pick sits outside a ``cond`` that the DMA
    lives in.  ``slot`` is resolved against the shared ``env`` (the slot
    index, e.g. ``step % num_slots``, was computed in that outer scope), so
    an ``async_copy`` into a slot and its matching ``async_wait`` hash equal.
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
        slots = _defining(node.args[0], bind)
        return (id(slots), _resolve(node.args[1][0], env))
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
    values of its ``output`` (for loop-carried threading).  A top-level node
    names the kernel (its ``meta['scope']``) every event under it is
    attributed to."""
    rs = ctx.rs
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op == "output":
            return _resolve(node.args[0], env)
        if ctx.depth == 0:
            scope = node.meta.get("scope")
            kernel = scope[0] if scope else node.name
            if kernel != rs.cur_kernel:
                rs.cur_kernel = kernel
                rs.launch(kernel)

        t = node.target
        if node.op == "call_module":
            rs.compute(node, path)
        elif t is WHILE_LOOP:
            env[node] = _run_loop(node, gm, env, ctx, path)
        elif t is COND:
            env[node] = _run_cond(node, gm, env, ctx, path)
        elif t is COMMIT:
            env[node] = _run_commit(node, gm, env, ctx, path)
        elif t is _ASYNC_COPY:
            buf, sizes, is_load = _dma_dir(node, ctx.bind)
            n_bytes = tile_bytes(buf, sizes)
            fill = node.meta.get(CSR_FILL_META)
            if fill is not None:
                n_bytes = math.ceil(n_bytes * fill)
            key = _sem_key(node.args[4], env, ctx.bind)
            post_count = _resolve(get_arg_value(node, 10, "post_count", 1), env)
            rs.async_copy(
                node,
                n_bytes,
                is_load,
                key,
                path,
                _buf_category(buf, ctx.bind),
                post_count=post_count,
            )
        elif t is _FILL:
            # A credit-seeded output store-sem: seed each slot's FIFO so
            # the first commit that waits the slot free draws the credit.
            value = int(get_arg_value(node, 2, "value", 0) or 0)
            slots = int(get_arg_value(node, 3, "num_slots", 1) or 1)
            for slot in range(slots):
                rs.seed_semaphore((id(node), slot), value)
        elif t is _ASYNC_WAIT:
            rs.async_wait(node, _sem_key(node.args[0], env, ctx.bind), path)
        elif t is _INSERT:
            # Destination-passing write: pure bookkeeping, zero-time -- the
            # producing compute already carries its destination.  (An async
            # producer posts its completion via voyager.commit's ``post``, not
            # the insert.)
            pass
        elif _produces_tensor(node) and is_compute_op(node):
            rs.compute(node, path)
        elif _is_dram_copy(node):
            reads, write = _copy_traffic(node, ctx.bind)
            rs.dram_copy(node, reads, write, path)
        elif _is_scratchpad_op(node, ctx.bind):
            rs.compute(node, path)
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


# --------------------------------------------------------------------------
# Loops: walk the fill, fold the steady state, walk the drain
# --------------------------------------------------------------------------


def _carried_delta(old, new) -> tuple:
    """How the loop-carried values changed over one iteration: a numeric
    difference per scalar, ``0`` for an object threaded through unchanged,
    and a fresh marker for anything else (which no other iteration can
    match)."""
    out = []
    for a, b in zip(old, new):
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            out.append(b - a)
        elif b is a:
            out.append(0)
        else:
            out.append(("obj", id(b)))
    return tuple(out)


@dataclass
class _Iteration:
    """One walked iteration as the folder remembers it."""

    key: tuple
    snapshot: tuple
    counters: tuple
    clock: int
    carried: list
    first_eid: int
    end_eid: int


@dataclass
class _Plan:
    """A confirmed period: what one more period does to the state."""

    period: int
    shift: int
    recipe: Recipe
    counters: tuple
    template: List[tuple]  # the period's iteration keys, in order
    snapshot: tuple  # the state at the end of the template period


@dataclass
class _Trial:
    """A jump under verification: the next ``period`` walked iterations
    must reproduce ``plan.template`` and land on the predicted state, or
    the jump is undone and retried one period shorter."""

    plan: _Plan
    repeats: int
    cp: Checkpoint
    step: int
    carried: list
    finder_len: int
    window_len: int
    keys_len: int
    matched: int = 0


class _Fold:
    """Fold one ``while_loop``'s steady state while it is walked.

    Every walked iteration is pushed as a key to a ``PeriodFinder``.  When
    the last ``2P`` iterations form two identical periods and the scheduler
    state after them is a pure time-shift of the state one period earlier,
    the whole periods ahead are skipped in one step -- but only as far as
    the loop's ``Structure`` shows every skipped iteration repeating the
    class of the one a period before it, so a store every K-th step or an
    edge row the two periods did not contain is walked, never folded.  The
    state advances by ``repeats`` periods under the recipe, the byte
    counters by ``repeats`` periods' worth, the carried scalars to their
    solved values, and a ``LoopSkip`` stands in for the records.  The next
    period is then walked and must match the template; if it does not, the
    jump is undone from a checkpoint and retried one period shorter.  At
    least two periods always remain to be walked after a jump.  A loop with
    no ``Structure`` is walked in full.

    ``keys`` is the iteration-key stream (walked keys plus skip markers) an
    enclosing loop hashes to recognise *its* period.
    """

    WINDOW = 2 * MAX_PERIOD + 1

    def __init__(
        self,
        rs: ResourceState,
        stats: LoopStats,
        steps: int,
        structure: Optional[Structure],
    ):
        self.rs = rs
        self.stats = stats
        self.steps = steps
        self.structure = structure
        self.finder = PeriodFinder(
            multiple_of=(
                max(1, structure.class_period()) if structure is not None else 1
            )
        )
        self.window: List[_Iteration] = []
        self.keys: list = []
        self.trial: Optional[_Trial] = None
        self.cap: Dict[int, int] = {}  # period -> repeats still allowed
        self.next_first = len(rs.records)

    def observe(self, key, step: int, carried: list):
        """Record the iteration just walked (``step`` is the next index);
        return the index and carried values to continue from."""
        rs = self.rs
        self.stats.walked += 1
        self.keys.append(key)
        if self.structure is None:
            return step, carried
        self.window.append(
            _Iteration(
                key,
                rs.snapshot(),
                rs.counters(),
                rs.clock(),
                carried,
                self.next_first,
                len(rs.records),
            )
        )
        self.next_first = len(rs.records)
        if self.trial is not None:
            return self._check(key, step, carried)
        if len(self.window) > self.WINDOW:
            del self.window[: len(self.window) - self.WINDOW]
        period = self.finder.push(key)
        if period is None or self.steps - step < 3 * period:
            return step, carried
        cap = self.cap.get(period)
        if cap is not None and cap <= 0:
            self.finder.reject(period)
            return step, carried
        plan = self._confirm(period)
        if plan is None:
            self.finder.reject(period)
            return step, carried
        repeats = (self.steps - step - 2 * period) // period
        if cap is not None:
            repeats = min(repeats, cap)
        return self._jump(plan, repeats, step, carried)

    def _confirm(self, period: int) -> Optional[_Plan]:
        w = self.window
        n = len(w)
        if n < 2 * period:
            return None
        a, b = n - 2 * period, n - period
        for m in range(period):
            if w[a + m].key != w[b + m].key:
                return None
        last_a, last_b = w[b - 1], w[n - 1]
        shift = last_b.clock - last_a.clock
        recipe = snapshot_recipe(last_a.snapshot, last_b.snapshot, shift)
        if recipe is None:
            return None
        return _Plan(
            period=period,
            shift=shift,
            recipe=recipe,
            counters=tuple(
                y - x for x, y in zip(last_a.counters, last_b.counters)
            ),
            template=[w[b + m].key for m in range(period)],
            snapshot=last_b.snapshot,
        )

    def _jump(self, plan: _Plan, repeats: int, step: int, carried: list):
        rs = self.rs
        period = plan.period
        # The skipped iterations and the period walked to verify the landing
        # must all repeat the class of the iteration a period before them.
        # A structure that breaks before the loop ends means the period is a
        # divisor of the true one (a store every K-th step, a row of tiles);
        # a longer period folds past the break, so this one is dropped after
        # the jump, or at once when the jump is not worth taking.
        wanted = repeats
        allowed = self.structure.periodic(step, period, (repeats + 1) * period)
        repeats = min(repeats, allowed // period - 1)
        if repeats < wanted:
            self.cap[period] = 0
            if repeats < 4:
                self.finder.reject(period)
                return step, carried
        advanced = advance_snapshot(plan.snapshot, plan.recipe, repeats)
        if advanced is None:
            self.cap[plan.period] = 0
            self.finder.reject(plan.period)
            return step, carried
        cp = rs.checkpoint()
        w = self.window
        n = len(w)
        rs.load_snapshot(advanced)
        rs.add_counters(plan.counters, repeats)
        rs.skips.append(
            LoopSkip(
                loop_uid=self.stats.loop_uid,
                kernel=rs.cur_kernel,
                after_eid=len(rs.records) - 1,
                first_step=step,
                period=plan.period,
                repeats=repeats,
                template=tuple(
                    range(w[n - plan.period].first_eid, w[n - 1].end_eid)
                ),
                shift=plan.shift,
                start=w[n - 1].clock,
                bytes=dict(
                    zip(
                        ("read", "write", "weight", "activation", "kv"),
                        plan.counters,
                    )
                ),
            )
        )
        self.stats.skipped += plan.period * repeats
        self.stats.period = plan.period
        self.stats.shift = plan.shift
        self.trial = _Trial(
            plan=plan,
            repeats=repeats,
            cp=cp,
            step=step,
            carried=carried,
            finder_len=len(self.finder),
            window_len=n,
            keys_len=len(self.keys),
        )
        self.keys.append((T_SKIP, plan.period, repeats, plan.shift))
        landing = step + plan.period * repeats
        return landing, self.structure.carried_at(landing)

    def _check(self, key, step: int, carried: list):
        t = self.trial
        plan = t.plan
        ok = key == plan.template[t.matched]
        t.matched += 1
        if ok and t.matched < plan.period:
            return step, carried
        if ok:
            want = advance_snapshot(plan.snapshot, plan.recipe, t.repeats + 1)
            ok = self.window[-1].snapshot == want
        if ok:
            self.trial = None
            return step, carried
        return self._undo()

    def _undo(self):
        t = self.trial
        self.trial = None
        self.rs.restore(t.cp)
        self.finder.truncate(t.finder_len)
        del self.window[t.window_len :]
        del self.keys[t.keys_len :]
        self.next_first = len(self.rs.records)
        repeats = t.repeats - 1
        self.cap[t.plan.period] = repeats
        if repeats <= 0:
            self.finder.reject(t.plan.period)
            return t.step, t.carried
        return self._jump(t.plan, repeats, t.step, t.carried)


def _run_loop(node: Node, gm: GraphModule, env, ctx: _Ctx, path):
    body = getattr(gm, str(node.args[1].target))
    carried = list(node.args[2])
    extra = list(node.args[3]) if len(node.args) > 3 else []
    phs = _placeholders(body)
    steps = _num_steps(node)
    rs = ctx.rs
    stats = rs.enter_loop(node, steps)
    prev_loop, rs.cur_loop = rs.cur_loop, id(node)
    outer_trace, outer_base = rs.trace, rs.trace_base
    entry = rs.clock()
    ctx.depth += 1

    def iteration(step, vals):
        _bind(phs, carried + extra, env, ctx, vals)
        out = _walk(body, env, ctx, tuple(path) + (step,))
        return out, (list(out) if isinstance(out, (list, tuple)) else [out])

    out = vals = [_resolve(c, env) for c in carried]
    if not ctx.fold:
        for step in range(steps):
            out, vals = iteration(step, vals)
        stats.walked += steps
    else:
        structure = (
            loop_structure(node, gm, env, vals, steps) if steps >= 4 else None
        )
        fold = _Fold(rs, stats, steps, structure)
        step = 0
        while step < steps:
            base = rs.clock()
            rs.trace, rs.trace_base = [], base
            out, new = iteration(step, vals)
            key = (
                tuple(rs.trace),
                rs.clock() - base,
                _carried_delta(vals, new),
            )
            step, vals = fold.observe(key, step + 1, new)
        rs.trace, rs.trace_base = outer_trace, outer_base
        if outer_trace is not None:
            outer_trace.append(
                (
                    T_LOOP,
                    id(node),
                    entry - outer_base,
                    tuple(fold.keys),
                    rs.clock() - outer_base,
                )
            )
    ctx.depth -= 1
    rs.exit_loop(stats)
    rs.cur_loop = prev_loop
    return out


def _run_cond(node: Node, gm: GraphModule, env, ctx: _Ctx, path):
    pred = _resolve(node.args[0], env)
    branch = node.args[1] if pred else node.args[2]
    branch_gm = getattr(gm, str(branch.target))
    operands = list(node.args[3]) if len(node.args) > 3 else []
    _bind(_placeholders(branch_gm), operands, env, ctx)
    ctx.depth += 1
    try:
        return _walk(branch_gm, env, ctx, path)
    finally:
        ctx.depth -= 1


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
        ctx.depth += 1
        try:
            result["out"] = _walk(sub, env, ctx, path)
        finally:
            ctx.depth -= 1
        if done_key is not None:
            ctx.rs.post_semaphore(done_key, ctx.rs.last_commit_node or node)

    ctx.rs.register_commit(node, deps, run)
    return result.get("out")


def estimate_schedule(
    model: GraphModule, config, *, full_walk: bool = False, calibration=None
) -> ScheduleResult:
    """Walk a bufferized + memory-planned FX graph and return its schedule:
    the walked timing records, the folded steady-state runs, total latency,
    and DRAM read / write bytes.

    Args:
        model: The graph; shapes come from its nodes' ``meta['val']`` /
            ``.value`` (set during bufferization), so nothing is re-run.
        config: The ``AcceleratorConfig``; ``cost.py`` converts its physical
            units to cycles.
        full_walk: Walk every loop iteration instead of folding the steady
            state -- the reference the fold must match exactly.
        calibration: A ``Calibration`` of RTL-measured kernel cycles, applied
            to each compute op as it is priced.
    """
    signatures = kernel_signatures(model, config)
    rs = ResourceState(config, calibration)
    rs.kernel_signatures = {k: s.key for k, s in signatures.items()}
    ctx = _Ctx(rs=rs, cost=config, bind={}, fold=not full_walk)
    _walk(model, {}, ctx, ())
    rs.assert_commits_drained("<graph end>")
    busy_compute, busy_dram, busy_any = busy_unions(rs.records, rs.skips)

    return ScheduleResult(
        records=rs.records,
        ops=list(rs.ops.values()),
        total_latency=rs.now,
        dram_read_bytes=rs.read_bytes,
        dram_write_bytes=rs.write_bytes,
        cost=config,
        dram_weight_bytes=rs.cat_bytes["weight"],
        dram_activation_bytes=rs.cat_bytes["activation"],
        dram_kv_bytes=rs.cat_bytes["kv"],
        skips=rs.skips,
        loops=list(rs.loop_stats.values()),
        busy_compute=busy_compute,
        busy_dram=busy_dram,
        busy_any=busy_any,
        kernel_signatures=signatures,
    )
