"""Exact per-iteration structure of a ``while_loop`` body.

Two matching periods say nothing about the iterations a fold then skips:
from inside a period, a store every K-th step or an edge tile at the end
of every row looks like any other step.  ``Structure`` settles that by
evaluating the body's scalar sub-language -- index arithmetic,
predicates, the counters the builders carry -- for every iteration at
once, as tensors over the trip count, and packing what decides an
iteration's shape (each ``cond``'s predicate, each semaphore slot, each
post count) into class words.  Two iterations with equal words run the
same nodes on the same semaphores, so the walk folds only over iterations
whose words repeat the template's, period for period.

The loop-carried scalars are solved in closed form from the update the
body computes for them: unchanged, ``x + c`` (affine in the step), or
``sym_ite(p, x + c, x)`` (``c`` times the running count of ``p``).  A body
whose scalars need anything else -- a data-dependent pointer, an op with
no tensor form, a nested loop -- has no ``Structure``, and the walk folds
nothing in that loop.
"""

import operator
from typing import Dict, List, Optional

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import get_arg_value, is_nop
from voyager_compiler.codegen.reporting.cost import _val
from voyager_compiler.codegen.transform.bufferize.emit import (
    COMMIT,
    COND,
    WHILE_LOOP,
)

_SUBVIEW = torch.ops.voyager.subview.default
_ASYNC_COPY = torch.ops.voyager.async_copy.default
_ASYNC_WAIT = torch.ops.voyager.async_wait.default
_FILL = torch.ops.voyager.fill.default
_DELINEARIZE = torch.ops.voyager.delinearize_index.default
_ALLOC = torch.ops.voyager.alloc.default
_ZEROS = torch.ops.voyager.zeros.default

_CHUNK = 1 << 20
_SLOT_BITS = 16  # an int-valued sink (a semaphore slot, a post count)
_WORD_BITS = 62
_MAX_CLASS_PERIOD = 1 << 16
_PREFIX = 32  # iterations at the loop's start the class period need not fit

_ELEMENTWISE = frozenset(
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.floordiv,
        operator.mod,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.and_,
        operator.or_,
        operator.neg,
        operator.getitem,
    ]
)


class Unsupported(Exception):
    """The body's control language has no tensor form here."""


def _delinearize(index, grid):
    coords = []
    for extent in reversed(list(grid)):
        coords.append(index % int(extent))
        index = index // int(extent)
    return list(reversed(coords))


def _apply(target, args, kwargs):
    """One scalar op, over tensors of every iteration's value."""
    if target in _ELEMENTWISE:
        return target(*args, **kwargs)
    if target is operator.not_:
        return ~args[0] if torch.is_tensor(args[0]) else not args[0]
    if target is torch.sym_ite:
        cond, a, b = args
        if torch.is_tensor(cond):
            return torch.where(cond, a, b)
        return a if cond else b
    if target is torch.sym_max:
        return torch.maximum(*(torch.as_tensor(a) for a in args))
    if target is torch.sym_min:
        return torch.minimum(*(torch.as_tensor(a) for a in args))
    if target is _DELINEARIZE:
        return _delinearize(args[0], args[1])
    raise Unsupported(target)


def _placeholders(gm: GraphModule) -> List[Node]:
    return [n for n in gm.graph.nodes if n.op == "placeholder"]


def _is_scalar_node(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target not in (_ALLOC, _ZEROS, _SUBVIEW)
        and not isinstance(_val(node), torch.Tensor)
    )


class Structure:
    """The class word of every iteration of one loop entry, and the
    loop-carried scalars at any iteration."""

    def __init__(self, loop: Node, gm: GraphModule, env, carried_vals, steps):
        self.steps = steps
        body = getattr(gm, str(loop.args[1].target))
        self.phs = _placeholders(body)
        carried = list(loop.args[2])
        extra = list(loop.args[3]) if len(loop.args) > 3 else []
        self.bind: Dict[Node, Node] = {}
        self.sinks: List[object] = []
        self.outputs = None
        self._scan(body)
        if self.outputs is None:
            raise Unsupported("loop body has no output")
        # Body placeholders: the carried slots, then the extra operands.
        self.constants: Dict[Node, object] = {}
        for i, ph in enumerate(self.phs):
            if i >= len(carried_vals):
                src = (carried + extra)[i]
                self.constants[ph] = (
                    (env[src] if src in env else _val(src))
                    if isinstance(src, Node)
                    else src
                )
        self.slots = self._solve(carried_vals)
        self.words: Optional[torch.Tensor] = None
        self.counts: Dict[int, torch.Tensor] = {}
        self._running: Dict[int, int] = {}
        self._evaluate_all()

    # -- scanning the body ------------------------------------------------

    def _slot_arg(self, sem_arg):
        """The node (or literal) giving a semaphore arg's slot index, as
        ``interpret._sem_key`` resolves it."""
        node = sem_arg
        while isinstance(node, Node) and node in self.bind:
            node = self.bind[node]
        while (
            isinstance(node, Node)
            and node.op == "call_function"
            and is_nop(node)
        ):
            node = node.args[0]
            while isinstance(node, Node) and node in self.bind:
                node = self.bind[node]
        if (
            isinstance(node, Node)
            and node.op == "call_function"
            and node.target is _SUBVIEW
        ):
            return node.args[1][0]
        return 0

    def _scan(self, gm: GraphModule) -> None:
        for node in gm.graph.nodes:
            if node.op == "output":
                if self.outputs is None:
                    args = node.args[0]
                    self.outputs = (
                        list(args)
                        if isinstance(args, (list, tuple))
                        else [args]
                    )
                continue
            if node.op != "call_function":
                continue
            t = node.target
            if t is WHILE_LOOP:
                raise Unsupported("nested while_loop")
            if t is COND:
                self.sinks.append(node.args[0])
                operands = list(node.args[3]) if len(node.args) > 3 else []
                for handle in node.args[1:3]:
                    branch = getattr(gm, str(handle.target))
                    for ph, src in zip(_placeholders(branch), operands):
                        self.bind[ph] = src
                    self._scan_nested(branch)
            elif t is COMMIT:
                sub = getattr(gm, str(node.args[0].target))
                for ph, src in zip(_placeholders(sub), node.args[1:]):
                    self.bind[ph] = src
                for d in node.kwargs.get("dependencies") or ():
                    self.sinks.append(self._slot_arg(d))
                post = node.kwargs.get("post")
                if post is not None:
                    self.sinks.append(self._slot_arg(post))
                self._scan_nested(sub)
            elif t is _ASYNC_COPY:
                self.sinks.append(self._slot_arg(node.args[4]))
                self.sinks.append(get_arg_value(node, 10, "post_count", 1))
            elif t is _ASYNC_WAIT:
                self.sinks.append(self._slot_arg(node.args[0]))
            elif t is _FILL:
                self.sinks.append(get_arg_value(node, 2, "value", 0))
                self.sinks.append(get_arg_value(node, 3, "num_slots", 1))

    def _scan_nested(self, gm: GraphModule) -> None:
        """A branch / commit region: its nodes matter, its output does not."""
        outputs = self.outputs
        self.outputs = outputs if outputs is not None else ()
        self._scan(gm)
        self.outputs = outputs

    # -- loop-carried scalars -----------------------------------------------

    def _cone(self, node) -> set:
        seen, stack = set(), [node]
        while stack:
            n = stack.pop()
            if not isinstance(n, Node) or n in seen:
                continue
            seen.add(n)
            if n in self.bind:
                stack.append(self.bind[n])
            stack.extend(a for a in n.args if isinstance(a, Node))
            stack.extend(
                x for a in n.args if isinstance(a, (list, tuple)) for x in a
            )
            stack.extend(v for v in n.kwargs.values() if isinstance(v, Node))
        return seen

    def _solve(self, carried_vals) -> List[tuple]:
        """Each carried slot's closed form: ``("const", c0)``,
        ``("affine", c0, k)`` or ``("cumsum", c0, k, pred, when_true)``."""
        slots = []
        for j, c0 in enumerate(carried_vals):
            ph = self.phs[j]
            out = self.outputs[j] if j < len(self.outputs) else None
            if out is ph or (out is None and j >= len(self.outputs)):
                slots.append(("const", c0))
                continue
            if not isinstance(out, Node) or not isinstance(c0, (int, bool)):
                if not isinstance(out, Node) and out == c0:
                    slots.append(("const", c0))
                    continue
                raise Unsupported(f"carried slot {j}")
            step = self._step_of(out, ph)
            if step is not None:
                slots.append(("affine", c0, step))
                continue
            if out.target is torch.sym_ite and len(out.args) == 3:
                pred, a, b = out.args
                for branch, other, when in ((a, b, True), (b, a, False)):
                    step = self._step_of(branch, ph)
                    if step is not None and other is ph:
                        if ph in self._cone(pred):
                            raise Unsupported(f"carried slot {j} recurses")
                        slots.append(("cumsum", c0, step, pred, when))
                        break
                else:
                    raise Unsupported(f"carried slot {j}")
                continue
            raise Unsupported(f"carried slot {j}")
        return slots

    @staticmethod
    def _step_of(node, ph) -> Optional[int]:
        """``k`` when ``node`` is ``ph + k`` for a literal ``k``."""
        if (
            isinstance(node, Node)
            and node.op == "call_function"
            and node.target is operator.add
            and len(node.args) == 2
        ):
            a, b = node.args
            if a is ph and isinstance(b, int):
                return b
            if b is ph and isinstance(a, int):
                return a
        return None

    # -- evaluation -------------------------------------------------------

    def _value(self, node, envv, lo, hi):
        """The value of ``node`` for iterations ``[lo, hi)``: a tensor per
        iteration, or a constant."""
        if not isinstance(node, Node):
            if isinstance(node, (list, tuple)):
                return type(node)(self._value(x, envv, lo, hi) for x in node)
            return node
        if node in envv:
            return envv[node]
        if node in self.bind:
            value = self._value(self.bind[node], envv, lo, hi)
        elif node in self.constants:
            value = self.constants[node]
        elif node.op == "placeholder":
            raise Unsupported(f"unsolved placeholder {node}")
        elif _is_scalar_node(node):
            args = [self._value(a, envv, lo, hi) for a in node.args]
            kwargs = {
                k: self._value(v, envv, lo, hi) for k, v in node.kwargs.items()
            }
            value = _apply(node.target, args, kwargs)
        else:
            value = _val(node)
        envv[node] = value
        return value

    def _evaluate_all(self) -> None:
        words = []
        for lo in range(0, self.steps, _CHUNK):
            hi = min(self.steps, lo + _CHUNK)
            words.append(self._evaluate(lo, hi))
        if words:
            self.words = torch.cat(words)
        else:
            self.words = torch.zeros(0, 1, dtype=torch.int64)

    def _evaluate(self, lo: int, hi: int) -> torch.Tensor:
        envv: Dict[Node, object] = {}
        step = torch.arange(lo, hi, dtype=torch.int64)
        pending = list(range(len(self.slots)))
        while pending:
            progress = False
            for j in list(pending):
                slot = self.slots[j]
                ph = self.phs[j]
                if slot[0] == "const":
                    envv[ph] = slot[1]
                elif slot[0] == "affine":
                    envv[ph] = slot[1] + slot[2] * step
                else:
                    _, c0, k, pred, when = slot
                    try:
                        p = self._value(pred, envv, lo, hi)
                    except Unsupported:
                        continue
                    if not when:
                        p = ~p if torch.is_tensor(p) else not p
                    if not torch.is_tensor(p):
                        p = torch.full_like(step, int(bool(p)))
                    hits = p.to(torch.int64)
                    before = self._running.get(j, 0)
                    exclusive = before + torch.cumsum(hits, 0) - hits
                    self._running[j] = before + int(hits.sum())
                    self.counts.setdefault(j, []).append(
                        exclusive.to(torch.int32)
                    )
                    envv[ph] = c0 + k * exclusive
                pending.remove(j)
                progress = True
            if not progress:
                raise Unsupported("carried scalars depend on each other")
        values = [self._value(s, envv, lo, hi) for s in self.sinks]
        return self._pack(values, hi - lo)

    @staticmethod
    def _pack(values, n: int) -> torch.Tensor:
        """Every sink's value per iteration packed into 62-bit words."""
        words, word, used = [], torch.zeros(n, dtype=torch.int64), 0
        for v in values:
            if torch.is_tensor(v):
                v = v.reshape(-1)
                if v.numel() == 1:
                    v = v.expand(n)
            else:
                v = torch.full((n,), int(v), dtype=torch.int64)
            if v.dtype == torch.bool:
                bits, v = 1, v.to(torch.int64)
            else:
                v = v.to(torch.int64)
                if v.numel() and (
                    int(v.max()) >= 1 << _SLOT_BITS or int(v.min()) < 0
                ):
                    raise Unsupported("slot value out of range")
                bits = _SLOT_BITS
            if used + bits > _WORD_BITS:
                words.append(word)
                word, used = torch.zeros(n, dtype=torch.int64), 0
            word = word | (v << used)
            used += bits
        words.append(word)
        return torch.stack(words, dim=1)

    # -- queries ------------------------------------------------------------

    def class_period(self) -> int:
        """The smallest period of the class words over the bulk of the loop
        (its first ``_PREFIX`` iterations and last two periods excepted), or
        ``0`` when none up to an eighth of the trip count fits.  Every
        timing period is a multiple of it.  Candidates are the distances at
        which the class of one bulk iteration recurs, each screened on a
        window around that iteration before the whole range is compared."""
        n = self.steps
        limit = min(_MAX_CLASS_PERIOD, n // 8)
        if limit < 1:
            return 0
        k0 = n // 3
        words = self.words
        recur = (words[k0 + 1 : k0 + 1 + limit] == words[k0]).all(dim=1)
        for c in (torch.nonzero(recur).flatten() + 1).tolist():
            window = min(max(32 * c, 1024), n - k0 - c)
            if not torch.equal(
                words[k0 : k0 + window], words[k0 + c : k0 + c + window]
            ):
                continue
            start = max(c, _PREFIX)
            if n - 2 * c <= start:
                return 0
            if torch.equal(words[start : n - 2 * c], words[start + c : n - c]):
                return c
        return 0

    def periodic(self, start: int, period: int, count: int) -> int:
        """How many of iterations ``[start, start + count)`` repeat, in
        order, the class of the iteration one ``period`` earlier -- the
        length of the prefix of that range a fold may cover."""
        count = min(count, self.steps - start)
        if count <= 0 or start < period:
            return 0
        a = self.words[start - period : start - period + count]
        b = self.words[start : start + count]
        same = (a == b).all(dim=1)
        if bool(same.all()):
            return count
        return int(torch.nonzero(~same)[0])

    def carried_at(self, i: int) -> list:
        """The loop-carried scalars at the start of iteration ``i``."""
        out = []
        for j, slot in enumerate(self.slots):
            if slot[0] == "const":
                out.append(slot[1])
            elif slot[0] == "affine":
                out.append(slot[1] + slot[2] * i)
            else:
                counts = self.counts[j]
                if isinstance(counts, list):
                    counts = self.counts[j] = torch.cat(counts)
                out.append(slot[1] + slot[2] * int(counts[i]))
        return out


def loop_structure(loop, gm, env, carried_vals, steps) -> Optional[Structure]:
    """The loop's ``Structure``, or ``None`` when its body cannot be
    evaluated over all iterations at once."""
    try:
        return Structure(loop, gm, env, carried_vals, steps)
    except Unsupported:
        return None
