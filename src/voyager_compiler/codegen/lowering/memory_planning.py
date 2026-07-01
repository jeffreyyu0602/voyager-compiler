"""
Baseline memory planner for the bufferized FX path.

The bufferization pass leaves every tensor node tagged with ``meta['space']``
(``"DRAM"`` / ``"Scratchpad"``) but with no concrete address.  This pass turns
those space-annotated buffers/tiles into a concrete map:

  * **DRAM** — the persistent params / inputs are placed first (no reuse), then
    the intermediate ``voyager.alloc`` activation buffers are packed with a
    greedy best-fit, shared-object allocator that reuses a slot whose lifetime
    is disjoint and whose size is closest (least fragmentation).
  * **Scratchpad** — every on-chip buffer is likewise an explicit
    ``voyager.alloc(SRAM)`` (the input / output banks — each a ``[num_banks,
    tile...]`` alloc — and the reduction scratch), so it is packed with the same
    greedy best-fit allocator as DRAM: a buffer reuses the slot of one whose
    lifetime is already dead, across region boundaries.  Bank separation is
    *structural* — distinct banks are distinct allocs, so simultaneously-live
    banks get distinct addresses automatically; no per-op banking strategy is
    needed.

The planner does not move values between DRAM and Scratchpad (that is fixed by
bufferization); it only decides addresses, reuse, and pool sizes.  Results are
written into ``node.meta['memory']`` (DRAM) and ``node.meta['scratchpad']``
(Scratchpad) as ``Segment``s, which the proto emitter (``set_tensor_field``)
reads directly.  See the roadmap in the design doc for the optimizing passes
that build on this baseline (intra-region reuse, store->load elision, double
buffering, the interstellar schedule).
"""

import logging
import math
import operator
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ..banking import require_allocation
from ..memory import MemorySpace, Segment, _align_size
from ...pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)

voyager = torch.ops.voyager
_ALLOC = voyager.alloc.default
_ZERO = voyager.zeros.default
_INCR = voyager.increment_indices.default
_WHILE = torch.ops.higher_order.while_loop
_GETITEM = operator.getitem


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _val(node) -> Optional[torch.Tensor]:
    """The single tensor a node produces (its ShapeProp ``value`` or, inside a
    loop body, the exported ``meta['val']``); ``None`` for tuples /
    non-tensors."""
    if not isinstance(node, Node):
        return None
    v = node.meta.get("val", getattr(node, "value", None))
    return v if isinstance(v, torch.Tensor) else None


def _nbytes(node, bank_width=None) -> int:
    """Byte size of a node's tensor, using the logical (quantized) dtype when
    set and aligning to ``bank_width``."""
    t = _val(node)
    if t is None:
        raise ValueError(f"{node} has no sized value to allocate memory for")
    dtype = node.meta.get("dtype") or t.dtype
    size = math.ceil(t.numel() * dtype_byte_size(dtype))
    return int(_align_size(size, bank_width))


def _submodule(gm: GraphModule, target) -> Optional[GraphModule]:
    try:
        sub = gm.get_submodule(str(target))
    except AttributeError:
        return None
    return sub if isinstance(sub, GraphModule) else None


@dataclass
class MemoryPlan:
    dram_bytes: int
    scratchpad_bytes: int


# ---------------------------------------------------------------------------
# Greedy best-fit shared-object allocator (used for the reusable DRAM region)
# ---------------------------------------------------------------------------


def _greedy_best_fit(
    items: List[Tuple[object, int, int, int]],
) -> Tuple[Dict[object, int], int]:
    """Assign each item the lowest address free of every overlapping-lifetime
    item.

    ``items`` is ``[(key, size, def_t, last_t), ...]``.  Process largest-first,
    and place each at the lowest offset that doesn't collide (in address) with
    an already-placed item whose lifetime overlaps — the standard "greedy by
    size" offset planner, whose high-water mark approaches the max concurrent
    demand (so sequential regions collapse onto the same low addresses, rather
    than each item getting a fresh slot).  Returns ``({key: offset},
    total_bytes)``.
    """
    placed: List[Tuple[object, int, int, int, int]] = (
        []
    )  # (key, lo, hi, start, end)
    total = 0
    for key, size, lo, hi in sorted(items, key=lambda it: (-it[1], it[2])):
        occupied = sorted(
            (start, end)
            for _k, a, b, start, end in placed
            if a <= hi and lo <= b
        )
        off = 0
        for start, end in occupied:
            if off + size <= start:  # fits in the gap before this block
                break
            off = max(off, end)
        placed.append((key, lo, hi, off, off + size))
        total = max(total, off + size)
    return {key: start for key, _lo, _hi, start, _end in placed}, total


# ---------------------------------------------------------------------------
# DRAM planning
# ---------------------------------------------------------------------------


def _is_param(node: Node, gm: GraphModule) -> bool:
    if node.op != "get_attr":
        return False
    if _submodule(gm, node.target) is not None:
        return False
    if str(node.target).startswith("lifted_tensor"):
        return False
    return require_allocation(node)


def _plan_dram(model: GraphModule, bank_width: Optional[int]) -> int:
    """Place all DRAM tensors: persistent params / inputs first (no reuse), then
    greedy best-fit over the intermediate ``alloc`` activation buffers.  Writes
    ``meta['memory']`` on each DRAM buffer root (threaded onto aliases later).
    """
    nodes = list(model.graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}

    # Walk every node and collect the DRAM tensors, split into two pools:
    #   persistent -- inputs + weights, live the whole run (placed once)
    #   reusable   -- alloc activation buffers, recycled once dead
    # ``buffer_of`` maps each DRAM node to the node owning its buffer.  Several
    # nodes can name the same physical buffer: an ``alloc`` owns itself, while a
    # ``getitem``/``copy_tile`` is just another handle to the buffer it pulls
    # out of a loop / writes to, so it resolves back to that owner.
    buffer_of: Dict[Node, Node] = {}
    persistent: List[Node] = []
    reusable: List[Node] = []
    for n in nodes:
        if n.op == "placeholder" and _val(n) is not None:
            buffer_of[n] = n  # model input
            persistent.append(n)
        elif _is_param(n, model) and _val(n) is not None:
            buffer_of[n] = n  # weight / codebook
            persistent.append(n)
        elif n.op == "call_function":
            if n.target is _ALLOC:
                buffer_of[n] = n  # new activation buffer
                reusable.append(n)
            elif n.target is _GETITEM:  # alias: loop result -> carried buffer
                src, idx = n.args[0], n.args[1]
                if (
                    isinstance(src, Node)
                    and src.target is _WHILE
                    and isinstance(idx, int)
                ):
                    carried = list(src.args[2])
                    if idx < len(carried) and isinstance(carried[idx], Node):
                        buffer_of[n] = buffer_of.get(carried[idx])

    # Lifetime of each buffer: from its def to the last top-level node consuming
    # it (or an alias).  A while_loop consuming a buffer keeps it live for the
    # loop.
    def_t = {b: pos[b] for b in set(buffer_of.values()) if b is not None}
    last_t = dict(def_t)
    for n in nodes:
        for inp in n.all_input_nodes:
            root = buffer_of.get(inp)
            if root is not None:
                last_t[root] = max(last_t[root], pos[n])

    # Persistent region first (params + inputs), linear, no reuse.
    offset = 0
    for b in persistent:
        size = _nbytes(b, bank_width)
        b.meta["memory"] = Segment(offset, offset + size, MemorySpace.DRAM, b)
        offset += size

    # Reusable activation buffers: greedy best-fit, laid out after the
    # persistent region.
    items = [(b, _nbytes(b, bank_width), def_t[b], last_t[b]) for b in reusable]
    placed, reuse_bytes = _greedy_best_fit(items)
    for b in reusable:
        start = offset + placed[b]
        size = _nbytes(b, bank_width)
        b.meta["memory"] = Segment(start, start + size, MemorySpace.DRAM, b)

    return offset + reuse_bytes


# ---------------------------------------------------------------------------
# Scratchpad planning (alloc-based, packed like DRAM)
# ---------------------------------------------------------------------------


# --- global schedule + per-buffer lifetimes (scratchpad allocated like DRAM) -


def _walk(gm: GraphModule):
    """Yield every node in execution order, descending into a ``while_loop``
    body at the loop's position and into a fused ``call_module`` submodule."""
    for n in gm.graph.nodes:
        yield n
        if n.op == "call_function" and n.target is _WHILE:
            body = _submodule(gm, n.args[1].target)
            if body is not None:
                yield from _walk(body)
        elif n.op == "call_module":
            sub = _submodule(gm, n.target)
            if sub is not None:
                yield from _walk(sub)


def _timestamps(model: GraphModule) -> Dict[Node, int]:
    """A global execution timestamp per node.  A loop body lands inside its
    loop's span, so a value carried through the loop spans the whole nest."""
    return {n: i for i, n in enumerate(_walk(model))}


class _UnionFind:
    def __init__(self):
        self.parent: Dict[Node, Node] = {}

    def find(self, x: Node) -> Node:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] is not root:
            root = self.parent[root]
        while self.parent[x] is not root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Node, b: Node) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra is not rb:
            self.parent[rb] = ra


def _buffer_identity(model: GraphModule) -> _UnionFind:
    """Merge the FX nodes that name the *same* physical buffer.  A value carried
    through a ``while_loop`` is one buffer — its carried operand, the body
    placeholder it binds, the body's returned value for that slot, and the
    loop-result ``getitem`` all collapse together.  So a scratch ``alloc``
    accumulator threaded through the reduction loop (alloc -> body arg ->
    accumulate-add -> getitem) becomes one buffer with one lifetime, instead of
    several co-live aliases."""
    uf = _UnionFind()

    def walk(gm: GraphModule):
        for n in gm.graph.nodes:
            uf.find(n)
            if n.op == "call_function" and n.target is _WHILE:
                body = _submodule(gm, n.args[1].target)
                if body is None:
                    continue
                phs = [p for p in body.graph.nodes if p.op == "placeholder"]
                carried = list(n.args[2])
                operands = carried + (
                    list(n.args[3]) if len(n.args) > 3 else []
                )
                out = next(
                    x for x in body.graph.nodes if x.op == "output"
                ).args[0]
                outs = list(out) if isinstance(out, (list, tuple)) else [out]
                for ph, o in zip(phs, operands):
                    if isinstance(o, Node):
                        uf.union(o, ph)
                for i, c in enumerate(carried):
                    if (
                        isinstance(c, Node)
                        and i < len(outs)
                        and isinstance(outs[i], Node)
                    ):
                        uf.union(c, outs[i])
                walk(body)
            elif n.op == "call_function" and n.target is _GETITEM:
                src, idx = n.args[0], n.args[1]
                if (
                    isinstance(src, Node)
                    and src.target is _WHILE
                    and isinstance(idx, int)
                ):
                    carried = list(src.args[2])
                    if idx < len(carried) and isinstance(carried[idx], Node):
                        uf.union(n, carried[idx])
            elif n.op == "call_module":
                sub = _submodule(gm, n.target)
                if sub is None:
                    continue
                phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
                for ph, a in zip(phs, n.args):
                    if isinstance(a, Node):
                        uf.union(a, ph)
                walk(sub)

    walk(model)
    return uf


@dataclass
class _Buf:
    size: int
    def_t: int
    last_t: int
    members: List[Node]


def _buffer_lifetimes(model, uf, order, bank_width) -> Dict[Node, _Buf]:
    """Per scratchpad buffer (a union-find root): byte size, birth, and last
    use.  Death follows the alias chain — a use of *any* member (e.g. the
    getitem of a carried accumulator, or the bias-add that reads it) extends the
    lifetime."""
    members: Dict[Node, List[Node]] = {}
    for n in _walk(model):
        members.setdefault(uf.find(n), []).append(n)

    bufs: Dict[Node, _Buf] = {}
    for root, mem in members.items():
        tiles = [m for m in mem if m.meta.get("space") == "Scratchpad"]
        if not tiles:
            continue
        def_t = min(order[m] for m in tiles)
        last_t = def_t
        for m in mem:  # scan all members' users for the death
            for u in m.users:
                if u in order:
                    last_t = max(last_t, order[u])
        size = max(_nbytes(m, bank_width) for m in tiles)
        bufs[root] = _Buf(size, def_t, last_t, tiles)
    return bufs


def _plan_scratchpad(
    model: GraphModule,
    uf: "_UnionFind",
    bufs: Dict[Node, "_Buf"],
    cache_size: int,
    num_banks: Optional[int],
    bank_width: Optional[int],
    unroll_dim: Optional[int],
):
    """Pack every Scratchpad buffer with greedy best-fit, exactly like DRAM.

    With the alloc-only model each on-chip buffer is an explicit
    ``voyager.alloc(SRAM)`` — the input / output banks (a ``[num_banks,
    tile...]`` alloc each) and the reduction scratch — so ``bufs`` already lists
    them with sizes and lifetimes.  A buffer reuses the address of one whose
    lifetime is already dead (across region boundaries).  Bank separation is
    *structural*: distinct banks are distinct allocs, so simultaneously-live
    banks land at distinct addresses automatically — no per-op banking strategy
    or region grouping is needed.
    """
    items = [
        (root, buf.size, buf.def_t, buf.last_t) for root, buf in bufs.items()
    ]
    bases, total = _greedy_best_fit(items)
    for root, base in bases.items():
        seg = Segment(
            base, base + bufs[root].size, MemorySpace.SCRATCHPAD, root
        )
        for m in bufs[root].members:
            m.meta.setdefault("scratchpad", seg)

    return int(total), {"buffers": len(items)}


def _buf_desc(buf: "_Buf", bank_width) -> str:
    """``name<shape x dtype>`` of the largest tile in a scratchpad buffer (the
    one that set its size), for the ``[MEM_ALLOC_FAIL]`` diagnostic."""
    m = max(buf.members, key=lambda x: _nbytes(x, bank_width))
    v = _val(m)
    dtype = m.meta.get("dtype") or (v.dtype if v is not None else "?")
    shape = "x".join(str(d) for d in v.shape) if v is not None else "?"
    return f"{m.name}<{shape}x{dtype}>"


def _peak_live_buffers(bufs: Dict[Node, "_Buf"]):
    """The scratchpad buffers simultaneously live at the busiest schedule step
    and their summed size -- the concurrency that drives the peak.  Returns
    ``(peak_bytes, [_Buf, ...] largest-first)``."""
    peak_bytes, peak_live = 0, []
    for t in sorted({b.def_t for b in bufs.values()}):
        live = [b for b in bufs.values() if b.def_t <= t <= b.last_t]
        total = sum(b.size for b in live)
        if total > peak_bytes:
            peak_bytes, peak_live = total, live
    return peak_bytes, sorted(peak_live, key=lambda b: -b.size)


# ---------------------------------------------------------------------------
# Thread Segments across loop / module boundaries onto the actual tile sites
# ---------------------------------------------------------------------------


def _thread_segments(model: GraphModule) -> None:
    """Propagate ``meta['memory']`` / ``meta['scratchpad']`` from buffer roots
    onto the body placeholders bound to them and onto in-place compute tiles, so
    every ``copy_tile`` / ``async_copy`` tile and op tile resolves to an
    address.
    Mirrors the space-threading recursion of ``annotate_tensor_spaces``.
    """

    def seg_of(n):
        if not isinstance(n, Node):
            return None
        return n.meta.get("scratchpad") or n.meta.get("memory")

    def walk(gm: GraphModule, bound: Dict[Node, Segment]):
        local: Dict[Node, Segment] = {}

        def get(n):
            return local.get(n) or seg_of(n)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                if (
                    node in bound
                    and "scratchpad" not in node.meta
                    and "memory" not in node.meta
                ):
                    local[node] = bound[node]
                    key = (
                        "scratchpad"
                        if bound[node].memory_space == MemorySpace.SCRATCHPAD
                        else "memory"
                    )
                    node.meta[key] = bound[node]
            elif node.op == "call_function":
                if node.target is _WHILE:
                    operands = list(node.args[2])
                    if len(node.args) > 3:
                        operands += list(node.args[3])
                    body = _submodule(gm, node.args[1].target)
                    if body is not None:
                        phs = [
                            p for p in body.graph.nodes if p.op == "placeholder"
                        ]
                        child = {
                            ph: get(o)
                            for ph, o in zip(phs, operands)
                            if get(o) is not None
                        }
                        walk(body, child)
                elif node.target is _GETITEM:
                    src, idx = node.args[0], node.args[1]
                    if (
                        isinstance(src, Node)
                        and src.target is _WHILE
                        and isinstance(idx, int)
                    ):
                        carried = list(src.args[2])
                        if idx < len(carried) and get(carried[idx]) is not None:
                            _set(node, get(carried[idx]), local)
                elif node.target not in (_ZERO, _INCR, _ALLOC):
                    # In-place compute tile: inherit the segment of its first
                    # scratchpad input (e.g. accumulate-add reuses the
                    # accumulator).
                    if (
                        node.meta.get("space") == "Scratchpad"
                        and "scratchpad" not in node.meta
                    ):
                        for inp in node.all_input_nodes:
                            s = get(inp)
                            if (
                                s is not None
                                and s.memory_space == MemorySpace.SCRATCHPAD
                            ):
                                _set(node, s, local)
                                break
            elif node.op == "call_module":
                sub = _submodule(gm, node.target)
                if sub is not None:
                    phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
                    child = {
                        ph: get(a)
                        for ph, a in zip(phs, node.args)
                        if get(a) is not None
                    }
                    walk(sub, child)

    walk(model, {})


def _set(node: Node, seg: Segment, local: Dict[Node, Segment]) -> None:
    local[node] = seg
    key = (
        "scratchpad" if seg.memory_space == MemorySpace.SCRATCHPAD else "memory"
    )
    node.meta.setdefault(key, seg)


# ---------------------------------------------------------------------------
# Invariant checker
# ---------------------------------------------------------------------------


def _check_overlaps(arena: str, items) -> None:
    """Warn if two simultaneously-live buffers share an address range.
    ``items`` is ``[(name, def_t, last_t, Segment), ...]`` — overlapping
    lifetimes must map to disjoint address ranges."""
    for i in range(len(items)):
        n1, a1, b1, s1 = items[i]
        for j in range(i + 1, len(items)):
            n2, a2, b2, s2 = items[j]
            if (
                a1 <= b2
                and a2 <= b1
                and s1.start < s2.end
                and s2.start < s1.end
            ):
                logger.warning(
                    "[MEM_OVERLAP] %s %s [%d,%d) and %s [%d,%d) overlap",
                    arena,
                    n1,
                    s1.start,
                    s1.end,
                    n2,
                    s2.start,
                    s2.end,
                )


def _check_invariants(model: GraphModule, bufs: Dict[Node, "_Buf"]) -> None:
    """Warn if two simultaneously-live buffers share an address range, per
    arena."""
    nodes = list(model.graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}

    dram = []
    for n in nodes:
        seg = n.meta.get("memory")
        if (
            seg is not None
            and n.meta.get("space") == "DRAM"
            and isinstance(_val(n), torch.Tensor)
        ):
            last = max([pos[n]] + [pos[u] for u in n.users if u in pos])
            dram.append((n.name, pos.get(n, 0), last, seg))
    _check_overlaps("DRAM", dram)

    scratch = []
    for root, bf in bufs.items():
        seg = next(
            (
                m.meta["scratchpad"]
                for m in bf.members
                if "scratchpad" in m.meta
            ),
            None,
        )
        if seg is not None:
            scratch.append((root.name, bf.def_t, bf.last_t, seg))
    _check_overlaps("Scratchpad", scratch)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def plan_memory(
    model: GraphModule,
    cache_size: Optional[int],
    *,
    num_banks: Optional[int] = None,
    bank_width: Optional[int] = None,
    unroll_dims=None,
) -> MemoryPlan:
    """Assign concrete DRAM / Scratchpad addresses to a bufferized FX graph.

    Writes ``meta['memory']`` (DRAM) and ``meta['scratchpad']`` (Scratchpad)
    ``Segment``s, threading them onto loop-body tile sites; returns the pool
    sizes. Warns, listing the buffers live at the peak, if the scratchpad
    plan exceeds ``cache_size``.
    """
    unroll_dim = unroll_dims[1] if unroll_dims else None

    dram_bytes = _plan_dram(model, bank_width)

    # Global schedule + per-buffer lifetimes drive scratchpad allocation and the
    # co-liveness check; compute them once.
    uf = _buffer_identity(model)
    bufs = _buffer_lifetimes(model, uf, _timestamps(model), bank_width)

    scratchpad_bytes = 0
    peak_region = None
    if cache_size is not None:
        scratchpad_bytes, peak_region = _plan_scratchpad(
            model, uf, bufs, cache_size, num_banks, bank_width, unroll_dim
        )

    _thread_segments(model)
    _check_invariants(model, bufs)

    if cache_size is not None and scratchpad_bytes > cache_size:
        peak_bytes, live = _peak_live_buffers(bufs)
        shown = live[:12]
        detail = "\n".join(
            f"    {_buf_desc(b, bank_width)}  {b.size} B  "
            f"[def={b.def_t} last={b.last_t}]"
            for b in shown
        )
        if len(live) > len(shown):
            detail += f"\n    ... (+{len(live) - len(shown)} more)"
        logger.warning(
            "[plan_memory] scratchpad plan needs %d bytes > cache_size %d; "
            "peak concurrency %d B across %d live scratchpad buffers:\n%s",
            scratchpad_bytes,
            cache_size,
            peak_bytes,
            len(live),
            detail,
        )

    logger.info(
        "Memory plan: DRAM=%d bytes, Scratchpad=%d/%s bytes (peak region %s)",
        dram_bytes,
        scratchpad_bytes,
        cache_size,
        peak_region,
    )
    return MemoryPlan(dram_bytes, scratchpad_bytes)
