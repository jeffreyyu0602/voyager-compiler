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
    ``voyager.alloc(SRAM)`` (the input / output banks — each a ``banks``-deep
    alloc — and the reduction scratch), so it is packed with the same greedy
    best-fit allocator as DRAM: a buffer reuses the slot of one whose lifetime
    is already dead, across region boundaries.  A software-pipeline bank is one
    buffer of ``banks`` slots, laid out contiguously: slot ``i`` sits at
    ``base + i * bank_stride``, so the slot a step writes can stay a *runtime*
    index.

The planner does not move values between DRAM and Scratchpad (that is fixed by
bufferization); it only decides addresses, reuse, and pool sizes.  Addresses are
written as ``Segment``s onto each buffer *root* — ``node.meta['memory']``
(DRAM) / ``node.meta['scratchpad']`` (Scratchpad), plus ``meta['bank_count']``
and ``meta['bank_stride']`` on a banked one.  A tile is not given an address of
its own: it is named by the buffer it lives in, which is what the code generator
serializes (a ``TensorBoxRef``, windowed at the bank it reads).  See the roadmap in the
design doc for the optimizing passes that build on this baseline (intra-region
reuse, store->load elision, double buffering, the interstellar schedule).
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from .bufferization import _viewed_buffer
from .utils import _collect_codebook_nodes, _passed_whole, _subgraph
from ..memory import MemorySpace, Segment, _align_size
from ..passes.utils import get_arg_value
from ...pt2e_utils import dtype_byte_size
from .ops import UNBANKED

logger = logging.getLogger(__name__)

voyager = torch.ops.voyager
_ALLOC = voyager.alloc.default
_ZERO = voyager.zeros.default
_FILL = voyager.fill.default
_WHILE = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond
_COMMIT = torch.ops.higher_order.commit

# Position of ``banks`` in each allocation primitive's schema:
_BANKS_ARG = {_ALLOC: 3, _ZERO: 2, _FILL: 3}


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


def _banks(node) -> int:
    """The software-pipeline depth of an ``alloc`` / ``zeros`` — how many banks
    its leading dimension holds.  ``UNBANKED`` (0) for every other node."""
    if not isinstance(node, Node) or node.op != "call_function":
        return UNBANKED
    index = _BANKS_ARG.get(node.target)
    if index is None:
        return UNBANKED
    return int(get_arg_value(node, index, "banks", UNBANKED))


def _bank_stride(node, bank_width=None) -> int:
    """Byte distance between adjacent banks of a banked buffer: the aligned size
    of *one* bank's payload.  The bank dimension leads the tensor, so the
    payload is the remaining ``numel // banks`` elements."""
    t = _val(node)
    dtype = node.meta.get("dtype") or t.dtype
    per_bank = math.ceil(t.numel() / _banks(node) * dtype_byte_size(dtype))
    return int(_align_size(per_bank, bank_width))


def _nbytes(node, bank_width=None) -> int:
    """Byte size of a node's tensor, using the logical (quantized) dtype when
    set and aligning to ``bank_width``.  A banked buffer is ``banks`` aligned
    payloads, so each bank starts on an aligned boundary."""
    t = _val(node)
    if t is None:
        raise ValueError(f"{node} has no sized value to allocate memory for")
    if _banks(node):
        return _banks(node) * _bank_stride(node, bank_width)
    dtype = node.meta.get("dtype") or t.dtype
    size = math.ceil(t.numel() * dtype_byte_size(dtype))
    return int(_align_size(size, bank_width))


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


def _is_param(node: Node, gm: GraphModule, codebooks: set) -> bool:
    if node.op != "get_attr":
        return False
    if _subgraph(gm, node.target) is not None:
        return False
    if str(node.target).startswith("lifted_tensor"):
        return False
    # A codebook / qmap or a scalar scale is passed whole and owns no storage.
    return not _passed_whole(node, codebooks)


def _materializes_dram(node: Node) -> bool:
    """A tensor the *host* produces rather than the accelerator — a ``pad`` to
    the hardware unrolling, a copy.  Bufferization leaves it in DRAM (it is no
    tile, and no ``insert`` stores it); the loop then loads tiles straight out
    of it, so it is a DRAM buffer and needs an address like any other.  A view
    of a buffer is excluded: it owns no bytes of its own.
    """
    return (
        node.op == "call_function"
        and node.target not in (_ALLOC, _ZERO)
        and node.meta.get("space") == "DRAM"
        and _val(node) is not None
        and _viewed_buffer(node) is None
    )


def _plan_dram(model: GraphModule, buffer_of, bank_width: Optional[int]) -> int:
    """Place all DRAM tensors: persistent params / inputs first (no reuse), then
    greedy best-fit over the intermediate ``alloc`` activation buffers.  Writes
    ``meta['memory']`` on each DRAM buffer root.
    """
    nodes = list(model.graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}

    codebooks = _collect_codebook_nodes(model)

    # The DRAM buffers, split into two pools:
    #   persistent -- inputs + weights, live the whole run (placed once)
    #   reusable   -- activation buffers, recycled once dead: an ``alloc``, or a
    #                 tensor the host materializes outside the accelerator (a
    #                 ``pad`` to the hardware unrolling) that the loop then loads
    #                 tiles from.
    persistent: List[Node] = []
    reusable: List[Node] = []
    for n in nodes:
        if n.op == "placeholder" and _val(n) is not None:
            persistent.append(n)  # model input
        elif _is_param(n, model, codebooks) and _val(n) is not None:
            persistent.append(n)  # weight
        elif n.op == "call_function" and (
            n.target is _ALLOC or _materializes_dram(n)
        ):
            if n.meta.get("space") != "Scratchpad":
                reusable.append(n)

    # Lifetime of each buffer: from its def to the last top-level node reading it
    # — through ``buffer_of``, so a read through a *name* of the buffer (a
    # reshape, the getitem handle of a loop result) is a read of the buffer, and
    # keeps it alive.  A while_loop reading one keeps it live for the loop.
    def_t = {b: pos[b] for b in persistent + reusable}
    last_t = dict(def_t)
    for n in nodes:
        for inp in n.all_input_nodes:
            root = buffer_of.get(inp, inp)
            if root in last_t:
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
    """Yield every node in execution order, descending at its position into a
    ``while_loop`` body, both branches of a ``cond``, and a fused
    ``call_module``.  ``_buffer_identity`` must know every region this descends
    into: a node it reaches but the union-find does not merge would look like a
    buffer of its own."""
    for n in gm.graph.nodes:
        yield n
        if n.op == "call_function" and n.target is _WHILE:
            body = _subgraph(gm, n.args[1].target)
            if body is not None:
                yield from _walk(body)
        elif n.op == "call_function" and n.target is _COND:
            for branch in (n.args[1], n.args[2]):
                sub = _subgraph(gm, branch.target)
                if sub is not None:
                    yield from _walk(sub)
        elif n.op == "call_function" and n.target is _COMMIT:
            sub = _subgraph(gm, n.args[0].target)
            if sub is not None:
                yield from _walk(sub)
        elif n.op == "call_module":
            sub = _subgraph(gm, n.target)
            if sub is not None:
                yield from _walk(sub)


def _timestamps(model: GraphModule) -> Dict[Node, int]:
    """A global execution timestamp per node.  A loop body lands inside its
    loop's span, so a value carried through the loop spans the whole nest."""
    return {n: i for i, n in enumerate(_walk(model))}


def _buffer_identity(model: GraphModule) -> Dict[Node, Node]:
    """The buffer each FX node names — a node absent from the map names itself.

    A buffer takes a new name every time it crosses a region boundary or is
    viewed, and each name would otherwise look like a buffer of its own, co-live
    with the rest.  Two rules resolve a name back to its buffer:

      * a **view** names the buffer it views — a bank ``subview``, a reshape, the
        ``getitem`` handle of a loop result;
      * a **region** binds its operands to its placeholders — a ``while_loop``
        (which also returns each carried buffer, written in place), a ``cond``
        (whose two branches share one operand list), a fused ``call_module``.

    Every one of those points from a *new* name to an *older* one, and ``_walk``
    is program order, so the source is always resolved by the time a name needs
    it — one pass, no fixpoint.

    So a scratch ``alloc`` accumulator threaded through the reduction loop
    (alloc -> body arg -> accumulate-add -> getitem) is one buffer with one
    lifetime, and the bank a ``cond`` branch writes through a slot ``subview`` is
    the bank itself, not a tile beside it.
    """
    buffer_of: Dict[Node, Node] = {}

    def bind(alias: Node, source) -> None:
        """``alias`` is another name for the buffer ``source`` names."""
        if not isinstance(source, Node):
            return
        owner = buffer_of.get(source, source)
        if buffer_of.setdefault(alias, owner) is not owner:
            raise ValueError(
                f"'{alias.name}' names two buffers, '{buffer_of[alias].name}' "
                f"and '{owner.name}': they would have to be one"
            )

    def walk(gm: GraphModule):
        for n in gm.graph.nodes:
            if (viewed := _viewed_buffer(n)) is not None:
                bind(n, viewed)

            if n.op == "call_function" and n.target is _WHILE:
                body = _subgraph(gm, n.args[1].target)
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
                    bind(ph, o)
                for i, c in enumerate(carried):
                    if isinstance(c, Node) and i < len(outs):
                        bind(outs[i], c)
                walk(body)
            elif n.op == "call_function" and n.target is _COND:
                operands = list(n.args[3]) if len(n.args) > 3 else []
                for branch in (n.args[1], n.args[2]):
                    sub = _subgraph(gm, branch.target)
                    if sub is None:
                        continue
                    phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
                    for ph, o in zip(phs, operands):
                        bind(ph, o)
                    walk(sub)
            elif n.op == "call_function" and n.target is _COMMIT:
                sub = _subgraph(gm, n.args[0].target)
                if sub is None:
                    continue
                phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
                for ph, o in zip(phs, n.args[1:]):
                    bind(ph, o)
                walk(sub)
            elif n.op == "call_module":
                sub = _subgraph(gm, n.target)
                if sub is None:
                    continue
                phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
                for ph, a in zip(phs, n.args):
                    bind(ph, a)
                walk(sub)

    walk(model)
    return buffer_of


@dataclass
class _Buf:
    size: int
    def_t: int
    last_t: int
    members: List[Node]


def _buffer_lifetimes(model, buffer_of, order, bank_width) -> Dict[Node, _Buf]:
    """Per scratchpad buffer: byte size, birth, and last use.  Death follows the
    names — a use of *any* of them (the getitem of a carried accumulator, the
    bias-add that reads it) extends the lifetime."""
    members: Dict[Node, List[Node]] = {}
    for n in _walk(model):
        members.setdefault(buffer_of.get(n, n), []).append(n)

    bufs: Dict[Node, _Buf] = {}
    for root, mem in members.items():
        # The members that own the bytes.  A view is in the group (it names this
        # buffer) but must not size it or start its life: a slot ``select`` is
        # one tile of a bank several tiles deep.  It still *ends* its life --
        # the death scan below reads every member.
        tiles = [
            m
            for m in mem
            if m.meta.get("space") == "Scratchpad" and _viewed_buffer(m) is None
        ]
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
# Banking metadata
# ---------------------------------------------------------------------------


def _stamp_banking(model: GraphModule, bank_width: Optional[int]) -> None:
    """Record each banked buffer's depth and bank pitch on its ``alloc`` /
    ``zeros`` node, so the code generator can serialize the bank dimension as
    ``bank_count`` / ``bank_stride_bytes`` rather than as a tensor dimension.
    A slot is then addressed ``base + bank * bank_stride_bytes``, which is what
    lets a runtime slot index (``step % num_banks``) stay a runtime value.
    """
    for node in _walk(model):
        if not (banks := _banks(node)):
            continue
        node.meta["bank_count"] = banks
        node.meta["bank_stride"] = _bank_stride(node, bank_width)


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

    Writes ``meta['memory']`` (DRAM) / ``meta['scratchpad']`` (Scratchpad)
    ``Segment``s on each buffer *root* — the ``alloc`` that owns the storage —
    plus ``meta['bank_count']`` / ``meta['bank_stride']`` on a banked one, and
    returns the pool sizes.  Nothing is threaded onto the tile sites: a tile is
    named by the buffer it lives in (and, for a bank, a runtime slot index), so
    the address belongs to the buffer, not to every reference to it.  Warns,
    listing the buffers live at the peak, if the scratchpad plan exceeds
    ``cache_size``.
    """
    unroll_dim = unroll_dims[1] if unroll_dims else None

    # Which buffer every name denotes, and the global schedule: both arenas need
    # them (a buffer dies at the last read of *any* of its names), so compute
    # them once.
    buffer_of = _buffer_identity(model)
    dram_bytes = _plan_dram(model, buffer_of, bank_width)
    bufs = _buffer_lifetimes(model, buffer_of, _timestamps(model), bank_width)

    scratchpad_bytes = 0
    peak_region = None
    if cache_size is not None:
        scratchpad_bytes, peak_region = _plan_scratchpad(
            model, bufs, cache_size, num_banks, bank_width, unroll_dim
        )

    _stamp_banking(model, bank_width)
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
