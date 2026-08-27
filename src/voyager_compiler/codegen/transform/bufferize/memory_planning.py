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
    ``voyager.alloc(SRAM)`` (the input / output buffers — each a
    ``num_slots``-deep alloc — and the reduction scratch), so it is packed
    with the same greedy best-fit allocator as DRAM: a buffer reuses the
    address of one whose lifetime is already dead, across region
    boundaries.  A software-pipelined buffer is one alloc of ``num_slots``
    slots, laid out contiguously: slot ``i`` sits at ``base + i *
    slot_stride``, so the slot a step writes can stay a *runtime* index.
    On a banked config, buffers stamped with the tile search's bank
    partition (``meta['bank_group']``) are placed in whole banks at
    bank-aligned addresses, each pipeline slot in its own bank(s) — see
    ``_plan_scratchpad``.

The planner does not move values between DRAM and Scratchpad (that is fixed by
bufferization); it only decides addresses, reuse, and pool sizes.  Addresses are
written as ``Segment``s onto each buffer *root* — ``node.meta['memory']``
(DRAM) / ``node.meta['scratchpad']`` (Scratchpad), plus ``meta['slot_count']``
and ``meta['slot_stride']`` on a pipelined one.  A tile is not given an
address of its own: it is named by the buffer it lives in, which is
what the code generator serializes (a ``TensorBoxRef``, windowed at the
slot it reads).  See the roadmap in the design doc for the optimizing
passes that build on this baseline (intra-region reuse, store->load
elision, double buffering, the interstellar schedule).
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import (
    get_arg_value,
    tensor_alloc_bytes,
)
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _viewed_buffer,
)
from voyager_compiler.codegen.transform.bufferize.ops import (
    UNPIPELINED,
    MemoryLevel,
)
from voyager_compiler.codegen.transform.bufferize.utils import (
    _collect_codebook_nodes,
    _passed_whole,
    _subgraph,
)

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: Union[float, int]
    end: Union[float, int]
    memory_space: Optional[int] = None
    node: Optional[torch.fx.Node] = None

    def __post_init__(self) -> None:
        s_raw = self.start
        e_raw = self.end

        s = int(s_raw)  # truncate toward zero (matches your original)
        e = math.ceil(e_raw)  # round end up

        if s != s_raw:
            logger.warning(
                "Segment start %r is not an integer. Rounding to %d.", s_raw, s
            )
        if e != e_raw:
            logger.warning(
                "Segment end %r is not an integer. Rounding up to %d.", e_raw, e
            )

        if e < s:
            raise ValueError(f"Segment end ({e}) is less than start ({s}).")

        self.start = s
        self.end = e


voyager = torch.ops.voyager
_ALLOC = voyager.alloc.default
_ZERO = voyager.zeros.default
_FILL = voyager.fill.default
_WHILE = torch.ops.higher_order.while_loop
_COND = torch.ops.higher_order.cond
_COMMIT = torch.ops.higher_order.commit

# Position of ``num_slots`` in each allocation primitive's schema:
_SLOTS_ARG = {_ALLOC: 3, _ZERO: 2, _FILL: 3}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _val(node) -> Optional[torch.Tensor]:
    """The single tensor a node produces -- the value ShapeProp stamped on it;
    ``None`` for tuples / non-tensors.

    ``value`` is the only source, the same one the emitter reads.  The exported
    ``meta['val']`` is deliberately not consulted: a pass that reshapes a node
    after export re-stamps ``value`` and leaves ``meta['val']`` at its
    export-time shape -- padding a sequence length is one -- so sizing a buffer
    from it reserves less than the emitter goes on to declare.
    """
    if not isinstance(node, Node):
        return None
    v = getattr(node, "value", None)
    return v if isinstance(v, torch.Tensor) else None


def _slots(node) -> int:
    """The software-pipeline depth of an ``alloc`` / ``zeros`` — how many slots
    its leading dimension holds.  ``UNPIPELINED`` (0) for every other node."""
    if not isinstance(node, Node) or node.op != "call_function":
        return UNPIPELINED
    index = _SLOTS_ARG.get(node.target)
    if index is None:
        return UNPIPELINED
    return int(get_arg_value(node, index, "num_slots", UNPIPELINED))


def _store_lanes(node, config) -> Optional[int]:
    """The store-beat width to size ``node`` with: the vector lanes for a
    Scratchpad tensor (the datapath writes it in whole beats, so its tail
    slack must be reserved), ``None`` for anything else (DRAM is written
    byte-exact by the DMA)."""
    if node.meta.get("space") != "Scratchpad":
        return None
    return config.vector_lanes


def _slot_stride(node, config) -> int:
    """Byte distance between adjacent slots of a pipelined buffer: the allocated
    size of *one* slot's payload -- its bytes plus the store path's tail-beat
    slack, aligned to ``config.bank_width`` (``tensor_alloc_bytes``).  The
    slot dimension leads the tensor, so the payload is the remaining
    ``numel // slots`` elements."""
    t = _val(node)
    dtype = node.meta.get("dtype") or t.dtype
    return tensor_alloc_bytes(
        t.numel() // _slots(node),
        dtype,
        config.bank_width,
        _store_lanes(node, config),
    )


def _nbytes(node, config) -> int:
    """Byte size of a node's tensor, using the logical (quantized) dtype when
    set: the payload plus, for a Scratchpad tensor, the store path's tail-beat
    slack, aligned to ``config.bank_width`` (``tensor_alloc_bytes``).  A
    pipelined buffer is ``slots`` such payloads, so each slot starts on an
    aligned boundary."""
    t = _val(node)
    if t is None:
        raise ValueError(f"{node} has no sized value to allocate memory for")
    if _slots(node):
        return _slots(node) * _slot_stride(node, config)
    dtype = node.meta.get("dtype") or t.dtype
    return tensor_alloc_bytes(
        t.numel(), dtype, config.bank_width, _store_lanes(node, config)
    )


@dataclass
class MemoryPlan:
    dram_bytes: int
    scratchpad_bytes: int


# ---------------------------------------------------------------------------
# Greedy best-fit shared-object allocator (used for the reusable DRAM region)
# ---------------------------------------------------------------------------


def _greedy_best_fit(
    items: List[Tuple[object, int, int, int, int]],
) -> Tuple[Dict[object, int], int]:
    """Assign each item the lowest address free of every overlapping-lifetime
    item.

    ``items`` is ``[(key, size, def_t, last_t, align), ...]``.  Process
    largest-first, and place each at the lowest ``align``-multiple offset that
    doesn't collide (in address) with an already-placed item whose lifetime
    overlaps — the standard "greedy by size" offset planner, whose high-water
    mark approaches the max concurrent demand (so sequential regions collapse
    onto the same low addresses, rather than each item getting a fresh slot).
    A bank-grouped scratchpad item passes ``bank_size`` so it occupies whole
    banks; everything else passes 1.  Returns ``({key: offset},
    total_bytes)``.
    """
    placed: List[Tuple[object, int, int, int, int]] = (
        []
    )  # (key, lo, hi, start, end)
    total = 0
    for key, size, lo, hi, align in sorted(
        items, key=lambda it: (-it[1], it[2])
    ):
        occupied = sorted(
            (start, end)
            for _k, a, b, start, end in placed
            if a <= hi and lo <= b
        )
        off = 0
        for start, end in occupied:
            candidate = math.ceil(off / align) * align
            if candidate + size <= start:  # fits in the gap before this block
                break
            off = max(off, end)
        off = math.ceil(off / align) * align
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


def _plan_dram(model: GraphModule, buffer_of, config) -> int:
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
        size = _nbytes(b, config)
        b.meta["memory"] = Segment(offset, offset + size, MemoryLevel.DRAM, b)
        offset += size

    # Reusable activation buffers: greedy best-fit, laid out after the
    # persistent region.
    items = [(b, _nbytes(b, config), def_t[b], last_t[b], 1) for b in reusable]
    placed, reuse_bytes = _greedy_best_fit(items)
    for b in reusable:
        start = offset + placed[b]
        size = _nbytes(b, config)
        b.meta["memory"] = Segment(start, start + size, MemoryLevel.DRAM, b)

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

      * a **view** names the buffer it views — a slot ``subview``, a reshape, the
        ``getitem`` handle of a loop result;
      * a **region** binds its operands to its placeholders — a ``while_loop``
        (which also returns each carried buffer, written in place), a ``cond``
        (whose two branches share one operand list), a fused ``call_module``.

    Every one of those points from a *new* name to an *older* one, and ``_walk``
    is program order, so the source is always resolved by the time a name needs
    it — one pass, no fixpoint.

    So a scratch ``alloc`` accumulator threaded through the reduction loop
    (alloc -> body arg -> accumulate-add -> getitem) is one buffer with one
    lifetime, and the slot a ``cond`` branch writes through a ``subview``
    is the slot itself, not a tile beside it.
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


def _buffer_lifetimes(model, buffer_of, order, config) -> Dict[Node, _Buf]:
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
        # one tile of a slot several tiles deep.  It still *ends* its life --
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
        size = max(_nbytes(m, config) for m in tiles)
        bufs[root] = _Buf(size, def_t, last_t, tiles)
    return bufs


def _bank_group_key(buf: "_Buf"):
    """The ``(scope, group)`` key of a buffer's searched bank group, or
    ``None`` for one the search left loose.  ``scope`` is the per-splice nest
    tag (``bufferize_graph``), so two nests spliced from one cached build
    never share a group."""
    for m in buf.members:
        if (group := m.meta.get("bank_group")) is not None:
            return (m.meta.get("scope"), group)
    return None


def _slot_payload(buf: "_Buf", config) -> Tuple[int, int]:
    """One slot's allocated bytes of a scratchpad buffer and its pipeline
    depth, ``(payload, slots)`` — the whole buffer at depth 1 for an
    unslotted one.  Sized off the member that set the buffer's size."""
    m = max(buf.members, key=lambda x: _nbytes(x, config))
    slots = _slots(m)
    if slots:
        return _slot_stride(m, config), slots
    return _nbytes(m, config), 1


def _plan_scratchpad(model: GraphModule, bufs: Dict[Node, "_Buf"], config):
    """Pack every Scratchpad buffer with greedy best-fit, exactly like DRAM.

    With the alloc-only model each on-chip buffer is an explicit
    ``voyager.alloc(SRAM)`` — the input / output slots (a ``[num_slots,
    tile...]`` alloc each) and the reduction scratch — so ``bufs`` already lists
    them with sizes and lifetimes.  A buffer reuses the address of one whose
    lifetime is already dead (across region boundaries).  Slot separation is
    *structural*: distinct slots are distinct allocs, so simultaneously-live
    slots land at distinct addresses automatically — no per-op
    pipelining strategy
    or region grouping is needed.

    A buffer carrying a searched bank group (``meta['bank_group']``, stamped
    by the builders from the tile search's winning partition) is packed as
    part of that group instead: the group's members lie side by side in a
    slot region rounded to whole banks, the region repeats once per pipeline
    slot (a member shallower than the group occupies region 0 only — the
    hole is exactly what the search charged), and the group is placed at a
    bank-aligned base.  Slot ``s`` of every member thus sits in its own whole
    bank(s), so a ping-ponged load never lands in the bank the compute is
    reading, and the byte total matches the search's bank-quantized fit
    check.  A group occupies whole banks at a bank-aligned base, so a loose
    buffer that is address-disjoint is bank-disjoint from it automatically —
    one allocator pass places both.
    """
    bank = config.bank_size
    groups: Dict[tuple, List[Node]] = {}
    if bank:
        for root, buf in bufs.items():
            if (key := _bank_group_key(buf)) is not None:
                groups.setdefault(key, []).append(root)
    in_group = {root for roots in groups.values() for root in roots}

    items = [
        (root, buf.size, buf.def_t, buf.last_t, 1)
        for root, buf in bufs.items()
        if root not in in_group
    ]

    # Slot-major group layout: member offsets within the slot region, the
    # region rounded to whole banks, and the group's depth.
    layouts = {}  # key -> (group_slot_bytes, {root: offset})
    for key, roots in groups.items():
        offsets, offset, depth = {}, 0, 1
        for root in roots:
            payload, slots = _slot_payload(bufs[root], config)
            offsets[root] = offset
            offset += payload
            depth = max(depth, slots)
        group_slot_bytes = math.ceil(offset / bank) * bank
        layouts[key] = (group_slot_bytes, offsets)
        items.append(
            (
                ("bank_group", key),
                depth * group_slot_bytes,
                min(bufs[r].def_t for r in roots),
                max(bufs[r].last_t for r in roots),
                bank,
            )
        )

    bases, total = _greedy_best_fit(items)

    reserved = config.scratchpad_offset
    if reserved:
        bases = {key: base + reserved for key, base in bases.items()}

    for root, buf in bufs.items():
        if root in in_group:
            continue
        base = bases[root]
        seg = Segment(base, base + buf.size, MemoryLevel.SRAM, root)
        for m in buf.members:
            m.meta.setdefault("scratchpad", seg)

    for key, roots in groups.items():
        base = bases[("bank_group", key)]
        group_slot_bytes, offsets = layouts[key]
        for root in roots:
            payload, slots = _slot_payload(bufs[root], config)
            start = base + offsets[root]
            end = start + payload + (slots - 1) * group_slot_bytes
            seg = Segment(start, end, MemoryLevel.SRAM, root)
            for m in bufs[root].members:
                m.meta.setdefault("scratchpad", seg)
                # The member's slots step by the group's region, not by its
                # own payload — ``_stamp_slots`` reads this override.
                if _slots(m):
                    m.meta["bank_group_stride"] = group_slot_bytes

    return int(total) + reserved, {
        "buffers": len(items),
        "bank_groups": len(groups),
    }


def _buf_desc(buf: "_Buf", config) -> str:
    """``name<shape x dtype>`` of the largest tile in a scratchpad buffer (the
    one that set its size), for the ``[MEM_ALLOC_FAIL]`` diagnostic."""
    m = max(buf.members, key=lambda x: _nbytes(x, config))
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
# Slot metadata
# ---------------------------------------------------------------------------


def _stamp_slots(model: GraphModule, config) -> None:
    """Record each pipelined buffer's depth and slot pitch on its ``alloc`` /
    ``zeros`` node, so the code generator can serialize the slot dimension as
    ``bank_count`` / ``bank_stride_bytes`` rather than as a tensor dimension.
    A slot is then addressed ``base + slot * meta['slot_stride']``, which is
    what lets a runtime slot index (``step % num_slots``) stay a runtime value.
    A bank-grouped buffer's slots step by the group's whole slot region
    (``meta['bank_group_stride']``, set by ``_plan_scratchpad``) rather than
    by its own payload.
    """
    for node in _walk(model):
        if not (slots := _slots(node)):
            continue
        node.meta["slot_count"] = slots
        node.meta["slot_stride"] = node.meta.get(
            "bank_group_stride"
        ) or _slot_stride(node, config)


# ---------------------------------------------------------------------------
# Invariant checker
# ---------------------------------------------------------------------------


def _check_overlaps(arena: str, items) -> None:
    """Warn if two simultaneously-live buffers share an address range.
    ``items`` is ``[(name, def_t, last_t, Segment, group), ...]`` —
    overlapping lifetimes must map to disjoint address ranges.  Two members
    of one searched bank group (equal non-``None`` ``group``) are exempt:
    their slot-interleaved segments overlap by construction."""
    for i in range(len(items)):
        n1, a1, b1, s1, g1 = items[i]
        for j in range(i + 1, len(items)):
            n2, a2, b2, s2, g2 = items[j]
            if g1 is not None and g1 == g2:
                continue
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
            dram.append((n.name, pos.get(n, 0), last, seg, None))
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
            scratch.append(
                (root.name, bf.def_t, bf.last_t, seg, _bank_group_key(bf))
            )
    _check_overlaps("Scratchpad", scratch)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def plan_memory(model: GraphModule, config) -> MemoryPlan:
    """Assign concrete DRAM / Scratchpad addresses to a bufferized FX graph.

    Writes ``meta['memory']`` (DRAM) / ``meta['scratchpad']`` (Scratchpad)
    ``Segment``s on each buffer *root* — the ``alloc`` that owns the storage —
    plus ``meta['slot_count']`` / ``meta['slot_stride']`` on a pipelined one,
    and returns the pool sizes.  Nothing is threaded onto the tile sites: a
    tile is named by the buffer it lives in (and, for a slot, a runtime
    slot index), so the address belongs to the buffer, not to every
    reference to it.

    ``config.scratchpad_size`` is the whole physical scratchpad, so the plan is
    compared against it directly: the ping-pong slots this places are already
    part of what it has to hold.

    Raises:
        ValueError: If the scratchpad plan exceeds ``scratchpad_size``, listing
            the buffers live at the peak.  A plan that does not fit describes a
            machine the accelerator does not have, so it is an error rather
            than a warning: the tile search has to pick smaller tiles.
    """
    capacity = config.scratchpad_size

    # Which buffer every name denotes, and the global schedule: both arenas need
    # them (a buffer dies at the last read of *any* of its names), so compute
    # them once.
    buffer_of = _buffer_identity(model)
    dram_bytes = _plan_dram(model, buffer_of, config)
    bufs = _buffer_lifetimes(model, buffer_of, _timestamps(model), config)

    scratchpad_bytes = 0
    peak_region = None
    if capacity is not None:
        scratchpad_bytes, peak_region = _plan_scratchpad(model, bufs, config)

    _stamp_slots(model, config)
    _check_invariants(model, bufs)

    if capacity is not None and scratchpad_bytes > capacity:
        peak_bytes, live = _peak_live_buffers(bufs)
        shown = live[:12]
        detail = "\n".join(
            f"    {_buf_desc(b, config)}  {b.size} B  "
            f"[def={b.def_t} last={b.last_t}]"
            for b in shown
        )
        if len(live) > len(shown):
            detail += f"\n    ... (+{len(live) - len(shown)} more)"
        reserved = config.scratchpad_offset
        note = (
            f" ({reserved} B of it reserved by scratchpad_offset)"
            if reserved
            else ""
        )
        raise ValueError(
            f"[plan_memory] scratchpad plan needs {scratchpad_bytes} bytes"
            f"{note} > scratchpad_size {capacity}; peak concurrency "
            f"{peak_bytes} B across {len(live)} live scratchpad buffers:"
            f"\n{detail}"
        )

    logger.info(
        "Memory plan: DRAM=%d bytes, Scratchpad=%d/%s bytes (peak region %s)",
        dram_bytes,
        scratchpad_bytes,
        capacity,
        peak_region,
    )
    return MemoryPlan(dram_bytes, scratchpad_bytes)
