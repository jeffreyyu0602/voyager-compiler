"""
Baseline memory planner for the bufferized FX path.

The bufferization pass leaves every tensor node tagged with ``meta['space']``
(``"DRAM"`` / ``"Scratchpad"``) but with no concrete address.  This pass turns
those space-annotated buffers/tiles into a concrete map:

  * **DRAM** — the persistent params / inputs are placed first (no reuse), then
    the intermediate ``voyager.alloc`` activation buffers are packed with a
    greedy best-fit, shared-object allocator that reuses a slot whose lifetime
    is disjoint and whose size is closest (least fragmentation).
  * **Scratchpad** — a single fixed ``cache_size`` pool, structured as
    ``num_banks`` banks.  Each top-level compute region (a ``while_loop`` nest
    or an untiled load/compute/store group) is laid out with the existing
    banking strategies (``codegen/banking.py``), which place an op's inputs /
    weight / output in *separate banks* for crossbar bandwidth.  Regions run
    sequentially, so each resets to offset 0 and the pool size is the max region
    footprint.

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

from ..banking import (
    get_banking_strategies_for_op,
    require_allocation,
    _get_scope,
)
from ..mapping import get_node_to_key_map
from ..mapping_utils import is_conv2d, is_gemm_op, is_nop
from ..memory import MemorySpace, Segment, _align_size
from ...pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)

voyager = torch.ops.voyager
_ALLOC = voyager.alloc.default
_ZERO = voyager.zero_tile.default
_LOAD = voyager.load_tile.default
_STORE = voyager.store_tile.default
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
        return 0
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
    # ``getitem``/``store_tile`` is just another handle to the buffer it pulls
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
            elif n.target is _STORE:  # alias: store -> its dest buffer
                buffer_of[n] = buffer_of.get(n.args[1])

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
# Scratchpad planning (bank-aware, region-scoped)
# ---------------------------------------------------------------------------


def _bodies(model: GraphModule, loop: Node) -> List[GraphModule]:
    """The body GraphModule of ``loop`` plus all nested loop / fused-module
    bodies."""
    out: List[GraphModule] = []
    body = _submodule(model, loop.args[1].target)
    if body is None:
        return out
    out.append(body)
    for n in body.graph.nodes:
        if n.op == "call_function" and n.target is _WHILE:
            out += _bodies(body, n)
        elif n.op == "call_module":
            sub = _submodule(body, n.target)
            if sub is not None:
                out.append(sub)
    return out


def _resolve_tile(n: Node, arg_map: Dict[Node, Node]) -> Optional[Node]:
    """Map a reference-op operand to the body ``load_tile`` that produces it: it
    is the operand itself for an in-body ref, or the call_module arg bound to it
    for a tail-fused submodule placeholder."""
    if isinstance(n, Node) and n.op == "call_function" and n.target is _LOAD:
        return n
    return arg_map.get(n)


def _region_nodes(bodies: List[GraphModule]) -> List[Node]:
    """All nodes across a region's bodies (and fused submodules), flattened."""
    return [n for g in bodies for n in g.graph.nodes]


def _region_roles(nodes: List[Node]):
    """Locate a region's reference compute op and its role -> tile map.

    Returns ``(ref, output_node, key_to_node, tile_shapes)`` or ``None``.
    ``ref`` is the GEMM/conv op (preferred) or the first scratchpad-producing
    op; ``output_node`` is the node that carries its output tile (the ref, or
    the call_module that wraps it); ``key_to_node`` maps ``f"{scope}::{role}"``
    to the ``load_tile`` for each operand; ``tile_shapes`` maps each of those
    tiles and the output node to its tile shape.
    """
    ref = next(
        (
            n
            for n in nodes
            if n.op == "call_function" and (is_gemm_op(n) or is_conv2d(n))
        ),
        None,
    )
    if ref is None:
        ref = next(
            (
                n
                for n in nodes
                if n.op == "call_function"
                and n.meta.get("space") == "Scratchpad"
                and n.target not in (_LOAD, _ZERO)
            ),
            None,
        )
    if ref is None:
        return None

    # If ref lives inside a fused call_module, the tiles feed that module; map
    # its placeholders back to the call_module args.
    host = None
    arg_map: Dict[Node, Node] = {}
    for n in nodes:
        if n.op != "call_module":
            continue
        sub = _owning_submodule(n)
        if sub is not None and any(x is ref for x in sub.graph.nodes):
            phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
            arg_map = {
                ph: a for ph, a in zip(phs, n.args) if isinstance(a, Node)
            }
            host = n
            break

    output_node = host if host is not None else ref
    scope = _get_scope(ref.target)
    try:
        node_to_key = get_node_to_key_map(ref)
    except (AttributeError, TypeError):
        # ``normalize_function`` can't resolve some custom-op schemas; without a
        # role map the region falls back to bump allocation (still correct).
        return None
    key_to_node: Dict[str, Node] = {}
    tile_shapes: Dict[Node, tuple] = {}
    for n, role in node_to_key.items():
        if role == "output":
            continue
        tile = _resolve_tile(n, arg_map)
        if tile is not None and _val(tile) is not None:
            key_to_node[f"{scope}::{role}"] = tile
            tile_shapes[tile] = tuple(_val(tile).shape)
    if _val(output_node) is not None:
        tile_shapes[output_node] = tuple(_val(output_node).shape)
    return ref, output_node, key_to_node, tile_shapes


def _owning_submodule(call_module_node: Node) -> Optional[GraphModule]:
    gm = call_module_node.graph.owning_module
    if gm is None:
        return None
    return _submodule(gm, call_module_node.target)


def _scratch_regions(model: GraphModule) -> List[List[Node]]:
    """The compute regions whose tiles share the pad while the region runs: one
    per top-level ``while_loop`` nest, plus one per top-level untiled op (the
    scratchpad cone behind its ``store_tile``).  A cone whose only compute is a
    view / nop (reshape, flatten, ...) is DRAM-resident addressing, not a pad
    occupant, so it is dropped — otherwise it would size the pad by a whole
    untiled tensor."""
    regions: List[List[Node]] = []
    for n in model.graph.nodes:
        if n.op == "call_function" and n.target is _WHILE:
            regions.append(_region_nodes(_bodies(model, n)))
    for store in model.graph.nodes:
        if store.op == "call_function" and store.target is _STORE:
            cone = _untiled_cone(store)
            compute = [
                x
                for x in cone
                if x.op == "call_function" and x.target not in (_LOAD, _ZERO)
            ]
            if compute and all(is_nop(x) for x in compute):
                continue
            regions.append(cone)
    return regions


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
    loop-result ``getitem`` all collapse together.  So a ``zero_tile``
    accumulator threaded through the reduction loop (zero_tile -> body arg ->
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


def _nop_view_tiles(model: GraphModule) -> set:
    """Scratchpad nodes whose only compute is a view / nop (reshape, flatten,
    ...) bufferized as load-whole/store-whole.  These are DRAM-resident
    addressing, not real pad occupants, so they are excluded from scratchpad
    allocation (otherwise a whole untiled tensor would size the pad)."""
    skip: set = set()
    for store in model.graph.nodes:
        if store.op == "call_function" and store.target is _STORE:
            cone = _untiled_cone(store)
            compute = [
                x
                for x in cone
                if x.op == "call_function" and x.target not in (_LOAD, _ZERO)
            ]
            if compute and all(is_nop(x) for x in compute):
                skip.update(cone)
    return skip


def _buffer_lifetimes(model, uf, order, bank_width) -> Dict[Node, _Buf]:
    """Per scratchpad buffer (a union-find root): byte size, birth, and last
    use.  Death follows the alias chain — a use of *any* member (e.g. the
    getitem of a carried accumulator, or the bias-add that reads it) extends the
    lifetime."""
    skip = _nop_view_tiles(model)
    members: Dict[Node, List[Node]] = {}
    for n in _walk(model):
        members.setdefault(uf.find(n), []).append(n)

    bufs: Dict[Node, _Buf] = {}
    for root, mem in members.items():
        tiles = [
            m
            for m in mem
            if m.meta.get("space") == "Scratchpad"
            and _val(m) is not None
            and m not in skip
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


# --- bank-aware placement: group an op's operands by bank, best-fit by time --


@dataclass(eq=False)
class _Group:
    """A bank-aligned scratchpad allocation: one or more buffers an op's banking
    strategy packs into the same bank, with a combined size / lifetime."""

    size: int
    def_t: int
    last_t: int
    members: List[
        Tuple[Node, int]
    ]  # (buffer root, byte offset within the group)
    region: int


def _build_groups(
    region_info, strat_idx, bufs, uf, bank_size, bank_width, unroll_dim
):
    """Turn each region's tiles into bank-aligned ``_Group``s under the
    region's currently-selected banking strategy: the ref op's operands are
    grouped by the bank ``evaluate`` assigns them (same bank -> packed together,
    different banks -> separate groups), and every other scratchpad buffer
    (accumulator / residual / bias) becomes its own group."""
    groups: List[_Group] = []
    placed: set = set()

    def _add(members, region):
        if not members:
            return
        size = max(off + bufs[r].size for r, off in members)
        if bank_size:
            size = int(_align_size(size, bank_size))
        groups.append(
            _Group(
                size,
                min(bufs[r].def_t for r, _ in members),
                max(bufs[r].last_t for r, _ in members),
                members,
                region,
            )
        )

    for rid, (nodes, roles) in enumerate(region_info):
        if roles is not None:
            ref, output_node, key_to_node, tile_shapes = roles
            # ``evaluate`` / ``compute_tensor_size`` read ``node.value``, which
            # body tiles lack (only ``meta['val']``); mirror it across
            # (codegen's later ShapeProp overwrites it).
            for n in tile_shapes:
                if not isinstance(getattr(n, "value", None), torch.Tensor):
                    v = n.meta.get("val")
                    if isinstance(v, torch.Tensor):
                        n.value = v
            strat_list = get_banking_strategies_for_op(ref.target)
            strategy = strat_list[min(strat_idx[rid], len(strat_list) - 1)]
            _total, node_to_seg = strategy.evaluate(
                key_to_node,
                output_node,
                tile_shapes,
                bank_width,
                bank_size,
                unroll_dim,
            )
            by_bank: Dict[int, List[Tuple[Node, int]]] = {}
            for tile, seg in node_to_seg.items():
                root = uf.find(tile)
                if root not in bufs or root in placed:
                    continue
                placed.add(root)
                bank = seg.start // bank_size if bank_size else 0
                off = seg.start % bank_size if bank_size else 0
                by_bank.setdefault(bank, []).append((root, off))
            for members in by_bank.values():
                _add(members, rid)

        # Every remaining scratchpad buffer in the region (accumulator,
        # residual, bias, or the whole region when no banking role was found) ->
        # its own bank.
        for n in nodes:
            if n.meta.get("space") != "Scratchpad":
                continue
            root = uf.find(n)
            if root in placed or root not in bufs:
                continue
            placed.add(root)
            _add([(root, 0)], rid)

    for root in bufs:  # defensive: anything outside a region
        if root not in placed:
            placed.add(root)
            _add([(root, 0)], -1)
    return groups


def _peak_live(groups: List[_Group]) -> Tuple[int, int, List[_Group]]:
    """The timestamp of maximum concurrent scratchpad demand and the groups
    live then.  Used to decide which ops to repack when the plan overflows
    ``cache_size``.
    """
    events = []
    for g in groups:
        events.append((g.def_t, 1, g.size))  # birth
        events.append((g.last_t + 1, 0, g.size))  # death (exclusive)
    cur = best = 0
    best_t = 0
    for t, birth, size in sorted(events):
        cur += size if birth else -size
        if cur > best:
            best, best_t = cur, t
    live = [g for g in groups if g.def_t <= best_t <= g.last_t]
    return best, best_t, live


def _plan_scratchpad(
    model: GraphModule,
    uf: "_UnionFind",
    bufs: Dict[Node, "_Buf"],
    cache_size: int,
    num_banks: Optional[int],
    bank_width: Optional[int],
    unroll_dim: Optional[int],
):
    """Allocate the scratchpad like DRAM: every tile is a buffer with a
    lifetime, and buffers are packed with greedy best-fit so a tile reuses a
    bank whose occupant is already dead — across region boundaries, not reset
    per region.  Bank separation comes from each op's banking strategy (operands
    grouped into bank-aligned ``_Group``s).  If the plan exceeds ``cache_size``,
    the ops live at the peak are repacked with a more-grouped strategy (fewer
    banks) and the placement is redone.
    """
    bank_size = cache_size // num_banks if num_banks else None

    region_info = [
        (nodes, _region_roles(nodes)) for nodes in _scratch_regions(model)
    ]
    strat_idx = [0] * len(region_info)  # most-separated strategy per region

    groups: List[_Group] = []
    bases: Dict[_Group, int] = {}
    total = 0
    for _ in range(64):
        groups = _build_groups(
            region_info, strat_idx, bufs, uf, bank_size, bank_width, unroll_dim
        )
        bases, total = _greedy_best_fit(
            [(g, g.size, g.def_t, g.last_t) for g in groups]
        )
        if total <= cache_size:
            break
        # Repack the ops contributing to the peak with a more-grouped strategy.
        _peak, _t, live = _peak_live(groups)
        changed = False
        for rid in {g.region for g in live if g.region >= 0}:
            _nodes, roles = region_info[rid]
            if roles is None:
                continue
            n_strat = len(get_banking_strategies_for_op(roles[0].target))
            if strat_idx[rid] < n_strat - 1:
                strat_idx[rid] += 1
                changed = True
        if not changed:
            break

    for g in groups:
        base = bases[g]
        for root, off in g.members:
            seg = Segment(
                base + off,
                base + off + bufs[root].size,
                MemorySpace.SCRATCHPAD,
                root,
            )
            for m in bufs[root].members:
                m.meta.setdefault("scratchpad", seg)

    return int(total), {"groups": len(groups), "downgrades": sum(strat_idx)}


def _untiled_cone(store: Node) -> List[Node]:
    """Scratchpad nodes feeding a top-level ``store_tile`` (its load/compute
    group); the backward walk stops at DRAM operands, isolating this op's
    region."""
    cone: List[Node] = []
    seen = set()
    stack = [store.args[0]]
    while stack:
        x = stack.pop()
        if not isinstance(x, Node) or x in seen:
            continue
        seen.add(x)
        if x.meta.get("space") != "Scratchpad":
            continue
        cone.append(x)
        stack.extend(x.all_input_nodes)
    return cone


# ---------------------------------------------------------------------------
# Thread Segments across loop / module boundaries onto the actual tile sites
# ---------------------------------------------------------------------------


def _thread_segments(model: GraphModule) -> None:
    """Propagate ``meta['memory']`` / ``meta['scratchpad']`` from buffer roots
    onto the body placeholders bound to them and onto in-place compute tiles, so
    every ``load_tile`` source / ``store_tile`` dest / op tile resolves to an
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
                elif node.target not in (_LOAD, _ZERO, _STORE, _INCR, _ALLOC):
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
    sizes.  Warns (consistent with the non-bufferized ``[MEM_ALLOC_FAIL]``) if
    the scratchpad plan exceeds ``cache_size``.
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
        logger.warning(
            "[MEM_ALLOC_FAIL] scratchpad plan needs %d bytes > cache_size %d",
            scratchpad_bytes,
            cache_size,
        )

    logger.info(
        "Memory plan: DRAM=%d bytes, Scratchpad=%d/%s bytes (peak region %s)",
        dram_bytes,
        scratchpad_bytes,
        cache_size,
        peak_region,
    )
    return MemoryPlan(dram_bytes, scratchpad_bytes)
