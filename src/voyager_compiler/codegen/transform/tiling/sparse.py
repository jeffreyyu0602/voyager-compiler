"""Joint slice-width choice for outlier-CSR producers and their consumers.

An outlier CSR is emitted per K slice, and its column indices are local to
the slice, so every GEMM consuming it must tile its reduction dim at exactly
that width; a GEMM whose fused tail *emits* a CSR cuts the slices from its
own output tile, so its column tile is the consumers' reduction tile.  One
number therefore couples each producer to its consumers, and a producer can
itself be a consumer (``up_proj -> down_proj -> lm_head``), so the coupled
ops form a forest along the CSR edges: every CSR has one producer, a
producer may fan out.

Searched op by op, each picks the tile that suits itself and the coupling
is imposed afterwards -- a producer's narrow column tile then strangles its
consumers.  Here the slice widths are chosen jointly: every op is searched
under each candidate width it could be handed (its reduction tile) and
could emit (its output tile), and a dynamic program from the leaves to the
roots picks the widths minimizing the summed modeled runtime.  The winners
are registered with the tiler, so the builders' own ``get_tiling`` calls
land on them, and stamped on each producer as ``meta["csr_slice"]``.
"""

import logging
import math
import time

import interstellar
from voyager_compiler.codegen.node_info import (
    csr_consumers,
    csr_quantize_node,
    gemm_produces_csr,
    get_anchor_node,
    is_fully_connected,
    is_gemm_op,
)
from voyager_compiler.codegen.transform.tiling.search import gemv_op_tiling
from voyager_compiler.codegen.transform.tiling.tiler import (
    _finish_search,
    _prepare_search,
    _run_search,
    _search_in_pool,
    get_tiling,
    SEARCH_TIMED_OUT,
    TileConstraint,
)

logger = logging.getLogger(__name__)
le = interstellar.le

# Narrowest CSR slice a consumer is handed when its reduction dim allows
# one: a step's fixed costs (gather DMAs, systolic fill and drain) dwarf the
# matrix work below this, whatever the rest of the tile.
DEFAULT_MIN_CSR_SLICE = 256


def _slice_candidates(producer, consumers, min_slice):
    """Slice widths ``producer``'s CSR may be cut at.

    Divisors of the consumers' reduction dim that hold a power of two of
    microscaling blocks, no narrower than ``min_slice`` -- or the whole dim
    when it is narrower than that.
    """
    block = csr_quantize_node(producer).args[3]
    depth = get_anchor_node(consumers[0]).args[0].value.shape[-1]
    floor = min(min_slice, depth)
    widths = []
    blocks = 1
    while blocks * block <= depth:
        width = blocks * block
        if depth % width == 0 and width >= floor:
            widths.append(width)
        blocks *= 2
    return widths or [depth]


def _constraint(s_in, s_out):
    """The constraint an op searches under when handed a ``s_in``-wide CSR
    and emitting one ``s_out`` wide (either ``None`` = no such CSR)."""
    return TileConstraint(
        exact=() if s_in is None else ((le.IC, s_in),),
        multiple=() if s_out is None else ((le.OC, s_out),),
    )


def _explicit_extent(anchor):
    """``extent(loop)`` of an anchor whose tiles were given outright
    (``meta["l2_tiling"]``), so the planner can tell which widths it admits;
    ``None`` when it carries none."""
    tiling = anchor.meta.get("l2_tiling")
    if tiling is None:
        return None
    n_n = tiling[1]
    n_k = tiling[2] if len(tiling) > 2 else 1
    tiles = {
        le.OC: anchor.value.shape[-1] // n_n,
        le.IC: anchor.args[0].value.shape[-1] // n_k,
    }
    return tiles.__getitem__


def _search_costs(searches, tiler):
    """Modeled runtime of every ``(node, constraint)`` search, in cycles.

    An anchor tiled outright (``l2_tiling``) prices at zero under the
    constraints its tiles satisfy and needs no tiler; a GEMV is priced by
    its own analytic search, inline; the interstellar searches go through
    the fork pool together.  A search that fits nothing prices at infinity.

    Identical layers share a cache key and are searched once; a search the
    pool's deadline cut off is run again here, serially.

    Returns:
        ``(costs, prepared)`` -- the runtime per ``(node, constraint)``, and
        for each interstellar one its ``(key, search, found)`` triple, which
        ``_finish_search`` turns into the cache entry of a winner.
    """
    costs, prepared = {}, {}
    pending = {}  # key -> (search, [(node, constraint), ...])
    for node, constraint in searches:
        anchor = get_anchor_node(node)
        extent = _explicit_extent(anchor)
        if extent is not None:
            costs[node, constraint] = (
                0.0 if constraint.allows(extent) else math.inf
            )
            continue
        if tiler is None:
            costs[node, constraint] = math.inf
            continue
        if is_fully_connected(anchor):
            try:
                gemv_op_tiling(node, tiler.config, constraint)
                costs[node, constraint] = node.meta["tiling_runtime"]
            except RuntimeError:
                costs[node, constraint] = math.inf
            continue
        prepared_search = _prepare_search(node, tiler, constraint)
        if prepared_search is None or prepared_search[1] is None:
            costs[node, constraint] = 0.0  # interstellar skips it
            continue
        key, search = prepared_search
        pending.setdefault(key, (search, []))[1].append((node, constraint))
    keys = list(pending)
    results = _search_in_pool([pending[key][0] for key in keys])
    for key, found in zip(keys, results):
        search, users = pending[key]
        if found is SEARCH_TIMED_OUT:
            try:
                found = _run_search(search)
            except RuntimeError:
                found = None
        for node, constraint in users:
            costs[node, constraint] = (
                found[1] if found is not None else math.inf
            )
            prepared[node, constraint] = (key, search, found)
    for (node, constraint), cost in costs.items():
        logger.debug("[csr] %s under %s: %.0f", node.name, constraint, cost)
    return costs, prepared


def plan_csr_slices(nodes, tiler, min_slice=DEFAULT_MIN_CSR_SLICE):
    """Choose every CSR's slice width jointly with the ops it couples.

    Args:
        nodes: The graph's nodes, before bufferization.
        tiler: The ``TilerContext``; receives the winners' constraints and
            their finished searches.  ``None`` when every coupled op is
            tiled outright.
        min_slice: Narrowest slice a consumer is handed (see
            ``DEFAULT_MIN_CSR_SLICE``).

    Raises:
        RuntimeError: No candidate width lets every op of some CSR tree fit.
    """
    consumers_of = {}
    for node in nodes:
        if csr_quantize_node(node) is None:
            continue
        consumers = csr_consumers(node)
        if consumers:
            consumers_of[node] = consumers
    producers = list(consumers_of)
    if not producers:
        return
    producer_of = {c: p for p in producers for c in consumers_of[p]}
    candidates = {
        p: _slice_candidates(p, consumers_of[p], min_slice) for p in producers
    }

    def is_searched(node):
        return is_gemm_op(get_anchor_node(node))

    def constraints_of(node):
        """Every constraint ``node``'s search may run under: each slice it
        could be handed times each it could emit."""
        ins = candidates[producer_of[node]] if node in producer_of else [None]
        outs = candidates[node] if node in consumers_of else [None]
        return [_constraint(s_in, s_out) for s_in in ins for s_out in outs]

    coupled = [n for n in nodes if n in producer_of or n in consumers_of]
    searches = [
        (node, constraint)
        for node in coupled
        if is_searched(node)
        for constraint in constraints_of(node)
    ]
    start = time.perf_counter()
    costs, prepared = _search_costs(searches, tiler)
    logger.info(
        "[csr] %d constrained searches over %d coupled ops in %.1fs",
        len(searches),
        len(coupled),
        time.perf_counter() - start,
    )

    def op_cost(node, constraint):
        return costs.get((node, constraint), 0.0) if is_searched(node) else 0.0

    best = {}

    def subtree(node, s_in):
        """Least summed runtime of ``node`` and everything downstream of its
        CSR, given the slice it is handed; memoized with its choice."""
        if (node, s_in) in best:
            return best[node, s_in][0]
        outs = candidates[node] if node in consumers_of else [None]
        choice = (math.inf, None)
        for s_out in outs:
            total = op_cost(node, _constraint(s_in, s_out))
            for child in consumers_of.get(node, ()):
                total += subtree(child, s_out)
            logger.debug(
                "[csr] %s handed %s emitting %s: subtree %.0f",
                node.name,
                s_in,
                s_out,
                total,
            )
            if total < choice[0]:
                choice = (total, s_out)
        best[node, s_in] = choice
        return choice[0]

    roots = [p for p in producers if p not in producer_of]
    for root in roots:
        if math.isinf(subtree(root, None)):
            raise RuntimeError(
                f"{root.name}: no CSR slice width in "
                f"{candidates[root]} fits every op coupled to it"
            )
        _commit(root, None, None, best, prepared, tiler, consumers_of)


def _commit(node, s_in, rows, best, prepared, tiler, consumers_of):
    """Register the winning constraint of ``node`` and its subtree.

    ``rows`` is the row block of the CSR ``node`` is handed (``None`` for a
    root, or below a row-swept producer, which sizes its blocks to its
    consumers): the node's row tile is aligned to it before its own
    consumers are committed against the row tile it ends up with.
    """
    total, s_out = best[node, s_in]
    constraint = _constraint(s_in, s_out)
    anchor = get_anchor_node(node)
    if is_gemm_op(anchor) and tiler is not None:
        tiler.constraints[anchor] = constraint
        if (node, constraint) in prepared:
            key, search, found = prepared[node, constraint]
            if key not in tiler.cache:
                tiler.cache[key] = _finish_search(search, found)
        if rows is not None:
            _align_rows(node, rows, tiler)
    if s_out is not None:
        node.meta["csr_slice"] = s_out
        logger.info(
            "[csr] %s: slice width %d (subtree runtime %.0f)",
            node.name,
            s_out,
            total,
        )
        block = None
        if gemm_produces_csr(node) and tiler is not None:
            block = _row_tile(node, tiler)
        for child in consumers_of[node]:
            _commit(child, s_out, block, best, prepared, tiler, consumers_of)


def _row_tile(node, tiler):
    """The row tile ``node``'s registered tiling gives it."""
    anchor = get_anchor_node(node)
    counts = get_tiling(node, tiler)[0]
    return anchor.args[0].value.shape[-2] // counts[-3]


def _align_rows(consumer, rows, tiler):
    """Make ``consumer``'s row tile a whole number of ``rows``.

    A GEMM epilogue emits its CSR in blocks of its own row tile, and a
    consumer gathers whole blocks, so its row tile must be a multiple.  The
    dynamic program does not see the producer's row tile (a search result),
    so a consumer that lands short of it is searched again under the same
    slice width plus that alignment.

    Raises:
        RuntimeError: No aligned tiling of the consumer fits on chip.
    """
    if _row_tile(consumer, tiler) % rows == 0:
        return
    anchor = get_anchor_node(consumer)
    constraint = tiler.constraints[anchor].merged(
        TileConstraint(multiple=((le.OX, rows),))
    )
    tiler.constraints[anchor] = constraint
    prepared = _prepare_search(consumer, tiler, constraint)
    if prepared is None or prepared[1] is None:
        return  # a GEMV or an explicit tiling: ``get_tiling`` applies it
    key, search = prepared
    try:
        found = _run_search(search)
    except RuntimeError as e:
        raise RuntimeError(
            f"{consumer.name}: no tiling with a row tile of whole "
            f"{rows}-row CSR blocks fits on chip"
        ) from e
    tiler.cache[key] = _finish_search(search, found)
    logger.info(
        "[csr] %s: row tile realigned to %d-row blocks", consumer.name, rows
    )
