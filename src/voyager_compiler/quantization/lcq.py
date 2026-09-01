"""Learned codebook quantization: fit a tensor's own levels.

A codebook is ``k`` distinct integers in ``[quant_min, quant_max]``, indexed
by ``log2(k)`` bits, so the stored element keeps its width and only the
levels it decodes to change.  It applies not to a tensor's raw values but to
``x / blockscale`` under microscaling, so every routine below works in that
normalized space.  How many levels there are, and what range they span, is
the caller's to say -- nothing here assumes a size or a grid.

Two solvers fit the levels to measured samples: ``weighted_lloyd`` refines
from a seed, and ``optimal_codebook`` returns the exact weighted-MSE minimum
by dynamic programming.  Weighting each sample by its channel's downstream
energy is what makes either minimize layer-output error rather than tensor
MSE.  Squared error alone is blind to zero and will spend that level
elsewhere, which under block scaling forces a block's many small values to
at least one step off zero -- pass ``pin_zero`` to forbid it.
"""

import dataclasses
import enum
import json
import logging
import math
import os

import numpy as np
import torch

from voyager_compiler.export_utils import get_node_name_to_scope
from voyager_compiler.ops.quantized import QMAP_SIZE, vmap
from voyager_compiler.quantization.dtypes import create_normal_map

logger = logging.getLogger(__name__)

EPS = 1e-12

#: Bins the range is divided into when a tensor's statistics are
#: accumulated.  The solver's candidate levels are integers, so this only
#: has to be fine enough that a bin boundary never falls where a decision
#: does; 4001 puts about 63 bins between consecutive integers.
HISTOGRAM_BINS = 4001

#: Elements one ``Histogram.add`` converts to float64 at a time.  Widening
#: an operand and forming the two products is the accumulator's whole
#: memory cost, and a model's largest weight runs to half a billion
#: elements, so the pass is taken in slices rather than whole.
ACCUMULATE_CHUNK = 1 << 25


@dataclasses.dataclass
class Histogram:
    """Weighted moments of one tensor's normalized values, per bin.

    The exact solver reduces whatever it is handed to three sums per bin --
    total weight, weighted sum, weighted sum of squares -- and reads nothing
    else.  Accumulating those directly over every element is therefore not
    an approximation of sampling but a replacement for it: no draw to vary
    between runs, no bound on how much data one fit may see, and a few
    kilobytes of state per tensor instead of gigabytes of retained samples.
    A fourth sum, the unweighted element count, is kept alongside them: the
    solver never reads it, but it says how many elements each bin's weight
    was gathered from, which is what separates a heavy bin from a crowded
    one.

    Attributes:
        quant_min: Smallest representable value, and the low edge.
        quant_max: Largest representable magnitude, and the high edge.
        bins: Resolution the range is divided into.
    """

    quant_min: float
    quant_max: float
    bins: int = HISTOGRAM_BINS

    def __post_init__(self):
        self.weight = None
        self.total = None
        self.square = None
        self.count = None

    def edges(self):
        """Return the bin boundaries the sums are taken between."""
        return np.linspace(
            self.quant_min - 0.5, self.quant_max + 0.5, self.bins + 1
        )

    def add(self, values, weights):
        """Accumulate one tensor's normalized values into the bins.

        Args:
            values: Flat tensor of ``x / blockscale``.
            weights: Per-element weight, broadcastable to ``values``.
        """
        if self.weight is None:
            self.weight, self.total, self.square = (
                torch.zeros(
                    self.bins, dtype=torch.float64, device=values.device
                )
                for _ in range(3)
            )
            self.count = torch.zeros(
                self.bins, dtype=torch.int64, device=values.device
            )
        edges = torch.linspace(
            self.quant_min - 0.5,
            self.quant_max + 0.5,
            self.bins + 1,
            device=values.device,
            dtype=torch.float32,
        )
        values = values.reshape(-1)
        weights = weights.reshape(-1).expand_as(values)
        for start in range(0, values.numel(), ACCUMULATE_CHUNK):
            piece = values[start : start + ACCUMULATE_CHUNK]
            share = weights[start : start + ACCUMULATE_CHUNK]
            index = (torch.bucketize(piece, edges, right=True) - 1).clamp_(
                0, self.bins - 1
            )
            # Float64 so that the order the bins are summed in -- which on a
            # GPU is decided by the scheduler -- cannot move a level.  That
            # same indifference to order is what lets the pass be sliced.
            piece = piece.double()
            share = share.double()
            self.weight.index_add_(0, index, share)
            self.total.index_add_(0, index, share * piece)
            self.square.index_add_(0, index, share * piece * piece)
            self.count += torch.bincount(index, minlength=self.bins)

    def cumulative(self):
        """Return the three running sums the segment cost is read from."""
        return tuple(
            np.concatenate([[0], moment.cpu().numpy().cumsum()])
            for moment in (self.weight, self.total, self.square)
        )


#: Candidates per level when fitting on the float grid.  The dynamic
#: program holds a cube of this many candidates, so the cost is n^3; an odd
#: count keeps an exact zero available for ``pin_zero``.
FLOAT_RESOLUTION = 257


class CodebookGrid(enum.Enum):
    """Where a fitted level is allowed to sit.

    ``INTEGER`` restricts every level to the integer grid the PE array
    decodes, which is what a deployed codebook must be.  ``FLOAT`` leaves
    them at full precision, which is how NormalFloat and its relatives are
    defined before anything projects them onto hardware -- useful for
    measuring what the projection itself costs, and for a format whose
    levels are stored as floats.
    """

    INTEGER = "integer"
    FLOAT = "float"


def block_scales(x, block_size, quant_max, scale_qmap=None):
    """Return the scale each microscaling block is divided by.

    Args:
        x: Tensor whose last axis is the contraction axis.
        block_size: Elements sharing one scale.
        quant_max: Largest magnitude the codebook can represent.
        scale_qmap: Codebook the scale is itself quantized into, as
            deployment quantizes it.  None returns the exact
            ``amax / quant_max``, which is not the number the decoder
            divides by: it can only store a scale its own codebook holds.

    Returns:
        One scale per block, shaped ``[blocks, 1]``.
    """
    amax = x.float().reshape(-1, block_size).abs().amax(-1, keepdim=True)
    if scale_qmap is None:
        return amax.clamp_min(EPS) / quant_max
    # Matching ``calculate_mx_qparam``, down to what it does with a scale
    # that underflows its codebook: replace it with one, not with a floor.
    scale = vmap(amax / quant_max, scale_qmap)
    return torch.where(scale > 0.0, scale, 1.0)


def unit_scale(x, block_size, quant_max, scale_qmap=None):
    """Return ``x / blockscale`` under block-amax microscaling.

    Args:
        x: Tensor whose last axis is the contraction axis.
        block_size: Elements sharing one scale.
        quant_max: Largest magnitude the codebook can represent.
        scale_qmap: Codebook the block scale is quantized into; see
            ``block_scales``.

    Returns:
        Tensor shaped like ``x``.  Its per-block maximum is ``quant_max``
        under an exact scale, and near it under a quantized one.
    """
    x = x.float()
    scale = block_scales(x, block_size, quant_max, scale_qmap)
    return (x.reshape(-1, block_size) / scale).reshape(x.shape)


def to_integer_codebook(values, quant_max, quant_min=None):
    """Project float levels onto distinct integers, endpoints pinned.

    The block maximum always lands on the outermost level, so both endpoints
    are forced to the range ends; letting them float measured worse.

    Args:
        values: Candidate levels, any order.
        quant_max: Largest representable magnitude.
        quant_min: Smallest representable value, or None for the symmetric
            ``-quant_max``.  Zero for a tensor that is never negative, whose
            codebook would otherwise spend half its levels on values that
            cannot occur.

    Returns:
        Sorted list of distinct integers of the same length as ``values``.

    Raises:
        ValueError: The range holds fewer integers than there are levels.
    """
    low = int(-quant_max if quant_min is None else quant_min)
    high = int(quant_max)
    if len(values) > high - low + 1:
        raise ValueError(
            f"{len(values)} levels cannot be distinct integers in "
            f"[{low}, {high}], which holds only {high - low + 1}"
        )
    # Ascending, each level at least one above the last, so a crowd rounding
    # onto the same integer spreads upwards instead of collapsing.
    out = []
    for value in sorted(int(round(v)) for v in values):
        out.append(max(value, low if not out else out[-1] + 1))
    # Spreading upwards can run past the top, so walk back down leaving room
    # for the levels above.  Both endpoints sit on the range ends: the block
    # maximum always lands on the outermost level.
    for index in reversed(range(len(out))):
        out[index] = min(out[index], high - (len(out) - 1 - index))
    out[0], out[-1] = low, high
    return out


def normal_float_levels(k, quant_max, grid=CodebookGrid.INTEGER):
    """Return NormalFloat's levels, scaled to the codebook's range.

    Args:
        k: Number of levels; NormalFloat is defined for a power of two.
        quant_max: Magnitude the outermost level sits at.
        grid: Whether to project onto integers or keep full precision.

    Returns:
        ``k`` ascending distinct levels, the seed every fit starts from.
    """
    levels = (create_normal_map(k=int(math.log2(k))) * quant_max).tolist()
    if grid is CodebookGrid.FLOAT:
        return levels
    return to_integer_codebook(levels, quant_max, -quant_max)


def weighted_lloyd(
    u,
    weights,
    k,
    quant_max,
    quant_min=None,
    iters=15,
    init=None,
    grid=CodebookGrid.INTEGER,
):
    """Fit a codebook by weighted Lloyd-Max on normalized samples.

    Weighting each sample by its input channel's downstream energy makes the
    fit minimize layer-output error rather than tensor MSE, which is where
    most of the accuracy comes from.

    Args:
        u: Flat tensor of ``x / blockscale`` samples.
        weights: Per-sample weights, same shape as ``u``.
        iters: Alternating assign/update rounds.
        k: Number of levels, used only when ``init`` is not given.
        init: Starting levels; defaults to NormalFloat, or to a uniform
            ladder when the range is not symmetric, which NF4 cannot seed.
        quant_max: Largest representable magnitude.
        quant_min: Smallest representable value; see ``to_integer_codebook``.
        grid: Whether the levels are snapped to integers or left at full
            precision.

    Returns:
        The fitted codebook, ascending and distinct: integers on the
        integer grid, floats on the float grid.
    """
    quant_min = -quant_max if quant_min is None else quant_min
    if init is None:
        init = (
            normal_float_levels(k, quant_max)
            if quant_min == -quant_max
            else torch.linspace(quant_min, quant_max, k).tolist()
        )
    codebook = torch.tensor(init, dtype=torch.float32, device=u.device)
    weights = (weights / weights.mean().clamp_min(EPS)).clamp(0, 100)
    for _ in range(iters):
        assignment = (u.unsqueeze(-1) - codebook).abs().argmin(-1)
        for level in range(codebook.numel()):
            member = assignment == level
            if member.any():
                codebook[level] = (u[member] * weights[member]).sum() / weights[
                    member
                ].sum().clamp_min(EPS)
        codebook = codebook.sort().values
    levels = codebook.tolist()
    if grid is CodebookGrid.INTEGER:
        return to_integer_codebook(levels, quant_max, quant_min)
    # The block maximum lands on the outermost level either way, so both
    # ends are pinned on the float grid too.
    levels[0], levels[-1] = float(quant_min), float(quant_max)
    return levels


def codebook_qmap(entries, device=None):
    """Build the 65536-entry bf16 lookup table a fake-quant consumes.

    Args:
        entries: The codebook's integers.
        device: Where to build the table.

    Returns:
        A bf16 tensor mapping every bf16 bit pattern to its nearest entry.
    """
    codebook = torch.tensor(entries, dtype=torch.float32, device=device)
    every = (
        torch.arange(2**16, dtype=torch.int16, device=device)
        .view(torch.bfloat16)
        .float()
    )
    index = (
        (every.clamp(codebook.min(), codebook.max()).unsqueeze(-1) - codebook)
        .abs()
        .argmin(-1)
    )
    return codebook[index].to(torch.bfloat16)


def optimal_codebook(
    histogram,
    k,
    pin_zero=False,
    grid=CodebookGrid.INTEGER,
    resolution=FLOAT_RESOLUTION,
):
    """Return the globally optimal codebook, by dynamic programming.

    Levels are ``k`` distinct candidates in ``[quant_min, quant_max]`` --
    every integer, or ``resolution`` points across the range -- and
    nearest-neighbour
    assignment puts the boundary between two chosen levels at their midpoint.
    Cluster membership is therefore an interval, which makes the exact
    minimizer of the weighted squared error a shortest path over
    ``(previous level, current level)`` states -- no initialization, no local
    minima.  Lloyd-Max lands 2-7% above this.

    Args:
        histogram: Accumulated weighted moments of the tensor's
            ``x / blockscale`` values.  A range starting at zero already
            pins the zero level, so ``pin_zero`` has nothing left to do and
            is ignored.
        k: Number of levels.
        grid: Whether a level may sit only on an integer or anywhere in the
            range at full precision.
        resolution: Candidates spanning the range on the float grid.  The
            search holds a cube of them, so its cost and memory grow as the
            cube; unused on the integer grid, whose candidates are fixed.
        pin_zero: Force the level ``0`` into the codebook.  Minimizing
            squared error alone almost always spends that level
            elsewhere, which leaves a block's many small values unable
            to quantize to zero.

    Returns:
        The optimal codebook as ``k`` ascending distinct levels.

    Raises:
        ValueError: Fewer candidates in the range than levels asked for, so
            no codebook of this size exists on this grid.
        RuntimeError: The shortest path did not reconstruct to ``k``
            distinct levels, which is a defect in the search rather than a
            property of the data.
    """
    quant_min, quant_max = histogram.quant_min, histogram.quant_max
    bins = histogram.bins
    candidates = (
        int(quant_max) - int(quant_min) + 1
        if grid is CodebookGrid.INTEGER
        else resolution
    )
    if k > candidates:
        raise ValueError(
            f"a {k}-level codebook cannot be drawn from the {candidates} "
            f"candidates in [{quant_min}, {quant_max}]"
        )
    if k <= 2:
        # Both endpoints are pinned, so there is nothing left to choose.
        ends = [float(quant_min), float(quant_max)][:k]
        if grid is CodebookGrid.FLOAT:
            return ends
        return to_integer_codebook(ends, quant_max, quant_min)
    edges = histogram.edges()
    cum_w, cum_s, cum_q = histogram.cumulative()

    levels = (
        np.arange(int(quant_min), int(quant_max) + 1)
        if grid is CodebookGrid.INTEGER
        else np.linspace(quant_min, quant_max, resolution)
    )
    n = levels.size
    midpoint = (levels[:, None] + levels[None, :]) / 2.0
    at = np.clip(
        np.searchsorted(edges, midpoint.ravel(), side="right") - 1, 0, bins
    ).reshape(n, n)

    def segment_cost(lo_bin, hi_bin, level):
        span_w = cum_w[hi_bin] - cum_w[lo_bin]
        span_s = cum_s[hi_bin] - cum_s[lo_bin]
        span_q = cum_q[hi_bin] - cum_q[lo_bin]
        return span_q - 2 * level * span_s + level * level * span_w

    # step[p, c, m]: cost of the samples level c owns when its neighbours are
    # p below and m above.
    step = segment_cost(at[:, :, None], at[None, :, :], levels[None, :, None])
    order = levels[:, None, None] < levels[None, :, None]
    step = np.where(
        order & (levels[None, :, None] < levels[None, None, :]), step, np.inf
    )

    # A second state axis records whether level 0 has been chosen yet, so
    # `pin_zero` can require it without giving up the exact minimum.  Without
    # the constraint that axis has length one and the recurrence is unchanged.
    first, last = 0, n - 1
    zero = int(np.argmin(np.abs(levels)))
    # Pinning is meaningless when zero is already the lowest level, since
    # that one is always chosen, or when the grid holds no exact zero.
    pin_zero = pin_zero and zero != first and levels[zero] == 0
    state = np.full((2 if pin_zero else 1, n, n), np.inf)
    state[0, first, :] = segment_cost(0, at[first, :], levels[first])
    state[0, first, first] = np.inf
    if pin_zero:
        state[1, first, zero] = state[0, first, zero]
        state[0, first, zero] = np.inf

    back_level, back_flag = [], []
    for _ in range(k - 3):
        total = state[:, :, :, None] + step[None]
        level_choice = np.argmin(total, axis=1)
        state = np.take_along_axis(total, level_choice[:, None], axis=1)[:, 0]
        flag_choice = np.zeros(state.shape, dtype=np.intp)
        if pin_zero:
            # Every level but 0 carries its flag through; 0 sets it, and is
            # reached from whichever flag was cheaper.
            flag_choice[1] = 1
            source = np.argmin(state[:, :, zero], axis=0)
            rows = np.arange(n)
            reached = state[source, rows, zero]
            # The level backpointer has to follow the same flag the cost was
            # taken from, or the reconstructed path is not the one measured.
            reached_level = level_choice[source, rows, zero]
            state[0, :, zero] = np.inf
            state[1, :, zero] = reached
            level_choice[1, :, zero] = reached_level
            flag_choice[1, :, zero] = source
        back_level.append(level_choice)
        back_flag.append(flag_choice)

    tail = segment_cost(at[:, last], bins, levels[last])
    final = state + step[None, :, :, last] + tail[None, None, :]
    if pin_zero:
        final[0] = np.inf
    flag, prev, cur = np.unravel_index(np.argmin(final), final.shape)

    chosen = [last, int(cur), int(prev)]
    for level_choice, flag_choice in zip(
        reversed(back_level), reversed(back_flag)
    ):
        step_back = int(level_choice[flag, prev, cur])
        flag = int(flag_choice[flag, prev, cur])
        chosen.append(step_back)
        prev, cur = step_back, prev
    chosen = sorted({float(levels[i]) for i in chosen})
    if len(chosen) != k:
        raise RuntimeError(
            f"the search returned {len(chosen)} distinct levels for a "
            f"{k}-level codebook"
        )
    if grid is CodebookGrid.FLOAT:
        return chosen
    return to_integer_codebook(chosen, quant_max, quant_min)


def _contraction_last(tensor, ch_axis):
    """Return the tensor with its contraction axis last.

    Args:
        tensor: The operand being quantized.
        ch_axis: Axis the microscaling blocks run along, as the spec gives
            it -- an int, or a one-element sequence.

    Returns:
        The tensor, transposed if its blocks do not already run along the
        last axis.
    """
    axis = ch_axis[0] if isinstance(ch_axis, (tuple, list)) else ch_axis
    if axis in (-1, tensor.dim() - 1):
        return tensor
    return tensor.transpose(axis, -1)


def _channel_energy(tensor, ch_axis):
    """Return the mean square of each channel along the contraction axis.

    This is what turns a tensor-MSE fit into an output-error fit: an error
    on channel ``j`` reaches the output scaled by the *other* operand's
    magnitude on ``j``, so samples are weighted by that.

    Args:
        tensor: The partner operand.
        ch_axis: Axis its microscaling blocks run along.

    Returns:
        One non-negative number per channel.
    """
    flat = _contraction_last(tensor.detach().float(), ch_axis)
    return flat.reshape(-1, flat.shape[-1]).pow(2).mean(0)


#: Ops that contract their first two arguments against each other.  Only
#: there is a codebook's job defined: the block runs along the contraction
#: axis, and the error a level makes reaches the output scaled by the other
#: operand.  The quantizer also annotates elementwise ops -- a residual add,
#: a softmax -- whose operands have neither.  Any remaining argument is a
#: bias or a stride, not an operand.
CONTRACTED_OPS = (
    torch.ops.aten.linear.default,
    torch.ops.aten.matmul.default,
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
)


def _quantized_operands(model, skip):
    """Find every tensor a prepared model quantizes through a codebook.

    Enumerating the fake-quants is what makes the search exhaustive: a
    module whose dtype keeps its levels in a table is one this can fit, and
    no traversal of compute nodes can miss one.  The op reading the tensor
    is consulted only to name it and to find the partner whose channel
    energy weights the fit.

    Args:
        model: Graph module from ``prepare_pt2e``.
        skip: Substrings of a name that exclude it.

    Returns:
        ``name -> (fake-quant module, operand node, partner node)``.
        Operands sharing a fake-quant appear once, under the first name
        found: the table belongs to the module, so they cannot hold
        different ones.
    """
    modules = dict(model.named_modules())
    scopes = get_node_name_to_scope(model)
    nodes = list(model.graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}

    # Which call of an op within its module a node is -- what tells Q @ K^T
    # from P @ V, neither of which has a weight to be named after.
    order, seen = {}, {}
    for node in nodes:
        if node.target in CONTRACTED_OPS:
            scope = scopes.get(node.name, ("", None, 0))[0]
            order[node] = seen.get((scope, node.target), 0)
            seen[(scope, node.target)] = order[node] + 1

    found, shared, loose = {}, set(), []
    for node in nodes:
        if node.op != "call_module":
            continue
        module = modules.get(node.target)
        if not getattr(module, "is_codebook_quantization", False):
            continue
        consumers = [
            user
            for user in sorted(node.users, key=position.__getitem__)
            if user.target in CONTRACTED_OPS and node in user.args[:2]
        ]
        if not consumers:
            loose.append(node.target)
            continue
        consumer = consumers[0]
        pair = consumer.args[:2]
        index = pair.index(node)
        scope = scopes.get(consumer.name, ("", None, 0))[0]
        name = _operand_name(scope, consumer, order[consumer], index)
        if any(excluded in name for excluded in skip) or module in shared:
            continue
        shared.add(module)
        # One tensor can feed several ops -- a normalized hidden state
        # reaches every projection reading it -- and its error reaches the
        # output through all of them, so every partner is kept.
        partners = tuple(
            user.args[:2][1 - user.args[:2].index(node)] for user in consumers
        )
        found[name] = (module, node, partners)

    if loose:
        logger.warning(
            "%d codebooks sit on a tensor no contracted op reads, so there "
            "is no contraction axis to block along and no partner to weight "
            "by, and they are left at their seed: %s%s",
            len(loose),
            ", ".join(loose[:3]),
            " ..." if len(loose) > 3 else "",
        )
    return found


#: How the two operands of an attention matmul are named, by which matmul
#: in the module it is and which side of it: Q @ K^T then P @ V.
ATTENTION_ROLES = {(0, 0): "q", (0, 1): "k", (1, 0): "p", (1, 1): "v"}


def _operand_name(scope, consumer, order, position):
    """Name one quantized operand, stably across runs.

    A linear's or conv's operands are the activation and the weight, so they
    take the module's own path.  A matmul of two activations has no weight to
    name them after, so it takes the enclosing module's path and its
    position.

    Args:
        scope: Module path export recorded for the consuming node.
        consumer: The contracted op reading the operand.
        order: Which call of that op within the module it is.
        position: Which of the consumer's two operands this is, 0 or 1.

    Returns:
        The operand's name, of the form ``<module path>|<role>``.
    """
    if consumer.target is not torch.ops.aten.matmul.default:
        return f"{scope}|{'act' if position == 0 else 'wgt'}"
    role = ATTENTION_ROLES.get((order, position))
    return f"{scope}|{role or f'mm{order}.{position}'}"


def _accumulate(
    tensor,
    module,
    energy,
    histogram,
    scale_weighted=False,
    quantized_scale=False,
):
    """Bin one operand's normalized values, weighted by its partner.

    Every element is counted, so what the fit sees is the tensor itself
    rather than a draw from it.  An operand whose fake-quant holds an
    outlier threshold is filtered the way deployment filters it: a value
    at or past the threshold leaves the block maximum and gets zero
    weight, so the fit sees the distribution the codebook will quantize.

    Args:
        tensor: The operand, as it enters its fake-quant.
        module: That fake-quant, which knows the block geometry.
        energy: The partner's per-channel energy, or None for a flat fit.
        histogram: Where the weighted moments accumulate.
        scale_weighted: Also weight each element by its block's squared
            scale.  The levels are fitted in normalized space, but
            deployment multiplies the residual back by the block scale,
            so a block's contribution to the output error grows with that
            scale squared.  Without it every block votes equally on where
            the levels sit, however large its values are.
        quantized_scale: Normalize by the scale deployment can store --
            the block maximum pushed through the scale codebook -- rather
            than the exact ``amax / quant_max``.  The decoder divides by
            the former, so the latter places the levels against values it
            never sees.
    """
    ordered = _contraction_last(tensor.detach(), module.ch_axis)
    if ordered.numel() < module.block_size:
        return
    threshold = getattr(module, "outlier_threshold", None)
    if isinstance(threshold, torch.Tensor) and threshold.numel() == 0:
        threshold = None
    kept = None
    if threshold is not None:
        kept = ordered.abs() < threshold
        ordered = ordered * kept
    # One scale serves both the normalization and the squared-scale weight,
    # so the two halves of the objective refer to the same number.
    scale = block_scales(
        ordered,
        module.block_size,
        module.quant_max,
        getattr(module, "scale_qmap", None) if quantized_scale else None,
    )
    flat = ordered.float().reshape(-1, module.block_size)
    normalized = (flat / scale).reshape(-1)
    if energy is None:
        weights = torch.ones_like(normalized)
    else:
        # The contraction axis is last, so an element's channel is its
        # position modulo the channel count -- which is what tiling the
        # energy across the flattened tensor spells.
        weights = energy.repeat(normalized.numel() // energy.numel())
    if kept is not None:
        weights = weights * kept.reshape(-1)
    if scale_weighted:
        weights = weights * (scale**2).expand_as(flat).reshape(-1)
    histogram.add(normalized, weights)


def _resolve_counts(counts, shape):
    """Align a granularity tuple to a shape and resolve its ``-1`` entries.

    The tuple aligns to the trailing axes the way ``Tensor.expand``'s
    does, so a caller names only the axes it means to split.

    Args:
        counts: Tables along each trailing axis -- ``-1`` for one per
            item, ``1`` to share one table across the whole axis.
        shape: The operand's shape.

    Returns:
        One count per axis of ``shape``, every ``-1`` resolved.

    Raises:
        ValueError: More counts than axes, a count below ``-1`` or zero,
            or a count that does not divide the axis it splits.
    """
    if len(counts) > len(shape):
        raise ValueError(
            f"granularity {tuple(counts)} names {len(counts)} axes, but "
            f"the tensor reaching it has {len(shape)}"
        )
    resolved = [1] * (len(shape) - len(counts)) + list(counts)
    for axis, count in enumerate(resolved):
        if count == -1:
            resolved[axis] = shape[axis]
        elif count < 1:
            raise ValueError(
                f"granularity {tuple(counts)} asks for {count} tables "
                f"along axis {axis}; use -1 for one per item"
            )
        elif shape[axis] % count:
            raise ValueError(
                f"granularity {tuple(counts)} splits axis {axis} into "
                f"{count}, which does not divide its {shape[axis]} items"
            )
    return tuple(resolved)


def _grid_cells(tensor, counts):
    """Split a tensor into the grid its granularity asks for.

    Args:
        tensor: The operand.
        counts: One resolved table count per axis.

    Returns:
        ``(index, slab)`` per cell in row-major order, where ``index``
        gives the cell's position along every axis.
    """
    cells = [((0,) * len(counts), tensor)]
    for axis, count in enumerate(counts):
        if count == 1:
            continue
        cells = [
            (index[:axis] + (position,) + index[axis + 1 :], piece)
            for index, slab in cells
            for position, piece in enumerate(slab.chunk(count, axis))
        ]
    return cells


def _padded_counts(counts, rank, head_axis):
    """Widen an older dump's table counts to one per operand axis.

    Args:
        counts: Counts as the dump spells them -- the head count alone,
            or a grid over the trailing axes.
        rank: The operand's rank, or None when the graph does not say.
        head_axis: The counts name the attention head axis, which leads
            from the front, rather than trailing axes.

    Returns:
        One count per axis, 1 where the dump splits nothing.
    """
    if rank is None or rank <= len(counts):
        return list(counts)
    if head_axis:
        padded = [1] * rank
        # Attention operands all enter their matmul as ``[batch, head,
        # ...]``, which is the axis those dumps indexed.
        padded[1] = counts[0]
        return padded
    return [1] * (rank - len(counts)) + list(counts)


def leading_split(counts):
    """Return the leading axis a granularity splits, or None.

    Counts align to the trailing axes, so anything split before the last
    two is a leading axis, along which a partner's energy has to be
    sliced to match.  Which it is settles from the tuple alone, without
    the operand's rank.

    Args:
        counts: A granularity tuple, as the caller wrote it.

    Returns:
        The split leading axis as an offset from the end, or None when
        the tuple splits only the last two axes.

    Raises:
        ValueError: More than one leading axis is split.  The partner
            energy is measured along a single axis, so the fit has no
            way to weight a cell split across two.
    """
    lead = [
        axis - len(counts)
        for axis, count in enumerate(counts[:-2])
        if count != 1
    ]
    if not lead:
        return None
    if len(lead) > 1:
        raise ValueError(
            f"granularity {tuple(counts)} splits {len(lead)} leading "
            f"axes; the partner energy is measured along one"
        )
    return lead[0]


def install_codebook(module, levels):
    """Write fitted levels into a fake-quant, in place.

    The levels replace the fake-quant's quantization map outright, since
    that map is what ``quantize`` reads and what ``convert_pt2e`` hands to
    the graph.  One table per tensor stays a lookup table, so nothing
    downstream sees a change; one table per head cannot be a lookup table
    -- an entry per bfloat16 bit pattern has no room for a second index --
    and is installed as the codebook itself, which ``quantize`` searches.

    Args:
        module: The fake-quant the levels belong to.
        levels: One table of integers, or one table per attention head.
    """
    device = module.qmap.device
    if isinstance(levels[0], (list, tuple)):
        module.qmap = torch.tensor(levels, dtype=torch.float32, device=device)
    else:
        module.qmap = codebook_qmap(levels, device)


#: The op a weight's tap wraps.  A conv's weight contracts over its
#: window as well as its input channels, so the same reduction would not
#: hold, and its weight keeps the partner-energy weighting.
TAPPED_OP = torch.ops.aten.linear.default


def _tap_output(call, note):
    """Wrap an op so its output gradient reaches ``note``, per token.

    A module hook cannot see this: fake-quants sit on the op's operands,
    and the gradient wanted is the one w.r.t. its *output*.  Pointing the
    node at a wrapper is the least invasive way to reach it -- the graph
    keeps its shape, and only the callable one node holds changes.

    Args:
        call: What the node calls today.
        note: Called during the backward as ``note(activation, grad)``
            with the op's first operand and the gradient of the loss
            w.r.t. its output, both still carrying their token axis.

    Returns:
        A stand-in with the same signature, for the node to point at.
    """

    def tapped(*args, **kwargs):
        output = call(*args, **kwargs)
        if output.requires_grad:
            output.register_hook(lambda grad: note(args[0], grad))
        return output

    return tapped


def _accumulate_fisher(
    model, calibration, operands, grid, fisher, tap_weights=False
):
    """Fill per-channel loss-gradient Fisher weights, by a backward pass.

    Weights each operand by the sum over calibration of its squared
    gradient of the final loss, per channel -- the Gauss-Newton diagonal,
    the operand's full downstream sensitivity.  Squaring happens per
    token and the squares are then summed, which is what makes it a
    curvature surrogate rather than a gradient magnitude.

    An activation's gradient arrives with its token axis intact, so its
    diagonal is read straight off the fake-quant's backward hook.  A
    parameter's does not: ``dL/dW[o, i]`` is already
    ``sum_t g[t, o] x[t, i]``, contracted by the op's own backward before
    any hook runs, and squaring that gives the square of a sum where the
    diagonal needs the sum of squares.  Summed over the output channels
    the wanted quantity factors,

        sum_o sum_t (g[t, o] x[t, i])^2 = sum_t x[t, i]^2 ||g_t||^2

    so a weight is served by tapping its op's output gradient, reducing
    it to one norm per token, and pairing that with the activation the
    backward is holding anyway.  A weight whose op is not a linear, or
    whose output never needs a gradient, is left out and keeps the
    partner-energy weighting -- the same diagonal under a uniform
    per-token weight.

    Args:
        model: Prepared graph module.
        calibration: Calibration inputs, each yielding a loss.
        operands: ``name -> (module, node, partners)`` from the caller.
        grid: ``name -> granularity tuple`` for the tensors fitted more
            than one table, whose diagonal is resolved per cell here.
        fisher: Dict filled with each operand's per-channel Fisher
            weight.  Operands absent from it fall back to their partner.
        tap_weights: Also weight the parameters, by the taps described
            above.  Off because it measured *worse* than leaving them on
            the partner energy (7.130313 against 7.112759): the exact
            diagonal carries a per-token ``||g_t||^2`` factor that a few
            high-loss calibration tokens dominate, and that concentration
            is a property of the calibration set rather than the weight.
    """

    def per_channel(grad, ch_axis):
        ordered = _contraction_last(grad, ch_axis)
        return ordered.reshape(-1, ordered.shape[-1]).pow(2).sum(0)

    def note_grad(name, module, grad):
        if grad is None:
            return
        grad = grad.detach().float()
        if name in grid:
            per = torch.stack(
                [
                    per_channel(slab, module.ch_axis)
                    for _, slab in _grid_cells(
                        grad, _resolve_counts(grid[name], grad.shape)
                    )
                ]
            )
        else:
            per = per_channel(grad, module.ch_axis)
        fisher[name] = per if name not in fisher else fisher[name] + per

    def note_weight(name, activation, grad):
        rows = activation.detach().reshape(-1, activation.shape[-1]).float()
        norms = grad.detach().reshape(-1, grad.shape[-1]).float().pow(2).sum(-1)
        per = (rows.pow(2) * norms.unsqueeze(1)).sum(0)
        fisher[name] = per if name not in fisher else fisher[name] + per

    handles, tapped = [], {}
    for name, (module, node, _) in operands.items():
        if getattr(node.args[0], "op", None) != "get_attr":
            handles.append(
                module.register_full_backward_hook(
                    lambda module, gi, go, name=name: note_grad(
                        name, module, gi[0]
                    )
                )
            )
            continue
        if not tap_weights:
            continue
        # A weight can feed several ops -- the same Linear called twice --
        # and its diagonal is the sum over all of them, so every use is
        # tapped and ``note_weight`` adds them.
        for consumer in [
            user for user in node.users if user.target is TAPPED_OP
        ]:
            tapped[consumer] = consumer.target
            consumer.target = _tap_output(
                consumer.target,
                lambda activation, grad, name=name: note_weight(
                    name, activation, grad
                ),
            )
    if tapped:
        model.recompile()

    # A full backward fills a gradient for every parameter and this pass
    # reads none of them, so restricting it to one anchor parameter looks
    # free.  It is not: ``inputs`` prunes the graph to what that anchor
    # needs, and an operand off that path stops being weighted at all --
    # anchoring on Q's weight drops K and V, whose gradients the path to Q
    # never visits.  The wasted gradients are the price of the coverage.
    was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        for entry in calibration:
            output = model(*entry)
            loss = output[0] if isinstance(output, tuple) else output.loss
            loss.backward()
            model.zero_grad(set_to_none=True)
    finally:
        torch.set_grad_enabled(was_enabled)
        for handle in handles:
            handle.remove()
        for consumer, target in tapped.items():
            consumer.target = target
        if tapped:
            model.recompile()


class Weighting(enum.Enum):
    """Which sensitivity weights each sample in a codebook fit.

    ``PARTNER`` weights a sample by its channel's energy in the operand it
    contracts against, which is what turns a tensor-MSE fit into a
    layer-output-error fit.  The two Fisher modes replace that with the
    operand's own Gauss-Newton diagonal, measured by a backward pass, so
    the fit minimizes the whole model's output error rather than the
    immediate layer's -- an operand no per-token gradient reaches falls
    back to ``PARTNER``.  ``FISHER_ALL`` extends the diagonal to the
    parameters by tapping each linear's output gradient, which measured
    worse than leaving them on partner energy.  ``NONE`` weights nothing,
    and recovers less than half as much.
    """

    NONE = "none"
    PARTNER = "partner"
    FISHER_ACTIVATIONS = "fisher_activations"
    FISHER_ALL = "fisher_all"


def fit_codebooks(
    model,
    calibration,
    skip=(),
    weighting=Weighting.PARTNER,
    weight_by=(),
    granularity=None,
    scale_weighted=False,
    quantized_scale=False,
    pin_zero=True,
    quantized_inputs=False,
    dump=None,
    histograms=None,
):
    """Fit a codebook to every tensor a prepared model quantizes, in place.

    Runs the calibration set twice: once to measure every operand's channel
    energy and sign, and once to bin every element of each operand weighted
    by its *partner's* energy, so the fit minimizes layer-output error
    rather than tensor MSE.  Nothing is sampled, so a fit is reproducible
    and sees the whole tensor.  Block size, contraction axis and range come
    from the fake-quant the quantizer already attached to each tensor, and
    the fitted levels are written back into that module's lookup table.

    Args:
        model: Graph module from ``prepare_pt2e``, before ``convert_pt2e``.
        calibration: Entries of positional arguments matching the graph's
            placeholders.
        skip: Substrings of operand names to leave alone.
        weighting: Which sensitivity weights each sample; see
            ``Weighting``.  ``PARTNER`` is the partner's channel energy,
            the two Fisher modes the operand's own loss-gradient diagonal,
            ``NONE`` no weighting at all.
        weight_by: Substrings picking which partners weight a tensor read
            by several ops; empty adds every partner's energy.
        granularity: ``name substring -> counts``, how many codebooks a
            tensor is fitted along each axis.  The tuple aligns to the
            trailing axes: ``-1`` gives one table per item of that axis,
            ``n`` splits it into ``n``, ``1`` shares one across it.
            ``(-1, 1, 1)`` is one table per attention head, ``(28, 16)``
            a grid over a weight's last two axes.
        scale_weighted: Weight each element by its block's squared scale, so
            the fit minimizes the error deployment makes.
        quantized_scale: Normalize by quantized block scale.
        pin_zero: Force a zero level.
        quantized_inputs: Sample with the fake-quants live.
        dump: Path to write the fitted tables to, as JSON.
        histograms: Dict filled with ``name -> accumulated histograms``,
            one per table the tensor is fitted.  Fresh by default.

    Returns:
        ``name -> the fitted levels``, for every tensor fitted.
    """
    operands = _quantized_operands(model, skip)
    logger.info(
        "fitting %d tensors over %d calibration inputs, weighting a shared "
        "tensor by %s",
        len(operands),
        len(calibration),
        (
            f"its {', '.join(weight_by)} partner"
            if weight_by
            else "every partner it feeds"
        ),
    )
    if quantized_inputs:
        logger.warning(
            "sampling with the fake-quants live: fitting against one "
            "configuration and deploying another measured worse"
        )
    if weighting is Weighting.NONE:
        logger.warning(
            "sampling unweighted: the fit minimizes tensor MSE rather than "
            "layer-output error, which recovers less than half as much"
        )
    named = {node: name for name, (_, node, _) in operands.items()}
    partners = {
        name: [
            found
            for partner in group
            if (found := named.get(partner)) is not None
            and (not weight_by or any(pick in found for pick in weight_by))
        ]
        or [named.get(partner) for partner in group]
        for name, (_, _, group) in operands.items()
    }
    grid = {}
    for pattern, counts in (granularity or {}).items():
        for name in operands:
            if pattern in name:
                grid[name] = tuple(counts)
    # Which axis, if any, each tensor's tables are indexed by.  Fitting
    # accepts any granularity; whether the result can be written into a
    # qmap is settled once, where the tables are installed.
    indexed = {name: leading_split(counts) for name, counts in grid.items()}
    # A split tensor's error reaches the output through its partner's own
    # slice of that axis, so the partner's energy is measured there too.
    leading = {name: axis for name, axis in indexed.items() if axis is not None}
    leading.update(
        {
            other: axis
            for name, axis in list(leading.items())
            for other in partners[name]
            if other
        }
    )
    if grid:
        logger.info(
            "fitting %d of them more than one codebook: %s",
            len(grid),
            ", ".join(sorted(grid)[:3]) + (" ..." if len(grid) > 3 else ""),
        )
    energies, floors, fisher_weights = {}, {}, {}
    # Settled the first time an operand is binned, once its rank is known.
    settled_counts = {}
    histograms = {} if histograms is None else histograms

    live = {
        module: module.fake_quant_enabled.clone()
        for module in model.modules()
        if hasattr(module, "qmap")
    }
    if not quantized_inputs:
        for module in live:
            module.fake_quant_enabled.zero_()

    def measure(name, module, tensor):
        axis = leading.get(name)
        if axis is not None:
            share = torch.stack(
                [
                    _channel_energy(piece, module.ch_axis)
                    for piece in tensor.unbind(axis)
                ]
            )
        else:
            share = _channel_energy(tensor, module.ch_axis)
        held = energies.get(name)
        energies[name] = share if held is None else held + share
        lowest = tensor.detach().min()
        held = floors.get(name)
        floors[name] = lowest if held is None else torch.minimum(held, lowest)

    def collect(name, module, tensor):
        counts = settled_counts.get(name)
        if counts is None:
            counts = (
                _resolve_counts(grid[name], tensor.shape)
                if name in grid
                else (1,) * tensor.dim()
            )
            settled_counts[name] = counts
        axis = leading.get(name)
        cells = _grid_cells(tensor, counts)
        if name not in histograms:
            # A tensor that never went negative -- a softmax output -- would
            # spend half its levels on values that cannot occur.  The floor
            # is read off the whole tensor in the pass before this one.
            floor = floors[name].item()
            histograms[name] = [
                Histogram(
                    0.0 if floor >= 0 else -module.quant_max, module.quant_max
                )
                for _ in cells
            ]
        energy = None
        # The operand's own loss-gradient sensitivity, not a partner's.
        # Only a Fisher mode fills that dict, so membership is the test.
        from_fisher = name in fisher_weights
        if from_fisher:
            energy = fisher_weights[name]
        elif weighting is not Weighting.NONE:
            # A tensor read by several ops reaches the output through all of
            # them, so the energies of its partners add.
            for share in (energies.get(other) for other in partners[name]):
                if share is not None:
                    energy = share if energy is None else energy + share
        for cell, ((index, slab), histogram) in enumerate(
            zip(cells, histograms[name])
        ):
            share = energy
            if share is not None and share.dim() == 2:
                if from_fisher:
                    # Already one row per cell, in this same order.
                    share = share[cell]
                elif axis is None:
                    # Measured per slice for a partner's sake, but this
                    # tensor is not split along that axis, so it pools
                    # every slice.
                    share = share.mean(0)
                else:
                    # Pooling a per-slice energy is averaging it: every
                    # slice contributes the same count to the mean, so a
                    # grouped split averages its group's rows.
                    per = share.shape[0] // counts[axis]
                    start = index[axis] * per
                    share = share[start : start + per].mean(0)
            if share is not None and counts[-1] > 1 and not from_fisher:
                # The energy indexes the contraction axis, so a cell is
                # weighted by the columns it holds -- the same slice for
                # every row band, which is what the column index picks out.
                share = share.chunk(counts[-1])[index[-1]]
            _accumulate(
                slab,
                module,
                share,
                histogram,
                scale_weighted,
                quantized_scale,
            )

    # A parameter is the same tensor on every calibration input, so binning
    # it once is binning it every time: the three moments scale together and
    # the cost the solver minimizes scales with them, which leaves its
    # minimum where it was.  Visiting one anyway is most of the work, since
    # a model's parameters outweigh one input's activations many times over.
    static = {
        name
        for name, (_, node, _) in operands.items()
        if getattr(node.args[0], "op", None) == "get_attr"
    }

    if weighting in (Weighting.FISHER_ACTIVATIONS, Weighting.FISHER_ALL):
        _accumulate_fisher(
            model,
            calibration,
            operands,
            grid,
            fisher_weights,
            tap_weights=weighting is Weighting.FISHER_ALL,
        )
        missed = sorted(set(operands) - set(fisher_weights))
        logger.info(
            "%d operands weighted by their loss-gradient Fisher%s",
            len(fisher_weights),
            (
                ""
                if not missed
                else (
                    f"; {len(missed)} reached no per-token gradient and keep "
                    f"the partner-energy weighting: {', '.join(missed[:3])}"
                    + (" ..." if len(missed) > 3 else "")
                )
            ),
        )
    passes = (measure, collect)

    with torch.no_grad():
        for visit in passes:
            settled = set()

            def once(name, module, tensor, visit=visit, settled=settled):
                if name in static:
                    if name in settled:
                        return
                    settled.add(name)
                visit(name, module, tensor)

            handles = [
                module.register_forward_pre_hook(
                    lambda _, inputs, name=name, module=module: once(
                        name, module, inputs[0]
                    )
                )
                for name, (module, _, _) in operands.items()
            ]
            step = max(1, len(calibration) // 10)
            for index, entry in enumerate(calibration):
                model(*entry)
                if (index + 1) % step == 0:
                    logger.info(
                        "%s pass: %d/%d windows",
                        visit.__name__,
                        index + 1,
                        len(calibration),
                    )
            for handle in handles:
                handle.remove()

    for module, enabled in live.items():
        module.fake_quant_enabled.copy_(enabled)

    # How concentrated the weighting is decides how much of the codebook a
    # few channels own: past roughly a hundredfold ratio the levels stop
    # serving the bulk of the tensor and serve those channels instead.
    for label, weights in (
        ("fisher", fisher_weights),
        ("partner energy", energies),
    ):
        spread = [
            (
                one.reshape(-1).max() / one.reshape(-1).median().clamp_min(EPS)
            ).item()
            for one in weights.values()
            if one is not None and one.numel() > 1
        ]
        if spread:
            spread = torch.tensor(spread)
            logger.info(
                "%s concentration over %d operands (max/median per "
                "operand): median %.3g, p90 %.3g, worst %.3g",
                label,
                len(spread),
                spread.median(),
                spread.quantile(0.9),
                spread.max(),
            )

    tables, unsigned = {}, []
    for name, (module, _, _) in operands.items():
        group = histograms.get(name)
        if group is None or group[0].weight is None:
            logger.warning(
                "%s: nothing accumulated, so it keeps the table it was "
                "seeded with",
                name,
            )
            continue
        if group[0].quant_min == 0.0:
            unsigned.append(name)
        # How many levels there are is the dtype's to say, and what the
        # module already carries says it: a lookup table by how many
        # distinct entries it holds, a codebook by its width.
        k = (
            module.qmap.shape[-1]
            if module.qmap.numel() < QMAP_SIZE
            else int(torch.unique(module.qmap).numel())
        )
        fitted = [
            optimal_codebook(histogram, k=k, pin_zero=pin_zero)
            for histogram in group
        ]
        # A codebook carries one axis per tensor axis, so the grid the
        # fit asked for is the shape the decoder dispatches on and every
        # granularity has a layout.
        counts = settled_counts[name]
        if all(count == 1 for count in counts):
            tables[name] = fitted[0]
            install_codebook(module, tables[name])
        else:
            module.qmap = torch.tensor(
                fitted, dtype=torch.float32, device=module.qmap.device
            ).reshape(*counts, k)
            tables[name] = {"granularity": list(counts), "levels": fitted}
        logger.debug("%s: %s", name, tables[name])

    logger.info("fitted %d of %d tensors", len(tables), len(operands))
    if unsigned:
        logger.info(
            "%d never went negative, so their levels span [0, quant_max]: "
            "%s%s",
            len(unsigned),
            ", ".join(unsigned[:3]),
            " ..." if len(unsigned) > 3 else "",
        )
    if dump is not None:
        with open(dump, "w") as handle:
            json.dump(tables, handle, indent=2, sort_keys=True)
        logger.info("wrote %d codebooks to %s", len(tables), dump)
    return tables


def load_codebooks(model, tables, skip=()):
    """Install already-fitted codebooks into a prepared model, in place.

    Args:
        model: Graph module from ``prepare_pt2e``.
        tables: ``name -> levels``, or the path of a JSON file holding it,
            as ``fit_codebooks`` writes with ``dump``.  One table is a
            list of levels; a grid of them is
            ``{"granularity": counts, "levels": ...}``, one count per
            operand axis.  Older dumps spell a grid over the trailing
            axes ``{"regions": [R, C], ...}`` or ``{"channel_groups": G}``,
            and a bare list of tables means one per attention head; both
            are padded out to the operand's rank.
        skip: Substrings of operand names to leave alone.

    Returns:
        The names installed.

    Raises:
        KeyError: The tables name a tensor this model does not quantize,
            which means they were fitted against a different scheme.
    """
    if isinstance(tables, (str, os.PathLike)):
        with open(tables) as handle:
            tables = json.load(handle)

    operands = _quantized_operands(model, skip)
    unknown = sorted(set(tables) - set(operands))
    if unknown:
        raise KeyError(
            f"{len(unknown)} codebooks do not match anything this model "
            f"quantizes, starting with {unknown[0]}"
        )
    for name, entry in tables.items():
        module, operand = operands[name][0], operands[name][1]
        value = operand.meta.get("val", getattr(operand, "value", None))
        rank = value.dim() if value is not None else None
        if not isinstance(entry, dict):
            if not isinstance(entry[0], (list, tuple)):
                install_codebook(module, entry)
                continue
            # A dump from before a codebook carried one axis per operand
            # axis: its rows indexed the attention head axis.
            counts = _padded_counts([len(entry)], rank, head_axis=True)
        elif "granularity" in entry:
            counts = list(entry["granularity"])
        else:
            counts = list(
                entry.get(
                    "regions", (1, entry.get("channel_groups", len(entry)))
                )
            )
            counts = _padded_counts(counts, rank, head_axis=False)
        levels = torch.tensor(
            entry["levels"] if isinstance(entry, dict) else entry,
            dtype=torch.float32,
            device=module.qmap.device,
        )
        module.qmap = levels.reshape(*counts, levels.shape[-1])
    logger.info("installed %d codebooks", len(tables))

    unfitted = sorted(set(operands) - set(tables))
    if unfitted:
        logger.warning(
            "%d tensors this model quantizes have no codebook and keep the "
            "table they were seeded with, starting with %s",
            len(unfitted),
            unfitted[0],
        )
    return sorted(tables)
