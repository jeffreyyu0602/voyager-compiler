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
from voyager_compiler.quantization.dtypes import create_normal_map

logger = logging.getLogger(__name__)

EPS = 1e-12

#: Bins the range is divided into when a tensor's statistics are
#: accumulated.  The solver's candidate levels are integers, so this only
#: has to be fine enough that a bin boundary never falls where a decision
#: does; 4001 puts about 63 bins between consecutive integers.
HISTOGRAM_BINS = 4001


@dataclasses.dataclass
class Histogram:
    """Weighted moments of one tensor's normalized values, per bin.

    The exact solver reduces whatever it is handed to three sums per bin --
    total weight, weighted sum, weighted sum of squares -- and reads nothing
    else.  Accumulating those directly over every element is therefore not
    an approximation of sampling but a replacement for it: no draw to vary
    between runs, no bound on how much data one fit may see, and a few
    kilobytes of state per tensor instead of gigabytes of retained samples.

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
        edges = torch.linspace(
            self.quant_min - 0.5,
            self.quant_max + 0.5,
            self.bins + 1,
            device=values.device,
            dtype=torch.float32,
        )
        index = (torch.bucketize(values, edges, right=True) - 1).clamp_(
            0, self.bins - 1
        )
        # Float64 so that the order the bins are summed in -- which on a GPU
        # is decided by the scheduler -- cannot move a level.
        values = values.double()
        weights = weights.double().expand_as(values)
        self.weight.index_add_(0, index, weights)
        self.total.index_add_(0, index, weights * values)
        self.square.index_add_(0, index, weights * values * values)

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


def unit_scale(x, block_size, quant_max):
    """Return ``x / blockscale`` under block-amax microscaling.

    Args:
        x: Tensor whose last axis is the contraction axis.
        block_size: Elements sharing one scale.
        quant_max: Largest magnitude the codebook can represent.

    Returns:
        Tensor shaped like ``x``, whose per-block maximum is ``quant_max``.
    """
    x = x.float()
    flat = x.reshape(-1, block_size)
    scale = flat.abs().amax(-1, keepdim=True).clamp_min(EPS) / quant_max
    return (flat / scale).reshape(x.shape)


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


def _accumulate(tensor, module, energy, histogram):
    """Bin one operand's normalized values, weighted by its partner.

    Every element is counted, so what the fit sees is the tensor itself
    rather than a draw from it.

    Args:
        tensor: The operand, as it enters its fake-quant.
        module: That fake-quant, which knows the block geometry.
        energy: The partner's per-channel energy, or None for a flat fit.
        histogram: Where the weighted moments accumulate.
    """
    ordered = _contraction_last(tensor.detach(), module.ch_axis)
    if ordered.numel() < module.block_size:
        return
    normalized = unit_scale(
        ordered, module.block_size, module.quant_max
    ).reshape(-1)
    if energy is None:
        weights = torch.ones_like(normalized)
    else:
        # The contraction axis is last, so an element's channel is its
        # position modulo the channel count -- which is what tiling the
        # energy across the flattened tensor spells.
        weights = energy.repeat(normalized.numel() // energy.numel())
    histogram.add(normalized, weights)


def fit_codebooks(
    model,
    calibration,
    skip=(),
    weighted=True,
    weight_by=(),
    pin_zero=True,
    quantized_inputs=False,
    dump=None,
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
        calibration: The calibration set; each entry is a tuple of
            positional arguments matching the graph's placeholders.
        skip: Substrings of operand names to leave alone.
        weighted: Weight each sample by its partner's channel energy, which
            is what makes the fit minimize layer-output error rather than
            tensor MSE.  Turning it off recovers less than half as much.
        weight_by: Substrings picking which partners weight a tensor read
            by several ops -- ``("v_proj", "up_proj")`` names one at each of
            a decoder layer's two sharing sites.  Empty adds every partner's
            energy, which is what the error actually does, and a site no
            substring matches falls back to that.
        pin_zero: Require a zero level.  Without it the optimum spends that
            level elsewhere, leaving a block's many small values unable to
            quantize to zero.
        quantized_inputs: Sample with the fake-quants live, so each tensor
            is measured as the quantized model produces it.  Off by default:
            fitting against one configuration and deploying another measured
            worse than ignoring the dependency.
        dump: Path to write the fitted tables to, as JSON.

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
    if not weighted:
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
    energies, floors, histograms = {}, {}, {}

    live = {
        module: module.fake_quant_enabled.clone()
        for module in model.modules()
        if hasattr(module, "qmap")
    }
    if not quantized_inputs:
        for module in live:
            module.fake_quant_enabled.zero_()

    def measure(name, module, tensor):
        share = _channel_energy(tensor, module.ch_axis)
        held = energies.get(name)
        energies[name] = share if held is None else held + share
        lowest = tensor.detach().min()
        held = floors.get(name)
        floors[name] = lowest if held is None else torch.minimum(held, lowest)

    def collect(name, module, tensor):
        if name not in histograms:
            # A tensor that never went negative -- a softmax output -- would
            # spend half its levels on values that cannot occur.  The floor
            # is read off the whole tensor in the pass before this one.
            floor = floors[name].item()
            histograms[name] = Histogram(
                0.0 if floor >= 0 else -module.quant_max, module.quant_max
            )
        energy = None
        if weighted:
            # A tensor read by several ops reaches the output through all of
            # them, so the energies of its partners add.
            for share in (energies.get(other) for other in partners[name]):
                if share is not None:
                    energy = share if energy is None else energy + share
        _accumulate(tensor, module, energy, histograms[name])

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

    with torch.no_grad():
        for visit in (measure, collect):
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

    tables, unsigned = {}, []
    for name, (module, _, _) in operands.items():
        histogram = histograms.get(name)
        if histogram is None or histogram.weight is None:
            logger.warning(
                "%s: nothing accumulated, so it keeps the table it was "
                "seeded with",
                name,
            )
            continue
        if histogram.quant_min == 0.0:
            unsigned.append(name)
        # How many levels the table holds is the dtype's to say, and the
        # seeded lookup already says it.
        k = int(torch.unique(module.qmap).numel())
        levels = optimal_codebook(histogram, k=k, pin_zero=pin_zero)
        tables[name] = levels
        module.qmap.copy_(codebook_qmap(levels, module.qmap.device))
        logger.debug("%s: %s", name, levels)

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
            as ``fit_codebooks`` writes with ``dump``.
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
    for name, levels in tables.items():
        module = operands[name][0]
        module.qmap.copy_(codebook_qmap(levels, module.qmap.device))
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
