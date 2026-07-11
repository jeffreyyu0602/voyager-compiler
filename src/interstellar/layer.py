"""
Layer specification.
"""


class Layer(object):
    """
    NN layer parameters.

    nifm: ifmap channels.
    nofm: ofmap channels.
    wifm: ifmap width.
    hifm: ifmap height.
    wofm: ofmap width.
    hofm: ofmap height.
    wfil: weight filter width.
    hfil: weight filter height.
    nimg: input images (batch).
    wstd: stride size in width dimension.
    hstd: stride size in height dimension.

    size_fn: how many *bytes* a tile of this layer occupies at a byte-pool level
        (L2+).  The loop nest gives element counts; everything that turns those
        into bytes -- element widths, microscaling scale tensors, fused post-op
        operands, and how operands are packed into banks -- is the caller's
        policy, so it lives here rather than in the cost model::

            size_fn(counts, point, level, partitioning_accum, bank_size,
                    num_banks) -> (if_bytes, of_bytes, fl_bytes)

        ``counts`` are the (input, output, weight) element counts at the level;
        ``point`` is the mapping (so the fn can derive its own output-dim
        extents and see whether the output is still a partial sum);
        ``partitioning_accum`` is the level's accumulated spatial partitioning,
        or None for a per-bank (one PE) size; ``bank_size`` is None when the
        level has no banking.

        None => the level is sized in element counts, unconverted.
    """

    def __init__(
        self,
        nifm,
        nofm,
        wofm,
        hofm,
        wfil,
        hfil,
        nimg=1,
        wstd=1,
        hstd=1,
        size_fn=None,
    ):
        self.nifm = nifm
        self.nofm = nofm
        self.wofm = wofm
        self.hofm = hofm
        self.wifm = wfil + (wofm - 1) * wstd
        self.hifm = hfil + (hofm - 1) * hstd
        self.wfil = wfil
        self.hfil = hfil
        self.nimg = nimg
        self.wstd = wstd
        self.hstd = hstd
        self.size_fn = size_fn
        assert self.wofm > 0
        assert self.hofm > 0
        assert self.nimg > 0
        self.sizes = [wfil, hfil, wofm, hofm, nofm, nifm, nimg]
