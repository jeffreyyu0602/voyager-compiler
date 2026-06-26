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
    if_dtype_bits: input activation element width in bits.
    fl_dtype_bits: weight/filter element width in bits.
    psum_dtype_bits: partial-sum accumulator width in bits.
    of_dtype_bits: output activation element width in bits.
    if_scale_bits: microscaling input-scale element width in bits (0 = none).
    fl_scale_bits: microscaling weight-scale element width in bits (0 = none).
    of_scale_bits: microscaling output-scale element width in bits (0 = none).
    block_size: microscaling group size (elements per scale); the scale tensor
        holds one scale per block_size value elements.
    fused_size_fn: optional callable ``fn(out_tile, bank_size) -> bytes``
        giving the L2+ storage (in bytes, bank-rounded) of fused post-op
        operands (residual, bias, ...) that share the output's bank domain.
        ``out_tile`` is the per-output-loop-dim tile at the level. None = no
        fused operands (contributes 0).
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
        if_dtype_bits=8,
        fl_dtype_bits=8,
        psum_dtype_bits=32,
        of_dtype_bits=8,
        if_scale_bits=0,
        fl_scale_bits=0,
        of_scale_bits=0,
        block_size=1,
        fused_size_fn=None,
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
        self.if_dtype_bits = if_dtype_bits
        self.fl_dtype_bits = fl_dtype_bits
        self.psum_dtype_bits = psum_dtype_bits
        self.of_dtype_bits = of_dtype_bits
        self.if_scale_bits = if_scale_bits
        self.fl_scale_bits = fl_scale_bits
        self.of_scale_bits = of_scale_bits
        self.block_size = block_size
        self.fused_size_fn = fused_size_fn
        assert self.wofm > 0
        assert self.hofm > 0
        assert self.nimg > 0
        self.sizes = [wfil, hfil, wofm, hofm, nofm, nifm, nimg]

    @classmethod
    def layer(cls, info):
        return cls(
            info["input_fmap_channel"],
            info["output_fmap_channel"],
            info["fmap_width"],
            info["fmap_height"],
            info["window_width"],
            info["window_height"],
            info["batch_size"],
            info["stride_width"],
            info["stride_height"],
        )


class FCLayer(Layer):
    """
    NN fully-connected layer parameters.

    (wifm, hifm) = (wfil, hfil), wstd = hstd = 1, wofm = hofm = 1.
    """

    def __init__(self, nifm, nofm, wfil, hfil, nimg=1):
        Layer.__init__(self, nifm, nofm, 1, 1, wfil, hfil, nimg)
