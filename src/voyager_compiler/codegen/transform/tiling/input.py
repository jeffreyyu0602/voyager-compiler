"""InputController buffer usage shared by the matrix backends."""

import interstellar

le = interstellar.le


def input_buffer_usage(mapping, stride, capacity=None):
    """Return input-buffer words and controller-limit violations for a tile."""
    b = mapping.loop_blockings
    sy, sx = stride
    fx, fy = b[le.FX][1], b[le.FY][1]
    # InputController fetcher/writer load a full stride after each output
    # position for filters wider than one. For a one-tap axis they fetch only
    # the sampled positions, so the buffer does not contain the skipped inputs.
    width = b[le.OX][1] * (sx if fx > 1 else 1) + fx - 1
    height = b[le.OY][1] * (sy if fy > 1 else 1) + fy - 1
    words = width * height * b[le.IC][1]
    reasons = []
    if sx != sy or not 1 <= sx <= 255:
        reasons.append("InputController requires equal strides from 1 to 255")
    if max(fx, fy) > 15:
        reasons.append("filter extent exceeds the 4-bit input-controller field")
    # The reader skips stride within the buffer only when both filter axes
    # have one tap. A single one-tap axis would disagree with the writer.
    if max(sx, sy) > 1 and (fx == 1) != (fy == 1):
        reasons.append(
            "strided one-dimensional filters are not implemented in InputController"
        )
    if words > min(capacity if capacity is not None else 65536, 65536):
        reasons.append("input tile including halo exceeds one input-buffer bank")
    if max(width, height, b[le.OX][2], b[le.OY][2]) > 512:
        reasons.append("input traversal exceeds the controller coordinate fields")
    if (
        (width - 1) * (sx if fx == 1 else 1) > 1023
        or (height - 1) * (sy if fy == 1 else 1) + b[le.FY][2] - 1 > 1023
    ):
        reasons.append("strided input coordinates exceed the 10-bit fetcher fields")
    return words, reasons
