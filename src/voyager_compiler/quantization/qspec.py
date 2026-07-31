"""The quantization spec-string vocabulary and its parser.

A datatype is configured with a comma-separated string —
``int8,qs=microscaling,bs=16,ax=-1,scale=fp8_e5m3``.  The first field is the
dtype; the rest are ``key=value``, where the key may be an abbreviation
(``_ABBREV_MAP``) and the value is coerced by ``_PARAMS_TYPE``.

``parse_spec_fields`` turns such a string into ``QuantizationSpec`` kwargs.  It
returns a plain dict rather than the object so this module stays a leaf: it
imports nothing from the package, which is what lets both ``fake_quantize`` and
``quantizer.quantizer`` depend on it without a cycle.
"""

import re
from enum import Enum

__all__ = [
    "QScheme",
    "parse_spec_fields",
]


class QScheme(Enum):
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"
    MICROSCALING = "microscaling"
    GROUP_WISE_AFFINE = "group_wise_affine"


_ABBREV_MAP = {
    "qmin": "quant_min",
    "qmax": "quant_max",
    "qs": "qscheme",
    "ahl": "amax_history_len",
    "ax": "ch_axis",
    "bs": "block_size",
    "scale": "scale_dtype",
    "othr": "outlier_threshold",
    "opct": "outlier_pct",
}


def _parse_int_or_list(value: str):
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        parts = tuple(int(v.strip()) for v in value[1:-1].split(","))
        return parts
    return int(value)


_PARAMS_TYPE = {
    "quant_min": float,
    "quant_max": float,
    "qscheme": QScheme,
    "amax_history_len": int,
    "ch_axis": _parse_int_or_list,
    "block_size": _parse_int_or_list,
    "scale_dtype": str,
    "outlier_threshold": float,
    "outlier_pct": float,
}


def _get_quant_min_max(dtype: str):
    # Signed integers
    if match := re.fullmatch(r"int(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        max_val = 2 ** (nbits - 1) - 1
        min_val = -(2 ** (nbits - 1))
        return min_val, max_val

    # Unsigned integers
    if match := re.fullmatch(r"uint(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        return 0, 2**nbits - 1

    # Floating-point like fpN_eXmY
    if match := re.fullmatch(r"fp(\d+)_e(\d+)m(\d+)", dtype, re.IGNORECASE):
        ebits = int(match.group(2))
        mbits = int(match.group(3)) + 2
        emax = 2 ** (ebits - 1) - 1 if ebits > 4 else 2 ** (ebits - 1)

        if dtype.lower() == "fp8_e4m3":
            max_val = 2**emax * 1.75  # max mantissa (1.75)
        else:
            max_val = 2**emax * (2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)

        return -max_val, max_val

    # Posit numbers
    if match := re.fullmatch(r"posit(\d+)_(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        es = int(match.group(2))
        max_val = (2 ** (2**es)) ** (nbits - 2)
        return -max_val, max_val

    # Normalized floats (NF)
    if match := re.fullmatch(r"nf(\d+)(?:_(\d+))?", dtype, re.IGNORECASE):
        if match.group(2) is not None:
            max_val = 2 ** (int(match.group(2)) - 1) - 1
        else:
            max_val = 1
        return -max_val, max_val

    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_spec_fields(s: str) -> dict:
    """Parse a spec string into ``QuantizationSpec`` keyword arguments.

    A qscheme implies the dtype's representable range, so ``quant_min`` /
    ``quant_max`` are filled in from it unless given, and the two amax-history
    schemes get a default history length.

    Args:
        s: e.g. ``"int8,qs=microscaling,bs=16"``.

    Returns:
        Keyword arguments for ``QuantizationSpec``.

    Raises:
        ValueError: If ``s`` is empty, a field is not ``key=value``, or the key
            is not a known parameter.
    """
    if not s:
        raise ValueError("String quantization_spec is None or empty")

    fields = re.split(r",(?![^()]*\))", s)
    params = {"dtype": fields[0]}

    for item in fields[1:]:
        if "=" not in item:
            raise ValueError(f"Expected key=value format but got '{item}'")
        key, value = item.split("=")
        key = _ABBREV_MAP.get(key, key)
        if key not in _PARAMS_TYPE:
            valid = ", ".join(_PARAMS_TYPE.keys())
            raise ValueError(f"Unknown argument '{key}'. Valid keys: {valid}")
        params[key] = _PARAMS_TYPE[key](value)

    if (qscheme := params.get("qscheme", None)) is not None:
        qmin, qmax = _get_quant_min_max(params["dtype"])
        params.setdefault("quant_min", float(qmin))
        params.setdefault("quant_max", float(qmax))
        if qscheme in [
            QScheme.PER_TENSOR_SYMMETRIC,
            QScheme.PER_CHANNEL_SYMMETRIC,
        ]:
            params.setdefault("amax_history_len", 16)

    return params
