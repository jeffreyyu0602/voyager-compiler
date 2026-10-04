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
from typing import NamedTuple, Optional, Tuple

__all__ = [
    "FloatFormat",
    "QScheme",
    "float_format",
    "int_range",
    "parse_codebook_dtype",
    "parse_spec_fields",
    "posit_format",
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
    "pot": "power_2_scale",
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


def _parse_bool(value: str) -> bool:
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"Expected 0, 1, false or true but got '{value}'")
    return value in ("1", "true")


_PARAMS_TYPE = {
    "quant_min": float,
    "quant_max": float,
    "qscheme": QScheme,
    "amax_history_len": int,
    "ch_axis": _parse_int_or_list,
    "block_size": _parse_int_or_list,
    "power_2_scale": _parse_bool,
    "scale_dtype": str,
    "outlier_threshold": float,
    "outlier_pct": float,
}


def int_range(dtype: str) -> Optional[Tuple[int, int]]:
    """Return the range of an integer dtype.

    Args:
        dtype: A dtype name; ``int<N>`` is signed, ``uint<N>`` unsigned.

    Returns:
        ``(min, max)``, or None when ``dtype`` is not an integer.
    """
    match = re.fullmatch(r"(u?)int(\d+)", dtype, re.IGNORECASE)
    if match is None:
        return None
    bits = int(match.group(2))
    if match.group(1):
        return 0, 2**bits - 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


class FloatFormat(NamedTuple):
    """A minifloat: ``bits`` wide, ``ebits`` exponent and ``mbits`` mantissa
    bits, and ``max_norm`` its largest finite magnitude."""

    bits: int
    ebits: int
    mbits: int
    max_norm: float

    @property
    def signed(self) -> bool:
        """Whether a bit is left over for the sign."""
        return self.bits == self.ebits + self.mbits + 1


def float_format(dtype: str) -> Optional[FloatFormat]:
    """Parse a minifloat dtype, ``fp<B>_e<X>m<Y>``.

    ``B = X + Y + 1`` is signed; ``B = X + Y`` is unsigned, as a scale is.
    A format with more than four exponent bits keeps its top exponent for
    inf and NaN; a narrower one spends it on finite values, and
    ``fp8_e4m3`` (torch's ``float8_e4m3fn``) gives up only its all-ones
    mantissa there, to NaN.

    Args:
        dtype: A dtype name.

    Returns:
        The format, or None when ``dtype`` is not a minifloat.

    Raises:
        ValueError: ``B`` is neither ``X + Y`` nor ``X + Y + 1``.
    """
    match = re.fullmatch(r"fp(\d+)_e(\d+)m(\d+)", dtype, re.IGNORECASE)
    if match is None:
        return None
    bits, ebits, mbits = map(int, match.groups())
    if bits not in (ebits + mbits, ebits + mbits + 1):
        raise ValueError(
            f"{dtype}: {bits} bits is neither {ebits} exponent plus {mbits} "
            "mantissa bits nor that plus a sign bit"
        )
    emax = 2 ** (ebits - 1) - 1 if ebits > 4 else 2 ** (ebits - 1)
    if dtype.lower() == "fp8_e4m3":
        max_norm = 2**emax * 1.75
    else:
        max_norm = 2**emax * (2 - 2.0**-mbits)
    return FloatFormat(bits, ebits, mbits, max_norm)


def posit_format(dtype: str) -> Optional[Tuple[int, int]]:
    """Parse a posit dtype, ``posit<N>_<es>``.

    Args:
        dtype: A dtype name.

    Returns:
        ``(bits, es)``, or None when ``dtype`` is not a posit.
    """
    match = re.fullmatch(r"posit(\d+)_(\d+)", dtype, re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _get_quant_min_max(dtype: str):
    if (bounds := int_range(dtype)) is not None:
        return bounds

    if (fmt := float_format(dtype)) is not None:
        return -fmt.max_norm, fmt.max_norm

    if (posit := posit_format(dtype)) is not None:
        bits, es = posit
        max_val = (2 ** (2**es)) ** (bits - 2)
        return -max_val, max_val

    # Lookup tables span their entries' range, or [-1, 1] when the entries
    # keep the model's dtype.
    if (codebook := parse_codebook_dtype(dtype)) is not None:
        entry = codebook[1]
        max_val = 1 if entry is None else _get_quant_min_max(entry)[1]
        return -max_val, max_val

    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_codebook_dtype(dtype: str):
    """Split a lookup-table dtype into its index width and entry dtype.

    ``lut<I>_to_<E>`` stores ``I``-bit indices into a table of ``2**I``
    entries of dtype ``E``, a signed ``int<N>`` or ``fp<B>_e<X>m<Y>``.  A bare
    ``lut<I>`` keeps the entries in the model's own dtype.

    Args:
        dtype: A dtype name.

    Returns:
        ``(index_bits, entry_dtype)``, ``entry_dtype`` None for a bare
        ``lut<I>``; None when ``dtype`` is not a lookup table.

    Raises:
        ValueError: The entry dtype is neither a signed integer nor a signed
            float.
    """
    match = re.fullmatch(r"lut(\d+)(?:_to_(\w+))?", dtype)
    if match is None:
        return None
    index_bits, entry_dtype = match.groups()
    if entry_dtype is None:
        return int(index_bits), entry_dtype
    bounds = int_range(entry_dtype)
    fmt = float_format(entry_dtype)
    if (bounds is None or bounds[0] == 0) and (fmt is None or not fmt.signed):
        raise ValueError(
            f"{dtype}: a lookup table's entries are int<N> or a signed "
            f"fp<bits>_e<exponent>m<mantissa>, not {entry_dtype}"
        )
    return int(index_bits), entry_dtype


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
