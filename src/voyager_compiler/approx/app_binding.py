"""
Helper functions for binding piecewise approximations to torch.nn.functional,
enabling end-to-end accuracy evaluation in inference scripts such as
examples/imagenet/main.py and examples/language_modeling/wikitext.py.

Typical usage:
    from voyager_compiler.approx.app_binding import bind_all_by_tag, unbind_all

    # bind every function registered under a tag at once
    bind_all_by_tag("kartik-thesis")
    # ... run inference ...
    unbind_all()

Or with the context manager:
    from voyager_compiler.approx.app_binding import approx_context

    with approx_context(tag="kartik-thesis"):
        validate(model, val_loader)

    # or a specific subset:
    with approx_context([("gelu", "kartik-thesis"), ("silu", "kartik-thesis")]):
        validate(model, val_loader)

Each function name must match both an entry in APPROXIMATION_REGISTRY and
either a torch.nn.functional attribute (e.g. "gelu" -> F.gelu) or a top-level
torch attribute (e.g. "exp" -> torch.exp).  torch.nn.functional is checked
first; torch is the fallback.
"""

import functools
from contextlib import contextmanager
from types import ModuleType
from typing import Callable, Union

import torch
import torch.nn.functional as F

from voyager_compiler.approx.app_param import (
    AppTemplateConfig,
    SUPPORTED_DTYPES,
    get_app_config,
    get_all_names_by_tag,
)
from voyager_compiler.approx.app_template import (
    get_compute_dtype,
    linear_app_template,
    quadratic_app_template,
)

# ---------------------------------------------------------------------------
# Map template_name -> segment evaluation function
# ---------------------------------------------------------------------------
_TEMPLATE_FN = {
    "quadratic_app_template": quadratic_app_template,
    "linear_app_template":    linear_app_template,
}

# Registry of active patches: name -> (module, original_fn)
# The module is whichever of F or torch the function was found on.
_original_fns: dict[str, tuple[ModuleType, Callable]] = {}

# When True, binding "exp" also replaces F.softmax with a pure-Python
# implementation that routes through torch.exp, so the patched exp is used
# inside softmax as well.
softmax_via_exp: bool = True


def _bind_softmax_via_exp() -> None:
    """Replace F.softmax with a numerically-stable pure-Python impl that
    calls torch.exp, so whatever patch is on torch.exp is also used inside
    softmax.  Idempotent: does nothing if softmax is already patched.
    """
    if "softmax" in _original_fns:
        return
    original_softmax = F.softmax

    @functools.wraps(original_softmax)
    def manual_softmax(input: torch.Tensor, dim: int = -1, **kwargs) -> torch.Tensor:
        x_max = input.max(dim=dim, keepdim=True).values
        exp_x = torch.exp(input - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    _original_fns["softmax"] = (F, original_softmax)
    F.softmax = manual_softmax


def _resolve_module(name: str) -> ModuleType:
    """Return the module (F or torch) that owns *name*, preferring F."""
    if hasattr(F, name):
        return F
    if hasattr(torch, name):
        return torch
    raise ValueError(
        f"Neither torch.nn.functional nor torch has an attribute '{name}'"
    )


# ---------------------------------------------------------------------------
# Piecewise approximation builder (uses app_template functions per segment)
# ---------------------------------------------------------------------------

def _build_piecewise_fn(config: AppTemplateConfig) -> Callable:
    """Return a callable implementing the piecewise approximation from *config*."""
    dtype         = config.param_dtype
    compute_dtype = get_compute_dtype(dtype)
    segments      = config.segments
    eval_segment  = _TEMPLATE_FN[config.template_name]

    # Pre-convert segment coefficients to CPU tensors once at build time.
    # They are moved to the input device lazily inside the closure so the
    # same piecewise function works for both CPU and GPU inputs.
    # bfloat16, float16, float32, and int32 are all supported on CUDA.
    # fp8 types use bfloat16 as compute_dtype (see get_compute_dtype),
    # so they also run correctly on GPU.
    x_min = segments[0].start
    x_max = segments[-1].end

    seg_coeff_tensors = [
        tuple(torch.tensor(c, dtype=dtype) for c in seg.coeffs)
        for seg in segments
    ]

    # Cache device-specific coefficients to avoid repeated .to(device) calls.
    device_coeff_cache: dict = {}

    def piecewise(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        orig_dtype = x.dtype
        device = x.device
        # Clamp out-of-range inputs to the segment boundaries so they are
        # evaluated by the first / last segment formula rather than being
        # undefined or zero.
        x_c = x.to(compute_dtype).clamp(x_min, x_max)

        if device not in device_coeff_cache:
            device_coeff_cache[device] = [
                tuple(c.to(device) for c in coeffs)
                for coeffs in seg_coeff_tensors
            ]
        coeff_tensors = device_coeff_cache[device]

        out = torch.zeros_like(x_c)
        for i, (seg, coeffs) in enumerate(zip(segments, coeff_tensors)):
            if i == 0:
                mask = x_c <= seg.end
            elif i == len(segments) - 1:
                mask = x_c > segments[i - 1].end
            else:
                mask = (x_c > segments[i - 1].end) & (x_c <= seg.end)

            seg_result = eval_segment(x_c, coeffs, dtype).to(compute_dtype)
            out = torch.where(mask, seg_result, out)

        return out.to(orig_dtype)

    return piecewise


# ---------------------------------------------------------------------------
# Quantized (dtype-cast) wrapper builder
# ---------------------------------------------------------------------------

def _build_quantized_fn(
    original_fn: Callable,
    dtype: torch.dtype,
    clamp_range: tuple[float, float] | None = None,
) -> Callable:
    """Return a wrapper that casts input to *dtype*, runs *original_fn* in
    ``compute_dtype``, then casts the output back to *dtype*.

    Parameters
    ----------
    clamp_range:
        When provided as ``(x_min, x_max)``, the input is clamped to this
        range after casting to *dtype* and before calling *original_fn*.
        Pass the segment boundaries from the approximation registry to
        isolate the effect of clamping from the effect of approximation.
    """
    compute_dtype = get_compute_dtype(dtype)

    @functools.wraps(original_fn)
    def quantized_fn(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_q = x.to(dtype)
        if clamp_range is not None:
            x_q = x_q.clamp(*clamp_range)
        out = original_fn(x_q.to(compute_dtype), *args, **kwargs)
        return out.to(dtype)

    return quantized_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bind_approx(name: str, tag: str) -> None:
    """Bind the piecewise approximation for *name*/*tag*.

    Looks up the function in torch.nn.functional first, then torch.

    Parameters
    ----------
    name:
        Function name as in APPROXIMATION_REGISTRY (e.g. ``"gelu"``,
        ``"exp"``).
    tag:
        Approximation tag (e.g. ``"kartik-thesis"``).
    """
    mod         = _resolve_module(name)
    config      = get_app_config(name, tag)
    original_fn = getattr(mod, name)
    approx_core = _build_piecewise_fn(config)

    @functools.wraps(original_fn)
    def approx_fn(*args, **kwargs):
        return approx_core(*args, **kwargs)

    if name not in _original_fns:
        _original_fns[name] = (mod, original_fn)

    setattr(mod, name, approx_fn)

    if name == "exp" and softmax_via_exp:
        _bind_softmax_via_exp()


def bind_approx_list(bindings: list[tuple[str, str]]) -> None:
    """Bind multiple approximations from a list of (name, tag) pairs.

    Parameters
    ----------
    bindings:
        Each element is a ``(name, tag)`` pair forwarded to
        :func:`bind_approx`.

    Example::

        bind_approx_list([
            ("gelu", "kartik-thesis"),
            ("silu", "kartik-thesis"),
        ])
    """
    for name, tag in bindings:
        bind_approx(name, tag)


def bind_all_by_tag(tag: str) -> None:
    """Bind every function registered under *tag*.

    Each function is resolved via torch.nn.functional first, then torch.
    Functions found in neither are silently skipped.

    Example::

        bind_all_by_tag("kartik-thesis")
        # ... run inference ...
        unbind_all()
    """
    bind_approx_list([
        (name, tag) for name in get_all_names_by_tag(tag)
        if hasattr(F, name) or hasattr(torch, name)
    ])


def bind_quantize(name: str, dtype_str: str, tag: str | None = None, clamp: bool = False) -> None:
    """Bind a dtype-cast wrapper for *name* using *dtype_str*.

    Looks up the function in torch.nn.functional first, then torch.
    The wrapper quantises the input to *dtype_str*, runs the **original**
    function in ``compute_dtype``, and requantises the output back to
    *dtype_str*.  No piecewise approximation is used.

    Parameters
    ----------
    name:
        Function name (e.g. ``"gelu"``, ``"exp"``).
    dtype_str:
        Key from ``SUPPORTED_DTYPES`` (e.g. ``"bf16"``, ``"fp8_e4m3"``,
        ``"int8"``).
    tag:
        Registry tag used to look up the segment boundaries when
        ``clamp=True``.  Required if ``clamp=True``.
    clamp:
        When ``True``, clamp the input to the segment range ``[x_min, x_max]``
        from the registry before calling the original function.  Useful for
        isolating the effect of clamping from the effect of approximation.

    Example::

        bind_quantize("gelu", "bf16")
        bind_quantize("gelu", "bf16", tag="kartik-thesis", clamp=True)
    """
    if dtype_str not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unknown dtype '{dtype_str}'. Choose from: {list(SUPPORTED_DTYPES)}"
        )
    if clamp and tag is None:
        raise ValueError("'tag' is required when clamp=True")

    clamp_range: tuple[float, float] | None = None
    if clamp:
        config      = get_app_config(name, tag)
        clamp_range = (config.segments[0].start, config.segments[-1].end)

    mod         = _resolve_module(name)
    dtype       = SUPPORTED_DTYPES[dtype_str]
    original_fn = getattr(mod, name)

    if name not in _original_fns:
        _original_fns[name] = (mod, original_fn)

    setattr(mod, name, _build_quantized_fn(original_fn, dtype, clamp_range))

    if name == "exp" and softmax_via_exp:
        _bind_softmax_via_exp()


def bind_all_quantize(dtype_str: str, tag: str = "kartik-thesis", clamp: bool = False) -> None:
    """Bind dtype-cast wrappers for every function registered under *tag*.

    This is for ablation studies to isolate the effect of quantization from
    the approximation.  Set ``clamp=True`` to additionally isolate the effect
    of clamping inputs to the segment boundaries.

    Functions found in neither torch.nn.functional nor torch are silently
    skipped.

    Parameters
    ----------
    dtype_str:
        Key from ``SUPPORTED_DTYPES`` (e.g. ``"bf16"``, ``"int8"``).
    tag:
        Registry tag used to select which function names to bind.
    clamp:
        When ``True``, each wrapper clamps its input to the ``[x_min, x_max]``
        range from the registry before calling the original function.

    Example::

        bind_all_quantize("bf16")                        # quantize only
        bind_all_quantize("bf16", clamp=True)            # quantize + clamp
        # ... run inference ...
        unbind_all()
    """
    for name in get_all_names_by_tag(tag):
        if hasattr(F, name) or hasattr(torch, name):
            bind_quantize(name, dtype_str, tag=tag, clamp=clamp)


def unbind_approx(name: str) -> None:
    """Restore the original function for *name*.  Does nothing if not bound.

    Unbinding ``exp`` also restores ``F.softmax`` if it was patched as a
    consequence of ``softmax_via_exp`` being set.
    """
    if name in _original_fns:
        mod, fn = _original_fns.pop(name)
        setattr(mod, name, fn)
    if name == "exp":
        unbind_approx("softmax")


def unbind_all() -> None:
    """Restore all patched functions in torch.nn.functional and torch."""
    for name, (mod, fn) in list(_original_fns.items()):
        setattr(mod, name, fn)
    _original_fns.clear()


@contextmanager
def approx_context(bindings: Union[str, list[tuple[str, str]]]):
    """Context manager: applies approximations on enter and restores on exit.

    Parameters
    ----------
    bindings:
        Either a tag string (e.g. ``"kartik-thesis"``) to bind every function
        registered under that tag, or a list of ``(name, tag)`` pairs for a
        specific subset.

    Example::

        with approx_context("kartik-thesis"):
            top1, top5 = validate(model, val_loader)

        with approx_context([("gelu", "kartik-thesis")]):
            top1, top5 = validate(model, val_loader)
    """
    if isinstance(bindings, str):
        tag = bindings
        bindings = [(name, tag) for name in get_all_names_by_tag(tag)]
    bound_names = [name for name, _ in bindings]
    try:
        bind_approx_list(bindings)
        yield
    finally:
        for name in bound_names:
            unbind_approx(name)
