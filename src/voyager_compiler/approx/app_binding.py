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
    quadratic_app_synth,
    quadratic_app_synth_mp,
    quadratic_stable_app_template,
)

# ---------------------------------------------------------------------------
# Map template_name -> segment evaluation function
# ---------------------------------------------------------------------------
_TEMPLATE_FN = {
    "quadratic_app_template": quadratic_app_template,
    "quadratic_app_synth":    quadratic_app_synth,
    "quadratic_app_synth_mp": quadratic_app_synth_mp,
    "linear_app_template":    linear_app_template,
    "quadratic_stable_app_template": quadratic_stable_app_template,
}

# Registry of active patches: name -> (module, original_fn)
# The module is whichever of F or torch the function was found on.
_original_fns: dict[str, tuple[ModuleType, Callable]] = {}

# Registry of extra patches for ops that have multiple entry points.
# e.g. "sigmoid" exists on F, torch, and torch.Tensor — nn.Sigmoid calls
# torch.sigmoid, some models call x.sigmoid().  Patching only F.sigmoid
# misses both.  Keys are "{name}@{location}" to avoid collisions.
_extra_originals: dict[str, tuple[object, str, Callable]] = {}

# Ops that need patching on torch AND torch.Tensor in addition to F.
_MULTI_ENTRY_OPS = {"sigmoid", "exp", "tanh"}

# When True, binding "exp" also replaces F.softmax with a pure-Python
# implementation that routes through torch.exp, so the patched exp is used
# inside softmax as well.
softmax_via_exp: bool = False


def _manual_softmax(input: torch.Tensor, dim: int = -1, **kwargs) -> torch.Tensor:
    """Numerically-stable softmax that routes through torch.exp so the
    patched exp is used."""
    x_max = input.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(input - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def _bind_softmax_via_exp() -> None:
    """Replace F.softmax, Tensor.softmax, and F.scaled_dot_product_attention
    with pure-Python implementations that route through torch.exp, so
    whatever patch is on torch.exp is also used inside softmax.

    This covers all three softmax paths found in TIMM and HuggingFace models:
      - F.softmax          (Swin, Lambda-ResNet, HF eager attention)
      - Tensor.softmax     (HaloNet, SeBotNet, CrossViT)
      - F.scaled_dot_product_attention  (ViT, DeiT — fused SDPA kernel)

    Idempotent: does nothing if softmax is already patched.
    """
    if "softmax" in _original_fns:
        return

    # 1. Patch F.softmax
    original_softmax = F.softmax
    _original_fns["softmax"] = (F, original_softmax)
    F.softmax = functools.wraps(original_softmax)(_manual_softmax)

    # 2. Patch Tensor.softmax
    original_tensor_softmax = torch.Tensor.softmax
    _extra_originals["softmax@Tensor"] = (torch.Tensor, "softmax", original_tensor_softmax)
    def _tensor_softmax(self, dim=-1, **kwargs):
        return _manual_softmax(self, dim=dim, **kwargs)
    torch.Tensor.softmax = _tensor_softmax

    # 3. Patch F.scaled_dot_product_attention with a manual implementation
    #    that uses the (now-patched) F.softmax → torch.exp path.
    if hasattr(F, "scaled_dot_product_attention"):
        original_sdpa = F.scaled_dot_product_attention
        _extra_originals["sdpa@F"] = (F, "scaled_dot_product_attention", original_sdpa)

        def manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                        is_causal=False, scale=None, **kwargs):
            import math
            L, S = query.size(-2), key.size(-2)
            scale = scale or (1.0 / math.sqrt(query.size(-1)))
            attn_weight = query @ key.transpose(-2, -1) * scale
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1
                )
                attn_weight = attn_weight.masked_fill(causal_mask, float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn_weight = attn_weight + attn_mask
            # This softmax now routes through torch.exp via our patch
            attn_weight = _manual_softmax(attn_weight, dim=-1)
            if dropout_p > 0.0:
                attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
            return attn_weight @ value

        F.scaled_dot_product_attention = manual_sdpa


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
    precisions    = config.precisions if config.precisions else None

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

            if precisions is not None:
                seg_result = eval_segment(x_c, coeffs, dtype, precisions).to(compute_dtype)
            else:
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
        orig_dtype = x.dtype
        x_q = x.to(dtype)
        if clamp_range is not None:
            x_q = x_q.clamp(*clamp_range)
        out = original_fn(x_q.to(compute_dtype), *args, **kwargs)
        return out.to(dtype).to(orig_dtype)

    return quantized_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _bind_extra_entry_points(name: str, replacement_fn: Callable) -> None:
    """Patch extra entry points (torch.xxx, torch.Tensor.xxx) for *name*.

    Some ops like sigmoid/exp/tanh are callable via F.xxx, torch.xxx, and
    x.xxx() (tensor method).  nn.Sigmoid calls torch.sigmoid; some models
    call x.sigmoid().  This ensures all paths are intercepted.
    """
    if name not in _MULTI_ENTRY_OPS:
        return

    # Patch torch.xxx — use the replacement directly (no recursion risk
    # because torch.xxx typically delegates to Tensor.xxx or a C++ call).
    torch_key = f"{name}@torch"
    if torch_key not in _extra_originals and hasattr(torch, name):
        _extra_originals[torch_key] = (torch, name, getattr(torch, name))
        setattr(torch, name, replacement_fn)

    # Patch torch.Tensor.xxx (tensor method).
    # IMPORTANT: we must NOT route through replacement_fn here if it calls
    # the original F.xxx or torch.xxx, because those may in turn call
    # input.xxx() → infinite recursion.  Instead, capture the *original*
    # Tensor method and build an independent wrapper that operates on it
    # directly.
    tensor_key = f"{name}@Tensor"
    if tensor_key not in _extra_originals and hasattr(torch.Tensor, name):
        orig_tensor_method = getattr(torch.Tensor, name)
        _extra_originals[tensor_key] = (torch.Tensor, name, orig_tensor_method)

        def _make_tensor_wrapper(orig_t_method, repl_fn):
            def _wrapper(self, *args, **kwargs):
                # Call replacement_fn which expects a tensor as first arg.
                # But replacement_fn might call F.xxx which calls Tensor.xxx
                # again.  To break the cycle, temporarily restore the original
                # Tensor method during the call.
                setattr(torch.Tensor, name, orig_t_method)
                try:
                    return repl_fn(self, *args, **kwargs)
                finally:
                    setattr(torch.Tensor, name, _wrapper)
            return _wrapper
        setattr(torch.Tensor, name, _make_tensor_wrapper(orig_tensor_method, replacement_fn))


def _unbind_extra_entry_points(name: str) -> None:
    """Restore extra entry points for *name*."""
    for suffix in ["@torch", "@Tensor"]:
        key = f"{name}{suffix}"
        if key in _extra_originals:
            obj, attr, orig = _extra_originals.pop(key)
            setattr(obj, attr, orig)


def bind_approx(name: str, tag: str) -> None:
    """Bind the piecewise approximation for *name*/*tag*.

    Looks up the function in torch.nn.functional first, then torch.
    For ops with multiple entry points (sigmoid, exp, tanh), also patches
    torch.xxx and torch.Tensor.xxx so that nn.Module wrappers and tensor
    methods are intercepted.

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
    _bind_extra_entry_points(name, approx_fn)

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

    quantized_fn = _build_quantized_fn(original_fn, dtype, clamp_range)
    setattr(mod, name, quantized_fn)
    _bind_extra_entry_points(name, quantized_fn)

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
    _unbind_extra_entry_points(name)
    if name in _original_fns:
        mod, fn = _original_fns.pop(name)
        setattr(mod, name, fn)
    if name == "exp":
        unbind_approx("softmax")


def unbind_all() -> None:
    """Restore all patched functions in torch.nn.functional and torch."""
    # Restore extra entry points first
    for key, (obj, attr, orig) in list(_extra_originals.items()):
        setattr(obj, attr, orig)
    _extra_originals.clear()
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
