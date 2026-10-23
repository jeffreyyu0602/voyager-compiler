# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Target profiles: the decisions a compilation target makes for itself.

A workload adapter under ``test/utils/models/`` builds a model, its inputs and
its calibration data. None of that depends on where the model is compiled to,
which is why one adapter already serves every quantization scheme.

What *does* depend on the target is the compilation policy applied to that
model: which operator sequences fuse into one pipelined stage, how operands
are quantized, and what the accelerator's resources are. Those live here, one
module per target, so adding a target adds a file rather than editing the
workload adapters.

A profile is a plain module exposing:

``TARGET``                  the ``--target`` name
``vector_pipeline(args)``   fusion pattern stages, passed to ``transform()``
``quantizer(args)``         the configured quantizer to prepare PT2E with
``accelerator_config(args)``the ``AcceleratorConfig`` the compiler consumes

It is deliberately not a class. There is no shared behaviour to inherit, and a
base class would invite targets to override pieces of each other's policy.
"""

from voyager_compiler.hardware_config import (
    STRUCTURE_DERIVED_FIELDS,
    AcceleratorConfig,
)

from . import gemmini, generic, magnet_alpha

_PROFILES = {
    generic.TARGET: generic,
    gemmini.TARGET: gemmini,
    magnet_alpha.TARGET: magnet_alpha,
}

DEFAULT_TARGET = generic.TARGET


def names() -> tuple:
    """Registered target names, for the CLI's ``choices``."""
    return tuple(sorted(_PROFILES))


def resolve(target: str):
    """The profile module for ``target``.

    Fail closed: an unknown target is a typo or an unregistered profile, and
    silently falling back to the default would compile for the wrong machine.
    """
    try:
        return _PROFILES[target]
    except KeyError:
        raise ValueError(
            f"unknown target {target!r}; registered: {', '.join(names())}"
        ) from None


def build_config(args, profile, *, extra=None):
    """An ``AcceleratorConfig`` from a target's declared hardware profile.

    ``profile`` is the target's authoritative machine description. A field the
    command line actually named wins; every other field comes from the profile.

    This is the single place that precedence is decided, so it cannot drift
    between targets -- and, more to the point, a declared value can now only be
    lost by being absent from the profile. It used to be possible to lose one
    silently: each target tested ``getattr(args, field) is None`` to mean "the
    CLI said nothing", but several machine flags carry a non-``None`` argparse
    default, so the test never fired and the target's own number was dropped.
    ``test_codegen`` clears the un-named ones before a target sees them, and
    routing every target through here keeps that contract in one place.

    Args:
        args: Parsed CLI namespace, already sentinel-cleared.
        profile: ``{field: value}`` the target's hardware fixes.
        extra: Fields set unconditionally -- capability declarations the CLI
            has no say over, such as ``compute_units``.

    Returns:
        The resolved config.
    """
    config = AcceleratorConfig.from_args(args)
    resolved = {
        field: getattr(config, field) for field in config.__dataclass_fields__
    }
    for field, value in profile.items():
        if getattr(args, field, None) is not None:
            continue  # the command line named it; leave it alone
        resolved[field] = value
        # A structural field fills the flat fields it describes, but only
        # where they are ``None``.  They carry real defaults, so blank the
        # ones this structure owns and the CLI did not name.
        for derived in STRUCTURE_DERIVED_FIELDS.get(field, ()):
            if getattr(args, derived, None) is None:
                resolved[derived] = None
    resolved.update(extra or {})
    return AcceleratorConfig(**resolved)


__all__ = ["DEFAULT_TARGET", "build_config", "names", "resolve"]
