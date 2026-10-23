# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Exact ``(workload, target)`` pairs and the policy each one carries.

A workload adapter under ``test/utils/models/`` builds a model; a target
profile under ``test/utils/targets/`` says how that target compiles. Neither
knows which quantization candidates have actually been *characterized* for a
given pairing -- that is a fact about evidence, not about the model or the
machine, so it lives here.

A pair module exposes:

``WORKLOAD``, ``TARGET``   the tuple it serves
``CANDIDATES``            the validated qspec tuples, keyed by candidate id
``DEFAULT_CANDIDATE``     the one used when the CLI names no qspec
Additional fields may describe target-required per-layer policy.

Selection is menu-driven, not override-driven: the pair declares what is
legal, the CLI picks among it, and anything else is refused. A qspec tuple
that was never characterized on this pairing is not a preference to honour.
"""

import copy

_PAIRS = {}


def register(module) -> None:
    """Register one pair module. Called by this package's imports below."""
    key = (module.WORKLOAD, module.TARGET)
    if key in _PAIRS:
        raise ValueError(f"pair {key} is already registered")
    _PAIRS[key] = module


def names() -> tuple:
    """Registered ``(workload, target)`` tuples."""
    return tuple(sorted(_PAIRS))


def resolve(workload: str, target: str):
    """The pair module for this tuple, or ``None`` if none is registered.

    ``None`` is not a failure: a workload may be compiled for a target with no
    pair-specific policy, in which case the target profile alone governs.
    """
    return _PAIRS.get((workload, target))


CORE_QSPEC_FIELDS = ("activation", "output_activation", "weight")


def select_candidate(args, candidates, default, *, label):
    """Resolve the CLI's qspecs against a pair's candidate menu.

    Unspecified qspecs are filled from ``default``; the result must match
    exactly one registered candidate. Shared by every pair so the rule cannot
    drift between them.

    Returns:
        ``(candidate_id, effective_args)`` -- a *copy* of ``args`` carrying the
        chosen candidate's qspecs. The caller's namespace is never mutated.

    Raises:
        ValueError: If the requested combination is not a registered candidate,
            or an explicit ``--bias`` contradicts the candidate's.
    """
    effective = copy.copy(args)
    requested = {
        field: (
            getattr(effective, field, None)
            if getattr(effective, field, None) is not None
            else candidates[default][field]
        )
        for field in CORE_QSPEC_FIELDS
    }
    matches = [
        candidate_id
        for candidate_id, qspecs in candidates.items()
        if requested == {f: qspecs[f] for f in CORE_QSPEC_FIELDS}
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{label} accepts only its registered candidates "
            f"({', '.join(sorted(candidates))}); received {requested!r}"
        )
    candidate_id = matches[0]
    chosen = candidates[candidate_id]

    supplied_bias = getattr(effective, "bias", None)
    if supplied_bias is not None and supplied_bias != chosen["bias"]:
        raise ValueError(
            f"CLI bias qspec conflicts with {candidate_id!r}: "
            f"{supplied_bias!r} != {chosen['bias']!r}"
        )
    for field, value in chosen.items():
        setattr(effective, field, value)
    return candidate_id, effective


from . import (  # noqa: E402
    resnet18_gemmini,
    yolo5n_0_5_magnet_alpha,
)

register(resnet18_gemmini)
register(yolo5n_0_5_magnet_alpha)

__all__ = [
    "CORE_QSPEC_FIELDS",
    "names",
    "register",
    "resolve",
    "select_candidate",
]
