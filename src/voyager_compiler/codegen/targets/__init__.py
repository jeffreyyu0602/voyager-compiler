# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler-side target backends.

A target profile under ``test/utils/targets/`` says how a target is *invoked*
-- its operand widths, its fusion pipeline, its quantizer. That runs before
compilation and never sees a graph.

This package is the other half: code that runs *during* compilation, with the
FX graph in hand. It is where a target rewrites what it must and declares what
it cannot execute, because both need the nodes.

A target module may define any of these; each is optional, and a target that
defines none behaves exactly as the untargeted compiler does:

``apply_graph_rewrites(model, config)``
    Rewrite the transformed graph for this target. May call
    ``declare_unsupported_operation`` on a call the hardware cannot run.

``layer_table(model, params)``
    How this target writes ``layers.txt``. The shared one describes a
    *bufferized* nest -- one row per loop nest, with the range of ops it
    spans -- which is not what a flat collateral has; and it reads a field
    (``op.name``) that the flat message does not carry, so it does not
    merely produce the wrong table, it raises. A target with its own
    terminal codegen answers for its own form.

``EXECUTION_FUSION_PATTERNS``
    The operator sequences this machine issues as one instruction, and so
    the sequences that must become one record. Read by a shared pass after
    the target's rewrites have built them. A target that declares none is
    unaffected -- which is every target on the shared path, whose records
    are whatever generic fusion produced.

``EXECUTION_MODES``
    The ``(scratchpad_mapping, execution_order)`` pairs this target's *own*
    terminal codegen implements. Read only when the target has one; a
    target on the shared path is described by what that path does, and
    says nothing. See ``codegen.execution_order``.

``unsupported_policy(node, error)``
    Consulted when a node cannot be scheduled. Return a reason code to
    declare that call a boundary and keep compiling, or ``None`` to let
    the failure stand. A target declares this only for failures it
    genuinely expects; anything else is a real error and must abort.

``prepared_domain_rules()``
    The quantization-domain relations this machine adds to the compiler's.
    A target declares one where its datapath reads an operation more
    narrowly than the operator alone implies -- a residual add whose
    addends must share a scale, a concatenation its shuffle cannot
    rescale. Runs before observers are calibrated, not with a graph in
    hand, which is why it takes no arguments.

``prepare_quantization(model, ...)``
    Apply target-owned observer preparation after shared PT2E observers have
    been inserted. A backend may wrap the shared region pass when ordering is
    part of its calibration contract.

``cleanup_quantization_preparation(model)``
    Remove calibration-only target state immediately before shared PT2E
    conversion. The executable reference graph must not retain temporary
    instrumentation.

``prepare_quantization_recalibration(model, reset_observers=...)``
    Configure an optional target-exact second calibration pass. Returning
    zero means that the selected target has no such pass.

Adding a target means adding a module here and registering it below. There is
no spec object and no required hook: a target implements the seam it needs and
stays silent everywhere else.
"""

from types import ModuleType
from typing import Optional

from voyager_compiler.codegen.targets import (
    gemmini,
    generic,
    magnet_alpha,
)

DEFAULT_TARGET = generic.TARGET

_TARGETS = {
    generic.TARGET: generic,
    gemmini.TARGET: gemmini,
    magnet_alpha.TARGET: magnet_alpha,
}

__all__ = [
    "DEFAULT_TARGET",
    "apply_graph_rewrites",
    "execution_fusion_patterns",
    "execution_modes",
    "finalize_quantization_calibration",
    "layer_table",
    "cleanup_quantization_preparation",
    "prepare_quantization",
    "prepare_quantization_recalibration",
    "terminal_codegen",
    "prepared_domain_rules",
    "scheduler_providers",
    "unsupported_policy",
    "names",
    "resolve",
]


def names() -> tuple:
    """Every registered target name."""
    return tuple(sorted(_TARGETS))


def resolve(target: Optional[str]) -> ModuleType:
    """The backend module for ``target``.

    Fail closed: an unknown name is a typo or an unregistered backend, and
    compiling for the default instead would silently build for the wrong
    machine.
    """
    if target is None:
        return _TARGETS[DEFAULT_TARGET]
    try:
        return _TARGETS[target]
    except KeyError:
        raise ValueError(
            f"unknown target {target!r}; registered: {', '.join(names())}"
        ) from None


def apply_graph_rewrites(model, config, target: Optional[str] = None) -> None:
    """Run ``target``'s graph rewrites, if it has any.

    Called once from ``transform()``, after the target-neutral passes have
    settled the graph and before it is handed to the tiler.
    """
    backend = resolve(target)
    hook = getattr(backend, "apply_graph_rewrites", None)
    if hook is not None:
        hook(model, config)


def terminal_codegen(target: Optional[str] = None):
    """``target``'s own terminal path, or ``None`` for the bufferized one.

    Almost every target ends the same way: bufferize, plan memory, emit the
    loop-nest IR. One does not. Alpha's collateral is a *flat* form -- one
    record per call, addresses resolved, no loop nests -- and its operators
    have no bufferized lowering at all, so running it down the shared path
    does not produce worse output, it produces none.

    Declared rather than branched on a name, and defaulting to ``None`` so
    that every target which fits the shared path keeps it untouched.
    """
    backend = resolve(target)
    return getattr(backend, "gen_code", None)


def execution_fusion_patterns(target: Optional[str] = None):
    """``target``'s one-instruction operator sequences, or none.

    Empty for every target whose records are what generic fusion already
    produces. A target declares these only when its datapath issues several
    operations as a single call, because then the grouping is not an
    optimization -- an ungrouped pair is a record the converter never sees.
    """
    return tuple(
        getattr(resolve(target), "EXECUTION_FUSION_PATTERNS", ())
    )


def execution_modes(target: Optional[str] = None):
    """The mode pairs ``target``'s own terminal codegen implements.

    Empty for a target that has no terminal codegen, and for one whose
    terminal codegen implements nothing yet -- which are different states
    with the same answer, deliberately: in both, a compile asking for a
    non-default pair has nothing that would honour it, and the honest reply
    is the same refusal.

    A path implementing a mode is not the same as a *machine* wanting it.
    Alpha's endpoint reads whole-model residency, and that is why the flag
    exists; it is declared here only once the code that carries it out is
    written, so the declaration cannot outrun the implementation.
    """
    return frozenset(getattr(resolve(target), "EXECUTION_MODES", frozenset()))


def layer_table(target: Optional[str] = None):
    """``target``'s own ``layers.txt`` writer, or ``None`` for the shared one.

    ``None`` for every target on the bufferized path, which is what the shared
    writer describes. It is a separate seam from ``terminal_codegen`` rather
    than a part of it because the two answer to different consumers -- the
    protobuf is what the endpoint executes, and this is the ordered list a
    converter and a group planner read to agree on what ran.
    """
    return getattr(resolve(target), "layer_table", None)


def scheduler_providers(target: Optional[str] = None):
    """``target``'s scheduling engines, or the ones the compiler ships with.

    A target that declares none plans with the generic engine, unchanged --
    which is every target that fits it. One whose datapath places loops
    differently, or that needs a different engine entirely, declares its own
    set; see ``scheduler.providers``.
    """
    from scheduler.providers import GENERIC_SCHEDULER_PROVIDERS

    return getattr(
        resolve(target), "SCHEDULER_PROVIDERS", GENERIC_SCHEDULER_PROVIDERS
    )


def prepared_domain_rules(target: Optional[str] = None):
    """``target``'s quantization-domain relations, plus the compiler's.

    A target that declares none gets the shared rules alone, which is what
    every generic lane forms its regions with.
    """
    from voyager_compiler.quantization.regions import shared_domain_rules

    hook = getattr(resolve(target), "prepared_domain_rules", None)
    return shared_domain_rules() if hook is None else hook()


def prepare_quantization(
    model,
    target: Optional[str] = None,
    *,
    region_qspec=None,
    domain_rules=None,
):
    """Apply the selected target's prepared-quantization contract.

    A backend hook owns any target-specific ordering around region formation.
    Without one, preserve the existing shared behavior exactly: form regions
    only when the caller requested a region domain or supplied explicit rules.
    """
    backend = resolve(target)
    hook = getattr(backend, "prepare_quantization", None)
    if hook is not None:
        return hook(
            model,
            region_qspec=region_qspec,
            domain_rules=domain_rules,
        )

    if region_qspec is not None or domain_rules is not None:
        from voyager_compiler.quantization.regions import (
            prepare_quantization_regions,
        )

        prepare_quantization_regions(
            model,
            region_qspec=region_qspec,
            rule_set=domain_rules,
        )
    return model


def cleanup_quantization_preparation(
    model, target: Optional[str] = None
):
    """Remove preparation-only state owned by ``target``."""
    hook = getattr(resolve(target), "cleanup_quantization_preparation", None)
    return model if hook is None else hook(model)


def finalize_quantization_calibration(
    model, target: Optional[str] = None
):
    """Finalize target-owned prepared domains after calibration."""
    hook = getattr(resolve(target), "finalize_quantization_calibration", None)
    return model if hook is None else hook(model)


def prepare_quantization_recalibration(
    model,
    target: Optional[str] = None,
    *,
    reset_observers: bool,
) -> int:
    """Configure one target-exact recalibration pass, when implemented."""
    hook = getattr(resolve(target), "prepare_quantization_recalibration", None)
    if hook is None:
        return 0
    return int(hook(model, reset_observers=reset_observers))


def unsupported_policy(target: Optional[str] = None):
    """``target``'s tolerance for unschedulable calls, or ``None``.

    ``None`` means the target expects every call to be schedulable, so a
    scheduling failure aborts the compile -- the untargeted behaviour.
    """
    return getattr(resolve(target), "unsupported_policy", None)
