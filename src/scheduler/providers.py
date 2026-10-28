# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The scheduling engines a target can be given, and the default set.

A *provider* builds the scheduler's view of one machine: it returns the tiler
context the compiler then maps every layer against. Which engine a target uses
is therefore a choice it makes once, rather than something the compiler decides
for it.

There is one engine here today. The seam exists because targets differ in how
far they diverge from it:

  * a target that matches it uses it unchanged (every lane does today);
  * a target whose datapath places loops differently *constrains* it, by
    supplying a provider that hands the engine a narrowed hint;
  * a target the engine cannot express at all *replaces* it, by registering a
    provider that builds something else entirely.

The third case is why this is a provider rather than a hook that edits the
engine's hint: a hint is private to the engine that reads it, and a target
needing a different engine has no hint to edit.
"""

from scheduler.types import SchedulerProviderSet

#: The engine the compiler ships with.
INTERSTELLAR_SCHEDULER = "interstellar"

__all__ = [
    "GENERIC_SCHEDULER_PROVIDERS",
    "INTERSTELLAR_SCHEDULER",
    "constrained_interstellar",
    "run_interstellar_scheduler",
]


def run_interstellar_scheduler(config, **options):
    """Build the interstellar view of ``config``, unconstrained.

    Args:
        config: The accelerator to plan against.
        **options: Passed through to the tiler (``dram_access_cost``,
            ``runtime_tolerance``).

    Returns:
        The tiler context every builder maps its anchor node against.
    """
    # Imported here: the tiler imports this module's registry through the
    # target package, so binding it at module scope would close the cycle.
    from voyager_compiler.codegen.transform.tiling.tiler import (
        build_interstellar_tiler,
    )

    return build_interstellar_tiler(config, **options)


def constrained_interstellar(constraint=None, issuable=None):
    """A provider running the interstellar engine under a target's narrowing.

    Two narrowings, because a target diverges from an engine in two ways.
    ``constraint`` says where loops may be *placed* -- a fact about the
    datapath the engine can express, so it is expressed in the engine's own
    hint. ``issuable`` says which of the mappings it would then return can
    actually be *issued* -- a fact about the machine's call interface, which
    the engine does not model at all and so cannot be told in advance.

    Args:
        constraint: ``(hint, config) -> hint`` -- given the engine's generic
            schedule hint, return the one this machine can honour. It is called
            with the hint the engine built, so a target narrows a placement it
            can see rather than restating one it cannot.
        issuable: ``(config) -> ((counts, point, level) -> bool)`` -- built
            once per compile against the machine, then asked of every
            candidate mapping the engine sizes, with the same resident element
            counts and mapping point the fit check reads. A ``False`` drops
            that candidate and the search goes on.

    Returns:
        A provider suitable for a :class:`SchedulerProviderSet`.
    """

    def provider(config, **options):
        from voyager_compiler.codegen.transform.tiling.tiler import (
            build_interstellar_tiler,
        )

        return build_interstellar_tiler(
            config,
            schedule_constraint=constraint,
            mapping_is_issuable=None if issuable is None else issuable(config),
            **options,
        )

    return provider


#: What a target gets when it names no engine of its own.
GENERIC_SCHEDULER_PROVIDERS = SchedulerProviderSet(
    default_backend=INTERSTELLAR_SCHEDULER,
    providers={INTERSTELLAR_SCHEDULER: run_interstellar_scheduler},
)
