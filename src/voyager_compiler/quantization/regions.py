# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Forming quantization regions, and joining their calibrated ranges.

A *region* is a maximal set of values whose codes must be interpreted against
one scale, because a data-movement operation puts them in one buffer. The
motivation is concrete: on an accelerator whose ``cat`` is an address
calculation and whose pooling selects among stored codes without decoding
them, two producers writing into one tensor at different scales produce a
tensor that means nothing. There is no rescale to insert -- the datapath has
no unit that could perform one at that point.

The pass runs in three movements.

**Discover.** :mod:`.contracts` turns declared rules into a transfer graph;
flooding it from each declared region root gives the values that must agree.

**Impose.** At each *frontier* -- a value entering the region from outside --
the region's domain is installed as an observer, and the in-region consumers
are rewired to read it. That is what makes the producer requantize into the
shared scale rather than its own.

**Join.** After calibration, every observer in a region takes the maximum
range any member saw. This is the step that gives "share a domain" its
numerical meaning: before it, region members merely have the same *kind* of
domain; after it, they have the same scale.

The join is deliberately a max rather than an average. A member whose range
is clipped produces saturated codes, which is a hard error in the output; a
member given more range than it needed loses a fraction of a bit.
"""

import copy
import logging
import operator
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.quantization.contracts import (
    QUANTIZATION_DOMAIN_ALIAS,
    QUANTIZATION_DOMAIN_DTYPE,
    QUANTIZATION_DOMAIN_ENTRY,
    QUANTIZATION_INPUT_REGION,
    QUANTIZATION_REGION,
    VSQ_JOIN_ALIAS,
    VSQ_JOIN_ENTRY,
    PreparedDomainInputFormat,
    PreparedDomainRelation,
    PreparedDomainRuleSet,
    PreparedDomainTransferRule,
    PreparedQuantizationContract,
    QuantizationContractTransferGraph,
)
from voyager_compiler.quantization.fake_quantize import FusedAmaxObsFakeQuantize

logger = logging.getLogger(__name__)

#: Where the count lands, for a workload profile that pins it.
QUANTIZATION_REGION_COUNT = "quantization_region_count"

__all__ = [
    "ACTIVATION_EPILOGUES",
    "QUANTIZATION_REGION_COUNT",
    "absorbed_by_a_matrix_call",
    "finalize_quantization_regions",
    "prepare_quantization_regions",
    "prepare_vsq_join_regions",
    "shared_domain_rules",
]


# These operations can be replayed independently over a VSQ payload and its
# block-scale tensor.  Keep this deliberately narrower than the general
# region rule set: a slice or pad may preserve a scalar quantization domain
# while changing vector blocks in a way that cannot be cloned onto the fine
# scale tensor.
_VSQ_JOIN_PASSTHROUGH_TARGETS = frozenset(
    {
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.upsample_nearest2d.vec,
        torch.ops.aten.view.default,
    }
)
_VSQ_JOIN_TARGETS = frozenset(
    {
        torch.ops.aten.cat.default,
        torch.ops.aten.stack.default,
    }
)


# -----------------------------------------------------------------------------
# The compiler's own rules
# -----------------------------------------------------------------------------


def _produces_a_sequence(node) -> bool:
    """Whether this value is a list of tensors rather than one tensor.

    Asked this way round on purpose. Only a *recorded* list is a list; a node
    with no shape metadata -- an observer this pass inserted moments ago, for
    one -- is a single tensor, because that is what every producer here is
    unless it says otherwise. Reading a missing annotation as "not a tensor"
    made every freshly inserted observer look unobservable.
    """
    return isinstance(node.meta.get("val"), (list, tuple))


def _is_a_tensor(node) -> bool:
    """Whether an observer can be placed on this value."""
    return not _produces_a_sequence(node)


def _extracts_a_tensor(node) -> bool:
    """A ``getitem`` pulling one tensor out of a multi-output producer."""
    source = node.args[0] if node.args else None
    if not isinstance(source, Node):
        return False
    return _produces_a_sequence(source) and not _produces_a_sequence(node)


def _is_zero_pad(node) -> bool:
    """A constant pad whose fill is zero.

    Only zero survives requantization unchanged: zero's code is zero in every
    symmetric domain, so a zero-padded tensor can join its source's domain
    without the padding having to be re-encoded. A pad with any other fill
    introduces a value the region's scale has not seen.

    The schema is ``pad(Tensor self, SymInt[] pad, str mode="constant",
    float? value=None)``, so the mode is argument two and the fill is argument
    three. This read position two for a while and therefore compared the
    string ``"constant"`` against zero: the rule could never fire, and every
    pad the padding pass emits was refused a domain it was entitled to.

    Both fields are read positionally *or* by keyword, because an FX graph is
    free to build ``pad(x, [0, 1], mode="reflect")`` with two positional
    arguments -- and a check that looked only at ``args`` would call that a
    zero pad, which is the same mistake as reading the wrong index, arrived
    at from the other direction.

    Correcting the index was not self-contained, which is why it waited. It
    admits convolutions that Alpha's lowering previously never saw, and one is
    a near-memory boundary whose bias arrives *already zero-padded* by the
    same pass. A projected scalar bias ``[c, c, ..., c]`` widened with zeros
    is no longer one value broadcast across lanes, so ``_scalar_bias`` refused
    it -- correctly, until it was told which of those entries are the layer's.
    It now is: the logical extent is recorded before the pad and travels with
    the value.

    It changes *generic* region formation too, since it widens which pads form
    a region, so it stands or falls on the byte gate rather than on a local
    argument.
    """
    mode = get_arg_value(node, 2, "mode", "constant")
    if mode not in (None, "constant"):
        return False  # a reflected or replicated edge is not a zero fill
    value = get_arg_value(node, 3, "value", None)
    return value is None or (isinstance(value, (int, float)) and value == 0)


def shared_domain_rules() -> PreparedDomainRuleSet:
    """Rules that hold on any machine, from operator semantics alone.

    Two kinds. A *passthrough* rearranges or selects among its input's
    elements without computing on them, so its result is its input's codes
    and it aliases the domain. A *join* places several inputs' elements in
    one result, so all of them and the result share one.

    ``max_pool2d`` and ``upsample_nearest2d`` are also region *roots*: they
    are the operations whose whole purpose is to move codes, so a region that
    exists only because of one is still a region worth forming. A plain
    ``view`` is not a root -- it joins a region that some other operation
    started, and rooting at it would form a region per reshape.
    """
    aten = torch.ops.aten
    passthrough = (
        aten.clone.default,
        aten.contiguous.default,
        aten.flatten.using_ints,
        aten.permute.default,
        aten.reshape.default,
        aten.squeeze.dim,
        aten.squeeze.dims,
        aten.transpose.int,
        aten.unsqueeze.default,
        aten.view.default,
        aten.slice.Tensor,
        aten.select.int,
    )
    rules = [
        PreparedDomainTransferRule(
            target=target,
            input_format=PreparedDomainInputFormat.FIRST,
            relation=PreparedDomainRelation.ALIAS,
        )
        for target in passthrough
    ]
    # Selection among stored codes. A root, because the operation exists to
    # move them and its output domain is its input's by construction.
    for target in (aten.max_pool2d.default, aten.upsample_nearest2d.vec):
        rules.append(
            PreparedDomainTransferRule(
                target=target,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.ALIAS,
                region_root=True,
                forcing_root=True,
            )
        )
    # Concatenation: the result *is* the inputs, adjacent.
    for target in (aten.cat.default, aten.stack.default):
        rules.append(
            PreparedDomainTransferRule(
                target=target,
                input_format=PreparedDomainInputFormat.SEQUENCE,
                relation=PreparedDomainRelation.JOIN,
                region_root=True,
            )
        )
    # A multi-output split and the extractions from it. Both are ALIAS and
    # neither is a root: a region must be able to walk *through* them to the
    # tensor underneath, because the split itself produces a list and there
    # is no single value there to carry a domain. Stopping at one would put
    # the region's frontier on a list, which nothing can observe.
    rules.append(
        PreparedDomainTransferRule(
            target=aten.chunk.default,
            input_format=PreparedDomainInputFormat.FIRST,
            relation=PreparedDomainRelation.ALIAS,
        )
    )
    rules.append(
        PreparedDomainTransferRule(
            target=operator.getitem,
            input_format=PreparedDomainInputFormat.FIRST,
            relation=PreparedDomainRelation.ALIAS,
            predicate=_extracts_a_tensor,
        )
    )
    rules.append(
        PreparedDomainTransferRule(
            target=aten.pad.default,
            input_format=PreparedDomainInputFormat.FIRST,
            relation=PreparedDomainRelation.MAP_COMPONENTS,
            region_root=True,
            forcing_root=True,
            predicate=_is_zero_pad,
        )
    )
    return PreparedDomainRuleSet(tuple(rules))


# -----------------------------------------------------------------------------
# Putting the observers where the datapath quantizes
# -----------------------------------------------------------------------------

#: Activations a matrix unit applies on the way out of a call, before it
#: truncates. They are part of the same instruction, so what the hardware
#: quantizes is the activation's result and not the accumulator's.
ACTIVATION_EPILOGUES = (
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
)

_GEMM_TARGETS = (
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
    torch.ops.aten.matmul.default,
)


def absorbed_by_a_matrix_call(node: Node) -> bool:
    """Whether this activation is a matrix call's drain rather than a value.

    A matrix unit applies its activation on the way out of the call, before
    it truncates: the convolution and the ReLU are one instruction, and there
    is no addressable value between them.

    Two passes need that fact from opposite sides.
    ``move_output_observers_after_epilogues`` uses it to put the observer
    where the datapath quantizes. A *target's domain rules* need it to keep a
    region's frontier off the accumulator: an activation declared to alias
    its input's domain becomes a region **member**, and then the value
    entering the region is the convolution's output rather than the
    activation's, so the entry observer is installed between the two. Nothing
    downstream can recover from that -- the boundary matcher looks for
    ``dequantize -> relu`` and finds ``dequantize -> quantize``, so it folds
    no activation and the record says the hardware applies none.

    Sole use is part of the question, not a caution: a matrix result read
    twice is a value in its own right, whatever else consumes it.
    """
    if node.op != "call_function" or node.target not in ACTIVATION_EPILOGUES:
        return False
    source = node.args[0] if node.args else None
    if not isinstance(source, Node):
        return False
    return (
        source.op == "call_function"
        and source.target in _GEMM_TARGETS
        and len(source.users) == 1
    )


def move_output_observers_after_epilogues(model: GraphModule) -> int:
    """Relocate a matrix op's output observer past its activation.

    Two reasons, and the second is why this runs before regions are formed.

    It is where the *hardware* quantizes. A matrix call applies its ReLU in
    the drain path and truncates afterwards, so the value that reaches the
    output's precision is the activation's, not the accumulator's. Observing
    the accumulator calibrates a range the datapath never stores.

    And it keeps observers out of region interiors. With the observer before
    the activation, an ``x -> observer -> relu -> cat`` chain makes the
    observer a *member* of the concatenation's region -- a quantization
    boundary inside a set of values that are supposed to share one domain,
    which is a contradiction the region model cannot express. Moving it past
    the activation puts it at the region's frontier, where a boundary
    belongs.

    Returns the number of observers moved.
    """
    observers = _observer_index(model)
    moved = 0
    for node, _module in list(observers.items()):
        source = node.args[0] if node.args else None
        if not isinstance(source, Node):
            continue
        if source.op != "call_function" or source.target not in _GEMM_TARGETS:
            continue
        users = list(node.users)
        if len(users) != 1:
            continue
        epilogue = users[0]
        if (
            epilogue.op != "call_function"
            or epilogue.target not in ACTIVATION_EPILOGUES
        ):
            continue

        # Capture before rewiring: the set changes as the edges move.
        downstream = [user for user in epilogue.users if user is not node]
        epilogue.replace_input_with(node, source)
        node.replace_input_with(source, epilogue)
        epilogue.append(node)
        for user in downstream:
            user.replace_input_with(epilogue, node)
        moved += 1

    if moved:
        model.graph.lint()
        model.recompile()
        logger.info("moved %d output observers past their activation", moved)
    return moved


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------


@dataclass
class _Plan:
    """One candidate region, before merging."""

    members: Tuple[Node, ...]
    contract: Optional[PreparedQuantizationContract]
    forcing: bool
    frontiers: Tuple = ()
    aliases: Tuple[Node, ...] = ()
    identifier: int = -1
    _merged: bool = field(default=False, repr=False)


def _observer_of(model: GraphModule, node: Node):
    """The observer module a ``call_module`` node runs, or ``None``."""
    if node.op != "call_module":
        return None
    module = dict(model.named_modules(remove_duplicate=False)).get(
        str(node.target)
    )
    return module if isinstance(module, FusedAmaxObsFakeQuantize) else None


def _observer_index(model: GraphModule) -> dict:
    """Every observer node, by node. Built once; the walk is not cheap."""
    modules = dict(model.named_modules(remove_duplicate=False))
    index = {}
    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        module = modules.get(str(node.target))
        if isinstance(module, FusedAmaxObsFakeQuantize):
            index[node] = module
    return index


def _contract_of(observers: dict, node: Node):
    module = observers.get(node)
    if module is None:
        return None
    return PreparedQuantizationContract.from_observer(module)


def _plan_contract(observers: dict, members, template):
    """The domain a plan will impose.

    A supplied ``template`` wins outright: a workload that declares one is
    saying "every data-movement region on this model uses this domain", which
    is a policy decision and not something to rediscover per region.

    Without one, the domain is *joined* from the members' own observers, and
    only if they already agree. Two structurally different domains inside one
    region is not something this pass may resolve by picking one -- it would
    silently reinterpret half the region's codes -- so the plan is dropped
    and the region is not formed.
    """
    if template is not None:
        return template
    contracts = {
        contract
        for contract in (_contract_of(observers, node) for node in members)
        if contract is not None
    }
    if len(contracts) != 1:
        return None
    return next(iter(contracts))


def _discover(model, graph, observers, template):
    plans = []
    for root in graph.roots:
        transfer = graph.transfer(root)
        if (
            transfer is not None
            and transfer.rule.relation is PreparedDomainRelation.INPUT_JOIN
        ):
            # Its inputs are tied to each other and its result is not tied to
            # anything, so there is no set of values here for a data-movement
            # region to be. ``_prepare_input_joins`` owns it.
            continue
        members, frontiers = graph.connected_component(root)
        # A region has to tie at least two values together to mean anything.
        # Counting members alone is not that test: a concatenation whose
        # inputs are already quantized has exactly one member -- itself --
        # and two frontiers, and tying those two frontiers to one domain is
        # the entire reason the region exists.
        if len(members) + len(frontiers) < 2:
            continue
        contract = _plan_contract(observers, members, template)
        if contract is None:
            logger.debug(
                "no single quantization domain across the region rooted at "
                "%s; leaving its members alone",
                root.name,
            )
            continue
        plans.append(
            _Plan(
                members=members,
                contract=contract,
                forcing=graph.is_forcing_root(root),
                frontiers=frontiers,
            )
        )
    return plans


def _merge(plans):
    """Union plans that overlap and agree on their domain.

    Overlap alone is not enough. Two regions can share a value and still
    describe different domains -- a ``view`` feeding both a VSQ concatenation
    and an INT8 pool -- and merging those would impose one of the two on
    values that cannot represent it. Such plans stay separate, and the shared
    value ends up with an entry observer on each side, which is the honest
    answer: a rescale really is required there.
    """
    merged = []
    for plan in plans:
        for existing in merged:
            if existing.contract != plan.contract:
                continue
            if not set(existing.members) & set(plan.members):
                continue
            order = {node: index for index, node in enumerate(existing.members)}
            combined = set(existing.members) | set(plan.members)
            existing.members = tuple(
                sorted(combined, key=lambda node: order.get(node, 1 << 30))
            )
            # A frontier of one plan can be a *member* of the other; the
            # merged region's frontier is only what is outside both.
            inside = set(existing.members)
            entering = {}
            for value, consumers in existing.frontiers + plan.frontiers:
                if value in inside:
                    continue
                entering.setdefault(value, []).extend(
                    consumer for consumer in consumers if consumer in inside
                )
            existing.frontiers = tuple(
                (value, tuple(dict.fromkeys(consumers)))
                for value, consumers in entering.items()
                if consumers
            )
            existing.forcing = existing.forcing or plan.forcing
            break
        else:
            merged.append(plan)
    return merged


# -----------------------------------------------------------------------------
# Imposition
# -----------------------------------------------------------------------------


def _new_observer(contract, template_spec):
    """One fresh observer carrying the region's domain."""
    from voyager_compiler.quantization.quantize_pt2e import (
        _create_obs_or_fq_from_qspec,
    )

    del contract
    return _create_obs_or_fq_from_qspec(template_spec, {}, False)


def _reusable_observer(model, observers, value, contract):
    """An existing observer on ``value`` that already carries the domain.

    Reused rather than duplicated when it is compatible, has one user, and
    has not already been claimed by another region. The single-user condition
    is what makes reuse safe: an observer read by something outside the
    region would have its range joined by a region it is not in.
    """
    for user in value.users:
        if user not in observers:
            continue
        if QUANTIZATION_REGION in user.meta:
            continue
        if len(user.users) != 1:
            continue
        if _contract_of(observers, user) != contract:
            continue
        return user
    return None


def _install_entry_observers(model, plan, observers, template_spec, counter):
    """Put the region's domain on every value entering it.

    Returns the number of observers inserted. The rewire is per *consumer*,
    not per value: a frontier value read by something outside the region
    keeps reading it unquantized, because that consumer is not bound by this
    region's domain.
    """
    inserted = 0
    for value, consumers in plan.frontiers:
        if value.op == "get_attr":
            # A parameter's domain is the weight quantizer's business; a
            # data-movement region has no say over it.
            continue
        if not _is_a_tensor(value):
            # The rules are supposed to walk through every multi-output
            # producer, so reaching one here means a relation is missing
            # rather than that this frontier does not matter. Say so: a
            # skipped frontier is a member quantized against the wrong scale.
            logger.warning(
                "quantization region %d has a non-tensor frontier %r; its "
                "consumers keep their own domain",
                plan.identifier,
                value.name,
            )
            continue
        entry = _reusable_observer(model, observers, value, plan.contract)
        if entry is None and template_spec is None:
            # Nothing compatible to reuse and no spec to build one from. A
            # region discovered from its own members can only *recognize* a
            # shared domain, never install one -- installing needs a spec,
            # and the only honest source of one is a caller that declared it.
            logger.debug(
                "quantization region %d cannot impose its domain on %s "
                "without a declared region qspec",
                plan.identifier,
                value.name,
            )
            continue
        if entry is None:
            name = f"quantization_region_{plan.identifier}_entry_{counter[0]}"
            counter[0] += 1
            model.add_submodule(name, _new_observer(plan.contract, template_spec))
            with model.graph.inserting_after(value):
                entry = model.graph.call_module(name, (value,))
            observers[entry] = getattr(model, name)
            inserted += 1
        for consumer in consumers:
            if consumer is entry:
                continue
            consumer.replace_input_with(value, entry)
        entry.meta[QUANTIZATION_REGION] = plan.identifier
        entry.meta[QUANTIZATION_DOMAIN_ENTRY] = True
        entry.meta[QUANTIZATION_DOMAIN_DTYPE] = plan.contract.dtype
    return inserted


def _install_exit_aliases(model, plan, graph, observers, template_spec, counter):
    """Observe a region's range without requantizing its encoded result.

    A region member already carries the region's codes.  Its output observer
    therefore records calibration data only: applying fake quantization there
    would introduce a second quantization boundary inside one physical
    domain.  Conversion recognizes the alias mark, removes the observer, and
    inserts dequantization only for consumers outside the region.

    The rewrite is per exit edge.  That preserves fan-out: a consumer outside
    the region receives the eventual dequantized value while any in-region
    consumer continues to read the raw payload.
    """
    inserted = 0
    inside = set(plan.members)
    for value in graph.exits(plan.members):
        for consumer in tuple(value.users):
            if consumer in inside:
                continue

            alias = None
            if (
                consumer in observers
                and consumer.args
                and consumer.args[0] is value
                and QUANTIZATION_REGION not in consumer.meta
                and _contract_of(observers, consumer) == plan.contract
            ):
                alias = consumer
            elif template_spec is not None:
                name = (
                    f"quantization_region_{plan.identifier}_alias_"
                    f"{counter[0]}"
                )
                counter[0] += 1
                model.add_submodule(
                    name, _new_observer(plan.contract, template_spec)
                )
                with model.graph.inserting_before(consumer):
                    alias = model.graph.call_module(name, (value,))
                observers[alias] = getattr(model, name)
                consumer.replace_input_with(value, alias)
                inserted += 1

            # A region inferred without a declared qspec can only reuse an
            # existing compatible observer; it has no policy-owned source
            # from which to manufacture another one.
            if alias is None:
                continue

            module = observers[alias]
            module.disable_fake_quant()
            module.enable_observer()
            alias.meta[QUANTIZATION_REGION] = plan.identifier
            alias.meta[QUANTIZATION_DOMAIN_ALIAS] = True
            alias.meta[QUANTIZATION_DOMAIN_DTYPE] = plan.contract.dtype
    return inserted


def _assert_no_interior_quantizer(plan, observers):
    """A region must not contain a quantizer.

    It cannot, by construction: only a node with a declared relation is a
    member, relations are declared on ``call_function`` targets, and an
    observer is a ``call_module``. An observer that reached the interior
    would be a second scale inside a set of values defined by sharing one,
    so this is checked rather than assumed -- a future rule matching a
    module would break the model silently otherwise.
    """
    interior = [
        node.name for node in plan.members if observers.get(node) is not None
    ]
    if interior:
        raise ValueError(
            f"quantization region {plan.identifier} contains quantizers "
            f"{interior}: a region is values that share one scale, and a "
            f"quantizer inside it is a second one"
        )


def prepare_quantization_regions(
    model: GraphModule,
    *,
    region_qspec=None,
    rule_set: Optional[PreparedDomainRuleSet] = None,
) -> int:
    """Form the model's quantization regions, in place.

    Args:
        model: A prepared graph -- observers inserted, not yet converted.
        region_qspec: The domain every data-movement region takes, as a
            :class:`QuantizationSpec` or a spec string. ``None`` derives each
            region's domain from its own members, and forms only the regions
            whose members already agree.
        rule_set: The declared relations. Defaults to the compiler's own; a
            target composes its rules *ahead* of these.

    Returns:
        The number of regions formed, including input joins.
    """
    if rule_set is None:
        rule_set = shared_domain_rules()

    template_spec = region_qspec
    if isinstance(template_spec, str):
        from voyager_compiler.quantization.quantizer.quantizer import (
            QuantizationSpec,
        )

        template_spec = QuantizationSpec.from_str(template_spec)
    template = None
    if template_spec is not None:
        template = PreparedQuantizationContract.from_observer(
            _new_observer(None, template_spec)
        )

    # Before anything is flooded: an observer sitting between a matrix op and
    # its activation would otherwise be discovered *inside* a region.
    move_output_observers_after_epilogues(model)

    observers = _observer_index(model)
    graph = QuantizationContractTransferGraph(model, rule_set)
    plans = _merge(_discover(model, graph, observers, template))

    # A region that does not force its domain may only be formed where every
    # exit already agrees with it. Otherwise forming it would replace a
    # boundary somebody else established -- a VSQ output, say -- with this
    # region's INT8, which is a reinterpretation rather than a placement.
    admitted = []
    for plan in plans:
        if plan.forcing:
            admitted.append(plan)
            continue
        # What the region's values are read as on the way out. A region
        # that does not force its domain may not silently replace one.
        inside = set(plan.members)
        contracts = [
            _contract_of(observers, user)
            for node in graph.exits(plan.members)
            for user in node.users
            if user not in inside
        ]
        if all(
            contract is None or contract == plan.contract
            for contract in contracts
        ):
            admitted.append(plan)
        else:
            # Not forming the region does not remove the constraint that
            # motivated it. The concatenation or shuffle still has no unit
            # that can rescale, so its producers still have to agree, and
            # this compile has just declined to make them. Say so at a level
            # somebody reads: a silent decline looks identical to there
            # having been nothing to do.
            logger.warning(
                "not forming the quantization region rooted near %s: it "
                "exits into a domain incompatible with its own and does not "
                "force it. The values it would have tied together are left "
                "on separate scales, which the datapath cannot reconcile at "
                "the operation that joins them.",
                plan.members[0].name,
            )

    counter = [0]
    inserted = 0
    aliases = 0
    for identifier, plan in enumerate(admitted):
        plan.identifier = identifier
        for node in plan.members:
            node.meta[QUANTIZATION_REGION] = identifier
            node.meta[QUANTIZATION_DOMAIN_DTYPE] = plan.contract.dtype
        inserted += _install_entry_observers(
            model, plan, observers, template_spec, counter
        )
        aliases += _install_exit_aliases(
            model,
            plan,
            graph,
            observers,
            template_spec,
            counter,
        )
        _assert_no_interior_quantizer(plan, observers)

    joins = _prepare_input_joins(
        model,
        graph,
        rule_set,
        observers,
        template_spec,
        template,
        start=len(admitted),
        counter=counter,
    )

    if inserted or aliases or joins:
        model.graph.lint()
        model.recompile()

    total = len(admitted) + joins
    model.meta[QUANTIZATION_REGION_COUNT] = total
    logger.info(
        "formed %d quantization regions (%d data-movement, %d input joins); "
        "inserted %d entry observers and %d exit aliases",
        total,
        len(admitted),
        joins,
        inserted,
        aliases,
    )
    return total


def _prepare_input_joins(
    model,
    graph,
    rule_set,
    observers,
    template_spec,
    template,
    *,
    start,
    counter,
):
    """Tie an ``INPUT_JOIN``'s inputs together without binding its result.

    A residual add is the case. Its two addends have to be commensurable --
    the datapath adds codes, not values -- but the sum's own range is larger
    than either addend's, and forcing it into their domain would clip it. So
    the region is recorded *on the add*, over its input edges only.
    """
    formed = 0
    for node in model.graph.nodes:
        resolved = rule_set.input_join(node)
        if resolved is None:
            continue
        if len(resolved.inputs) < 2:
            continue
        contract = template
        if contract is None:
            contracts = {
                found
                for found in (
                    _contract_of(observers, source)
                    for source in resolved.inputs
                )
                if found is not None
            }
            if len(contracts) != 1:
                continue
            contract = next(iter(contracts))

        identifier = start + formed

        # Every entry is resolved before any of them is stamped. Stamping as
        # we went and abandoning the join partway would leave inputs carrying
        # a region id that the *next* join then reuses -- and the calibration
        # join would give those strangers one shared scale. A half-tied join
        # is worse than an untied one.
        entries = []
        for source in resolved.inputs:
            entry = _reusable_observer(model, observers, source, contract)
            if entry is None and template_spec is None:
                # Nothing compatible to reuse and no spec to build one from,
                # so these inputs cannot be tied at all.
                entries = None
                break
            entries.append((source, entry))
        if entries is None:
            logger.debug(
                "%s: its inputs cannot be tied without a declared region "
                "qspec; leaving them on their own domains",
                node.name,
            )
            continue

        for source, entry in entries:
            if entry is None:
                name = f"quantization_input_join_{identifier}_{counter[0]}"
                counter[0] += 1
                model.add_submodule(
                    name, _new_observer(contract, template_spec)
                )
                with model.graph.inserting_after(source):
                    entry = model.graph.call_module(name, (source,))
                observers[entry] = getattr(model, name)
                node.replace_input_with(source, entry)
            entry.meta[QUANTIZATION_REGION] = identifier
            entry.meta[QUANTIZATION_DOMAIN_ENTRY] = True
            entry.meta[QUANTIZATION_DOMAIN_DTYPE] = contract.dtype
        node.meta[QUANTIZATION_INPUT_REGION] = identifier
        formed += 1
    return formed


# -----------------------------------------------------------------------------
# Vector-scaled joins
# -----------------------------------------------------------------------------


def _prepared_vsq_observer(node: Node, modules: dict):
    """Return a prepared per-tensor-coarse VSQ observer, if ``node`` is one."""
    if node.op != "call_module" or not isinstance(node.target, str):
        return None
    observer = modules.get(node.target)
    if not isinstance(observer, FusedAmaxObsFakeQuantize):
        return None
    contract = PreparedQuantizationContract.from_observer(observer)
    if contract is None or contract.qscheme != "vsq":
        return None
    if contract.coarse_qscheme != "per_tensor_symmetric":
        return None
    if contract.coarse_ch_axis is not None:
        return None
    return observer


def _vsq_join_data_inputs(node: Node) -> tuple[Node, ...]:
    """The tensor inputs of an operation legal on each VSQ component."""
    if node.target in _VSQ_JOIN_TARGETS:
        values = node.args[0] if node.args else ()
        if not isinstance(values, (tuple, list)):
            return ()
        return tuple(value for value in values if isinstance(value, Node))
    if node.target in _VSQ_JOIN_PASSTHROUGH_TARGETS:
        value = node.args[0] if node.args else None
        return (value,) if isinstance(value, Node) else ()
    return ()


def _prepared_vsq_join_exit(join: Node, modules: dict):
    """Find ``join -> supported views -> VSQ observer`` with no fan-out."""
    path = [join]
    current = join
    while len(current.users) == 1:
        user = next(iter(current.users))
        observer = _prepared_vsq_observer(user, modules)
        if observer is not None and user.args[0] is current:
            return tuple(path), user, observer
        if user.target not in _VSQ_JOIN_PASSTHROUGH_TARGETS:
            return None
        if _vsq_join_data_inputs(user) != (current,):
            return None
        path.append(user)
        current = user
    return None


def _node_tensor_shape(node: Node) -> Optional[tuple[int, ...]]:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        value = getattr(node, "value", None)
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(value.shape)


def _normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def _prepared_vsq_join_has_legal_layout(
    join: Node,
    path: tuple[Node, ...],
    observer: FusedAmaxObsFakeQuantize,
) -> bool:
    """Whether component-wise replay preserves this VSQ block layout."""
    if not isinstance(observer.ch_axis, int):
        return False
    if not isinstance(observer.block_size, int):
        return False

    inputs = _vsq_join_data_inputs(join)
    shapes = tuple(_node_tensor_shape(node) for node in inputs)
    if not shapes or any(shape is None for shape in shapes):
        return False
    reference = shapes[0]
    if any(len(shape) != len(reference) for shape in shapes[1:]):
        return False

    channel_axis = _normalize_axis(observer.ch_axis, len(reference))
    if not 0 <= channel_axis < len(reference):
        return False

    if join.target is torch.ops.aten.cat.default:
        join_axis = int(get_arg_value(join, 1, "dim", 0))
        join_axis = _normalize_axis(join_axis, len(reference))
        if join_axis != channel_axis:
            return False
    elif join.target is torch.ops.aten.stack.default:
        join_axis = int(get_arg_value(join, 1, "dim", 0))
        join_axis = _normalize_axis(join_axis, len(reference) + 1)
        if join_axis != channel_axis or len(path) < 2:
            return False
        flatten = path[1]
        if flatten.target is not torch.ops.aten.flatten.using_ints:
            return False
        start_dim = int(get_arg_value(flatten, 1, "start_dim", 0))
        end_dim = int(get_arg_value(flatten, 2, "end_dim", -1))
        if (start_dim, end_dim) != (channel_axis, channel_axis + 1):
            return False
    else:
        return False

    block_size = observer.block_size
    channel_sizes = tuple(shape[channel_axis] for shape in shapes)
    if any(
        not isinstance(size, int) or size % block_size
        for size in channel_sizes
    ):
        return False

    output_shape = _node_tensor_shape(path[-1])
    return output_shape is not None and output_shape[channel_axis] == sum(
        channel_sizes
    )


def _prepared_vsq_join_input(
    node: Node,
    join: Node,
    modules: dict,
    template: FusedAmaxObsFakeQuantize,
    is_output_boundary: Callable[[Node], bool],
):
    """Resolve one exclusive encoded producer boundary entering ``join``."""
    observer = _prepared_vsq_observer(node, modules)
    if observer is not None:
        if (
            PreparedQuantizationContract.from_observer(template)
            != PreparedQuantizationContract.from_observer(observer)
        ):
            return None
        source = node.args[0] if node.args else None
        if not isinstance(source, Node) or not is_output_boundary(source):
            return None
        if tuple(node.users) != (join,) or tuple(source.users) != (node,):
            return None
        return node, observer

    if not is_output_boundary(node):
        return None
    if tuple(node.users) != (join,):
        return None
    return node, None


def _insert_vsq_join_observer(
    model: GraphModule,
    modules: dict,
    value: Node,
    join: Node,
    observer: FusedAmaxObsFakeQuantize,
) -> Node:
    stem = f"{value.name}_vsq_output"
    name, suffix = stem, 0
    while hasattr(model, name):
        suffix += 1
        name = f"{stem}_{suffix}"
    model.add_submodule(name, observer)
    modules[name] = observer
    with model.graph.inserting_before(join):
        observer_node = model.graph.call_module(name, (value,))
    observer_node.meta = copy.copy(value.meta)
    join.replace_input_with(value, observer_node)
    return observer_node


def prepare_vsq_join_regions(
    model: GraphModule,
    *,
    is_output_boundary: Callable[[Node], bool],
    rule_set: Optional[PreparedDomainRuleSet] = None,
) -> int:
    """Prepare legal channel joins as one vector-scaled physical domain.

    The shared pass owns observer topology and component-wise join semantics;
    the target supplies only the fact it alone knows: which prepared nodes are
    real encoded output boundaries.  Returns the number of VSQ joins formed.
    """
    if not callable(is_output_boundary):
        raise TypeError("is_output_boundary must be callable")
    if rule_set is None:
        rule_set = shared_domain_rules()

    modules = dict(model.named_modules(remove_duplicate=False))
    next_region = int(model.meta.get(QUANTIZATION_REGION_COUNT, 0))
    used_observers = set()
    formed = 0

    for join in list(model.graph.nodes):
        if join.target not in _VSQ_JOIN_TARGETS:
            continue
        resolved = rule_set.resolve(join)
        if (
            resolved is None
            or resolved.rule.relation is not PreparedDomainRelation.JOIN
        ):
            continue
        if resolved.rule.forcing_root:
            # A target already owns this as mandatory data movement (for
            # example Alpha's channel shuffle), so do not reinterpret it as a
            # transparent VSQ component join.
            continue

        exit_match = _prepared_vsq_join_exit(join, modules)
        if exit_match is None:
            continue
        path, exit_node, exit_observer = exit_match
        if exit_node in used_observers:
            continue
        if not _prepared_vsq_join_has_legal_layout(join, path, exit_observer):
            continue

        inputs = tuple(resolved.inputs)
        if len(inputs) < 2 or len(set(inputs)) != len(inputs):
            continue
        matches = tuple(
            _prepared_vsq_join_input(
                input_node,
                join,
                modules,
                exit_observer,
                is_output_boundary,
            )
            for input_node in inputs
        )
        if any(match is None for match in matches):
            continue
        if any(match[0] in used_observers for match in matches):
            continue

        entries = []
        for input_node, observer in matches:
            if observer is None:
                observer = copy.deepcopy(exit_observer)
                observer.amax_history.resize_(0)
                observer.scale.resize_(1).fill_(1.0)
                input_node = _insert_vsq_join_observer(
                    model, modules, input_node, join, observer
                )
            entries.append(input_node)

        region_id = next_region + formed
        formed += 1
        contract = PreparedQuantizationContract.from_observer(exit_observer)

        for node in path:
            node.meta[QUANTIZATION_REGION] = region_id
            node.meta[QUANTIZATION_DOMAIN_DTYPE] = contract.dtype
        for entry in entries:
            entry.meta[QUANTIZATION_REGION] = region_id
            entry.meta[QUANTIZATION_DOMAIN_DTYPE] = contract.dtype
            entry.meta[VSQ_JOIN_ENTRY] = True
        exit_node.meta[QUANTIZATION_REGION] = region_id
        exit_node.meta[QUANTIZATION_DOMAIN_DTYPE] = contract.dtype
        exit_node.meta[VSQ_JOIN_ALIAS] = True
        exit_observer.disable_fake_quant()
        exit_observer.enable_observer()
        used_observers.update((*entries, exit_node))

    if formed:
        model.graph.lint()
        model.recompile()
    model.meta[QUANTIZATION_REGION_COUNT] = next_region + formed
    logger.info("formed %d vector-scaled join regions", formed)
    return formed


# -----------------------------------------------------------------------------
# The join
# -----------------------------------------------------------------------------


def finalize_quantization_regions(model: GraphModule) -> int:
    """Give every region one calibrated scale. Returns the regions joined.

    Run after calibration and before conversion. Until this runs, region
    members share the *kind* of a domain but not its range, and a converter
    that assumed otherwise would be reading a scale nobody had reconciled.

    Observers whose histories have different shapes are not joined. That is a
    per-channel observer meeting a per-tensor one, which the contract check
    should already have rejected; reaching here means the region was formed
    with a template that does not match a member, and broadcasting one into
    the other would invent per-channel ranges from a scalar.
    """
    observers = _observer_index(model)
    grouped = {}
    for node, module in observers.items():
        identifier = node.meta.get(QUANTIZATION_REGION)
        if identifier is None:
            continue
        grouped.setdefault(identifier, []).append((node, module))

    joined = 0
    for identifier, members in sorted(grouped.items()):
        histories = [
            module.amax_history
            for _node, module in members
            if module.amax_history.numel()
        ]
        if len(histories) < 2:
            continue
        shapes = {tuple(history.shape) for history in histories}
        if len(shapes) != 1:
            logger.warning(
                "quantization region %d holds observers with amax histories "
                "of shapes %s; not joining it",
                identifier,
                sorted(shapes),
            )
            continue
        # The region's range is the widest any member saw, reduced over the
        # history dimension so what every member adopts is one amax rather
        # than one history.
        group_amax = torch.stack(histories, dim=0).amax(dim=0).amax(dim=0)
        for _node, module in members:
            module.join_amax(group_amax)
        joined += 1

    logger.info("joined the calibrated range of %d quantization regions", joined)
    return joined
