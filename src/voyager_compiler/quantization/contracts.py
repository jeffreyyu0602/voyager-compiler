# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""What it means for two values to share a quantization domain.

A quantized tensor is a payload plus the scale that reconstructs it. Most
passes only ever look at one value at a time, and for arithmetic that is
enough. Data *movement* is different: when a ``cat`` writes two producers'
codes side by side into one buffer, or a ``max_pool2d`` selects among codes
without touching them, the result is only meaningful if every contributing
value was encoded against the *same* scale. Otherwise the operation has to
rescale -- which on an accelerator whose concatenation is an address
calculation is not something it can do at all.

This module states that relation. It defines:

* :class:`PreparedQuantizationContract` -- the structural identity of a
  domain. Two observers describe the same domain exactly when their contracts
  are equal, which is why it is a frozen dataclass with no tensor in it: the
  *ranges* differ before calibration converges, the *domain* must not.
* :class:`PreparedDomainRelation` -- how an operation relates its inputs'
  domains to its output's.
* :class:`PreparedDomainRuleSet` -- a target's and the compiler's declarations
  of which operations carry which relation, composed so a target's rule wins.
* :class:`QuantizationContractTransferGraph` -- the dataflow those rules
  induce, which is what region formation floods over.

Nothing here inspects a graph or decides policy. It is vocabulary, and the
pass that uses it is :mod:`voyager_compiler.quantization.regions`.

Why a *declared* rule rather than a pattern match: an operation's effect on a
quantization domain is not derivable from its schema. ``aten.cat`` joins its
inputs' domains, and so does a channel shuffle -- but ``aten.add`` joins the
domains of its *inputs* without its result belonging to them, and no amount
of looking at the operator signature distinguishes those.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple

#: ``node.meta`` keys. They ride the graph because region identity has to
#: survive from observer insertion, through PT2E conversion, into lowering --
#: a target's fusion, residual and QDQ passes all read it.
QUANTIZATION_REGION = "quantization_region"
QUANTIZATION_INPUT_REGION = "quantization_input_region"
QUANTIZATION_DOMAIN_DTYPE = "quantization_domain_dtype"
QUANTIZATION_DOMAIN_ENTRY = "quantization_domain_entry"
QUANTIZATION_DOMAIN_ALIAS = "quantization_domain_alias"

# A vector-scaled join is prepared as several encoded producer boundaries and
# one output observer that aliases their joined representation.  These marks
# survive PT2E preparation so conversion can materialize separate payload and
# fine-scale joins without teaching a target lowering how to rediscover the
# prepared observer topology.
VSQ_JOIN_ENTRY = "vsq_join_entry"
VSQ_JOIN_ALIAS = "vsq_join_alias"

#: The fine/coarse endpoint pair one vector-scaled operand was split against,
#: as ``(fine_quant_max, coarse_scale_quant_max)``.
#:
#: **Only for an operand whose quantizer is not in the graph.** An activation's
#: ``quantize_vsq`` carries both as call arguments, and a call argument
#: survives what a meta key does not, so reading the operator is right wherever
#: the operator exists. A *weight* has no such node: the converter knew the
#: parameter and precomputed all three components into ``get_attr``s, erasing
#: the only thing that recorded the split.
#:
#: Falling back to the machine's declared defaults there is right for a pairing
#: that uses them and *silently wrong* for one that does not -- the endpoints
#: fix the rebalance between the stored mantissa and the coarse scale, so a
#: wrong pair is every channel scaled by a constant factor, with nothing
#: downstream able to notice. So the converter writes them down where it still
#: knows them.
VSQ_SCALE_ENDPOINTS = "vsq_scale_endpoints"


__all__ = [
    "PreparedDomainInputFormat",
    "PreparedDomainRelation",
    "PreparedDomainRuleSet",
    "PreparedDomainTransfer",
    "PreparedDomainTransferRule",
    "PreparedQuantizationContract",
    "QUANTIZATION_DOMAIN_ALIAS",
    "QUANTIZATION_DOMAIN_DTYPE",
    "QUANTIZATION_DOMAIN_ENTRY",
    "QUANTIZATION_INPUT_REGION",
    "QUANTIZATION_REGION",
    "QuantizationContractTransferGraph",
    "VSQ_SCALE_ENDPOINTS",
    "VSQ_JOIN_ALIAS",
    "VSQ_JOIN_ENTRY",
]


@dataclass(frozen=True)
class PreparedQuantizationContract:
    """The structural identity of one prepared quantization domain.

    Everything that decides how a code is *interpreted*, and nothing that
    decides what the code happens to be. The calibrated range is deliberately
    absent: two observers that have seen different data still describe the
    same domain, and joining them is precisely the operation regions exist to
    perform. Including the range would make every pair incompatible until the
    join that compatibility is a precondition for.
    """

    dtype: Optional[str] = None
    qscheme: Optional[str] = None
    quant_min: Optional[float] = None
    quant_max: Optional[float] = None
    rounding: str = "nearest"
    ch_axis: Optional[int] = None
    block_size: Optional[int] = None
    scale_dtype: Optional[str] = None
    scale_quant_max: Optional[float] = None
    force_scale_power_of_two: bool = False
    coarse_dtype: Optional[str] = None
    coarse_qscheme: Optional[str] = None
    coarse_ch_axis: Optional[int] = None
    coarse_quant_max: Optional[float] = None
    coarse_scale_quant_max: Optional[float] = None
    coarse_rounding: str = "nearest"

    @classmethod
    def from_observer(cls, observer) -> Optional["PreparedQuantizationContract"]:
        """Canonicalize one observer, or ``None`` if it is not one.

        ``qscheme`` is normalized to its string value so an enum member and
        the string it wraps do not describe two different domains.
        """
        if observer is None or not hasattr(observer, "dtype"):
            return None
        coarse = getattr(observer, "coarse", None) or {}
        return cls(
            dtype=_text(getattr(observer, "dtype", None)),
            qscheme=_text(getattr(observer, "qscheme", None)),
            quant_min=_number(getattr(observer, "quant_min", None)),
            quant_max=_number(getattr(observer, "quant_max", None)),
            rounding=_text(getattr(observer, "rounding", "nearest")),
            ch_axis=_integer(getattr(observer, "ch_axis", None)),
            block_size=_integer(getattr(observer, "block_size", None)),
            scale_dtype=_text(getattr(observer, "scale_dtype", None)),
            scale_quant_max=_number(
                getattr(observer, "scale_quant_max", None)
            ),
            force_scale_power_of_two=bool(
                getattr(observer, "force_scale_power_of_two", False)
            ),
            coarse_dtype=_text(coarse.get("dtype")),
            coarse_qscheme=_text(coarse.get("qscheme")),
            coarse_ch_axis=_integer(coarse.get("ch_axis")),
            coarse_quant_max=_number(coarse.get("quant_max")),
            coarse_scale_quant_max=_number(coarse.get("scale_quant_max")),
            coarse_rounding=_text(coarse.get("rounding", "nearest")),
        )


def _text(value) -> Optional[str]:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _number(value) -> Optional[float]:
    return None if value is None else float(value)


def _integer(value) -> Optional[int]:
    return None if value is None else int(value)


class PreparedDomainInputFormat(Enum):
    """Where an operation's tensor inputs sit in its arguments."""

    #: The first positional argument, e.g. ``relu(x)``.
    FIRST = "first"
    #: One sequence argument holding all of them, e.g. ``cat([a, b])``.
    SEQUENCE = "sequence"
    #: The first ``input_count`` positional arguments, e.g. ``add(a, b)``.
    PREFIX = "prefix"


class PreparedDomainRelation(Enum):
    """How an operation relates its inputs' domain to its result's."""

    #: The result *is* the input's codes, moved. One input, and the result
    #: belongs to the input's domain.
    ALIAS = "alias"
    #: The result is built from the inputs' codes with a declared component
    #: mapping -- a zero-valued pad, whose added elements are the domain's
    #: representation of zero rather than a new value.
    MAP_COMPONENTS = "map_components"
    #: Every input, *and the result*, share one domain. ``cat``, ``stack``, a
    #: channel shuffle: the result is literally the inputs' bytes adjacent.
    JOIN = "join"
    #: Every input shares one domain, and the result does **not** join it.
    #: A residual add: the two addends must be commensurable to be summed,
    #: but the sum's range is its own, and forcing the result into the
    #: addends' domain would quantize it against the wrong scale.
    INPUT_JOIN = "input_join"


@dataclass(frozen=True)
class PreparedDomainTransferRule:
    """One declaration: this operation carries this relation.

    ``region_root`` marks an operation that *starts* a region -- flood
    filling begins at it. ``forcing_root`` additionally says the region's
    domain wins over an existing incompatible one at its frontier, which is
    how a target states "my shuffle cannot rescale, so its neighbours move".

    ``predicate`` narrows a target to a subset of its matches; a ``cat`` that
    is a channel shuffle behaves differently from one that is not, and the
    difference is in the graph rather than in the target name.
    """

    target: object
    input_format: PreparedDomainInputFormat = PreparedDomainInputFormat.FIRST
    relation: PreparedDomainRelation = PreparedDomainRelation.ALIAS
    input_count: int = 1
    region_root: bool = False
    forcing_root: bool = False
    predicate: Optional[Callable] = None

    def __post_init__(self):
        if self.input_count < 1:
            raise ValueError("a transfer rule needs at least one input")
        if (
            self.input_format is PreparedDomainInputFormat.FIRST
            and self.input_count != 1
        ):
            raise ValueError("the FIRST format carries exactly one input")
        if self.forcing_root and not self.region_root:
            raise ValueError("a forcing root must also be a region root")


@dataclass(frozen=True)
class PreparedDomainTransfer:
    """One resolved rule together with the node it matched."""

    node: object
    rule: PreparedDomainTransferRule
    inputs: Tuple = ()


@dataclass(frozen=True)
class PreparedDomainRuleSet:
    """The declared rules, in precedence order.

    A target's rules are composed *ahead* of the shared ones so that a target
    can narrow an operation the compiler already describes -- Alpha's
    ``cat`` is a plain join unless it is a channel shuffle, in which case it
    forces its domain -- without the shared rule having to know about it.
    """

    rules: Tuple[PreparedDomainTransferRule, ...] = ()

    def compose(self, *others: "PreparedDomainRuleSet") -> "PreparedDomainRuleSet":
        """This set's rules first, then each of ``others`` in turn."""
        composed = list(self.rules)
        for other in others:
            composed.extend(other.rules)
        return PreparedDomainRuleSet(tuple(composed))

    def resolve(self, node) -> Optional[PreparedDomainTransfer]:
        """The first rule matching ``node``, with its inputs resolved."""
        for rule in self.rules:
            if not _matches(node, rule):
                continue
            inputs = _inputs(node, rule)
            if inputs is None:
                continue
            return PreparedDomainTransfer(node=node, rule=rule, inputs=inputs)
        return None

    def transfer(self, node) -> Optional[PreparedDomainTransfer]:
        """The resolved rule, unless it is an ``INPUT_JOIN``.

        Excluded on purpose. An ``INPUT_JOIN`` does not transfer a domain to
        its result, so a caller walking result-ward must not see it -- that
        is the whole distinction the relation exists to draw.
        """
        resolved = self.resolve(node)
        if resolved is None:
            return None
        if resolved.rule.relation is PreparedDomainRelation.INPUT_JOIN:
            return None
        return resolved

    def input_join(self, node) -> Optional[PreparedDomainTransfer]:
        """The resolved rule, only if it *is* an ``INPUT_JOIN``."""
        resolved = self.resolve(node)
        if resolved is None:
            return None
        if resolved.rule.relation is not PreparedDomainRelation.INPUT_JOIN:
            return None
        return resolved


def _matches(node, rule: PreparedDomainTransferRule) -> bool:
    if getattr(node, "op", None) != "call_function":
        return False
    if node.target is not rule.target:
        return False
    if rule.predicate is not None and not rule.predicate(node):
        return False
    return True


def _inputs(node, rule: PreparedDomainTransferRule) -> Optional[Tuple]:
    """The tensor inputs the rule names, or ``None`` if they are not there.

    Returning ``None`` rather than raising lets a later rule match the same
    target with a different arity, which is what makes the rule list a
    precedence order rather than a lookup table.
    """
    from torch.fx import Node

    args = getattr(node, "args", ())
    if rule.input_format is PreparedDomainInputFormat.FIRST:
        if not args or not isinstance(args[0], Node):
            return None
        return (args[0],)
    if rule.input_format is PreparedDomainInputFormat.PREFIX:
        if len(args) < rule.input_count:
            return None
        prefix = args[: rule.input_count]
        if not all(isinstance(argument, Node) for argument in prefix):
            return None
        return tuple(prefix)
    if not args or not isinstance(args[0], (list, tuple)):
        return None
    sequence = tuple(
        argument for argument in args[0] if isinstance(argument, Node)
    )
    if len(sequence) < 2:
        return None
    return sequence


@dataclass
class QuantizationContractTransferGraph:
    """The dataflow the declared rules induce over one prepared graph.

    Built once and asked three kinds of question:

    * is this an edge -- do these two values have to agree?
    * what is connected to this root, following only edges?
    * what enters and what leaves that connected set?

    Deliberately *undirected* when flooding. A domain constraint is
    symmetric: if a ``cat``'s result must share its inputs' scale, then
    changing either end changes the other, and a directed walk would find
    only half the values a decision affects.
    """

    model: object
    rule_set: PreparedDomainRuleSet
    _transfers: dict = field(default_factory=dict, init=False)
    _neighbours: dict = field(default_factory=dict, init=False)
    _roots: list = field(default_factory=list, init=False)
    _forcing: set = field(default_factory=set, init=False)

    def __post_init__(self):
        for node in self.model.graph.nodes:
            resolved = self.rule_set.resolve(node)
            if resolved is None:
                continue
            self._transfers[node] = resolved
            if resolved.rule.region_root:
                self._roots.append(node)
            if resolved.rule.forcing_root:
                self._forcing.add(node)
            if resolved.rule.relation is PreparedDomainRelation.INPUT_JOIN:
                # Its inputs are tied to each other, not to it. Linking them
                # through the node would drag the sum into the addends'
                # domain, which is exactly what INPUT_JOIN denies.
                for left in resolved.inputs:
                    for right in resolved.inputs:
                        if left is not right:
                            self._link(left, right)
                continue
            for source in resolved.inputs:
                self._link(node, source)

    def _link(self, left, right) -> None:
        self._neighbours.setdefault(left, set()).add(right)
        self._neighbours.setdefault(right, set()).add(left)

    @property
    def roots(self) -> tuple:
        """Region roots, in graph order."""
        return tuple(self._roots)

    def transfer(self, node) -> Optional[PreparedDomainTransfer]:
        return self._transfers.get(node)

    def is_edge(self, node) -> bool:
        return node in self._transfers

    def is_forcing_root(self, node) -> bool:
        return node in self._forcing

    def connected_component(self, root) -> tuple:
        """``(members, frontiers)`` of the region rooted at ``root``.

        Flooding reaches two kinds of node, and conflating them is a real
        error rather than a tidiness question.

        A node with a *declared relation* carries the domain onward: it is a
        **member**, and its codes are in the region.

        A node reached only because some member's rule named it as an input
        has no relation of its own. It transfers nothing. Its output enters
        the region and its own operands do not -- a convolution pulled in as
        a ReLU's input contributes its result, not its weights. So it is a
        **frontier**: the place the region's domain is imposed, and the place
        the region stops.

        Reading those the other way round is what put a convolution's weight
        quantizer inside a concatenation's region, and then a second
        quantizer on top of it.

        Frontiers are returned as ``(value, in-region consumers)``: a
        frontier read by something outside the region keeps feeding it
        unquantized, because that consumer is not bound by this domain.
        """
        seen = {root}
        pending = [root]
        while pending:
            current = pending.pop()
            for neighbour in self._neighbours.get(current, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    pending.append(neighbour)

        order = {node: index for index, node in enumerate(self.model.graph.nodes)}
        ordered = sorted(seen, key=lambda node: order.get(node, -1))
        members = tuple(node for node in ordered if node in self._transfers)
        inside = set(members)

        entering = {}
        for node in members:
            for source in self._transfers[node].inputs:
                if source in inside:
                    continue
                entering.setdefault(source, []).append(node)
        frontiers = tuple(
            (source, tuple(consumers))
            for source, consumers in sorted(
                entering.items(), key=lambda item: order.get(item[0], -1)
            )
        )
        return members, frontiers

    def exits(self, members) -> tuple:
        """Region values read by something outside it."""
        inside = set(members)
        leaving = [
            node
            for node in members
            if any(user not in inside for user in node.users)
        ]
        order = {node: index for index, node in enumerate(self.model.graph.nodes)}
        return tuple(sorted(leaving, key=lambda node: order.get(node, -1)))
