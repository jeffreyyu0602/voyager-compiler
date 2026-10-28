# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The accelerator hardware description.

``AcceleratorConfig`` bundles every hardware knob the compiler needs — the PE
array, the on-chip L1 systolic buffers, the L2 scratchpad, DRAM, and the clock —
into one frozen object that ``transform()`` / ``compile()`` and their callees
pass around, instead of threading a dozen loose, drift-prone arguments through
every layer.

Physical units: ``dram_bandwidth`` is GB/s, ``dram_access_latency`` ns,
``frequency`` GHz, so bytes/cycle is ``dram_bandwidth / frequency`` and
per-transfer latency in cycles is ``dram_access_latency * frequency``.  The
reporting model reads this object directly as its cost knobs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, FrozenSet, Optional, Tuple

DEFAULT_PE_ARRAY_SIZE = (32, 32)
#: Input-channel counts a CNN's first layer typically has. A matrix unit with a
#: partial-input path masks its unused lanes for these rather than being fed a
#: padded operand, which is why they are the historical default.
COMMON_IMAGE_INPUT_CHANNELS = frozenset({1, 3})
DEFAULT_FREQUENCY_GHZ = 1.0
DEFAULT_INPUT_BUFFER_SIZE = 1024
DEFAULT_WEIGHT_BUFFER_SIZE = 1024
DEFAULT_ACCUM_BUFFER_SIZE = 1024
DEFAULT_SCRATCHPAD_OFFSET = 0
DEFAULT_DOUBLE_BUFFERED_L2 = True
DEFAULT_DRAM_SIZE_GB = 16.0
DEFAULT_DRAM_BANDWIDTH_GBS = 64.0
DEFAULT_DRAM_ACCESS_LATENCY_NS = 100.0


@dataclass(frozen=True)
class DTypeMode:
    """How the matrix datapath behaves at one operand width.

    A narrower operand packs more reduction lanes into the same array, and
    changes the output-channel granularity the datapath requires. A config
    that declares no modes has one implicit mode described by
    ``pe_array_size``.
    """

    operand_bits: int
    #: Reduction (input-channel) lanes at this width.
    input_lanes: int
    #: Output-channel granularity the datapath requires at this width.
    channel_alignment: int
    #: Physical banks one logical value occupies at this width. A machine
    #: that packs one value per bank leaves this ``None``; one that spreads a
    #: wider operand across several says so, because the bank a tensor lands
    #: in is then a function of its width and not of its size alone.
    banks_per_value: Optional[int] = None

    def __post_init__(self):
        for name in ("operand_bits", "input_lanes", "channel_alignment"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.banks_per_value is not None and self.banks_per_value <= 0:
            raise ValueError("banks_per_value must be positive")


class TensorRole(Enum):
    """What a memory holds, so an instance can say which operands it serves."""

    ACTIVATION = "activation"
    WEIGHT = "weight"
    OUTPUT = "output"
    PARTIAL_SUM = "partial_sum"
    #: A requantization scale or bias the datapath reads beside its operands.
    #: They are separate roles because a machine may accept them in a memory
    #: that refuses activations -- Alpha's weight SRAM takes a bias and no
    #: activation -- and a role-blind check cannot express that.
    SCALE = "scale"
    BIAS = "bias"


@dataclass(frozen=True)
class UnrollDimension:
    """One dimension a compute unit unrolls in space, and by how much.

    A matrix unit unrolls the reduction (``IC``) and output-channel (``OC``)
    dimensions; naming them is what lets a unit describe a non-square or
    non-systolic datapath without the shape being read out of a bare tuple.
    """

    name: str
    size: int
    #: ``(operand_bits, size)`` pairs for widths whose unrolling differs from
    #: ``size``. A datapath that packs two 4-bit operands into the slot one
    #: 8-bit operand occupies unrolls twice as far at the narrow width, and
    #: that is a fact about this dimension rather than about the whole unit.
    #: Empty means the unrolling does not depend on the operand width.
    dtype_extents: Tuple[Tuple[int, int], ...] = ()

    def __post_init__(self):
        if not self.name:
            raise ValueError("an unroll dimension needs a name")
        if self.size <= 0:
            raise ValueError(f"{self.name}: size must be positive")
        extents = tuple(
            (int(bits), int(extent)) for bits, extent in self.dtype_extents
        )
        widths = [bits for bits, _ in extents]
        if len(set(widths)) != len(widths):
            raise ValueError(f"{self.name}: duplicate dtype_extents widths")
        for bits, extent in extents:
            if bits <= 0 or extent <= 0:
                raise ValueError(
                    f"{self.name}: dtype_extents must be positive"
                )
        object.__setattr__(self, "dtype_extents", extents)

    def extent(self, operand_bits: Optional[int] = None) -> int:
        """This dimension's unrolling, at ``operand_bits`` if it declares one.

        Fail open, not closed: a width with no declared extent unrolls by
        ``size``. Declaring every width would make the common case -- a
        datapath whose unrolling does not vary -- restate the same number
        once per mode.
        """
        if operand_bits is not None:
            for bits, value in self.dtype_extents:
                if bits == operand_bits:
                    return value
        return self.size


@dataclass(frozen=True)
class MemoryAccess:
    """One operand role a memory serves, and for which unit.

    ``accepted_roles`` answers "may a weight live here". On a machine with
    two compute units that is not the whole question: Alpha's activation SRAM
    serves weights *for the NMP* and activations for both, while its weight
    SRAM serves weights for the matrix unit only. A role alone cannot say
    that, so the access is qualified by the unit that performs it.

    ``compute_unit`` of ``None`` means "any unit, or no unit at all" -- the
    right answer for a bias or scale the control processor stages.
    """

    role: TensorRole
    compute_unit: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "role", TensorRole(self.role))
        if self.compute_unit is not None and not self.compute_unit:
            raise ValueError("compute_unit must be a name or None")

    @property
    def key(self) -> Tuple[str, Optional[str]]:
        return (self.role.value, self.compute_unit)


@dataclass(frozen=True)
class MemoryInstance:
    """One physical memory: how big it is, and what it may hold.

    Capacity can be stated three ways, and all three are legitimate because
    different levels genuinely know different things:

    * **By geometry** -- ``bank_count`` x ``depth_per_bank`` x
      ``bits_per_bank_entry``. A published SRAM organisation.
    * **By elements** -- ``size_elements`` with ``slot_bits``. A slot array,
      where what the hardware fixes is how many operands fit, and the width of
      one is a property of the datapath. This is how L0/L1 are sized.
    * **By plain size** -- ``size_bits`` or ``size_bytes`` alone, when only the
      capacity is known or published.

    Whichever is given, ``size_bits`` is the canonical answer and the others
    are derived from it. Bits rather than bytes because a 4-bit operand array
    is not byte-granular: sixteen INT4 slots are 64 bits, and rounding that to
    bytes before it is summed loses the sub-byte remainder.

    Anything undescribed stays ``None`` rather than defaulting, so a pass that
    needs banking (``require_banking``) or an element view
    (``capacity_elements``) finds out rather than silently planning against an
    invented geometry.
    """

    name: str
    #: Canonical capacity. Derived from whichever description was given.
    size_bits: Optional[int] = None
    #: Capacity in bytes. Derived; may be given instead of ``size_bits``.
    size_bytes: Optional[int] = None
    #: Capacity in element slots, for a memory sized by operand count.
    size_elements: Optional[int] = None
    #: Width of one element slot. An element occupies a fixed-width slot
    #: whatever its dtype, so this is the datapath's operand width, not the
    #: layer's.
    slot_bits: Optional[int] = None
    bank_count: Optional[int] = None
    depth_per_bank: Optional[int] = None
    bits_per_bank_entry: Optional[int] = None
    #: Operands this instance serves. Empty means it is not role-restricted.
    #: Derived from ``accepted_accesses`` when only that is given, so a caller
    #: that states the qualified form does not restate the unqualified one.
    accepted_roles: FrozenSet[TensorRole] = frozenset()
    #: The same claim, qualified by which unit performs the access. A machine
    #: with one compute unit gains nothing by stating it and may leave this
    #: empty; one with two cannot express its routing without it.
    accepted_accesses: Tuple[MemoryAccess, ...] = ()
    double_buffered: bool = False
    #: Relative cost of one access, in the scheduler's own units. ``None``
    #: means the machine has not been characterised, not that access is free.
    access_cost: Optional[int] = None
    #: Capacity the allocator may use, when it is less than the physical
    #: capacity -- a reserved region, or a geometry the addressing cannot
    #: reach. ``None`` means all of it.
    logical_capacity: Optional[int] = None
    #: Address alignment an allocation in this memory must satisfy.
    alignment_bytes: Optional[int] = None
    #: Width of one access. Distinct from ``bits_per_bank_entry``: a memory
    #: may be read a whole line at a time while its banks are narrower.
    access_granularity_bits: Optional[int] = None
    #: Sustained bandwidth, in the profile's own units. A write bandwidth of
    #: zero is a positive claim -- Alpha's weight SRAM is not written by the
    #: datapath -- and is therefore distinct from ``None``.
    read_bandwidth: Optional[float] = None
    write_bandwidth: Optional[float] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("a memory instance needs a name")
        for field in (
            "size_bits",
            "size_bytes",
            "size_elements",
            "slot_bits",
            "bank_count",
            "depth_per_bank",
            "bits_per_bank_entry",
        ):
            value = getattr(self, field)
            if value is not None and value <= 0:
                raise ValueError(f"{self.name}: {field} must be positive")

        for field in (
            "access_cost",
            "logical_capacity",
            "alignment_bytes",
            "access_granularity_bits",
        ):
            value = getattr(self, field)
            if value is not None and value <= 0:
                raise ValueError(f"{self.name}: {field} must be positive")
        for field in ("read_bandwidth", "write_bandwidth"):
            value = getattr(self, field)
            if value is not None and value < 0:
                raise ValueError(f"{self.name}: {field} must not be negative")

        accesses = tuple(
            access
            if isinstance(access, MemoryAccess)
            else MemoryAccess(*access)
            for access in self.accepted_accesses
        )
        if len({access.key for access in accesses}) != len(accesses):
            raise ValueError(f"{self.name}: duplicate accepted_accesses")
        object.__setattr__(self, "accepted_accesses", accesses)
        # The two forms describe one fact, so they may not disagree: an
        # instance that states both must state the same roles in each.
        implied = frozenset(access.role for access in accesses)
        if accesses and self.accepted_roles and implied != self.accepted_roles:
            raise ValueError(
                f"{self.name}: accepted_roles "
                f"{sorted(role.value for role in self.accepted_roles)} does "
                f"not match accepted_accesses "
                f"{sorted(role.value for role in implied)}"
            )
        if accesses and not self.accepted_roles:
            object.__setattr__(self, "accepted_roles", implied)

        bits = self._resolve_bits()
        object.__setattr__(self, "size_bits", bits)
        object.__setattr__(self, "size_bytes", -(-bits // 8))
        if self.slot_bits is not None and self.size_elements is None:
            object.__setattr__(self, "size_elements", bits // self.slot_bits)

    def _resolve_bits(self) -> int:
        """The one capacity every given description has to agree on."""
        claims = {}

        geometry = (
            self.bank_count,
            self.depth_per_bank,
            self.bits_per_bank_entry,
        )
        if any(part is not None for part in geometry):
            if any(part is None for part in geometry):
                raise ValueError(
                    f"{self.name}: partial banking; give all of bank_count / "
                    f"depth_per_bank / bits_per_bank_entry, or none"
                )
            claims["banking"] = (
                self.bank_count
                * self.depth_per_bank
                * self.bits_per_bank_entry
            )

        if self.size_elements is not None:
            if self.slot_bits is None:
                raise ValueError(
                    f"{self.name}: size_elements needs slot_bits to say how "
                    f"wide an element is"
                )
            claims["elements"] = self.size_elements * self.slot_bits

        if self.size_bits is not None:
            claims["size_bits"] = self.size_bits
        if self.size_bytes is not None:
            claims["size_bytes"] = self.size_bytes * 8

        if not claims:
            raise ValueError(
                f"{self.name}: state a capacity -- banking, "
                f"size_elements + slot_bits, size_bits, or size_bytes"
            )

        # ``size_bytes`` is byte-rounded, so it agrees with a sub-byte
        # capacity to within the rounding rather than exactly.
        exact = {k: v for k, v in claims.items() if k != "size_bytes"}
        canonical = next(iter(exact.values()), claims.get("size_bytes"))
        for source, value in exact.items():
            if value != canonical:
                disagreeing = ", ".join(
                    f"{k}={v} b" for k, v in sorted(claims.items())
                )
                raise ValueError(
                    f"{self.name}: descriptions disagree ({disagreeing})"
                )
        if "size_bytes" in claims and exact:
            if -(-canonical // 8) != self.size_bytes:
                raise ValueError(
                    f"{self.name}: size_bytes {self.size_bytes} contradicts "
                    f"the {canonical} b this memory is otherwise described as"
                )
        return canonical

    @property
    def banked(self) -> bool:
        """Whether this memory's banking is described at all."""
        return self.bank_count is not None

    @property
    def bank_width_bytes(self) -> Optional[int]:
        """One bank's entry width, or ``None`` when banking is undescribed."""
        if self.bits_per_bank_entry is None:
            return None
        return self.bits_per_bank_entry // 8

    def capacity_elements(
        self, slot_bits: Optional[int] = None
    ) -> Optional[int]:
        """Capacity in element slots, or ``None`` when the width is unknown.

        A slot array is sized by how many elements it holds, not how many
        bytes: a 4-bit operand and a 16-bit one occupy the same slot. That is
        what interstellar checks L0/L1 against.

        Args:
            slot_bits: Override for this instance's own width, for a caller
                that knows the design's widest operand.
        """
        width = slot_bits if slot_bits is not None else self.slot_bits
        if width is None:
            return None
        return self.size_bits // width

    def require_banking(self, reason: str) -> None:
        """Assert this memory's banking is described, or say who needed it.

        Args:
            reason: What needs it, so the error names the requirement rather
                than the missing field.

        Raises:
            ValueError: If the instance was described by capacity alone.
        """
        if not self.banked:
            raise ValueError(
                f"{self.name}: {reason} needs this memory's banking, but it "
                f"is described by capacity alone ({self.size_bits} b). Give "
                f"bank_count / depth_per_bank / bits_per_bank_entry."
            )


@dataclass(frozen=True)
class MemoryLevelConfig:
    """One level of the hierarchy, as one or more named instances.

    A level is *unified* when a single instance serves every operand -- the
    banked L2 scratchpad Voyager assumes. It is *split* when each operand has
    its own memory, as some split-SRAM accelerators do. Modelling both shapes here is
    what lets a split machine be described without inventing a monolithic
    capacity it does not have.
    """

    name: str
    instances: Tuple[MemoryInstance, ...] = ()
    #: Which instance serves each role, by name. Unset on a unified level.
    input_memory: Optional[str] = None
    output_memory: Optional[str] = None
    weight_memory: Optional[str] = None
    #: Bytes reserved at the base, which the compiler may not spend.
    offset_bytes: int = 0

    def __post_init__(self):
        if not self.instances:
            raise ValueError(f"{self.name}: a level needs an instance")
        names = [instance.name for instance in self.instances]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate instance names {names}")
        for role in ("input_memory", "output_memory", "weight_memory"):
            named = getattr(self, role)
            if named is not None and named not in names:
                raise ValueError(
                    f"{self.name}: {role}={named!r} names no instance; "
                    f"declared: {sorted(names)}"
                )
        if self.offset_bytes < 0:
            raise ValueError(f"{self.name}: offset_bytes is negative")

    @property
    def operand_partitions(self) -> Tuple[int, int, int]:
        """Which instance serves (input, output, weight), by index.

        The scheduler describes a level this way: three equal indices mean one
        memory serves every operand, three distinct ones mean each has its own.
        A level that binds no roles is treated as unified on its first
        instance, since that is the only memory there is to serve them.
        """
        order = (
            TensorRole.ACTIVATION,
            TensorRole.OUTPUT,
            TensorRole.WEIGHT,
        )
        indices = []
        for role in order:
            served = self.for_role(role)
            indices.append(
                self.instances.index(served) if served is not None else 0
            )
        return tuple(indices)

    def capacity_elements(self, slot_bits) -> Tuple[int, ...]:
        """Each instance's capacity in element slots.

        ``slot_bits`` is a ``(input, output, weight)`` triple: an instance is
        sized by the widest operand bound to it, which is conservative where
        one memory serves several roles.
        """
        widest = {}
        for role, bits in zip(
            (TensorRole.ACTIVATION, TensorRole.OUTPUT, TensorRole.WEIGHT),
            slot_bits,
        ):
            served = self.for_role(role)
            if served is not None and bits:
                widest[served.name] = max(widest.get(served.name, 0), bits)
        return tuple(
            instance.capacity_elements(
                widest.get(instance.name) or instance.slot_bits
            )
            or instance.size_bits
            for instance in self.instances
        )

    def instance(self, name: str) -> MemoryInstance:
        """The named instance, or a ValueError naming what exists."""
        for candidate in self.instances:
            if candidate.name == name:
                return candidate
        known = sorted(i.name for i in self.instances)
        raise ValueError(f"{self.name}: no instance {name!r}; declared: {known}")

    def for_role(self, role: TensorRole) -> Optional[MemoryInstance]:
        """The instance serving ``role``, by explicit binding then by claim."""
        bound = {
            TensorRole.ACTIVATION: self.input_memory,
            TensorRole.OUTPUT: self.output_memory,
            TensorRole.PARTIAL_SUM: self.output_memory,
            TensorRole.WEIGHT: self.weight_memory,
        }.get(role)
        if bound is not None:
            return self.instance(bound)
        for candidate in self.instances:
            if role in candidate.accepted_roles:
                return candidate
        return self.unified

    @property
    def unified(self) -> Optional[MemoryInstance]:
        """The lone instance when this level is one memory, else ``None``."""
        return self.instances[0] if len(self.instances) == 1 else None

    @property
    def total_bytes(self) -> int:
        """Every instance summed.

        Only a meaningful capacity on a *unified* level. On a split level the
        sum is not addressable as one pool, which is why the compiler's
        ``scratchpad_size`` is taken from ``unified`` and not from here.
        """
        return sum(instance.size_bytes for instance in self.instances)


@dataclass(frozen=True)
class ExternalMemoryConfig:
    """Off-chip memory: capacity, bandwidth and latency.

    A later system-modelling extension rather than part of the accelerator
    template, so every field is optional and the cost model treats an absent
    one as "not modelled".
    """

    size_gb: Optional[float] = DEFAULT_DRAM_SIZE_GB
    bandwidth_gbs: Optional[float] = DEFAULT_DRAM_BANDWIDTH_GBS
    access_latency_ns: Optional[float] = DEFAULT_DRAM_ACCESS_LATENCY_NS


@dataclass(frozen=True)
class OutputCollector:
    """The staging register file a matrix unit drains through.

    Distinct from a ``MemoryInstance`` with the OUTPUT role, and deliberately
    so: what a schedule needs from it is not a capacity but a shape -- how
    many accumulator words deep it is and how many lanes wide one word is --
    because those are what bound a single call's output tile. Sizing it as
    bytes would lose exactly that.
    """

    name: str
    depth_words: int
    lanes_per_word: int
    accumulator_bits: int

    def __post_init__(self):
        if not self.name:
            raise ValueError("an output collector needs a name")
        for field in ("depth_words", "lanes_per_word", "accumulator_bits"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{self.name}: {field} must be positive")

    @property
    def size_bits(self) -> int:
        return self.depth_words * self.lanes_per_word * self.accumulator_bits


@dataclass(frozen=True)
class HierarchyBandwidth:
    """One directed route between two named levels, and what it carries.

    A single number per level cannot describe a machine whose weight path is
    four times its activation path, nor one where a read and a write of the
    same memory differ. Naming the route, its direction and the roles it
    serves states each of those once.

    ``shared_group`` names routes that contend for one physical resource;
    ``None`` means this route is not shared with any other.
    """

    name: str
    source: str
    destination: str
    bandwidth_bits_per_cycle: int
    roles: FrozenSet[TensorRole] = frozenset()
    shared_group: Optional[str] = None

    def __post_init__(self):
        for field in ("name", "source", "destination"):
            if not getattr(self, field):
                raise ValueError(f"a hierarchy route needs a {field}")
        if self.bandwidth_bits_per_cycle <= 0:
            raise ValueError(f"{self.name}: bandwidth must be positive")
        if self.source == self.destination:
            raise ValueError(f"{self.name}: a route must cross levels")
        object.__setattr__(
            self, "roles", frozenset(TensorRole(role) for role in self.roles)
        )


@dataclass(frozen=True)
class ComputeUnit:
    """One execution resource.

    Some accelerators pair a matrix unit with a narrower one that owns the
    operations the matrix unit cannot express -- pooling, softmax, depthwise
    convolution. Which unit owns an operation is a hardware fact, so it is
    declared here rather than inferred from the operation's shape.
    """

    name: str
    #: Parallel lanes this unit provides -- its output-channel unrolling.
    lanes: int
    #: Operations this unit owns. Empty means "whatever is not claimed".
    accepted_operations: frozenset = frozenset()
    #: Operand widths this unit accepts. Empty means no restriction.
    accepted_operand_bits: frozenset = frozenset()
    #: Dimensions this unit unrolls in space, e.g. ``IC`` x ``OC``. A unit
    #: that declares them describes its own shape; one that does not is the
    #: implicit square array ``pe_array_size`` names.
    unroll_dimensions: Tuple[UnrollDimension, ...] = ()
    #: Clock, when this unit runs on its own rather than the accelerator's.
    frequency_ghz: Optional[float] = None
    #: Width of one operand word the unit is fed. The unrolling says how many
    #: values arrive together; this says how wide the delivery is, which is
    #: what a memory's line has to be an integral number of.
    input_word_bits: Optional[int] = None
    #: Width of one scalar the unit reads beside its operands -- a per-channel
    #: bias, a multiplier. Narrower than the accumulator by design.
    scalar_bits: Optional[int] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("a compute unit needs a name")
        if self.lanes <= 0:
            raise ValueError(f"{self.name}: lanes must be positive")
        for field in ("input_word_bits", "scalar_bits"):
            value = getattr(self, field)
            if value is not None and value <= 0:
                raise ValueError(f"{self.name}: {field} must be positive")
        if self.frequency_ghz is not None and self.frequency_ghz <= 0:
            raise ValueError(f"{self.name}: frequency_ghz must be positive")
        names = [d.name for d in self.unroll_dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate unroll dims {names}")

    def unroll(
        self,
        name: str,
        operand_bits: Optional[int] = None,
    ) -> Optional[int]:
        """This unit's unrolling along ``name``, or ``None`` if it has none.

        ``operand_bits`` selects the width-dependent extent where the
        dimension declares one; without it the width-independent size is
        returned, which is what every existing caller wants.
        """
        for dimension in self.unroll_dimensions:
            if dimension.name == name:
                return dimension.extent(operand_bits)
        return None

    @property
    def array_shape(self) -> Optional[Tuple[int, int]]:
        """``(reduction, output)`` unrolling -- the compiler's PE array shape.

        ``None`` unless the unit declares both, in which case the shape is a
        fact about this unit rather than a tuple the config carries for it.
        """
        reduction, output = self.unroll("IC"), self.unroll("OC")
        if reduction is None or output is None:
            return None
        return (reduction, output)


# -----------------------------------------------------------------------------
# The datapath: what a machine's units are made of, and what drains into what
# -----------------------------------------------------------------------------
#
# A ``ComputeUnit`` above says what a unit *accepts*. That is enough to refuse
# an operation, and not enough to say what may ride along with it. Which
# sequences a machine can issue as one call is a consequence of how its blocks
# are wired, so it is described here once rather than restated as a hand-written
# fusion table that can drift from the hardware it claims to describe.
#
# These carry operation-name STRINGS and predicate callables, never an
# ``OpMatcher`` -- this module depends on nothing but ``dataclasses``, ``enum``
# and ``typing``, and that is worth keeping. Matcher construction lives in
# ``codegen/targets/datapath.py``.


@dataclass(frozen=True)
class Block:
    """One functional block: the finest thing a datapath applies in one pass.

    ``ops`` is an ORDERED tuple, never a set. ``OpMatcher.targets`` order is
    observable -- ``find_sequential_nodes_`` walks it and takes the first
    sequential candidate -- so a set here would make fusion depend on hash
    order. It is also why ``ComputeUnit.accepted_operations``, a frozenset,
    can never be the source of a stage.

    ``source`` is where the claim came from. It is never matched against; its
    job is to let a reader check the block against the thing it was
    transcribed from. A block whose source is not in this tree must say so --
    an uncheckable citation that does not admit it is worse than none.
    """

    name: str
    #: Operation names this block claims, in the order they are matched.
    ops: Tuple[str, ...] = ()
    #: Extra condition on a node carrying one of ``ops``.
    predicate: Optional[Callable] = None
    #: The subset of ``ops`` that no FX node ever names. The cross-tile
    #: accumulator add is today's only instance: implied by the loop nest,
    #: never written in the graph. Naming it in ``ops`` is what lets
    #: ``_accumulate_fusible`` find it by target membership; ``predicate``
    #: must then refuse every real node carrying it.
    virtual_ops: Tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self):
        for label, value in (
            ("ops", self.ops),
            ("virtual_ops", self.virtual_ops),
        ):
            if not isinstance(value, tuple):
                raise TypeError(
                    f"{self.name}: {label} must be an ordered tuple, got "
                    f"{type(value).__name__} -- stage order is observable"
                )
        missing = set(self.virtual_ops) - set(self.ops)
        if missing:
            raise ValueError(
                f"{self.name}: virtual ops {sorted(missing)} are not in ops, "
                "so no stage would name them"
            )
        if self.virtual_ops and self.predicate is None:
            raise ValueError(
                f"{self.name}: virtual ops need a predicate that refuses every "
                "real node carrying them, or the operation is admitted for real"
            )


@dataclass(frozen=True)
class Unit:
    """A node in the datapath graph.

    ``blocks`` is the unit's internal connectivity: block *i* drives block
    *i+1*, in declared order, no reordering and no back edges. That fixed
    order is what makes a walk yield an ordered stage list rather than a set.

    ``compute_unit`` is the very ``ComputeUnit`` the config already carries --
    this describes that unit's internals, not a second machine. A drain-only
    unit (an accumulator, a post-processing unit, a collector) leaves it
    ``None``: it is not independently schedulable.
    """

    name: str
    blocks: Tuple[Block, ...] = ()
    #: Present only on a unit that ISSUES a call; a walk starts here.
    issue: Optional[Block] = None
    compute_unit: Optional["ComputeUnit"] = None


@dataclass(frozen=True)
class Link:
    """A wire: ``source``'s last block drives ``destination``'s first."""

    source: str
    destination: str


@dataclass(frozen=True)
class Datapath:
    """A machine's compute units and what they drain through.

    Hashable by construction -- frozen, tuple fields, function-object
    predicates -- which is what lets the derivation be cached and hand back an
    identical object every call, the same identity a module-level literal has.
    Adding a dict field here would silently take that away.
    """

    units: Tuple[Unit, ...]
    links: Tuple[Link, ...] = ()
    #: Units at which a fused sequence may end.
    sinks: Tuple[str, ...] = ()

    def unit(self, name: str) -> "Unit":
        for candidate in self.units:
            if candidate.name == name:
                return candidate
        raise KeyError(f"{name}: no such unit in this datapath")

    @property
    def compute_units(self) -> Tuple["ComputeUnit", ...]:
        return tuple(
            unit.compute_unit
            for unit in self.units
            if unit.compute_unit is not None
        )


#: Which flat fields each structural field derives.  Published so a caller
#: assembling a config can blank exactly the fields a structure will fill --
#: derivation only touches a field left ``None``, so anything it must own has
#: to be cleared first.  Keeping the map here means the two cannot drift.
STRUCTURE_DERIVED_FIELDS = {
    "compute": ("pe_array_size", "frequency"),
    "local_memory": (
        "input_buffer_size",
        "weight_buffer_size",
        "accum_buffer_size",
    ),
    "global_memory": ("scratchpad_size", "num_banks", "bank_width"),
    "external_memory": (
        "dram_size",
        "dram_bandwidth",
        "dram_access_latency",
    ),
}


@dataclass(frozen=True)
class AcceleratorConfig:
    """Compiler-visible hardware parameters for Voyager accelerators.

    Voyager couples two programmable compute engines: a weight-stationary
    2-D systolic array for convolution and GEMM, and a multi-stage vector
    unit for elementwise operations, reductions, nonlinear activations,
    normalization, and quantization. During a matrix tile, weights remain
    in the processing elements, activations flow horizontally, and partial
    sums propagate vertically. The matrix output can stream directly into
    the vector unit for operator fusion or pass through a double-buffered
    accumulation buffer to decouple the two engines.

    The systolic array uses dedicated input, weight, and accumulation
    buffers backed by a banked L2 scratchpad. This configuration describes
    the accelerator's compute parallelism, clock frequency, and on-chip
    memory hierarchy. The optional DRAM capacity, bandwidth, and latency
    parameters are later system-modeling extensions, rather than parameters
    of the Voyager accelerator template described in the paper.
    """

    # Compute
    pe_array_size: Tuple[int, int] = DEFAULT_PE_ARRAY_SIZE
    vector_unit_width: Optional[int] = None  # None -> pe_array_size[1]
    frequency: float = DEFAULT_FREQUENCY_GHZ  # accelerator clock
    # L1 systolic buffers, per lane, in ELEMENTS -- not bytes.  L0/L1 are slot
    # arrays: an element occupies a fixed-width slot whatever its dtype, so
    # interstellar checks their fit in element counts, while L2/L3 are flat
    # byte pools.  See ``cost_model.get_bank_size``, which branches on exactly
    # that (``level <= 1`` -> counts, else ``_level_bytes``).  The tiler passes
    # these through ``buf_capacity_list``, whose per-level unit is therefore
    # mixed by design.
    input_buffer_size: Optional[int] = DEFAULT_INPUT_BUFFER_SIZE
    weight_buffer_size: Optional[int] = DEFAULT_WEIGHT_BUFFER_SIZE
    accum_buffer_size: Optional[int] = DEFAULT_ACCUM_BUFFER_SIZE
    double_buffered_accum_buffer: bool = False
    # L2 scratchpad
    scratchpad_size: Optional[int] = None
    scratchpad_offset: int = DEFAULT_SCRATCHPAD_OFFSET  # bytes at the base
    num_banks: Optional[int] = None
    bank_width: Optional[int] = None
    double_buffered_l2: bool = DEFAULT_DOUBLE_BUFFERED_L2
    # L3 DRAM
    dram_size: Optional[float] = DEFAULT_DRAM_SIZE_GB
    dram_bandwidth: Optional[float] = DEFAULT_DRAM_BANDWIDTH_GBS
    dram_access_latency: Optional[float] = DEFAULT_DRAM_ACCESS_LATENCY_NS

    # Optional hardware description.  Both default to empty, in which
    # case the machine is the single implicit matrix unit that
    # ``pe_array_size`` describes -- which is every target that predates
    # these fields.
    dtype_modes: Tuple[DTypeMode, ...] = ()
    compute_units: Tuple[ComputeUnit, ...] = ()
    #: The units above, plus what they drain through. ``None`` means the
    #: machine has not been inventoried -- which is a different thing from a
    #: machine that can fuse nothing -- and whatever table the target authors
    #: stands. ``from_args`` can never set this, so a config assembled from the
    #: command line alone always leaves it ``None``.
    datapath: Optional[Datapath] = None
    #: Input-channel extents the matrix unit can run by *masking* its unused
    #: lanes, instead of being given an operand padded up to the array width.
    #: A machine with no such path must declare the empty set: its narrow
    #: layers are then padded to fill the reduction, which is what makes them
    #: schedulable at all. The default preserves the long-standing behaviour
    #: of never padding a 1- or 3-channel first layer.
    partial_matrix_input_channels: FrozenSet[int] = COMMON_IMAGE_INPUT_CHANNELS
    #: Whether an adaptive pool is tiled on (N, C) rather than run whole.
    #: Only its channels may be cut -- an adaptive window is proportional to
    #: ``H_in / H_out``, so a spatial tile has no fixed input extent. A machine
    #: whose on-chip memory holds the whole tensor gains nothing by tiling and
    #: would only change its collateral, so this is off unless a target says
    #: its memory cannot.
    tiles_adaptive_pooling: bool = False

    # Structural description of the same machine the flat fields above
    # describe.  A level supplied here *derives* the flat fields it covers,
    # but only where they were left ``None`` -- so a value named on the
    # command line still wins, and a profile states each number once.
    #: The unit the flat compute fields describe, when one is singled out.
    compute: Optional[ComputeUnit] = None
    #: L1: the systolic input / weight / accumulator buffers.
    local_memory: Optional[MemoryLevelConfig] = None
    #: L2: the scratchpad, unified or split per operand.
    global_memory: Optional[MemoryLevelConfig] = None
    #: L3: off-chip memory.
    external_memory: Optional[ExternalMemoryConfig] = None
    #: The register file the matrix unit drains through, when the machine has
    #: one distinct from its L1 output memory.
    output_collector: Optional[OutputCollector] = None
    #: The routes between the levels above. Empty means the machine has not
    #: been characterised, and a cost model falls back to whatever single
    #: number it used before.
    hierarchy_bandwidths: Tuple[HierarchyBandwidth, ...] = ()
    #: ``(role, unit_name)`` pairs saying which declared unit performs a named
    #: kind of work -- ``matrix``, ``nmp``. A machine with one unit has
    #: nothing to bind and leaves this empty; serialization and conversion
    #: read it to resolve a unit without guessing from its name.
    compute_unit_bindings: Tuple[Tuple[str, str], ...] = ()
    #: Process node, for an energy model that needs one.
    technology_nm: Optional[int] = None
    #: Whether a depthwise convolution's channels are padded up to the
    #: unit's lane count. Off by default: a machine that can mask its unused
    #: lanes gains nothing, and padding would change its collateral.
    pad_depthwise_channels: bool = False
    #: Levels the machine does not have. Naming one here is a positive claim
    #: of absence, which is different from leaving it undescribed: the tile
    #: search leaves an absent level unconstrained instead of sizing it. A
    #: machine whose on-chip storage is entirely per-role -- three
    #: SRAMs -- has no unified pool above it, and sizing one from the same
    #: bytes would count that storage twice.
    absent_memory_levels: FrozenSet[str] = frozenset()
    #: Levels the caller described, as opposed to ones synthesized from the
    #: flat fields. Set during construction; not something to pass in.
    declared_memory_levels: FrozenSet[str] = frozenset()

    def unit(self, role: str) -> Optional[ComputeUnit]:
        """The unit bound to ``role``, or ``None`` if nothing is bound.

        Fail closed on a binding that names a unit the machine does not
        declare: that is a description that disagrees with itself, and
        resolving it to the first unit would silently plan for the wrong one.
        """
        names = dict(self.compute_unit_bindings)
        bound = names.get(role)
        if bound is None:
            return None
        for unit in self.compute_units:
            if unit.name == bound:
                return unit
        raise ValueError(
            f"compute_unit_bindings binds {role!r} to {bound!r}, which is "
            f"not a declared unit; declared: "
            f"{sorted(unit.name for unit in self.compute_units)}"
        )

    def __post_init__(self):
        """Derive the flat fields from any structure given, then validate.

        Inert at the default of 0, so a config that names no scratchpad at
        all still constructs.
        """
        bindings = tuple(
            (str(role), str(name)) for role, name in self.compute_unit_bindings
        )
        if len({role for role, _ in bindings}) != len(bindings):
            raise ValueError("compute_unit_bindings names a role twice")
        object.__setattr__(self, "compute_unit_bindings", bindings)
        if self.technology_nm is not None and self.technology_nm <= 0:
            raise ValueError("technology_nm must be positive")
        # Which levels the caller actually described, recorded before
        # synthesis fills the rest.  Synthesis exists so every config can be
        # *asked* for its structure, but a synthesized level restates the flat
        # fields rather than claiming anything about the machine -- and a pass
        # that changes behaviour on the strength of a description must only do
        # so for one a target really made.
        object.__setattr__(
            self,
            "declared_memory_levels",
            frozenset(
                name
                for name, level in (
                    ("local", self.local_memory),
                    ("global", self.global_memory),
                    ("external", self.external_memory),
                )
                if level is not None
            ),
        )
        self._absorb_structure()
        self._synthesize_structure()
        if self.scratchpad_offset < 0:
            raise ValueError(
                f"scratchpad_offset {self.scratchpad_offset} is negative"
            )
        if not self.scratchpad_offset:
            return
        if self.scratchpad_size is None:
            raise ValueError(
                "scratchpad_offset needs a scratchpad_size to reserve from"
            )
        if self.scratchpad_offset >= self.scratchpad_size:
            raise ValueError(
                f"scratchpad_offset {self.scratchpad_offset} leaves nothing of "
                f"scratchpad_size {self.scratchpad_size}"
            )
        # A reservation that split a bank would put the planner's bank-aligned
        # groups and the tile search's bank budget on different geometries.
        bank = self.bank_size
        if bank is not None and self.scratchpad_offset % bank:
            raise ValueError(
                f"scratchpad_offset {self.scratchpad_offset} is not a multiple "
                f"of the {bank} B bank size"
            )

    # --- structural derivation ---------------------------------------------

    def _set(self, field: str, value) -> None:
        """Fill ``field`` from a structure, but only if nothing named it.

        ``None`` is the one honest "unset" marker available on a frozen
        dataclass whose fields carry real defaults.  A caller that named a
        value -- a CLI override, say -- leaves it non-``None`` and keeps it.
        """
        if value is not None and getattr(self, field) is None:
            object.__setattr__(self, field, value)

    def _absorb_structure(self) -> None:
        """Derive every flat field a supplied level or unit already describes.

        This is what makes the structures the single statement of a machine:
        A target may say its activation buffer is 1 bank x 1024 words x 64 bits, and
        ``scratchpad_size`` / ``num_banks`` / ``bank_width`` follow from that
        instead of being restated beside it and drifting.
        """
        unit = self.compute
        if unit is not None:
            self._set("pe_array_size", unit.array_shape)
            self._set("frequency", unit.frequency_ghz)
            if not self.compute_units:
                object.__setattr__(self, "compute_units", (unit,))

        # A datapath names the units as part of saying how they are wired, so
        # a machine that supplies one does not restate them beside it.
        if self.datapath is not None:
            if not self.compute_units:
                object.__setattr__(
                    self, "compute_units", self.datapath.compute_units
                )
            # ``pe_array_size`` is deliberately NOT derived here. ``compute``
            # already owns it, and one field with two owners is the drift this
            # whole change exists to remove. A machine supplying a datapath
            # supplies ``compute`` too; they name the same ``ComputeUnit``.

        level = self.local_memory
        if level is not None:
            # These three are ELEMENT counts *per lane*, not byte counts:
            # the tiler multiplies each by the array dimension it feeds
            # (``input_buffer_size * ic_dim`` and so on).  So a level derives
            # them by converting its capacity to elements at the widest
            # declared operand, then dividing by the lanes that share it.  A
            # memory that cannot say how wide its slots are derives nothing,
            # and the generic default stands.
            widest = (
                max(mode.operand_bits for mode in self.dtype_modes)
                if self.dtype_modes
                else None
            )
            reduction, output = (
                self.pe_array_size if self.pe_array_size else (None, None)
            )
            for field, role, lanes in (
                ("input_buffer_size", TensorRole.ACTIVATION, reduction),
                ("weight_buffer_size", TensorRole.WEIGHT, output),
                ("accum_buffer_size", TensorRole.OUTPUT, output),
            ):
                instance = level.for_role(role)
                if instance is None or not lanes:
                    continue
                # The accumulator holds partial sums, which are wider than an
                # operand; an instance that states its own slot width is
                # believed over the datapath's.
                elements = instance.capacity_elements(
                    None if instance.slot_bits else widest
                )
                self._set(field, elements // lanes if elements else None)
            accumulator = level.for_role(TensorRole.OUTPUT)
            if accumulator is not None and accumulator.double_buffered:
                object.__setattr__(
                    self, "double_buffered_accum_buffer", True
                )

        level = self.global_memory
        if level is not None:
            # The compiler addresses L2 as one pool.  A split level has no
            # such pool, so only a unified one yields a capacity; a split one
            # contributes its per-role instances and leaves the monolithic
            # fields to whoever is willing to name a stand-in.
            pool = level.unified
            if pool is not None:
                self._set("scratchpad_size", pool.size_bytes)
                # ``None`` when the pool is described by capacity alone, in
                # which case these stay unset rather than defaulting to one
                # bank -- a pass that needs them says so via require_banking.
                self._set("num_banks", pool.bank_count)
                self._set("bank_width", pool.bank_width_bytes)
            if level.offset_bytes and not self.scratchpad_offset:
                object.__setattr__(
                    self, "scratchpad_offset", level.offset_bytes
                )

        dram = self.external_memory
        if dram is not None:
            self._set("dram_size", dram.size_gb)
            self._set("dram_bandwidth", dram.bandwidth_gbs)
            self._set("dram_access_latency", dram.access_latency_ns)

    def _synthesize_structure(self) -> None:
        """The inverse of ``_absorb_structure``: describe what only the flat
        fields state.

        A config assembled straight from the command line -- the generic
        target, which declares no machine because it *is* whatever the CLI
        says -- would otherwise have no structural view at all. Synthesizing
        one makes the description total: every config can be asked for its
        memory levels, however it was built.

        The geometry is recovered where it is determined and skipped where it
        is not. ``depth = size / (banks x width)`` is exact, so a unified
        scratchpad round-trips; an L1 buffer states only a byte count, so it
        is described as that many unbanked bytes rather than given invented
        banking.
        """
        if self.compute is None and self.pe_array_size is not None:
            reduction, output = self.pe_array_size
            object.__setattr__(
                self,
                "compute",
                ComputeUnit(
                    name="pe_array",
                    lanes=output,
                    unroll_dimensions=(
                        UnrollDimension("IC", reduction),
                        UnrollDimension("OC", output),
                    ),
                    frequency_ghz=self.frequency,
                ),
            )

        if self.local_memory is None and self.dtype_modes:
            # The flat L1 fields are element counts.  Turning one into a
            # ``MemoryInstance`` needs a slot width, so this synthesizes only
            # where the config declares its operand widths; otherwise L1 stays
            # undescribed, which is honest -- an element count alone does not
            # say how big the memory is.
            slot_bits = max(mode.operand_bits for mode in self.dtype_modes)
            reduction, output = (
                self.pe_array_size if self.pe_array_size else (1, 1)
            )
            instances = tuple(
                MemoryInstance(
                    name=name,
                    # The flat field is per *lane* -- absorption divided by the
                    # array dimension the memory feeds, so synthesis multiplies
                    # it back.  Without this the two directions disagree and a
                    # 16x16 machine describing 1024 per lane round-trips to
                    # 1024 total instead of 16384.
                    size_elements=elements * lanes,
                    slot_bits=slot_bits,
                    accepted_roles=frozenset(roles),
                )
                for name, elements, roles, lanes in (
                    (
                        "input_buffer",
                        self.input_buffer_size,
                        {TensorRole.ACTIVATION},
                        reduction,
                    ),
                    (
                        "weight_buffer",
                        self.weight_buffer_size,
                        {TensorRole.WEIGHT},
                        output,
                    ),
                    (
                        "accum_buffer",
                        self.accum_buffer_size,
                        {TensorRole.OUTPUT, TensorRole.PARTIAL_SUM},
                        output,
                    ),
                )
                if elements
            )
            if instances:
                object.__setattr__(
                    self,
                    "local_memory",
                    MemoryLevelConfig(name="l1", instances=instances),
                )

        if self.global_memory is None and self.scratchpad_size:
            # Recover the banking only when it is both given and exact;
            # otherwise state the capacity and leave the banking undescribed,
            # which is what the flat fields actually said.
            banking = {}
            if self.num_banks and self.bank_width:
                stride = self.num_banks * self.bank_width
                if self.scratchpad_size % stride == 0:
                    banking = {
                        "bank_count": self.num_banks,
                        "depth_per_bank": self.scratchpad_size // stride,
                        "bits_per_bank_entry": self.bank_width * 8,
                    }
            object.__setattr__(
                self,
                "global_memory",
                MemoryLevelConfig(
                    name="l2",
                    instances=(
                        MemoryInstance(
                            name="scratchpad",
                            size_bytes=self.scratchpad_size,
                            accepted_roles=frozenset(TensorRole),
                            **banking,
                        ),
                    ),
                    offset_bytes=self.scratchpad_offset,
                ),
            )

        if self.external_memory is None and self.has_memory_level("external"):
            object.__setattr__(
                self,
                "external_memory",
                ExternalMemoryConfig(
                    size_gb=self.dram_size,
                    bandwidth_gbs=self.dram_bandwidth,
                    access_latency_ns=self.dram_access_latency,
                ),
            )

    # --- derived views ------------------------------------------------------

    @property
    def vector_lanes(self) -> int:
        """Vector-unit lane count: its own width, else the PE array columns."""
        if self.vector_unit_width is not None:
            return self.vector_unit_width
        return self.pe_array_size[1]

    @property
    def bytes_per_cycle(self) -> float:
        return self.dram_bandwidth / self.frequency

    @property
    def access_latency_cycles(self) -> float:
        return self.dram_access_latency * self.frequency

    @property
    def num_slots(self) -> int:
        """Banks one buffer occupies: two when it is ping-ponged, else one."""
        return 2 if self.double_buffered_l2 else 1

    @property
    def bank_size(self) -> Optional[int]:
        if self.num_banks is None:
            return None
        return self.scratchpad_size // self.num_banks

    @property
    def usable_scratchpad_size(self) -> Optional[int]:
        """What the plan may spend: the SRAM above ``scratchpad_offset``."""
        if self.scratchpad_size is None:
            return None
        return self.scratchpad_size - self.scratchpad_offset

    @property
    def usable_banks(self) -> Optional[int]:
        """Banks above the reservation.

        Counted off ``num_banks`` rather than divided out of the usable
        bytes, so it is exactly ``num_banks`` at an offset of 0 even when the
        banks do not divide the SRAM evenly.
        """
        if self.num_banks is None or self.bank_size is None:
            return None
        return self.num_banks - self.scratchpad_offset // self.bank_size

    def dtype_mode(self, operand_bits: int) -> Optional[DTypeMode]:
        """The datapath's behaviour at ``operand_bits``.

        ``None`` when the config declares no modes, meaning the single
        implicit mode. Fail closed once modes *are* declared: a width the
        hardware cannot run is a compile-time error, not a substitution.
        """
        if not self.dtype_modes:
            return None
        for mode in self.dtype_modes:
            if mode.operand_bits == operand_bits:
                return mode
        accepted = sorted(m.operand_bits for m in self.dtype_modes)
        raise ValueError(
            f"no {operand_bits}-bit operand mode; accepted: {accepted}"
        )

    def input_lanes(self, operand_bits: Optional[int] = None) -> int:
        """Reduction lanes, at ``operand_bits`` when modes are declared."""
        if operand_bits is not None:
            mode = self.dtype_mode(operand_bits)
            if mode is not None:
                return mode.input_lanes
        return self.pe_array_size[0]

    def channel_alignment(self, operand_bits: Optional[int] = None) -> int:
        """Output-channel granularity, at ``operand_bits`` when declared."""
        if operand_bits is not None:
            mode = self.dtype_mode(operand_bits)
            if mode is not None:
                return mode.channel_alignment
        return self.pe_array_size[1]

    def output_lanes(self, operand_bits: Optional[int] = None) -> int:
        """Output-channel lanes the datapath drives.

        Deliberately not the same question as ``channel_alignment``. The lanes
        are how wide the datapath *is*; the alignment is the granularity a
        resident operand has to respect. Alpha drives 32 lanes at both operand
        widths and aligns to 16 at eight bits, so a 16-channel weight is a
        legal resident extent inside a 32-lane unit -- and collapsing the two
        would round it up to 32 and emit a weight twice the size the endpoint
        expects.

        Read from the matrix unit's ``OC`` unrolling when one is declared,
        because that is where a width-dependent extent would be stated, and
        from ``pe_array_size`` otherwise -- which is the value every caller
        used before this existed.
        """
        unit = self.compute
        if unit is not None:
            extent = unit.unroll("OC", operand_bits)
            if extent is not None:
                return extent
        return self.pe_array_size[1]

    def supports_partial_matrix_output(
        self,
        channels: int,
        operand_bits: Optional[int] = None,
    ) -> bool:
        """Whether a resident K narrower than the datapath may mask lanes.

        The output-side counterpart of ``supports_partial_matrix_input``, and
        the same kind of claim: ``True`` means the unit drives all its lanes
        and discards the ones above ``channels``, so the operand does not have
        to be widened to fill them.

        Narrower *and* aligned, both: an extent that is not a multiple of the
        granularity the datapath requires is not a partial vector, it is a
        misaligned one, and the machine has no way to run it.
        """
        lanes = self.output_lanes(operand_bits)
        alignment = self.channel_alignment(operand_bits)
        return 0 < channels < lanes and channels % alignment == 0

    def matrix_output_resident_channels(
        self,
        channels: int,
        operand_bits: Optional[int] = None,
    ) -> int:
        """The resident output-channel extent ``channels`` must pad up to.

        Align first, then fill the datapath only if the aligned extent cannot
        be masked. On a machine whose alignment equals its lane count -- every
        target but Alpha -- the first step already lands on a multiple of the
        lanes and the second never fires, so this returns exactly
        ``ceil(channels / K_unroll) * K_unroll``: the value the flat pass has
        always computed.
        """
        if channels <= 0:
            raise ValueError("matrix output channels must be positive")
        lanes = self.output_lanes(operand_bits)
        alignment = self.channel_alignment(operand_bits)
        aligned = -(-channels // alignment) * alignment
        if aligned % lanes == 0 or self.supports_partial_matrix_output(
            aligned, operand_bits
        ):
            return aligned
        return -(-aligned // lanes) * lanes

    def describes_memory_level(self, level: str) -> bool:
        """Whether ``level`` was described by the caller rather than derived."""
        return level in self.declared_memory_levels

    def has_memory_level(self, level: str) -> bool:
        """Whether the machine has ``level`` at all (``"local"`` / ``"global"``).

        ``False`` means the level is declared absent, so a pass that would
        otherwise give it a capacity must leave it unconstrained rather than
        invent one.
        """
        return level not in self.absent_memory_levels

    def supports_partial_matrix_input(self, channels: int) -> bool:
        """Whether matrix input lanes may be masked for this channel extent.

        ``False`` means the operand has to be padded up to the array width
        before the layer can be mapped -- the datapath has no way to ignore
        the lanes it would otherwise read as garbage.
        """
        return channels in self.partial_matrix_input_channels

    def compute_unit(self, name: str) -> ComputeUnit:
        """The named compute unit, or a ValueError naming what exists."""
        for unit in self.compute_units:
            if unit.name == name:
                return unit
        known = sorted(u.name for u in self.compute_units)
        raise ValueError(f"no compute unit {name!r}; declared: {known}")

    def unit_for_operation(self, operation: str) -> Optional[ComputeUnit]:
        """The unit that owns ``operation``.

        ``None`` when no unit claims it: either the config declares no units,
        or the operation belongs to whichever unit claims nothing specific.
        """
        for unit in self.compute_units:
            if operation in unit.accepted_operations:
                return unit
        return None

    @classmethod
    def from_args(cls, args) -> "AcceleratorConfig":
        """Build the config from parsed CLI args (``add_compile_args``)."""
        return cls(
            pe_array_size=args.pe_array_size,
            vector_unit_width=args.vector_unit_width,
            frequency=args.frequency,
            input_buffer_size=args.input_buffer_size,
            weight_buffer_size=args.weight_buffer_size,
            accum_buffer_size=args.accum_buffer_size,
            double_buffered_accum_buffer=args.double_buffered_accum_buffer,
            scratchpad_size=args.scratchpad_size,
            scratchpad_offset=args.scratchpad_offset,
            num_banks=args.num_banks,
            bank_width=args.bank_width,
            double_buffered_l2=args.double_buffered_l2,
            dram_size=args.dram_size,
            dram_bandwidth=args.dram_bandwidth,
            dram_access_latency=args.dram_access_latency,
        )
