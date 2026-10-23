# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The scratchpad address Alpha's endpoint decodes.

The field layout below defines how the Alpha endpoint decodes each address it
is given::

    Addr:1010d83 Base:1000000 LocalAddr:183 VirtualBankAddr:3
                 BankClusterSize:1 BankAddr:3

so a run yields a labelled sample of the encoding, and the layout is whatever
reproduces all of them. Searching (shift, width) against 135 distinct samples:

===================  ==============  ==========================
field                bits            agreement
===================  ==============  ==========================
entry                ``[9:0]``       135/135, and *uniquely* fitted
virtual bank         ``[15:10]``     135/135
bank cluster         ``[18:16]``     135/135
operand width        ``[19]``        set on all 51 cluster-2 samples
===================  ==============  ==========================

and the virtual-to-physical mapping falls out of the same samples::

    physical bank = virtual bank * bank cluster        # 135/135

which is what a cluster means: a wider operand occupies more than one bank, so
the virtual index is scaled by the cluster it sits in.

**Two widths the samples do not pin, stated rather than hidden.** No sample has
a virtual bank above 22 or a cluster above 2, so `[15:10]` fits the data just as
well as `[14:10]`, and `[18:16]` as well as `[17:16]`. The wider reading is
taken here because it is what the endpoint's own field layout uses; the data
only rules out anything *narrower*. Likewise bit 19 is pinned by the data but
its *meaning* is not -- the samples show it set on every cluster-2 address and
on 33 of 84 cluster-1 ones, which is consistent with "the operand is 8-bit"
-- and a second sample settles it. The reference compiler's own
`magnet_alpha_bank_placements` records 206 emitted addresses with the bank each
was meant for; all 206 decode and re-encode exactly, all 206 agree on
`physical_bank`, and the 70 with bit 19 *clear* are precisely the 70 backings
whose layout is `alpha_vsq_embedded` -- a VSQ pair whose primary payload is
int4. So bit 19 is the payload width, not a correlate of the cluster.

One trap for anyone repeating the fit: the model prints `VirtualBankAddr`,
`BankClusterSize` and `BankAddr` in **hex with no prefix**. Read as decimal
they agree 134/135, and the single disagreement looks exactly like a real edge
case. Read as hex, both relations are exact.
"""

from dataclasses import dataclass

__all__ = [
    "ALPHA_ACTIVATION_BASE",
    "ALPHA_MEMORY_BASES",
    "ALPHA_WEIGHT_BASE",
    "AlphaAddress",
    "alpha_entry_bytes",
    "alpha_place",
]

ALPHA_ENTRY_BITS = 10
ALPHA_VIRTUAL_BANK_BITS = 6
ALPHA_BANK_CLUSTER_BITS = 3

ALPHA_VIRTUAL_BANK_SHIFT = ALPHA_ENTRY_BITS
ALPHA_BANK_CLUSTER_SHIFT = ALPHA_VIRTUAL_BANK_SHIFT + ALPHA_VIRTUAL_BANK_BITS
ALPHA_OPERAND_WIDTH_SHIFT = (
    ALPHA_BANK_CLUSTER_SHIFT + ALPHA_BANK_CLUSTER_BITS
)

#: The two global memories, and the only two bases the reference ever emits.
#: The converter refuses anything below the lower one.
ALPHA_WEIGHT_BASE = 0x800000
ALPHA_ACTIVATION_BASE = 0x1000000

#: Which operand width sets bit 19. Established from the reference: the 70
#: addresses with it clear are exactly the int4 VSQ payloads.
ALPHA_WIDE_OPERAND_BITS = 8


def _field(width):
    return (1 << width) - 1


@dataclass(frozen=True)
class AlphaAddress:
    """One scratchpad address, as the endpoint reads it.

    ``base`` is the global memory the address sits in; the rest is packed into
    the low twenty bits. Nothing above bit 19 is ever set -- checked against
    every sample -- so an encoded address is exactly ``base + word``.
    """

    base: int
    operand_bits: int
    cluster: int
    virtual_bank: int
    entry: int

    def __post_init__(self):
        for name, value, width in (
            ("entry", self.entry, ALPHA_ENTRY_BITS),
            ("virtual_bank", self.virtual_bank, ALPHA_VIRTUAL_BANK_BITS),
            ("cluster", self.cluster, ALPHA_BANK_CLUSTER_BITS),
        ):
            if not 0 <= value <= _field(width):
                raise ValueError(
                    f"{name}={value} does not fit its {width}-bit field"
                )
        if self.base < ALPHA_WEIGHT_BASE:
            # The converter refuses these, so refuse them here instead of
            # emitting one and learning about it two stages downstream.
            raise ValueError(
                f"base {self.base:#x} is below {ALPHA_WEIGHT_BASE:#x}"
            )

    @property
    def physical_bank(self) -> int:
        """The bank the endpoint will actually address."""
        return self.virtual_bank * self.cluster

    def encode(self) -> int:
        word = (
            self.entry
            | (self.virtual_bank << ALPHA_VIRTUAL_BANK_SHIFT)
            | (self.cluster << ALPHA_BANK_CLUSTER_SHIFT)
        )
        if self.operand_bits == ALPHA_WIDE_OPERAND_BITS:
            word |= 1 << ALPHA_OPERAND_WIDTH_SHIFT
        return self.base + word

    @classmethod
    def decode(cls, address: int, base: int) -> "AlphaAddress":
        word = address - base
        if word < 0:
            raise ValueError(f"{address:#x} is below its base {base:#x}")
        if word >> (ALPHA_OPERAND_WIDTH_SHIFT + 1):
            raise ValueError(
                f"{address:#x} sets bits above the encoded field"
            )
        wide = (word >> ALPHA_OPERAND_WIDTH_SHIFT) & 1
        return cls(
            base=base,
            operand_bits=ALPHA_WIDE_OPERAND_BITS if wide else 4,
            cluster=(word >> ALPHA_BANK_CLUSTER_SHIFT)
            & _field(ALPHA_BANK_CLUSTER_BITS),
            virtual_bank=(word >> ALPHA_VIRTUAL_BANK_SHIFT)
            & _field(ALPHA_VIRTUAL_BANK_BITS),
            entry=word & _field(ALPHA_ENTRY_BITS),
        )


# ---------------------------------------------------------------------------
# From a byte offset to an address
# ---------------------------------------------------------------------------
#
# The encoding above says how the four fields pack. This says which values
# they take for a tensor placed at a byte offset -- and it is *derived from
# the declared machine*, not fitted.
#
# ``hardware.GLOBAL`` states both memories outright::
#
#     global_activation   64 banks x 1024 entries x 128 bits   = 1 MiB
#     global_weight       16 banks x 1024 entries x 512 bits   = 1 MiB
#
# so an entry is ``bits_per_bank_entry // 8`` -- 16 bytes in the activation
# memory, which is one input word, and 64 in the weight memory, which is four
# of them. A cluster multiplies it: a cluster of two addresses a pair of banks
# as one, so its entry spans twice the bytes and its virtual bank twice the
# span.
#
# **An earlier reading of this was fitted and wrong in a way that mattered.**
# It took the *bank span in bytes* -- 65536 for the weight memory -- for the
# bank itself, which reproduces every address and implies 64 banks of 64 KiB,
# or 4 MiB, in a memory the machine says is 1 MiB. The arithmetic is
# identical; the geometry is not, and only one of them survives being asked
# how many banks there are.

#: Where each global memory starts. The only two bases a reference compile
#: emits, and the converter refuses anything below the lower one.
ALPHA_MEMORY_BASES = {
    "global_weight": ALPHA_WEIGHT_BASE,
    "global_activation": ALPHA_ACTIVATION_BASE,
}


def alpha_entry_bytes(memory, cluster: int = 1) -> int:
    """How many bytes one addressable entry covers.

    Args:
        memory: A ``MemoryInstance`` from the declared machine.
        cluster: How many banks are addressed as one.

    Returns:
        ``cluster * bits_per_bank_entry // 8``.

    Raises:
        ValueError: If the cluster is not positive, or the memory declares an
            entry that is not a whole number of bytes.
    """
    if cluster <= 0:
        raise ValueError(f"a bank cluster must be positive, not {cluster}")
    bits = int(memory.bits_per_bank_entry)
    if bits % 8:
        raise ValueError(
            f"{memory.name}: a {bits}-bit bank entry is not a whole number "
            f"of bytes"
        )
    return cluster * bits // 8


def alpha_place(memory, byte_offset: int, *, cluster: int, operand_bits: int):
    """The address a tensor at ``byte_offset`` in ``memory`` is read from.

    The offset is in the memory's own bytes, counted from its base. It must
    land on an entry: the address has no sub-entry field, so an unaligned
    offset would encode as the entry below it and read the wrong words.
    *Measured:* all 206 allocations of a reference compile are entry-aligned,
    so this refuses rather than rounds.

    Args:
        memory: A ``MemoryInstance`` from the declared machine.
        byte_offset: Where the tensor starts, from the memory's base.
        cluster: How many banks are addressed as one.
        operand_bits: The width bit 19 records.

    Returns:
        An :class:`AlphaAddress`.

    Raises:
        ValueError: If the offset is negative, not entry-aligned, or beyond
            what the memory's banks can address.
    """
    base = ALPHA_MEMORY_BASES.get(memory.name)
    if base is None:
        raise ValueError(f"{memory.name}: no Alpha base address is declared")
    if byte_offset < 0:
        raise ValueError(f"{memory.name}: negative offset {byte_offset}")
    entry_bytes = alpha_entry_bytes(memory, cluster)
    if byte_offset % entry_bytes:
        raise ValueError(
            f"{memory.name}: offset {byte_offset} is not a multiple of the "
            f"{entry_bytes}-byte entry a cluster of {cluster} addresses; the "
            f"address has no field for the remainder"
        )
    slot = byte_offset // entry_bytes
    depth = int(memory.depth_per_bank)
    virtual_bank = slot // depth
    if virtual_bank * cluster >= int(memory.bank_count):
        raise ValueError(
            f"{memory.name}: offset {byte_offset} needs physical bank "
            f"{virtual_bank * cluster} of {memory.bank_count}"
        )
    return AlphaAddress(
        base=base,
        operand_bits=operand_bits,
        cluster=cluster,
        virtual_bank=virtual_bank,
        entry=slot % depth,
    )
