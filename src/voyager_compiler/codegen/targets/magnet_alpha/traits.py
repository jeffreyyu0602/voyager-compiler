# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""What Alpha's datapath implies about domains, fusion and operands.

The compiler's shared rules describe what an operation means. These describe
what *this machine* can do about it, which is a different question and
sometimes a narrower answer.

Two kinds of declaration live here: the quantization-domain relations Alpha
adds to the shared set, and the operator sequences its datapath issues as one
instruction. Operand roles arrive with the operators they describe.
"""

import torch

from voyager_compiler.op_matcher import OpMatcher
from voyager_compiler.codegen.targets.magnet_alpha.graph import (
    is_channel_shuffle_cat,
)
from voyager_compiler.quantization.contracts import (
    PreparedDomainInputFormat,
    PreparedDomainRelation,
    PreparedDomainRuleSet,
    PreparedDomainTransferRule,
)
from voyager_compiler.codegen.targets.magnet_alpha.calibration import (
    ALPHA_CALIBRATION_REFERENCE,
)
from voyager_compiler.quantization.regions import (
    ACTIVATION_EPILOGUES,
    absorbed_by_a_matrix_call,
    shared_domain_rules,
)

__all__ = [
    "EXECUTION_FUSION_PATTERNS",
    "NON_ALLOCATING_OPERAND_META",
    "PREPARED_DOMAIN_RULES",
    "prepared_domain_rules",
]


def _drains_a_matrix_call(node) -> bool:
    """Whether this activation sits on a matrix call's drain.

    Two shapes, because calibration puts an instrument in the middle of one.
    Directly after the convolution is the plain case. After a *calibration
    reference* is the same boundary with the Q8.8 epilogue reference spliced
    in, and the ReLU is no less absorbed for it -- reading only the plain
    case would put the frontier back on the accumulator the moment the
    reference was inserted, which is the defect the reference exists around.
    """
    if absorbed_by_a_matrix_call(node):
        return True
    if node.op != "call_function" or node.target not in ACTIVATION_EPILOGUES:
        return False
    source = node.args[0] if node.args else None
    return (
        source is not None
        and getattr(source, "op", None) == "call_module"
        and bool(source.meta.get(ALPHA_CALIBRATION_REFERENCE))
        and len(source.users) == 1
    )


def _stands_alone(node) -> bool:
    """Whether this activation is a value of its own, not a matrix drain.

    The negative of ``_drains_a_matrix_call``. A free-standing ReLU still
    aliases its input's domain -- the encoding argument for that is unchanged
    -- but one the array absorbs is not a value a region can put a frontier
    on, because there is no addressable tensor between the convolution and
    its epilogue.
    """
    return not _drains_a_matrix_call(node)


#: Alpha's own relations, composed *ahead* of the shared ones so a narrower
#: reading of an operation the compiler already describes wins.
PREPARED_DOMAIN_RULES = PreparedDomainRuleSet(
    (
        # A residual add. Its two addends have to be in one domain -- the
        # PPU adds codes and has no second scale to reconcile them with --
        # while the sum's own range is wider than either addend's, so the
        # result must not be dragged into their domain. That distinction is
        # the whole reason INPUT_JOIN exists.
        *(
            PreparedDomainTransferRule(
                target=target,
                input_format=PreparedDomainInputFormat.PREFIX,
                relation=PreparedDomainRelation.INPUT_JOIN,
                input_count=2,
                region_root=True,
                forcing_root=True,
            )
            for target in (
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add_.Tensor,
            )
        ),
        # ReLU on codes. Alpha's PPU clamps at zero on the way out of a
        # matrix call, and clamping a symmetric zero-point domain at zero is
        # exactly the identity on the encoding: ReLU(Q(x)) == Q(ReLU(x)).
        # So the activation does not change the domain, it aliases it.
        #
        # **Only when it is a value.** The encoding argument is about the
        # codes and says nothing about where a boundary may sit, and the two
        # are not the same question. An activation the array drains through
        # is part of the matrix instruction; aliasing makes it a region
        # *member*, and then the value entering the region is the
        # accumulator, so the entry observer lands between the convolution
        # and its own ReLU. ``_pe_boundary`` then matches
        # ``dequantize -> quantize`` instead of ``dequantize -> relu``,
        # folds nothing, and emits ``relu=False`` over an activation left in
        # the graph for nobody to run. *Measured before this predicate:* 37
        # of 66 matrix records carried ``relu=False`` against a reference
        # compile's 3, and 33 ``aten.relu.default`` nodes survived with no
        # record performing them.
        *(
            PreparedDomainTransferRule(
                target=target,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.ALIAS,
                predicate=_stands_alone,
            )
            for target in (
                torch.ops.aten.relu.default,
                torch.ops.aten.relu_.default,
            )
        ),
        # Alpha's shuffle is physically a two-input INT8 join.  The source
        # graph still spells it as cat/view/transpose/contiguous/view during
        # PT2E preparation, so the cat has to force the shared domain before
        # lowering collapses that spine into the target operator.
        PreparedDomainTransferRule(
            target=torch.ops.aten.cat.default,
            input_format=PreparedDomainInputFormat.SEQUENCE,
            relation=PreparedDomainRelation.JOIN,
            region_root=True,
            forcing_root=True,
            predicate=is_channel_shuffle_cat,
        ),
        # Directly prepared target graphs carry the same physical relation.
        PreparedDomainTransferRule(
            target=torch.ops.magnet_alpha.channel_shuffle.default,
            input_format=PreparedDomainInputFormat.PREFIX,
            relation=PreparedDomainRelation.JOIN,
            input_count=2,
            region_root=True,
            forcing_root=True,
        ),
    )
)


def prepared_domain_rules() -> PreparedDomainRuleSet:
    """Alpha's rules, then the compiler's."""
    return PREPARED_DOMAIN_RULES.compose(shared_domain_rules())


#: The sequences Alpha's datapath issues as **one instruction**, and therefore
#: the sequences that must become one record.
#:
#: These are not a performance fusion. A matrix call *is* an accumulator plus
#: its epilogue -- the converter emits the pair as a single
#: ``gemm_convolution`` -- so a graph where the two are separate nodes does not
#: describe a slower machine, it describes records that do not exist. Every
#: unfused pair is a lost record, silently: ``execution_kind`` recognizes the
#: *sequence*, and a lone accumulator is not one.
#:
#: The head is an alternation because the two convolution heads differ only in
#: how the operands are scaled; both drain through the same epilogue and both
#: convert to the same record kind.
EXECUTION_FUSION_PATTERNS = (
    (
        OpMatcher(
            torch.ops.magnet_alpha.conv2d_int8_accumulator.default,
            torch.ops.magnet_alpha.conv2d_vsq.default,
        ),
        OpMatcher(torch.ops.magnet_alpha.ppu_fixed_relu.default),
        OpMatcher(torch.ops.magnet_alpha.pvs_quantize.default),
    ),
    (
        OpMatcher(
            torch.ops.magnet_alpha.conv2d_int8_accumulator.default,
            torch.ops.magnet_alpha.conv2d_vsq.default,
        ),
        OpMatcher(torch.ops.magnet_alpha.ppu_fixed_relu.default),
    ),
    (
        OpMatcher(torch.ops.magnet_alpha.conv2d_int8_accumulator.default),
        OpMatcher(torch.ops.magnet_alpha.nmp_fixed_requantize.default),
        OpMatcher(torch.ops.magnet_alpha.pvs_quantize.default),
    ),
    (
        OpMatcher(torch.ops.magnet_alpha.conv2d_int8_accumulator.default),
        OpMatcher(torch.ops.magnet_alpha.nmp_fixed_requantize.default),
    ),
)


#: Meta keys marking an operand the endpoint does **not** fetch from an
#: address.
#:
#: A per-channel multiplier or an encoded bias is folded into the instruction
#: that reads it, so it occupies no storage and must not appear among the
#: model's parameters -- the reference collateral lists none of them. This is
#: the same distinction ``require_allocation`` draws for quantization tables
#: and scalar scales, applied to operands only this target creates, so only
#: this target can say so.
#:
#: Two keys because there are two reasons: an *instruction* operand is
#: programmed into the call, and a *semantic* one is kept for reference and
#: validation but never issued.
NON_ALLOCATING_OPERAND_META = (
    "magnet_alpha_instruction_operand",
    "magnet_alpha_semantic_operand",
)
