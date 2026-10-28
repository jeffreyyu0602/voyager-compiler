# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Alpha's schedulers, and the provider set that offers them.

Two seams, because Alpha's two units need different answers.

Its **near-memory processor** owns depthwise convolution, pooling, upsampling
and channel shuffle. For that work there is nothing to search: the unit reads
a tensor and writes it back, and what a schedule has to say is how many
passes over the channels and positions that takes. So the NMP is served by an
*operation provider*, which claims those operations outright and never
consults a GEMM engine.

Its **matrix unit** runs GEMM-shaped work, and that does need a search. It is
served here by Alpha's deterministic rule-based GEMM engine.

The operation provider runs whichever engine is selected, because it is about
which *unit* owns the work rather than which engine plans it. That is why the
reference lane's performance estimates carry two backends at once: 66 GEMM
records from the selected engine and 78 from the NMP.
"""

from scheduler.magnet_alpha_scheduler.providers import (
    MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    MAGNET_ALPHA_SCHEDULER_PROVIDERS,
)
from scheduler.magnet_alpha_scheduler.scheduler import (
    MagnetAlphaSchedulingError,
    project_gemm_layer,
    single_call_gemm_schedule,
    single_call_gemm_schedule_result,
)

__all__ = [
    "MAGNET_ALPHA_RULE_BASED_SCHEDULER",
    "MAGNET_ALPHA_SCHEDULER_PROVIDERS",
    "MagnetAlphaSchedulingError",
    "project_gemm_layer",
    "single_call_gemm_schedule",
    "single_call_gemm_schedule_result",
]
