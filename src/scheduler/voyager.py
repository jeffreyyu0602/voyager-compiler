# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The narrow boundary where a scheduler result is attached to Voyager FX.

Scheduler engines never receive graph nodes.  A target first freezes the
facts it recognizes into ``OperationWorkload``, dispatch returns one immutable
``LocalScheduleResult``, and this module binds that pair to the final record
owner.  Mapping geometry therefore travels with the record that will execute
it without becoming Interstellar metadata or serializer-owned policy.
"""

from scheduler.types import LocalScheduleResult, OperationWorkload

LOCAL_SCHEDULE_META = "local_schedule"
LOCAL_SCHEDULER_WORKLOAD_META = "local_scheduler_workload"


def bind_local_schedule(node, result, workload) -> None:
    """Bind one immutable scheduling decision exactly once."""
    if not isinstance(result, LocalScheduleResult):
        raise TypeError("a bound local schedule must be LocalScheduleResult")
    if not isinstance(workload, OperationWorkload):
        raise TypeError("a bound scheduler workload must be OperationWorkload")

    previous = node.meta.get(LOCAL_SCHEDULE_META)
    if previous is not None and previous != result:
        raise ValueError(f"{node.name}: conflicting local schedules")
    previous_workload = node.meta.get(LOCAL_SCHEDULER_WORKLOAD_META)
    if previous_workload is not None and previous_workload != workload:
        raise ValueError(f"{node.name}: conflicting scheduler workloads")
    node.meta[LOCAL_SCHEDULE_META] = result
    node.meta[LOCAL_SCHEDULER_WORKLOAD_META] = workload


__all__ = [
    "LOCAL_SCHEDULE_META",
    "LOCAL_SCHEDULER_WORKLOAD_META",
    "bind_local_schedule",
]
