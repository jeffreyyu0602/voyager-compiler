"""Chrome / Perfetto trace export (``chrome://tracing`` JSON).

Emits every walked event: compute split onto a Matrix (systolic array) and
a Vector (vector unit) track, the DRAM interface on another, control
(waits) on a third.  A fused pass that runs vector ops alongside its
matrix anchor (a GEMM + elementwise epilogue) draws on BOTH compute
tracks.  A folded steady state -- iterations the walk skipped as repeats
of the period before them -- appears as one bar on the Control track
spanning the skipped time, so the trace stays readable at any trip count.
Time is in cycles (the trace ``ts``/``dur`` unit is arbitrary).
"""

import json
from typing import Dict

from voyager_compiler.codegen.reporting.model import ScheduleResult

_TID = {"mma": 0, "vector": 1, "dram": 2, "control": 3}
_TRACK = {0: "Matrix", 1: "Vector", 2: "DRAM", 3: "Control"}


def perfetto_dict(result: ScheduleResult) -> Dict:
    events = []
    for tid, name in _TRACK.items():
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "args": {"name": name},
            }
        )
    for r in result.records:
        for lane in r.resource:
            events.append(
                {
                    "name": f"{r.node_name} {r.iteration_path}",
                    "cat": r.kind,
                    "ph": "X",
                    "ts": r.start,
                    "dur": max(0, r.end - r.start),
                    "pid": 0,
                    "tid": _TID.get(lane, _TID["control"]),
                    "args": {
                        "eid": r.eid,
                        "kind": r.kind,
                        "kernel": r.kernel,
                        "bytes": r.bytes,
                        "is_read": r.is_read,
                        "iteration": list(r.iteration_path),
                    },
                }
            )
    for s in result.skips:
        events.append(
            {
                "name": f"{s.kernel} x{s.repeats} periods folded",
                "cat": "fold",
                "ph": "X",
                "ts": s.start,
                "dur": s.end - s.start,
                "pid": 0,
                "tid": _TID["control"],
                "args": {
                    "kernel": s.kernel,
                    "first_step": s.first_step,
                    "iterations": s.iterations,
                    "period": s.period,
                    "repeats": s.repeats,
                    "cycles_per_period": s.shift,
                },
            }
        )
    return {"traceEvents": events, "displayTimeUnit": "ns"}


def write_perfetto(result: ScheduleResult, path: str) -> str:
    """Write the trace JSON to ``path`` (open it in ``chrome://tracing`` or
    ``ui.perfetto.dev``) and return the path."""
    with open(path, "w") as f:
        json.dump(perfetto_dict(result), f)
    return path
