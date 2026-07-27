"""Chrome / Perfetto trace export (``chrome://tracing`` JSON).

Emits the full, uncompressed event stream so the exact schedule can be
inspected: compute split onto a Matrix (systolic array) and a Vector
(vector unit) track, the DRAM interface on another, control (waits) on a
third.  A fused pass that runs vector ops alongside its matrix anchor (a
GEMM + elementwise epilogue) draws on BOTH compute tracks.  Time is in
cycles (the trace ``ts``/``dur`` unit is arbitrary).
"""

import json
from typing import Dict

from .model import ScheduleResult

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
                        "bytes": r.bytes,
                        "is_read": r.is_read,
                        "iteration": list(r.iteration_path),
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
