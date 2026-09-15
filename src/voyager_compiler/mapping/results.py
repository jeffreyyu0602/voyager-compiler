# Write selected mappings and evaluation reports to the requested build directory
import json
import os
from pathlib import Path
import tempfile

from google.protobuf import text_format
from .reporting import render_report
from dataclasses import asdict
from voyager_compiler.codegen import tiling_pb2


# Serialize resolved settings and reports consistently
def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


# Publish complete files atomically and preserve unchanged output timestamps
def write_if_changed(path, content):
    path = Path(path)
    data = content.encode()
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


# Serialize the retained winner without invoking its model again
def serialize(name, target, mapping, inputs):
    evaluation = mapping.evaluation
    result = evaluation.metadata
    schedule = result.schedule
    tiling = tiling_pb2.Tiling(name=name)
    if target.backend == "sa":
        for level in (1, 2):
            output = tiling.level_tilings.add()
            for loop in sorted(range(7), key=lambda loop: mapping.loop_orders[loop][level]):
                output.loop_bounds.add(loop=loop, bound=mapping.loop_blockings[loop][level])
    else:
        for level in (schedule.l1, schedule.l2):
            output = tiling.level_tilings.add()
            for loop in level.order:
                output.loop_bounds.add(loop=getattr(tiling_pb2, loop), bound=level.bound(loop))
    traffic = result.traffic
    if target.backend == "cim":
        buffer_outputs = (traffic.buffer_accum_reads + traffic.buffer_accum_intermediate_writes
                          + traffic.buffer_accum_final_writes + traffic.buffer_output_reads)
        counts = ((traffic.input_buffer_reads * target.k, buffer_outputs * target.n, traffic.b_write_bytes),
                  (traffic.input_requested_bytes,
                   0 if traffic.vector_output_vectors else (traffic.direct_output_vectors + traffic.buffer_output_reads) * target.n,
                   traffic.weight_requested_bytes))
    else:
        counts = traffic["accesses"][1:3]
    for input_count, output_count, weight_count in counts:
        tiling.level_access_counts.add(input_access_count=input_count, output_access_count=output_count,
                                      weight_access_count=weight_count)
    context = inputs.candidate_evaluator
    if target.backend == "cim":
        detail = asdict(result)
        detail.pop("details")
        detail.pop("schedule")
        detail["runtime_cycles"] = result.runtime_cycles
        return tiling, dict(workload=asdict(context.workload), schedule=asdict(schedule), evaluation=detail,
                            search=context.statistics())
    detail = asdict(evaluation)
    detail.pop("metadata")
    return tiling, dict(evaluation=detail, vector_unit=context.vector_timing.report(), **result.details,
                       workload=dict(input_channels=context.workload.input_channels,
                                     output_channels=context.workload.output_channels,
                                     output_x=context.workload.output_x, output_y=context.workload.output_y))


# Write an already selected set of tilings and evaluation reports
def write_tilings(tilings, report, output_dir):
    output_dir = Path(output_dir)
    write_if_changed(output_dir / "tilings.txtpb", text_format.MessageToString(tilings))
    write_if_changed(output_dir / "mapping-evaluations.json", json_text(report))
    write_if_changed(output_dir / "mapping-report.md", render_report(report, tilings))
