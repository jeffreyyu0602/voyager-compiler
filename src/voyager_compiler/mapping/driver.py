# Drive operation traversal, callback-based search, retained reports, and artifacts
import json
from pathlib import Path
from time import perf_counter

from google.protobuf.json_format import MessageToDict
from voyager_compiler.codegen import tiling_pb2
from .operations import channel_metadata, matrix_operation, parse_operation, skip_reason
from .search import prepare_search, search_mapping
from .results import serialize, write_tilings






# Remove identity-only operation fields while retaining shape, route, and dtype
def operation_key(operation):
    # Normalize nested protobuf values without discarding list order
    def normalize(value):
        if isinstance(value, dict):
            return {key: {} if key == "memory" else normalize(item) for key, item in value.items()
                    if key not in ("name", "node", "scratchpad")}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    return json.dumps(normalize(MessageToDict(operation, preserving_proto_field_name=True)), sort_keys=True)




# Resolve only the options supported by the selected hardware evaluator
def resolve_timing(target, options=None):
    from .models import sa
    from .models.cim_timing import TimingOptions
    option_type = TimingOptions if target.backend == "cim" else sa.TimingOptions
    if options is None or isinstance(options, dict):
        return option_type(**(options or {}))
    if type(options) is not option_type:
        raise ValueError(f"{target.backend} requires {option_type.__name__}")
    return options


# Map both backends through the same traversal, invocation cache, and reporting
def generate_tilings(model, target, *, timing_options=None):
    timing = resolve_timing(target, timing_options)
    tilings = tiling_pb2.ModelTiling(backend=target.backend, target_configuration=target.configuration_json)
    report = dict(target=target.to_dict(), operations=[], skipped=[])
    cache, names = {}, set()
    for operation in model.ops:
        matrix = matrix_operation(operation)
        name = matrix.name if operation.HasField("op") else operation.fused_op.name
        reason = skip_reason(matrix)
        if reason:
            report["skipped"].append(dict(name=name, reason=reason))
            continue
        if name in names:
            raise ValueError(f"duplicate matrix operation name: {name}")
        names.add(name)
        key = operation_key(operation)
        reused = key in cache
        try:
            if not reused:
                workload, vector_timing = parse_operation(target, operation)
                inputs = prepare_search(target, workload, vector_timing=vector_timing, options=timing)
                started = perf_counter()
                mapping = search_mapping(inputs)
                cache[key] = mapping, inputs, perf_counter() - started
            mapping, inputs, elapsed = cache[key]
            tiling, detail = serialize(name, target, mapping, inputs)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        logical, padded = channel_metadata(matrix)
        tiling.logical_channels.extend(logical)
        tiling.padded_channels.extend(padded)
        tilings.tilings.append(tiling)
        evaluation = mapping.evaluation
        metrics = dict(ideal_cycles=evaluation.ideal_cycles, runtime_cycles=evaluation.runtime_cycles,
                       spatial_utilization=evaluation.spatial_utilization,
                       useful_work_fraction=evaluation.useful_work_fraction,
                       effective_utilization=evaluation.effective_utilization)
        report["operations"].append(dict(name=name, reused=reused, search_seconds=elapsed,
                                         metrics=metrics, useful_work_basis="compiler logical channels" if logical else "supplied extents without padding metadata", **detail))
        print(f"{name}: {evaluation.runtime_cycles:g} cycles; ideal {evaluation.ideal_cycles:g}; "
              f"spatial {evaluation.spatial_utilization:.2%}; useful work {evaluation.useful_work_fraction:.2%}; "
              f"effective {evaluation.effective_utilization:.2%}; search {elapsed:.3f}s", flush=True)
    return tilings, report


# Parse the common mapping command line without inspecting hardware sources
def main(argv=None):
    import argparse
    from google.protobuf import text_format
    from voyager_compiler.codegen import param_pb2
    from .target import load_target
    parser = argparse.ArgumentParser(description="Run the Voyager mapping driver")
    parser.add_argument("--codegen_dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True, help="Resolved mapping-target.json")
    parser.add_argument("--backend", choices=("sa", "cim"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--timing_options", type=Path, help="JSON timing assumptions in cycles")
    args = parser.parse_args(argv)
    target = load_target(args.target)
    if args.backend is not None and args.backend != target.backend:
        parser.error("selected backend does not match mapping target")
    model = text_format.Parse((args.codegen_dir / "model.txt").read_text(), param_pb2.Model())
    timing = json.loads(args.timing_options.read_text()) if args.timing_options else None
    tilings, report = generate_tilings(model, target, timing_options=timing)
    write_tilings(tilings, report, args.output_dir)
