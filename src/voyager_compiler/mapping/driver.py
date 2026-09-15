# Drive operation traversal, callback-based search, retained reports, and artifacts
import json
import subprocess
from pathlib import Path
from time import perf_counter

from google.protobuf.json_format import MessageToDict
from voyager_compiler.codegen import tiling_pb2
from .operations import channel_metadata, matrix_operation, parse_operation, skip_reason
from .reporting import utilization_metrics
from .search import prepare_search, search_mapping
from .results import serialize, write_tilings
from .models.cim_timing import TimingOptions
from .models.output import OutputOptions


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
    option_type = TimingOptions if target.backend == "cim" else OutputOptions
    if options is None or isinstance(options, dict):
        return option_type(**(options or {}))
    if type(options) is not option_type:
        raise ValueError(f"{target.backend} requires {option_type.__name__}")
    return options


# Export compiler-selected epilogue stages and transfers for all matrix-search operations
def export_epilogues(model, executable):
    from google.protobuf import text_format
    selected = type(model)()
    for operation in model.ops:
        if skip_reason(matrix_operation(operation)) is None:
            selected.ops.add().CopyFrom(operation)
    result = subprocess.run([str(Path(executable).resolve())],
                            input=text_format.MessageToString(selected),
                            text=True, capture_output=True)
    if result.returncode:
        raise ValueError("epilogue export failed: " + result.stderr.strip())
    return json.loads(result.stdout)


# Map both backends through the same traversal, invocation cache, and reporting
def generate_tilings(model, target, *, timing_options=None, verbose=0, epilogues=None):
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
        epilogue = None if epilogues is None else epilogues[name]
        key = operation_key(operation), json.dumps(epilogue, sort_keys=True)
        reused = key in cache
        try:
            if not reused:
                workload, vector_timing = parse_operation(target, operation, epilogue)
                inputs = prepare_search(target, workload, vector_timing=vector_timing, options=timing)
                if verbose:
                    layer = inputs.layer
                    print(f"Searching {name}: backend={target.backend}; lanes={target.k}x{target.n}; "
                          f"IC={layer.nifm} OC={layer.nofm} OX={layer.wofm} OY={layer.hofm} "
                          f"FX={layer.wfil} FY={layer.hfil}", flush=True)
                started = perf_counter()
                mapping = search_mapping(inputs, verbose=verbose)
                cache[key] = mapping, inputs, perf_counter() - started
            elif verbose:
                print(f"Reusing an identical operation's mapping for {name}", flush=True)
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
        metrics.update(utilization_metrics(metrics, detail["evaluation"]))
        report["operations"].append(dict(name=name, reused=reused, search_seconds=elapsed,
                                         metrics=metrics, useful_work_basis="compiler logical channels" if logical else "supplied extents without padding metadata", **detail))
        print(f"{name}: {evaluation.runtime_cycles:g} cycles; "
              f"dense ideal {metrics['dense_ideal_cycles']:g}; useful ideal {metrics['useful_ideal_cycles']:g}; "
              f"utilization {metrics['utilization']:.2%}; true utilization {metrics['true_utilization']:.2%}; "
              f"search {elapsed:.3f}s", flush=True)
        if verbose and "search" in detail:
            print(f"{name} search statistics: {json.dumps(detail['search'], sort_keys=True)}", flush=True)
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
    parser.add_argument("--epilogue_exporter", type=Path, help="Host exporter for epilogue execution descriptions")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--timing_options", type=Path, help="JSON timing assumptions in cycles")
    parser.add_argument("--verbose", type=int, choices=range(4), default=0,
                        help="0: results, 1: progress, 2: tile factors and best updates, 3: every evaluated candidate")
    args = parser.parse_args(argv)
    target = load_target(args.target)
    if args.backend is not None and args.backend != target.backend:
        parser.error("selected backend does not match mapping target")
    model = text_format.Parse((args.codegen_dir / "model.txt").read_text(), param_pb2.Model())
    timing = json.loads(args.timing_options.read_text()) if args.timing_options else None
    epilogues = export_epilogues(model, args.epilogue_exporter) if args.epilogue_exporter else None
    tilings, report = generate_tilings(model, target, timing_options=timing, verbose=args.verbose, epilogues=epilogues)
    write_tilings(tilings, report, args.output_dir)
