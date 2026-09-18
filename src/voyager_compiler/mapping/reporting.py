# Present selected mappings and analytical estimates without search bookkeeping
import argparse
import json
from pathlib import Path


LOOPS = ("OX", "OY", "IC", "OC", "FX", "FY")


# Separate full dense work from useful work excluding modeled zero padding
def utilization_metrics(metrics, evaluation):
    runtime = metrics["runtime_cycles"]
    useful = metrics["ideal_cycles"]
    fraction = metrics.get("useful_work_fraction")
    dense = evaluation.get("timing", {}).get("compute_cycles")
    if dense is None and useful is not None and fraction:
        dense = useful / fraction
    return dict(dense_ideal_cycles=dense, useful_ideal_cycles=useful,
                utilization=dense / runtime if dense is not None and runtime else None,
                true_utilization=useful / runtime if useful is not None and runtime else None)


# Format missing measurements explicitly rather than treating them as zero
def number(value, digits=0):
    return "n/a" if value is None else f"{value:,.{digits}f}"


# Display dimensionless utilization as a percentage
def percent(value):
    return "n/a" if value is None else f"{value:.2%}"


# Escape operation names used in Markdown tables
def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


# Resolve the array geometry from the recorded configuration
def geometry(target):
    if target["backend"] == "cim":
        k = target["ch_in"] * target["tile_input_axis_elements"] * target["input_axis_tiles"]
        n = target["ch_out"] // (8 // target["base_b_width"])
        return k, n * target["tile_output_axis_elements"] * target["output_axis_tiles"]
    return target["k"], target["n"]


# Show energy only when the evaluator declares a physical pJ cost
def energy_pj(evaluation):
    if evaluation.get("energy") is not None:
        return evaluation["energy"]["total_pj"]
    return evaluation.get("cost") if evaluation.get("cost_unit") == "pJ" else None


# Render one compact comparison row per mapped operation
def summary(report):
    lines = ["| Layer | Cycles | Energy (µJ) | Utilization | True utilization | Mapping reused |",
             "|---|---:|---:|---:|---:|---|"]
    for op in report["operations"]:
        metrics = utilization_metrics(op["metrics"], op["evaluation"])
        energy = energy_pj(op["evaluation"])
        lines.append(f"| {cell(op['name'])} | {number(op['metrics']['runtime_cycles'])} | "
                     f"{number(energy / 1e6 if energy is not None else None, 3)} | "
                     f"{percent(metrics['utilization'])} | {percent(metrics['true_utilization'])} | "
                     f"{'yes' if op.get('reused') else 'no'} |")
    return "\n".join(lines)


# Read selected loop factors in their actual inner-to-outer order
def levels(operation, tilings):
    if "schedule" in operation:
        return [[(loop, dict(zip(LOOPS, operation["schedule"][name]["bounds"]))[loop])
                 for loop in operation["schedule"][name]["order"]] for name in ("l1", "l2")]
    if tilings is not None:
        from voyager_compiler.codegen import tiling_pb2
        for tiling in tilings.tilings:
            if tiling.name == operation["name"]:
                return [[(tiling_pb2.Loop.Name(item.loop), item.bound) for item in level.loop_bounds]
                        for level in tiling.level_tilings]
    return []


# Explain one operation's performance, selected loops, and storage use
def layer_report(operation, target, tilings=None):
    op, evaluation = operation, operation["evaluation"]
    metrics = utilization_metrics(op["metrics"], evaluation)
    workload = op.get("workload", {})
    k, n = geometry(target)
    lines = [f"## {cell(op['name'])}", ""]
    if "input_x" in workload:
        stride, padding = workload["stride"], workload["padding"]
        ox = (workload["input_x"] + 2 * padding - workload["filter_x"]) // stride + 1
        oy = (workload["input_y"] + 2 * padding - workload["filter_y"]) // stride + 1
        lines += [f"Input X×Y×C: {workload['input_x']}×{workload['input_y']}×{workload['input_channels']}; "
                  f"output: {ox}×{oy}×{workload['output_channels']}; "
                  f"filter X×Y: {workload['filter_x']}×{workload['filter_y']}; stride {stride}; padding {padding}.", ""]
    else:
        lines += [f"Channels: {workload.get('input_channels', '?')} → {workload.get('output_channels', '?')}; "
                  f"output X×Y: {workload.get('output_x', '?')}×{workload.get('output_y', '?')}.", ""]
    logical, padded = workload.get("logical_channels"), workload.get("padded_channels")
    if logical and padded and logical != padded:
        lines += [f"Channel padding IC/OC: {logical[0]}/{logical[1]} → {padded[0]}/{padded[1]}.", ""]
    runtime, energy = op["metrics"]["runtime_cycles"], energy_pj(evaluation)
    energy_text = (f"**{number(energy / 1e6, 3)} µJ** ({number(energy, 1)} pJ)"
                   if energy is not None else "**unavailable (uncharacterized)**")
    lines += [f"Estimated cycles: **{number(runtime)}**; energy: {energy_text}.", "",
              f"- Utilization: **{percent(metrics['utilization'])}** = "
              f"{number(metrics['dense_ideal_cycles'])} dense ideal cycles / {number(runtime)} cycles",
              f"- True utilization: **{percent(metrics['true_utilization'])}** = "
              f"{number(metrics['useful_ideal_cycles'])} useful ideal cycles / {number(runtime)} cycles", "",
              "### Selected mapping", "", f"Fixed spatial lanes: IC={k}, OC={n}.", ""]
    selected = levels(op, tilings)
    for label, level in zip(("L1", "L2"), selected):
        nest = " → ".join(f"{loop} ×{factor}" for loop, factor in level)
        lines += [f"**{label}, inner → outer:** `{nest}`", ""]
    if not selected:
        lines += ["Loop factors are unavailable in this report; inspect tilings.txtpb.", ""]
    elif len(selected) == 2:
        l1, l2 = map(dict, selected)
        lines += [f"L1 output tile: {l1.get('OX', 1)}×{l1.get('OY', 1)} spatial positions × "
                  f"{l1.get('OC', 1) * n} output channels; reduction tile: "
                  f"{l1.get('IC', 1) * k} input channels × {l1.get('FX', 1)}×{l1.get('FY', 1)} filter positions.", "",
                  "L2 factors repeat L1 tiles; they are counts, not absolute tensor dimensions.", ""]
    timing = evaluation.get("timing", {})
    if timing:
        lines += ["### Timing and storage", "",
                  f"Startup: {number(timing['startup_cycles'])} cycles; drain: {number(timing['drain_cycles'])} cycles. "
                  "Resource work totals overlap and must not be added together.", "",
                  "| Stage | Service cycles |", "|---|---:|"]
        for stage in ("compute", "result_slots", "input", "weight", "accumulation", "output", "bias"):
            if stage in timing["resource_cycles"]:
                lines.append(f"| {stage.replace('_', ' ')} | {number(timing['resource_cycles'][stage])} |")
        lines += [""]
        readiness = timing.get("readiness", {})
        if readiness:
            lines += [f"Weight readiness adds {number(readiness['weight_wait_cycles'])} cycles to MAC issue time; "
                      f"input-bank readiness adds {number(readiness['input_wait_cycles'])}. "
                      "These overlapping constraints are not additive.", ""]
            if "weight_fill_cycles" in readiness:
                lines += [f"Resident weight-set fill time: {number(readiness['weight_fill_cycles'])} cycles; "
                          f"release-to-reuse delay: {number(readiness['weight_release_cycles'])}; "
                          f"fill-to-ready delay: {number(readiness['weight_ready_cycles'])}. "
                          "Control delays are explicit timing assumptions.", ""]
            if readiness.get("input_max_fill_bound"):
                lines += ["Input timing uses a maximum-fill bound for variable boundary tiles.", ""]
            if any(readiness.get(key) for key in ("weight_serialized_bound", "input_serialized_bound", "output_bank_serialized_bound")):
                lines += ["Timing reached its fixed state/work limit and uses a serialized upper bound.", ""]
            spacing = readiness.get("accumulation_feedback_spacing_cycles", 0)
            if spacing:
                safety = "checked against the supplied latency" if readiness.get("accumulation_feedback_safe") else "not checked: SRAM feedback latency is unspecified"
                lines += [f"Minimum SRAM feedback spacing: {number(spacing)} cycles; {safety}.", ""]
    bias = timing.get("readiness", {}) if timing else op.get("bias_timing", {})
    if "bias_wait_cycles" in bias:
        lines += ["### Bias loading", "",
                  f"One bias vector needs {number(bias['bias_cycles_per_vector'])} transfer cycles; "
                  f"{number(bias['bias_prefetch_vectors'])} complete vector can be prefetched. "
                  f"First-reduction demand adds {number(bias['bias_wait_cycles'])} wait cycles.", ""]
    inputs = op.get("input_loading")
    if inputs:
        lines += ["### Input loading", "",
                  f"{inputs['lane_elements']} elements per bank word, {inputs['element_bits']} bits per source element; "
                  f"{inputs['pack_factor']} words per request on a {inputs['port_bits']}-bit port.", "",
                  f"{number(inputs['requests'])} external requests / {number(inputs['external_beats'])} beats; "
                  f"{number(inputs['writes'])} buffer-word writes including zero padding. "
                  f"{number(inputs['fills'])} bank fills; first fill {number(inputs['first_fill_cycles'])} cycles; "
                  f"fill time {number(inputs['min_fill_cycles'])}–{number(inputs['max_fill_cycles'])} cycles.", "",
                  f"Input-bank readiness adds {number(inputs['wait_cycles'])} cycles to the compute schedule. "
                  "This overlaps other stage constraints and must not be added to their waits.", ""]
        if inputs['max_fill_bound']:
            lines += ["Variable boundary fills use a maximum-fill timing bound.", ""]
        if inputs['serialized_bound']:
            lines += ["Input readiness reached its work limit and uses a serialized upper bound.", ""]
    vector = op.get("vector_unit")
    if vector:
        lines += ["### Vector timing", "",
                  f"One result vector contains {vector['elements_per_vector']} output elements; "
                  f"input/output precision: {vector['input_element_bits']}/{vector['output_element_bits']} bits per element. "
                  f"Shared-resource throughput bound: {vector['cycles_per_vector']} cycles per result vector.", ""]
        for index, stage_pass in enumerate(vector["passes"], 1):
            stages = ", ".join(f"stage {i}: {name}" for i, name in enumerate(stage_pass["stages"]) if name) or "forwarding"
            lines += [f"- Pass {index}: source {stage_pass['source']}; {stages}; "
                      f"{stage_pass['cycles_per_vector']} cycles per result vector"]
        lines += [""]
    output_ready = timing.get("readiness", {}) or op.get("output_timing", {})
    if "output_stall_cycles" in output_ready:
        lines += [f"Finite output buffering adds **{number(output_ready['output_stall_cycles'])} producer stall cycles**. "
                  f"Explicit storage capacity: {number(output_ready['output_capacity_vectors'])} result vectors of "
                  f"{number(output_ready['output_elements_per_vector'])} elements; output processing time: "
                  f"{number(output_ready['output_cycles_per_vector'])} cycles per result vector. "
                  f"Remaining consumer work: {number(output_ready['output_backlog_cycles'])} cycles.", ""]
        storage = "; ".join(f"{name.replace('_', ' ')}: {number(elements)} elements"
                            for name, elements in output_ready["output_storage_elements"].items() if elements)
        lines += [storage + ".", "",
                  "Capacity counts exported result storage. HLS-inserted pipeline registers, "
                  "control-only queues, and partial-sum contexts are excluded.", ""]
    policy = evaluation.get("policy")
    if policy:
        lines += [f"Resident weight sequence: {policy['sequence_sets']} sets / {target['b_sets']} available; "
                  f"{'fits in resident storage' if policy['fits'] else 'uses singleton refetch'}. "
                  f"Compute replays: {policy['compute_replays']}; fetch replays: {policy['fetch_replays']}.", "",
                  f"Input footprint: {evaluation['input_footprint']} / {target['input_buffer_words']} words; "
                  f"live output footprint: {evaluation['accumulation_footprint']} / {target['accum_buffer_words']} vectors. "
                  f"Local contexts required: {evaluation['local_accum_footprint']}; available: {target['local_accum_contexts']}. "
                  "Excess local contexts use accumulation SRAM.", ""]
    if "schedule" in op:
        lines += [f"Write output to accumulation buffer: "
                  f"{'yes' if op['schedule']['write_output_to_accum_buffer'] else 'no'}.", ""]
    search = op.get("search")
    if search:
        origin = "Reused mapping; counts describe its original search" if op.get("reused") else "Search"
        lines += [f"{origin}: {number(search['evaluated'])} evaluated, {number(search['legal'])} legal, "
                  f"{number(search['capacity_rejected'])} early capacity rejections; "
                  f"{number(op.get('search_seconds'), 3)} seconds.", ""]
    return "\n".join(lines)


# Assemble a readable report with definitions and explicit coverage
def render_report(report, tilings=None):
    target = report["target"]
    k, n = geometry(target)
    lines = ["# Mapping report", "",
             f"Target: **{target['backend'].upper()} {k}×{n}, {target['datatype']}**; "
             f"external ports: {target['ic_port_bits']}/{target['oc_port_bits']} bits; "
             f"{len(report['operations'])} mapped operations.", ""]
    if target["backend"] == "cim":
        lines += [f"Macro A/B/C widths: {target['base_a_width']}/{target['base_b_width']}/{target['base_c_width']} bits; "
                  f"resident sets: {target['b_sets']}; MAC latency: {target['mac_latency']}; "
                  f"result slots per output lane: {target['result_slots_per_output_lane']}.", ""]
    lines += ["Utilization = dense ideal cycles / estimated cycles, including padding work. "
              "True utilization = useful ideal cycles / estimated cycles, excluding modeled zero padding. "
              "Both backends exclude modeled spatial convolution padding and compiler channel padding.", "",
              "Cycles cover the modeled matrix path, not the complete fused operator or end-to-end network. "
              "Energy is unavailable without hardware characterization; n/a does not mean zero. "
              "Stored reports with declared pJ estimates display those values. "
              "Search ranks by cycles and retains the first candidate on an exact tie.", "",
              "## Layer summary", "", summary(report), "",
              "## Reading the mapping", "",
              "OX/OY are output X/Y positions; IC/OC are input/output channel blocks; FX/FY are filter X/Y positions. "
              "IC/OC temporal factors multiply the fixed spatial lane counts. Each level is written inner → outer; "
              "all L1 loops execute inside the L2 loops. Unit factors are shown explicitly. ON, when present, is batch.", ""]
    for op in report["operations"]:
        lines += [layer_report(op, target, tilings)]
    if report.get("skipped"):
        lines += ["## Operations outside this mapping report", "", "| Operation | Reason |", "|---|---|"]
        lines += [f"| {cell(op['name'])} | {cell(op['reason'])} |" for op in report["skipped"]]
    return "\n".join(lines) + "\n"


# Format saved results without invoking the compiler or mapping search
def main(argv=None):
    parser = argparse.ArgumentParser(description="Render a readable per-layer mapping report")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--layer", help="Print one layer's selected mapping and statistics")
    args = parser.parse_args(argv)
    from google.protobuf import text_format
    from voyager_compiler.codegen import tiling_pb2
    from .results import write_if_changed
    path = args.results_dir / "mapping-evaluations.json"
    if not path.is_file():
        parser.error(f"missing {path}; run make network-proto first")
    report = json.loads(path.read_text())
    tiling_path = args.results_dir / "tilings.txtpb"
    tilings = (text_format.Parse(tiling_path.read_text(), tiling_pb2.ModelTiling(), allow_unknown_field=True)
               if tiling_path.is_file() else None)
    if args.layer:
        operation = next((op for op in report["operations"] if op["name"] == args.layer), None)
        if operation is None:
            parser.error(f"no mapped layer named {args.layer}")
        print(layer_report(operation, report["target"], tilings))
    else:
        output = args.results_dir / "mapping-report.md"
        write_if_changed(output, render_report(report, tilings))
        print(summary(report))
        print(f"\nFull report: {output}")


if __name__ == "__main__":
    main()
