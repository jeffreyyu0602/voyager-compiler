# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The compile's auxiliary artifacts, beside ``model.txt``.

``model.txt`` says what the accelerator executes. Three questions it
deliberately does not answer have consumers of their own:

* **what machine was this compiled for** -- ``hardware_config.json``. A
  collateral converter has to check that the schedule it is reading was
  planned against the endpoint it is about to launch, and it cannot do that
  from a graph.
* **what did the scheduler predict** -- ``performance_estimates.json``. Used
  to partition a workload into kernel calls; it is an estimate and says so in
  its own units field.
* **where does every tensor live** -- ``storage_plan.json`` and the
  target-owned payloads in ``supplemental_artifacts.json``.

Each writer is a *projection* of state the compiler already holds. None of
them derives anything: a value that is not already decided is absent, and an
artifact with nothing to say removes itself rather than being written empty.
That is what keeps a stale file from a previous compile out of a fresh dump.
"""

import json
import logging
import os
import traceback
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

#: Set to derive a storage plan when no producer attached one.
STORAGE_PLAN_ENV = "VOYAGER_STORAGE_PLAN"

HARDWARE_CONFIG_SCHEMA_VERSION = 1
SUPPLEMENTAL_ARTIFACTS_SCHEMA_VERSION = 1
PERFORMANCE_ESTIMATES_SCHEMA_VERSION = 1
STORAGE_PLAN_SCHEMA_VERSION = 1

_SUPPLEMENTAL_ARTIFACTS_META = "supplemental_compile_artifacts"

__all__ = [
    "HARDWARE_CONFIG_SCHEMA_VERSION",
    "PERFORMANCE_ESTIMATES_SCHEMA_VERSION",
    "STORAGE_PLAN_SCHEMA_VERSION",
    "SUPPLEMENTAL_ARTIFACTS_SCHEMA_VERSION",
    "discard_supplemental_artifact",
    "dump_compile_artifacts",
    "dump_hardware_config",
    "dump_performance_estimates",
    "dump_storage_plan",
    "dump_supplemental_artifacts",
    "serialize_hardware_config",
    "set_supplemental_artifact",
]


# -----------------------------------------------------------------------------
# Target-owned payloads
# -----------------------------------------------------------------------------


def set_supplemental_artifact(model, name: str, payload: Mapping) -> None:
    """Record one target-owned payload under ``name``.

    Every payload carries its own ``schema_version``, checked here rather than
    at the reader: a manifest holding several targets' payloads has no single
    version, and a reader that has to guess whether a field is missing or
    merely old is the thing this prevents.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("a supplemental artifact needs a name")
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError(
            f"supplemental artifact {name!r} needs a positive schema_version"
        )
    model.meta.setdefault(_SUPPLEMENTAL_ARTIFACTS_META, {})[name] = dict(payload)


def discard_supplemental_artifact(model, name: str) -> None:
    """Drop one payload, and the manifest itself once it is empty."""
    artifacts = model.meta.get(_SUPPLEMENTAL_ARTIFACTS_META)
    if artifacts is None:
        return
    artifacts.pop(name, None)
    if not artifacts:
        model.meta.pop(_SUPPLEMENTAL_ARTIFACTS_META, None)


# -----------------------------------------------------------------------------
# The machine
# -----------------------------------------------------------------------------


def _unroll_dimension_to_dict(dimension) -> dict:
    return {
        "name": dimension.name,
        "extent": dimension.size,
        "dtype_extents": [
            {"input_bits": bits, "extent": extent}
            for bits, extent in dimension.dtype_extents
        ],
    }


def _compute_unit_to_dict(unit) -> dict:
    return {
        "name": unit.name,
        "unroll_dimensions": [
            _unroll_dimension_to_dict(dimension)
            for dimension in unit.unroll_dimensions
        ],
        "accepted_input_bits": sorted(unit.accepted_operand_bits),
        "accepted_operations": sorted(unit.accepted_operations),
        "input_word_bits": unit.input_word_bits,
        "scalar_bits": unit.scalar_bits,
    }


def _memory_instance_to_dict(memory) -> dict:
    return {
        "name": memory.name,
        "bank_count": memory.bank_count,
        "depth_per_bank": memory.depth_per_bank,
        "bits_per_bank_entry": memory.bits_per_bank_entry,
        "accepted_accesses": [
            {"role": role, "compute_unit": unit}
            for role, unit in sorted(
                (access.role.value, access.compute_unit)
                for access in memory.accepted_accesses
            )
        ],
        "access_cost": memory.access_cost,
        "logical_capacity": memory.logical_capacity,
        "alignment_bytes": memory.alignment_bytes,
        "access_granularity_bits": memory.access_granularity_bits,
        "read_bandwidth": memory.read_bandwidth,
        "write_bandwidth": memory.write_bandwidth,
        "double_buffered": memory.double_buffered,
    }


def _memory_level_to_dict(level) -> dict:
    return {
        "name": level.name,
        "instances": [
            _memory_instance_to_dict(memory) for memory in level.instances
        ],
        "input_memory": level.input_memory,
        "output_memory": level.output_memory,
        "weight_memory": level.weight_memory,
    }


def _external_memory_to_dict(memory) -> dict:
    return {
        "size_gb": memory.size_gb,
        "bandwidth_gbs": memory.bandwidth_gbs,
        "access_latency_ns": memory.access_latency_ns,
    }


def serialize_hardware_config(hardware, target=None) -> dict:
    """Project the typed machine into the endpoint's own vocabulary.

    A ``None`` field is written out rather than omitted. The reader has to
    distinguish "this machine has no external memory" from "this compiler did
    not say", and dropping the key collapses the two.
    """
    return {
        "schema_version": HARDWARE_CONFIG_SCHEMA_VERSION,
        "config_type": type(hardware).__name__,
        "target": target,
        "compute_unit_bindings": dict(
            sorted(hardware.compute_unit_bindings)
        ),
        "compute_units": [
            _compute_unit_to_dict(unit) for unit in hardware.compute_units
        ],
        "dtype_modes": [
            {
                "operand_bits": mode.operand_bits,
                "banks_per_value": mode.banks_per_value,
                "output_channel_alignment": mode.channel_alignment,
            }
            for mode in hardware.dtype_modes
        ],
        "local_memory": (
            _memory_level_to_dict(hardware.local_memory)
            if hardware.local_memory is not None
            else None
        ),
        "global_memory": (
            _memory_level_to_dict(hardware.global_memory)
            if hardware.global_memory is not None
            else None
        ),
        "external_memory": (
            _external_memory_to_dict(hardware.external_memory)
            if hardware.external_memory is not None
            else None
        ),
        "output_collector": (
            {
                "name": hardware.output_collector.name,
                "depth_words": hardware.output_collector.depth_words,
                "lanes_per_word": hardware.output_collector.lanes_per_word,
                "accumulator_bits": (
                    hardware.output_collector.accumulator_bits
                ),
            }
            if hardware.output_collector is not None
            else None
        ),
        "hierarchy_bandwidths": [
            {
                "name": route.name,
                "source": route.source,
                "destination": route.destination,
                "bandwidth_bits_per_cycle": route.bandwidth_bits_per_cycle,
                "roles": sorted(role.value for role in route.roles),
                "shared_group": route.shared_group,
            }
            for route in hardware.hierarchy_bandwidths
        ],
        "mapping_num_banks": hardware.num_banks,
        "mapping_bank_width": hardware.bank_width,
        "technology_nm": hardware.technology_nm,
        "pad_depthwise_channels": hardware.pad_depthwise_channels,
        "partial_matrix_input_channels": sorted(
            hardware.partial_matrix_input_channels
        ),
    }


def dump_hardware_config(hardware, output_dir, target=None) -> None:
    """Write ``hardware_config.json``, or remove a stale one."""
    path = Path(output_dir) / "hardware_config.json"
    if hardware is None:
        path.unlink(missing_ok=True)
        return
    payload = serialize_hardware_config(hardware, target=target)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# -----------------------------------------------------------------------------
# What the scheduler predicted
# -----------------------------------------------------------------------------


def dump_performance_estimates(model, output_dir) -> None:
    """Write the per-operation cycles and accesses a scheduler reported.

    Only operations that were actually scheduled appear. An operation the
    host runs has no estimate, and inventing a zero for it would let a
    downstream partitioner treat host work as free accelerator work.
    """
    path = Path(output_dir) / "performance_estimates.json"
    layers = {}
    for node in model.graph.nodes:
        result = node.meta.get("local_schedule")
        if result is None:
            continue
        if result.runtime is None and not result.access_counts:
            continue
        record = {"backend": result.backend}
        if result.runtime is not None:
            record["execution_cycles"] = float(result.runtime)
        if result.access_counts:
            record["level_access_counts"] = [
                {
                    "input": int(level.input),
                    "output": int(level.output),
                    "weight": int(level.weight),
                }
                for level in result.access_counts
            ]
        layers[node.name] = record

    if not layers:
        path.unlink(missing_ok=True)
        return
    payload = {
        "schema_version": PERFORMANCE_ESTIMATES_SCHEMA_VERSION,
        "count_unit": "scalar_accesses",
        "level_order": "inner_to_outer",
        "runtime_unit": "cycles",
        "layers": dict(sorted(layers.items())),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


# -----------------------------------------------------------------------------
# Where everything lives
# -----------------------------------------------------------------------------


def dump_storage_plan(model, output_dir) -> None:
    """Write the logical and physical tensor identities, for inspection.

    This one is a diagnostic. Nothing downstream reads it, and nothing should
    start to: the authoritative placement is the address in ``model.txt``, and
    a second copy that could disagree with it is worth less than no copy.
    """
    path = Path(output_dir) / "storage_plan.json"
    plan = model.meta.get("storage_plan")
    if plan is None and os.environ.get(STORAGE_PLAN_ENV):
        # Nothing in the tree attaches a plan yet, so without this the artifact
        # never appears. It is *derived on request* rather than always: this is
        # the one artifact whose producer would run at dump time, and a
        # diagnostic nothing reads should not cost every compile in the
        # regression a graph walk and a large JSON file. A target that attaches
        # its own plan gets it written either way, which is the path Alpha's
        # terminal serialization will take.
        try:
            from voyager_compiler.codegen.storage.analysis import StorageGraph

            plan = StorageGraph(model).finalize()
        except Exception:
            logger.warning(
                "storage plan not derivable for this graph:\n%s",
                traceback.format_exc(),
            )
            path.unlink(missing_ok=True)
            return
    if plan is None:
        # Nothing attached one and nothing asked for one.
        path.unlink(missing_ok=True)
        return
    payload = {"schema_version": STORAGE_PLAN_SCHEMA_VERSION, **plan.to_dict()}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def dump_supplemental_artifacts(model, output_dir) -> None:
    """Write the target-owned payloads under one versioned manifest."""
    path = Path(output_dir) / "supplemental_artifacts.json"
    artifacts = model.meta.get(_SUPPLEMENTAL_ARTIFACTS_META)
    if not artifacts:
        path.unlink(missing_ok=True)
        return
    payload = {
        "schema_version": SUPPLEMENTAL_ARTIFACTS_SCHEMA_VERSION,
        "artifacts": dict(sorted(artifacts.items())),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_compile_artifacts(model, hardware, output_dir, target=None) -> None:
    """Write every auxiliary artifact one compile produces."""
    output_dir = Path(output_dir)
    dump_hardware_config(hardware, output_dir, target=target)
    dump_performance_estimates(model, output_dir)
    dump_storage_plan(model, output_dir)
    dump_supplemental_artifacts(model, output_dir)
