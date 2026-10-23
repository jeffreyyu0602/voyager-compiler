#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Convert a Voyager text dump into Magnet Alpha C-model layer collateral.

The converter is intentionally outside ``voyager_compiler``. It consumes the
stable dump schema, serialized hardware configuration, graph connections,
and tensor files without importing or changing compiler implementation code.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, NoReturn, Sequence

import numpy as np
import utils as cmod_utils
from google.protobuf import text_format

HARDWARE_CONFIG_SCHEMA_VERSION = 1
CONVERSION_REPORT_SCHEMA_VERSION = 3
CMOD_EXECUTION_BUILDERS = {
    "gemm_convolution": "build_gemm",
    "nmp_depthwise_convolution": "build_depthwise",
    "nmp_channel_shuffle": "build_shuffle",
    "nmp_copy": "build_copy",
    "nmp_max_pool": "build_pool",
    "nmp_nearest_upsample": "build_upsample",
}
_CMOD_KIND_BY_SINGLE_TARGET = {
    "channel_shuffle": "nmp_channel_shuffle",
    cmod_utils.CMOD_COPY_TARGET: "nmp_copy",
    "max_pool2d": "nmp_max_pool",
    "upsample_nearest2d": "nmp_nearest_upsample",
}
_LEGACY_COPY_FIELDS = frozenset(
    {
        "block_channels",
        "copy_block_payload",
        "copy_destination_entry",
        "copy_destination_payload",
        "copy_source_entry",
        "copy_source_output",
        "copy_source_payload",
    }
)
_FUSION_CANDIDATE = "fusion_candidate"
_HOST_LAYOUT_CANDIDATE = "host_layout_candidate"
_SERIALIZED_UNSUPPORTED_OPERATION = "unsupported"


class SkipLayer(RuntimeError):
    """Reject one layer after recording its conversion error."""


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    layer: str
    message: str


@dataclass
class LayerResult:
    name: str
    kind: str
    schedule: dict[str, Any]
    inputs: list[str]
    files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ExecutionRecord:
    """One CMOD execution projection derived from canonical dump facts."""

    operation: Any
    name: str
    target: str
    disposition: str
    kind: str | None
    skip_reason: str | None
    fusion_owners: tuple[str, ...]


def encode_cmod_vector_scale(
    values: np.ndarray,
    *,
    kind: str,
    output_lanes: int | None = None,
) -> np.ndarray:
    """Encode one raw rank-4 scale tensor in Alpha CMOD lane order."""
    values = np.asarray(values)
    if values.ndim != 4:
        raise ValueError(f"{kind} vector scale must be rank 4")
    if kind in {"input", "output"}:
        return values.transpose(0, 3, 2, 1).copy()
    if kind != "weight":
        raise ValueError(f"unknown CMOD vector-scale kind {kind!r}")
    if not isinstance(output_lanes, int) or output_lanes <= 0:
        raise ValueError("weight vector scale needs positive output lanes")
    k, c_word, r, s = values.shape
    if k % output_lanes:
        raise ValueError(
            f"weight vector scale K={k} is not aligned to "
            f"{output_lanes} lanes"
        )
    return (
        values.reshape(-1, output_lanes, c_word, r, s)
        .transpose(0, 1, 4, 3, 2)
        .copy()
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_proto_module(repo_root: Path) -> Any:
    path = repo_root / "src/voyager_compiler/codegen/param_pb2.py"
    spec = importlib.util.spec_from_file_location(
        "voyager_dump_param_pb2", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Voyager dump schema: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact_integer(value: Any) -> int:
    """Return an integer scalar without truncating or accepting booleans."""
    if isinstance(value, bool):
        raise ValueError("booleans are not integer operands")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(value)
    raise ValueError(f"{value!r} is not an exact integer")


def _config_integer(
    value: Any,
    field: str,
    *,
    minimum: int = 1,
) -> int:
    """Read an exact bounded integer from serialized hardware config."""
    try:
        result = _exact_integer(value)
    except ValueError as error:
        raise ValueError(
            f"hardware_config.json {field} must be an exact integer"
        ) from error
    if result < minimum:
        raise ValueError(
            f"hardware_config.json {field} must be at least {minimum}"
        )
    return result


def _pair(value: Any, default: int = 1) -> tuple[int, int]:
    if value is None:
        return default, default
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) == 1:
            scalar = _exact_integer(value[0])
            return scalar, scalar
        if len(value) == 2:
            return _exact_integer(value[0]), _exact_integer(value[1])
        raise ValueError(f"Expected one or two integer values; found {value!r}")
    scalar = _exact_integer(value)
    return scalar, scalar


def _numel(shape: Sequence[int]) -> int:
    return math.prod(int(item) for item in shape)


class _Converter:
    """Fail-closed Voyager-to-C-model dump translator."""

    def __init__(
        self,
        model_dir: Path,
        output_dir: Path,
        network: str,
        write_tensors: bool,
    ) -> None:
        self.repo_root = _repo_root()
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.network = network
        self.write_tensors = write_tensors
        self.tensor_dir = model_dir / "tensor_files"
        self.issues: list[Issue] = []
        self.results: list[LayerResult] = []
        self.array_cache: dict[str, np.ndarray] = {}

        proto = _load_proto_module(self.repo_root)
        self.model = proto.Model()
        text_format.Parse((model_dir / "model.txt").read_text(), self.model)
        hardware_path = model_dir / "hardware_config.json"
        if not hardware_path.exists():
            raise ValueError(
                "hardware_config.json is required for Magnet Alpha conversion"
            )
        data = json.loads(hardware_path.read_text())
        if data.get("schema_version") != HARDWARE_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "hardware_config.json has an unsupported schema version"
            )
        if data.get("target") != "magnet-alpha":
            raise ValueError(
                "hardware_config.json is not a Magnet Alpha configuration"
            )
        self.hardware_config = data

        self.operations = list(self.model.ops)

    def error(self, code: str, layer: str, message: str) -> None:
        """Record a conversion error without rejecting the current layer."""
        self.issues.append(Issue("error", code, layer, message))

    def fail(self, code: str, layer: str, message: str) -> NoReturn:
        """Record the reason that the current layer cannot be converted."""
        self.error(code, layer, message)
        raise SkipLayer(layer)

    def require_integer(
        self,
        value: Any,
        layer: str,
        field: str,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
        code: str = "invalid-serialized-integer",
    ) -> int:
        """Require an exact integer scalar before applying any range check."""
        try:
            result = _exact_integer(value)
        except ValueError:
            result = None
        if (
            result is None
            or (minimum is not None and result < minimum)
            or (maximum is not None and result > maximum)
        ):
            bounds = ""
            if minimum is not None and maximum is not None:
                bounds = f" in [{minimum}, {maximum}]"
            elif minimum is not None:
                bounds = f" greater than or equal to {minimum}"
            elif maximum is not None:
                bounds = f" less than or equal to {maximum}"
            self.fail(
                code,
                layer,
                f"{field}={value!r}; expected exact integer{bounds}",
            )
        return result

    def require_pair(
        self,
        value: Any,
        layer: str,
        field: str,
        *,
        default: int | None = None,
        minimum: int | None = None,
    ) -> tuple[int, int]:
        """Require a one- or two-element exact integer geometry field."""
        if value is None and default is None:
            self.fail(
                "invalid-serialized-integer-pair",
                layer,
                f"{field}: missing",
            )
        try:
            result = _pair(value, 1 if default is None else default)
        except ValueError:
            result = None
        if result is None or (
            minimum is not None and any(item < minimum for item in result)
        ):
            qualifier = (
                f" with values greater than or equal to {minimum}"
                if minimum is not None
                else ""
            )
            self.fail(
                "invalid-serialized-integer-pair",
                layer,
                f"{field}={value!r}; expected one or two exact integers"
                f"{qualifier}",
            )
        return result

    def require_boolean(
        self,
        value: Any,
        layer: str,
        field: str,
        *,
        code: str = "invalid-serialized-boolean",
    ) -> bool:
        """Require a real boolean rather than relying on truthiness."""
        if not isinstance(value, bool):
            self.fail(
                code,
                layer,
                f"{field}={value!r}; expected boolean",
            )
        return value

    def load_tensor(
        self,
        tensor: Any,
        layer: str,
    ) -> np.ndarray:
        if tensor.node in self.array_cache:
            return self.array_cache[tensor.node].copy()
        path = self.tensor_dir / f"{tensor.node}.bin"
        if not path.exists():
            self.fail(
                "missing-tensor-file",
                layer,
                f"{tensor.node}: payload missing at {path}",
            )
        values = np.fromfile(path, dtype=np.float32)
        expected = _numel(tensor.shape)
        if values.size != expected:
            self.fail(
                "tensor-size-mismatch",
                layer,
                f"{tensor.node}: values={values.size}; expected={expected} "
                f"for shape={list(tensor.shape)}",
            )
        values = values.reshape(tuple(tensor.shape))
        self.array_cache[tensor.node] = values
        return values.copy()

    def load_exact_integer_tensor(
        self,
        tensor: Any,
        layer: str,
        role: str,
        dtype: str,
        low: int,
        high: int,
    ) -> np.ndarray:
        """Load a compiler-finalized integer tensor without converting it."""
        if tensor.dtype != dtype:
            self.fail(
                "unfinalized-target-payload",
                layer,
                f"{role} {tensor.node}: dtype={tensor.dtype}; "
                f"expected={dtype}",
            )
        return self.load_exact_integer_values(
            tensor,
            layer,
            role,
            low,
            high,
        )

    def load_exact_integer_values(
        self,
        tensor: Any,
        layer: str,
        role: str,
        low: int,
        high: int,
    ) -> np.ndarray:
        """Load integral target operands whose dump storage dtype is float."""
        values = self.load_tensor(tensor, layer)
        rounded = np.rint(values)
        valid = (
            values.size > 0
            and np.all(np.isfinite(values))
            and np.array_equal(values, rounded)
            and np.all(rounded >= low)
            and np.all(rounded <= high)
        )
        if not valid:
            self.fail(
                "invalid-target-payload",
                layer,
                f"{role} {tensor.node}: expected exact integers "
                f"in [{low}, {high}]",
            )
        return rounded.astype(np.int64, copy=False)

    def load_exact_payload(
        self,
        tensor: Any,
        layer: str,
        role: str,
        precision: int,
        *,
        weight: bool = False,
    ) -> np.ndarray:
        """Load one compiler-finalized signed activation/weight payload."""
        if precision == 4:
            low = -7 if weight else -8
            high = 7
        elif precision == 8:
            low, high = -128, 127
        else:
            self.fail(
                "unsupported-target-precision",
                layer,
                f"{role}: precision={precision}; unsupported",
            )
        return self.load_exact_integer_tensor(
            tensor,
            layer,
            role,
            f"int{precision}",
            low,
            high,
        )

    def require_shape(
        self,
        values: np.ndarray,
        expected: tuple[int, ...],
        layer: str,
        role: str,
    ) -> None:
        """Reject target operands whose structural extent is ambiguous."""
        if values.shape == expected:
            return
        self.fail(
            "target-operand-shape-mismatch",
            layer,
            f"{role}: shape={values.shape}; expected={expected}",
        )

    def require_address(
        self,
        tensor: Any,
        layer: str,
        description: str,
        operand_role: str,
        compute_role: str,
    ) -> int:
        address = cmod_utils.tensor_address(tensor)
        if address is None:
            self.fail(
                "missing-scratchpad-address",
                layer,
                f"{description} {tensor.node}: scratchpad address missing",
            )
        self.access_memory_instance(
            tensor,
            operand_role,
            compute_role,
            layer,
        )
        return address

    def write_values(
        self,
        result: LayerResult,
        filename: str,
        values: np.ndarray,
        low: int,
        high: int,
    ) -> None:
        values = np.asarray(values)
        rounded = np.rint(values)
        valid = (
            np.all(np.isfinite(values))
            and np.array_equal(values, rounded)
            and np.all(rounded >= low)
            and np.all(rounded <= high)
        )
        if not valid:
            self.fail(
                "invalid-cmodel-payload",
                result.name,
                f"{filename}: expected exact integers in [{low}, {high}]",
            )
        integers = rounded.astype(np.int64, copy=False)
        text = ",".join(str(int(value)) for value in integers.ravel())
        round_trip = np.fromstring(text, sep=",", dtype=np.int64)
        if not np.array_equal(round_trip, integers.ravel()):
            self.fail(
                "payload-round-trip-failed",
                result.name,
                f"{filename}: CMOD text round trip changed values",
            )
        if not self.write_tensors:
            return
        path = self.output_dir / result.name / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        result.files.append(filename)

    def write_payload(
        self,
        result: LayerResult,
        filename: str,
        values: np.ndarray,
        precision: int,
        weight: bool = False,
    ) -> None:
        if precision == 4:
            low = -7 if weight else -8
            self.write_values(result, filename, values, low, 7)
        elif precision == 8:
            self.write_values(result, filename, values, -128, 127)
        else:
            self.fail(
                "unsupported-target-precision",
                result.name,
                f"{filename}: precision={precision}; unsupported",
            )

    def load_serialized_artifact(
        self,
        operation: Any,
        field: str,
        layer: str,
        role: str,
        expected_shape: tuple[int, ...],
        *,
        dtype: str = "uint8",
        low: int = 0,
        high: int = 255,
    ) -> np.ndarray:
        """Load one canonical nonresident compiler artifact verbatim."""
        values = self.load_nonresident_artifact(
            operation,
            field,
            layer,
            role,
            dtype=dtype,
            low=low,
            high=high,
        )
        self.require_shape(values, expected_shape, layer, role)
        return values

    def load_nonresident_artifact(
        self,
        operation: Any,
        field: str,
        layer: str,
        role: str,
        *,
        dtype: str = "uint8",
        low: int = 0,
        high: int = 255,
    ) -> np.ndarray:
        """Load a canonical artifact that has no runtime residence."""
        tensor = self.serialized_tensor_role(
            operation,
            field,
            required=True,
            output_only=False,
        )
        if tensor.HasField("memory") or tensor.HasField("scratchpad"):
            self.fail(
                "resident-serialized-artifact",
                layer,
                f"{role} {tensor.node}: runtime memory instance not allowed",
            )
        values = self.load_exact_integer_tensor(
            tensor,
            layer,
            role,
            dtype,
            low,
            high,
        )
        return values

    def load_cmod_vector_scale(
        self,
        operation: Any,
        field: str,
        layer: str,
        role: str,
        raw_expected: np.ndarray,
        *,
        kind: str,
        output_lanes: int | None = None,
    ) -> np.ndarray:
        """Encode a raw scale artifact, accepting old preencoded dumps."""
        raw_expected = np.asarray(raw_expected)
        try:
            encoded_expected = encode_cmod_vector_scale(
                raw_expected,
                kind=kind,
                output_lanes=output_lanes,
            )
        except ValueError as error:
            self.fail(
                "invalid-vector-scale-geometry",
                layer,
                f"{role}: {error}",
            )
        values = self.load_nonresident_artifact(
            operation,
            field,
            layer,
            role,
        )
        if values.shape == raw_expected.shape and np.array_equal(
            values, raw_expected
        ):
            return encoded_expected
        if values.shape == encoded_expected.shape and np.array_equal(
            values, encoded_expected
        ):
            return values
        self.fail(
            "invalid-vector-scale-artifact",
            layer,
            f"{role}: artifact is neither canonical raw data nor a "
            "compatible legacy CMOD encoding",
        )

    def operation_inputs(self, operation: Any) -> list[str]:
        internal = {op.name for op in cmod_utils.inner_operations(operation)}
        returned = {
            tensor.node for tensor in cmod_utils.output_tensors(operation)
        }
        seen: set[str] = set()
        inputs: list[str] = []
        for tensor in cmod_utils.input_tensors(operation):
            if (
                tensor.node in internal
                or tensor.node in returned
                or tensor.node in seen
            ):
                continue
            seen.add(tensor.node)
            inputs.append(tensor.node)
        return inputs

    def require_precision(
        self,
        tensor: Any,
        layer: str,
        role: str,
    ) -> int:
        precision = cmod_utils.dtype_precision(tensor.dtype)
        if precision in (4, 8):
            return precision
        self.fail(
            "unsupported-target-precision",
            layer,
            f"{role} {tensor.node}: dtype={tensor.dtype}; unsupported",
        )

    def serialized_tensor_role(
        self,
        operation: Any,
        field: str,
        *,
        required: bool,
        output_only: bool,
    ) -> Any | None:
        layer = cmod_utils.operation_name(operation)
        values = [
            cmod_utils.operation_kwarg(op, field)
            for op in cmod_utils.inner_operations(operation)
            if cmod_utils.operation_kwarg(op, field) is not None
        ]
        tensors = [
            value for value in values if hasattr(value, "node") and value.node
        ]
        nodes = {tensor.node for tensor in tensors}
        if not tensors:
            if required:
                self.fail(
                    "missing-serialized-tensor-role",
                    layer,
                    f"{field}: role missing",
                )
            return None
        if len(nodes) != 1 or len(tensors) != len(values):
            self.fail(
                "ambiguous-serialized-tensor-role",
                layer,
                f"{field}: expected one tensor",
            )
        tensor = tensors[0]
        if output_only and tensor.node not in {
            output.node for output in cmod_utils.output_tensors(operation)
        }:
            self.fail(
                "invalid-serialized-output-role",
                layer,
                f"{field} {tensor.node}: not an executable output",
            )
        return tensor

    def output_payload(self, operation: Any) -> Any:
        return self.serialized_tensor_role(
            operation,
            "output_payload",
            required=True,
            output_only=True,
        )

    def output_vector_scale(self, operation: Any) -> Any | None:
        return self.serialized_tensor_role(
            operation,
            "output_vector_scale",
            required=False,
            output_only=True,
        )

    def inner_by_target(
        self,
        operation: Any,
        role: str,
        *,
        targets: frozenset[str],
        required: bool,
    ) -> Any | None:
        """Resolve one unique inner operation from canonical targets."""
        layer = cmod_utils.operation_name(operation)
        matches = [
            op
            for op in cmod_utils.inner_operations(operation)
            if op.target in targets
        ]
        if not matches and not required:
            return None
        if len(matches) != 1:
            self.fail(
                "invalid-cmod-inner-target",
                layer,
                f"{role}: expected one target in {sorted(targets)}; "
                f"found {[op.target for op in matches]}",
            )
        return matches[0]

    def output_quantizer(self, operation: Any) -> tuple[Any, Any] | None:
        """Return only a quantizer contained by this executable record."""
        quantizer = self.inner_by_target(
            operation,
            "PVS quantizer",
            targets=frozenset({"pvs_quantize"}),
            required=False,
        )
        return None if quantizer is None else (operation, quantizer)

    def explicit_ppu_parameters(
        self,
        operation: Any,
        layer: str,
    ) -> dict[str, Any] | None:
        """Read already-lowered Alpha PPU operands without deriving policy."""
        ppu = self.inner_by_target(
            operation,
            "PPU epilogue",
            targets=frozenset({"ppu_fixed_relu"}),
            required=False,
        )
        if ppu is None:
            return None
        if cmod_utils.operation_kwarg(ppu, "integer_accumulator") is not True:
            self.fail(
                "noninteger-pe-output",
                layer,
                "PPU output is not an exact integer",
            )
        required = (
            "primary_multiplier",
            "primary_shift",
            "secondary_multiplier",
            "secondary_shift",
            "output_precision",
            "relu",
        )
        if any(
            cmod_utils.operation_kwarg(ppu, field) is None for field in required
        ):
            self.fail(
                "missing-pe-fixed-point-operands",
                layer,
                "missing finalized PPU multiplier/shift/output/activation",
            )
        multiplier_tensor = cmod_utils.operation_kwarg(
            ppu,
            "primary_multiplier",
        )
        values = self.load_exact_integer_values(
            multiplier_tensor,
            layer,
            "Alpha PE primary multiplier",
            0,
            65535,
        ).reshape(-1)
        primary_shift = self.require_integer(
            cmod_utils.operation_kwarg(ppu, "primary_shift"),
            layer,
            "primary_shift",
            minimum=0,
            maximum=255,
            code="invalid-pe-fixed-point-operands",
        )
        secondary_multiplier = self.require_integer(
            cmod_utils.operation_kwarg(ppu, "secondary_multiplier"),
            layer,
            "secondary_multiplier",
            minimum=0,
            maximum=65535,
            code="invalid-pe-fixed-point-operands",
        )
        secondary_shift = self.require_integer(
            cmod_utils.operation_kwarg(ppu, "secondary_shift"),
            layer,
            "secondary_shift",
            minimum=0,
            maximum=255,
            code="invalid-pe-fixed-point-operands",
        )
        output_precision = self.require_integer(
            cmod_utils.operation_kwarg(ppu, "output_precision"),
            layer,
            "output_precision",
            code="invalid-pe-fixed-point-operands",
        )
        if output_precision not in (4, 8):
            self.fail(
                "invalid-pe-fixed-point-operands",
                layer,
                "PPU operands exceed hardware ranges",
            )
        return {
            "scale_int": values,
            "binary_point": primary_shift,
            "nonlinear_multiplier": secondary_multiplier,
            "nonlinear_shift": secondary_shift,
            "bias": cmod_utils.operation_kwarg(ppu, "bias_payload"),
            "relu": self.require_boolean(
                cmod_utils.operation_kwarg(ppu, "relu"),
                layer,
                "relu",
                code="invalid-pe-fixed-point-operands",
            ),
            "output_precision": output_precision,
        }

    def explicit_eadd_operand(
        self,
        operation: Any,
        layer: str,
        output: Any,
        ppu_parameters: dict[str, Any],
    ) -> Any | None:
        """Validate and return one compiler-finalized Alpha EADD operand."""
        eadd = self.inner_by_target(
            operation,
            "EADD epilogue",
            targets=frozenset({"saturating_add"}),
            required=False,
        )
        if eadd is None:
            return None
        input1 = cmod_utils.operation_kwarg(eadd, "input1")
        input2 = cmod_utils.operation_kwarg(eadd, "input2")
        if input1 is None or input2 is None:
            missing = [
                role
                for role, tensor in (("input1", input1), ("input2", input2))
                if tensor is None
            ]
            self.fail(
                "missing-eadd-operand",
                layer,
                f"missing={','.join(missing)}",
            )
        ppu = self.inner_by_target(
            operation,
            "PPU epilogue",
            targets=frozenset({"ppu_fixed_relu"}),
            required=True,
        )
        internal_names = {
            inner.name for inner in cmod_utils.inner_operations(operation)
        }
        if input1.node != ppu.name or input2.node in internal_names:
            self.fail(
                "invalid-eadd-dataflow",
                layer,
                "EADD input1 must be the contained PPU result and input2 "
                "must be an external operand.",
            )
        if ppu_parameters["output_precision"] != 8:
            self.fail(
                "unsupported-eadd-output-precision",
                layer,
                "Alpha PE EADD requires an O8 PPU result.",
            )
        if ppu_parameters["relu"]:
            self.fail(
                "unsupported-eadd-ppu-relu",
                layer,
                "Alpha PE EADD requires PPU ReLU to be disabled.",
            )

        tensors = {
            "PPU result": input1,
            "EADD operand2": input2,
            "EADD output": output,
        }
        shapes = {}
        for role, tensor in tensors.items():
            if tensor.dtype != "int8":
                self.fail(
                    "unsupported-eadd-payload-domain",
                    layer,
                    f"{role} {tensor.node}: dtype={tensor.dtype}; "
                    "expected signed int8",
                )
            try:
                shape = tuple(_exact_integer(value) for value in tensor.shape)
            except ValueError:
                shape = ()
            if len(shape) != 4 or any(value <= 0 for value in shape):
                self.fail(
                    "invalid-eadd-tensor-extents",
                    layer,
                    f"{role}: shape={tuple(tensor.shape)}; "
                    "expected positive rank-4 NCHW",
                )
            shapes[role] = shape
        if len(set(shapes.values())) != 1:
            self.fail(
                "invalid-eadd-tensor-extents",
                layer,
                f"Alpha PE EADD shapes must match exactly; found={shapes}",
            )

        input2_address = self.require_address(
            input2,
            layer,
            "EADD operand2",
            "activation",
            "matrix",
        )
        self.access_memory_instance(
            input2,
            "activation",
            "nmp",
            layer,
        )
        output_address = self.require_address(
            output,
            layer,
            "EADD output",
            "output",
            "matrix",
        )
        if (
            input2_address != output_address
            or input2.scratchpad != output.scratchpad
        ):
            self.fail(
                "invalid-eadd-output-alias",
                layer,
                "EADD output must use exactly the operand2 scratchpad "
                "instance, partition, and encoded address.",
            )
        return input2

    def explicit_nmp_parameters(
        self,
        operation: Any,
        layer: str,
    ) -> dict[str, Any]:
        """Read compiler-lowered DWC operands without deriving target math."""
        epilogue = self.inner_by_target(
            operation,
            "NMP epilogue",
            targets=frozenset({"nmp_fixed_requantize"}),
            required=False,
        )
        required = (
            "scale_factor",
            "binary_point",
            "bias_factor",
            "output_precision",
            "relu",
        )
        if epilogue is None or any(
            cmod_utils.operation_kwarg(epilogue, field) is None
            for field in required
        ):
            self.fail(
                "missing-depthwise-fixed-point-operands",
                layer,
                "missing finalized DWC scale/shift/bias/output operands",
            )
        result = {
            "scale_factor": self.require_integer(
                cmod_utils.operation_kwarg(epilogue, "scale_factor"),
                layer,
                "scale_factor",
                minimum=0,
                maximum=65535,
                code="invalid-depthwise-fixed-point-operands",
            ),
            "binary_point": self.require_integer(
                cmod_utils.operation_kwarg(epilogue, "binary_point"),
                layer,
                "binary_point",
                minimum=0,
                maximum=255,
                code="invalid-depthwise-fixed-point-operands",
            ),
            "bias_factor": self.require_integer(
                cmod_utils.operation_kwarg(epilogue, "bias_factor"),
                layer,
                "bias_factor",
                minimum=-128,
                maximum=127,
                code="invalid-depthwise-fixed-point-operands",
            ),
            "output_precision": self.require_integer(
                cmod_utils.operation_kwarg(epilogue, "output_precision"),
                layer,
                "output_precision",
                code="invalid-depthwise-fixed-point-operands",
            ),
            "relu": self.require_boolean(
                cmod_utils.operation_kwarg(epilogue, "relu"),
                layer,
                "relu",
                code="invalid-depthwise-fixed-point-operands",
            ),
        }
        if result["output_precision"] not in (4, 8):
            self.fail(
                "invalid-depthwise-fixed-point-operands",
                layer,
                "DWC fixed-point operands exceed hardware ranges",
            )
        return result

    def concat_fields(
        self,
        operations: Iterable[Any],
        *,
        copy: bool = False,
    ) -> dict[str, int]:
        operations = list(operations)
        layer = (
            cmod_utils.operation_name(operations[0])
            if operations
            else "unknown"
        )
        fields = (
            ("is_to_concat", "concat_total_size_0")
            if copy
            else cmod_utils.CONCAT_FIELDS
        )
        result: dict[str, int] = {}
        for name in fields:
            raw_values = [
                value
                for operation in operations
                for op in cmod_utils.inner_operations(operation)
                if (value := cmod_utils.operation_kwarg(op, name)) is not None
            ]
            if name == "is_to_concat":
                if any(not isinstance(value, bool) for value in raw_values):
                    self.fail(
                        "invalid-serialized-storage-field",
                        layer,
                        "is_to_concat: expected boolean",
                    )
                values = [int(value) for value in raw_values]
            else:
                values = [
                    self.require_integer(
                        value,
                        layer,
                        name,
                        minimum=0,
                        code="invalid-serialized-storage-field",
                    )
                    for value in raw_values
                ]
            if len(set(values)) > 1:
                self.fail(
                    "conflicting-serialized-storage-field",
                    layer,
                    f"{name}: conflicting values",
                )
            if values:
                result[name] = values[0]
        required = fields[1:]
        if result.get("is_to_concat") and any(
            name not in result for name in required
        ):
            self.fail(
                "incomplete-serialized-concat-view",
                layer,
                "concat view: fields incomplete",
            )
        return result

    def serialized_value(
        self,
        operation: Any,
        field: str,
        *,
        required: bool = True,
    ) -> Any:
        layer = cmod_utils.operation_name(operation)
        values = [
            value
            for op in cmod_utils.inner_operations(operation)
            if (value := cmod_utils.operation_kwarg(op, field)) is not None
        ]
        if not values:
            if required:
                self.fail(
                    "missing-serialized-storage-field",
                    layer,
                    f"{field}: missing",
                )
            return None
        if any(value != values[0] for value in values[1:]):
            self.fail(
                "conflicting-serialized-storage-field",
                layer,
                f"{field}: conflicting values",
            )
        return values[0]

    def nmp_storage_fields(
        self,
        operation: Any,
    ) -> dict[str, int | float]:
        layer = cmod_utils.operation_name(operation)
        result: dict[str, int | float] = {}
        for field in ("nmp_c2", "nmp_k2"):
            value = self.serialized_value(
                operation,
                field,
                required=False,
            )
            if value is None:
                continue
            value = self.require_integer(
                value,
                layer,
                field,
                minimum=0,
                code="invalid-serialized-storage-field",
            )
            if value:
                result[field] = value

        split_value = self.serialized_value(
            operation,
            "is_from_split",
            required=False,
        )
        if split_value is None:
            return result
        is_from_split = self.require_boolean(
            split_value,
            layer,
            "is_from_split",
            code="invalid-serialized-storage-field",
        )
        if not is_from_split:
            return result
        result["is_from_split"] = 1

        for field in (
            "split_channel_idx",
            "split_cur_size",
            "split_idx",
            "split_total_size",
        ):
            result[field] = self.require_integer(
                self.serialized_value(operation, field),
                layer,
                field,
                minimum=0,
                code="invalid-serialized-storage-field",
            )
        split_scale = self.serialized_value(operation, "split_scale")
        if (
            isinstance(split_scale, bool)
            or not isinstance(split_scale, (int, float))
            or not math.isfinite(float(split_scale))
            or float(split_scale) <= 0.0
        ):
            self.fail(
                "invalid-serialized-storage-field",
                layer,
                "split_scale: expected finite positive value",
            )
        result["split_scale"] = split_scale
        return result

    def nmp_group_fields(
        self,
        operation: Any,
    ) -> dict[str, int]:
        layer = cmod_utils.operation_name(operation)
        result = {}
        for serialized, schedule in (
            ("nmp_groups", "groups"),
            ("nmp_channels_per_group", "channels_per_group"),
        ):
            value = self.serialized_value(
                operation,
                serialized,
                required=False,
            )
            if value is None:
                continue
            result[schedule] = self.require_integer(
                value,
                layer,
                serialized,
                minimum=1,
                code="invalid-serialized-storage-field",
            )
        return result

    def padding_fields(
        self,
        operation: Any,
        *,
        spatial: bool,
    ) -> dict[str, int]:
        layer = cmod_utils.operation_name(operation)
        names = ["padding_c", "padding_k"]
        if spatial:
            names.extend(
                (
                    "pad_h_up",
                    "pad_h_down",
                    "pad_w_left",
                    "pad_w_right",
                )
            )
        result = {}
        for field in names:
            value = self.serialized_value(operation, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                self.fail(
                    "invalid-serialized-padding",
                    layer,
                    f"{field}: expected non-negative integer",
                )
            result[field] = value
        return result

    def spatial_input_geometry_fields(
        self,
        layer: str,
        input_shape: Sequence[int],
        kernel: tuple[int, int],
        output: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        padding: dict[str, int],
        *,
        failure_code: str = "invalid-gemm-spatial-coverage",
    ) -> dict[str, int]:
        """Validate convolution coverage without changing resident storage."""
        input_height, input_width = int(input_shape[2]), int(input_shape[3])
        output_height, output_width = output
        effective_height = dilation[0] * (kernel[0] - 1) + 1
        effective_width = dilation[1] * (kernel[1] - 1) + 1
        padded_height = (
            input_height + padding["pad_h_up"] + padding["pad_h_down"]
        )
        padded_width = (
            input_width + padding["pad_w_left"] + padding["pad_w_right"]
        )
        required_height = (output_height - 1) * stride[0] + effective_height
        required_width = (output_width - 1) * stride[1] + effective_width
        valid_height = (
            input_height > 0
            and required_height <= padded_height < required_height + stride[0]
        )
        valid_width = (
            input_width > 0
            and required_width <= padded_width < required_width + stride[1]
        )
        if not valid_height or not valid_width:
            self.fail(
                failure_code,
                layer,
                "serialized input storage "
                f"{input_height}x{input_width} with padded extent "
                f"{padded_height}x{padded_width} cannot produce output "
                f"{output_height}x{output_width} for kernel "
                f"{kernel[0]}x{kernel[1]}, stride={stride}, and "
                f"dilation={dilation}.",
            )
        return {
            "input_height": input_height,
            "input_width": input_width,
        }

    def base_schedule(self, name: str) -> dict[str, Any]:
        return {
            "network": self.network,
            "layer": name,
            "test_id": 1,
        }

    def global_memory(self, layer: str) -> dict[str, Any]:
        """Return the serialized global-memory description."""
        memory = self.hardware_config.get("global_memory")
        if not isinstance(memory, dict):
            self.fail(
                "invalid-hardware-memory-config",
                layer,
                "global_memory: missing",
            )
        instances = memory.get("instances")
        if not isinstance(instances, list) or not instances:
            self.fail(
                "invalid-hardware-memory-config",
                layer,
                "global_memory.instances: empty",
            )
        names = [
            instance.get("name")
            for instance in instances
            if isinstance(instance, dict)
        ]
        if (
            len(names) != len(instances)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            self.fail(
                "invalid-hardware-memory-config",
                layer,
                "global_memory instance names: expected unique, non-empty "
                "values",
            )
        return memory

    def validate_memory_instance_access(
        self,
        selected_name: str,
        operand_role: str,
        compute_role: str,
        layer: str,
        description: str,
    ) -> str:
        """Validate one compiler-selected instance without choosing storage."""
        memory = self.global_memory(layer)
        unit_name = self.compute_unit(compute_role)["name"]
        selected = next(
            (
                instance
                for instance in memory["instances"]
                if instance["name"] == selected_name
            ),
            None,
        )
        if selected is None:
            self.fail(
                "invalid-scratchpad-instance",
                layer,
                f"{description}: instance={selected_name!r}; "
                "absent from global_memory",
            )
        accesses = selected.get("accepted_accesses")
        allowed = isinstance(accesses, list) and any(
            isinstance(access, dict)
            and access.get("role") == operand_role
            and "compute_unit" in access
            and access["compute_unit"] in (None, unit_name)
            for access in accesses
        )
        if not allowed:
            self.fail(
                "invalid-hardware-memory-access",
                layer,
                f"{description}: instance={selected_name!r}; "
                f"{compute_role}/{operand_role} access denied",
            )
        return selected_name

    def access_memory_instance(
        self,
        tensor: Any,
        operand_role: str,
        compute_role: str,
        layer: str,
    ) -> str:
        """Validate a serialized tensor's compiler-selected memory instance."""
        return self.validate_memory_instance_access(
            tensor.scratchpad.instance,
            operand_role,
            compute_role,
            layer,
            f"{operand_role} tensor {tensor.node}",
        )

    def compute_unit(self, role: str) -> dict[str, Any]:
        """Resolve one compute unit through its serialized role binding."""
        bindings = self.hardware_config.get("compute_unit_bindings")
        if not isinstance(bindings, dict):
            raise ValueError("hardware_config.json lacks compute_unit_bindings")
        unit_name = bindings.get(role)
        if not unit_name:
            raise ValueError(
                "hardware_config.json lacks compute-unit binding " f"{role}"
            )
        units = {
            unit["name"]: unit
            for unit in self.hardware_config.get("compute_units", [])
        }
        unit = units.get(unit_name)
        if unit is None:
            raise ValueError(
                "hardware_config.json binds "
                f"{role} to unknown compute unit {unit_name}"
            )
        return unit

    def accepted_input_bits(
        self,
        role: str,
    ) -> frozenset[int]:
        """Return serialized operand widths for one compute-unit role."""
        unit = self.compute_unit(role)
        widths = unit.get("accepted_input_bits")
        if widths is None:
            raise ValueError(
                "hardware_config.json lacks accepted_input_bits for "
                f"{unit['name']}"
            )
        if not isinstance(widths, list):
            raise ValueError(
                "hardware_config.json accepted_input_bits must be a list"
            )
        return frozenset(
            _config_integer(
                width,
                f"{unit['name']}.accepted_input_bits",
            )
            for width in widths
        )

    def unroll_extent(
        self,
        role: str,
        dimension: str,
        input_bits: int | None = None,
    ) -> int:
        unit = self.compute_unit(role)
        accepted = self.accepted_input_bits(role)
        if input_bits is not None and accepted and input_bits not in accepted:
            raise ValueError(
                f"{unit['name']}: unsupported {input_bits}-bit input"
            )
        dimensions = {
            item["name"]: item for item in unit.get("unroll_dimensions", [])
        }
        descriptor = dimensions.get(dimension)
        if descriptor is None:
            raise ValueError(f"hardware_config.json lacks {role} {dimension}")
        raw_dtype_extents = descriptor.get("dtype_extents", [])
        if not isinstance(raw_dtype_extents, list) or any(
            not isinstance(item, dict) for item in raw_dtype_extents
        ):
            raise ValueError(
                "hardware_config.json dtype_extents must be a list of "
                "objects"
            )
        dtype_extents = {
            _config_integer(
                item.get("input_bits"),
                f"{unit['name']}.{dimension}.input_bits",
            ): _config_integer(
                item.get("extent"),
                f"{unit['name']}.{dimension}.extent",
            )
            for item in raw_dtype_extents
        }
        if input_bits is not None and input_bits in dtype_extents:
            return dtype_extents[input_bits]
        return _config_integer(
            descriptor.get("extent"),
            f"{unit['name']}.{dimension}.extent",
        )

    def matrix_input_lanes(self, input_bits: int) -> int:
        return self.unroll_extent("matrix", "IC", input_bits)

    def matrix_output_lanes(self, input_bits: int) -> int:
        return self.unroll_extent("matrix", "OC", input_bits)

    def nmp_lanes(self, input_bits: int) -> int:
        return self.unroll_extent("nmp", "CHANNEL", input_bits)

    def require_nmp_tensor_shape(
        self,
        tensor: Any,
        layer: str,
        role: str,
    ) -> tuple[int, int, int, int]:
        """Return one positive NCHW extent from a finalized dump tensor."""
        try:
            shape = tuple(_exact_integer(value) for value in tensor.shape)
        except ValueError:
            shape = ()
        if len(shape) != 4 or any(value <= 0 for value in shape):
            self.fail(
                "invalid-nmp-tensor-extents",
                layer,
                f"{role}: shape={tuple(tensor.shape)}; "
                "expected positive rank-4 NCHW",
            )
        return shape

    def reject_nmp_tensor_extents(
        self,
        layer: str,
        execution_kind: str,
        detail: str,
    ) -> None:
        self.fail(
            "invalid-nmp-tensor-extents",
            layer,
            f"{execution_kind}: {detail}",
        )

    def validate_schedule_semantics(
        self,
        layer: str,
        execution_kind: str,
        factors: dict[str, int],
        expected: dict[str, int],
    ) -> None:
        """Assert compiler-selected factors against serialized semantics."""
        mismatches = {
            field: (factors[field], expected_value)
            for field, expected_value in expected.items()
            if factors[field] != expected_value
        }
        if not mismatches:
            return
        self.fail(
            "target-schedule-semantic-mismatch",
            layer,
            f"{execution_kind}: schedule mismatch={mismatches}",
        )

    def require_block_size(
        self,
        operation: Any,
        layer: str,
        role: str,
    ) -> int:
        """Read one compiler-selected VSQ block size without target policy."""
        value = cmod_utils.operation_kwarg(operation, "block_size")
        return self.require_integer(
            value,
            layer,
            f"{role} block_size",
            minimum=1,
            code="invalid-vsq-block-size",
        )

    def validate_matrix_output_storage(
        self,
        name: str,
        resident_k: int,
        input_bits: int,
        schedule: dict[str, Any],
    ) -> None:
        """Require Alpha's scheduled output vectors to cover resident K."""
        vector_count = (
            int(schedule["k1"]) * int(schedule["k2"]) * int(schedule["k3"])
        )
        valid_lanes = self.matrix_output_lanes(input_bits) - int(
            schedule["padding_k"]
        )
        storage_k = vector_count * valid_lanes
        if storage_k == resident_k:
            return
        self.fail(
            "target-output-storage-mismatch",
            name,
            f"scale/bias count={storage_k}; resident K={resident_k}",
        )

    def validate_matrix_input_storage(
        self,
        name: str,
        resident_c: int,
        input_bits: int,
        schedule: dict[str, Any],
    ) -> None:
        vector_count = (
            int(schedule["c1"]) * int(schedule["c2"]) * int(schedule["c3"])
        )
        lanes = self.matrix_input_lanes(input_bits)
        padding = int(schedule["padding_c"])
        storage_c = vector_count * (lanes - padding)
        if 0 <= padding < lanes and storage_c == resident_c:
            return
        self.fail(
            "target-input-storage-mismatch",
            name,
            f"input channels={storage_c}; resident C={resident_c}",
        )

    def require_nmp_input_bits(
        self,
        layer: str,
        **operands: Any,
    ) -> dict[str, int]:
        """Validate physical inputs against the Alpha CMOD contract."""
        result = {}
        accepted = self.accepted_input_bits("nmp")
        for role, tensor in operands.items():
            bits = cmod_utils.dtype_precision(tensor.dtype)
            if accepted and bits not in accepted:
                self.fail(
                    "unsupported-nmp-input-precision",
                    layer,
                    f"{role} {tensor.node}: dtype={tensor.dtype}; "
                    f"accepted bits={sorted(accepted)}",
                )
            result[role] = bits
        return result

    def schedule_factors(
        self,
        operation: Any,
        names: Sequence[str],
    ) -> dict[str, int]:
        """Read final compiler-owned named factors without using dump tiling."""
        layer = cmod_utils.operation_name(operation)
        factors = {}
        for name in names:
            field = f"{cmod_utils.SCHEDULE_FACTOR_PREFIX}{name}"
            value = self.serialized_value(
                operation,
                field,
                required=False,
            )
            if value is None:
                self.fail(
                    "missing-target-schedule-factor",
                    layer,
                    f"{field}: missing",
                )
            factors[name] = self.require_integer(
                value,
                layer,
                field,
                minimum=1,
                code="invalid-target-schedule-factor",
            )
        return factors

    def validate_gemm_schedule_limits(
        self,
        layer: str,
        factors: dict[str, int],
        input_bits: int,
    ) -> None:
        """Reject schedule factors that CMOD cannot represent safely."""
        if factors["batch_size"] != 1:
            self.fail(
                "unsupported-alpha-schedule-mapping",
                layer,
                "Alpha CMOD one-PE GEMM requires batch_size=1.",
            )
        outer_factors = {
            name: factors[name] for name in ("c2", "c3", "k2", "k3")
        }
        if any(value != 1 for value in outer_factors.values()):
            self.fail(
                "unsupported-alpha-schedule-mapping",
                layer,
                "One CMOD GEMM call requires C2=C3=K2=K3=1; "
                f"found={outer_factors}.",
            )
        if factors["p1"] % factors["p0"] or factors["q1"] % factors["q0"]:
            self.fail(
                "unsupported-alpha-schedule-mapping",
                layer,
                "Alpha GEMM requires integral P1/P0 and Q1/Q0.",
            )

        kernel_channel_words = factors["r"] * factors["s"] * factors["c1"]
        temporal_tile_count = factors["p2"] * factors["q2"]
        weight_tiles_per_spatial_step = kernel_channel_words * factors["k1"]
        derived = {
            "temporal_tile_count": temporal_tile_count,
            "weight_tiles_per_spatial_step": weight_tiles_per_spatial_step,
            "weight_tile_count": (
                temporal_tile_count
                * (factors["p1"] // factors["p0"])
                * (factors["q1"] // factors["q0"])
                * weight_tiles_per_spatial_step
            ),
            "level0_weight_read_count": weight_tiles_per_spatial_step,
            "level1_weight_read_count": (
                factors["p1"] * factors["q1"] * weight_tiles_per_spatial_step
            ),
            "mac_iteration_count": (
                factors["p1"]
                * factors["q1"]
                * weight_tiles_per_spatial_step
                * temporal_tile_count
            ),
        }

        local_weight_word_bits = factors["c0"] * input_bits
        global_memory = self.global_memory(layer)
        weight_memory_name = global_memory.get("weight_memory")
        weight_memory = next(
            (
                instance
                for instance in global_memory["instances"]
                if instance["name"] == weight_memory_name
            ),
            None,
        )
        if weight_memory is None:
            self.fail(
                "invalid-hardware-memory-config",
                layer,
                f"global weight memory {weight_memory_name!r}: missing",
            )
        dimensions = {}
        for name in ("bank_count", "depth_per_bank", "bits_per_bank_entry"):
            try:
                dimensions[name] = _config_integer(
                    weight_memory.get(name),
                    f"{weight_memory_name}.{name}",
                )
            except ValueError as error:
                self.fail(
                    "invalid-hardware-memory-config",
                    layer,
                    str(error),
                )
        bits_per_entry = dimensions["bits_per_bank_entry"]
        if bits_per_entry % local_weight_word_bits:
            self.fail(
                "unsupported-alpha-schedule-mapping",
                layer,
                "The global weight line is not an integral number of local "
                "weight vectors.",
            )
        vectors_per_line = bits_per_entry // local_weight_word_bits
        if not vectors_per_line or factors["k0"] % vectors_per_line:
            self.fail(
                "unsupported-alpha-schedule-mapping",
                layer,
                "Alpha output lanes do not contain an integral number of "
                "global weight lines.",
            )
        derived["global_weight_sweep_lines"] = weight_tiles_per_spatial_step * (
            factors["k0"] // vectors_per_line
        )
        derived["weight_payload_bits"] = (
            weight_tiles_per_spatial_step
            * factors["k0"]
            * local_weight_word_bits
        )
        limits = {
            "temporal_tile_count": (1 << 16) - 1,
            "weight_tile_count": (1 << 16) - 1,
            "level0_weight_read_count": (1 << 16) - 1,
            "level1_weight_read_count": (1 << 16) - 1,
            "mac_iteration_count": 1 << 32,
            "global_weight_sweep_lines": min(
                2 * dimensions["bank_count"] * dimensions["depth_per_bank"],
                (1 << 15) - 1,
            ),
            "weight_payload_bits": (
                dimensions["bank_count"]
                * dimensions["depth_per_bank"]
                * bits_per_entry
            ),
        }
        if kernel_channel_words > 1 << 8 or factors["k1"] > 8:
            self.fail(
                "alpha-schedule-limit-exceeded",
                layer,
                "R*S*C1 or K1 exceeds the Alpha CMOD counter range.",
            )
        overflow = {
            name: (derived[name], maximum)
            for name, maximum in limits.items()
            if derived[name] > maximum
        }
        if overflow:
            self.fail(
                "alpha-schedule-limit-exceeded",
                layer,
                f"derived CMOD counters exceed their ranges: {overflow}",
            )

    def emit_schedule(self, result: LayerResult) -> None:
        layer_dir = self.output_dir / result.name
        layer_dir.mkdir(parents=True, exist_ok=True)
        text = cmod_utils.render_schedule(result.schedule)
        for index in (0, 1, 10):
            (layer_dir / f"schedule{index}.txt").write_text(text)

    def materialized_input_channels(
        self,
        layer: str,
        operation: Any,
    ) -> tuple[int, int] | None:
        """Return compiler-declared valid/resident input channels."""
        valid_value = self.serialized_value(
            operation,
            "materialized_input_valid_channels",
            required=False,
        )
        resident_value = self.serialized_value(
            operation,
            "materialized_input_resident_channels",
            required=False,
        )
        if valid_value is None and resident_value is None:
            return None
        if valid_value is None or resident_value is None:
            self.fail(
                "invalid-materialized-input-channels",
                layer,
                "materialized input channel storage requires both valid "
                "and resident channel declarations",
            )
        valid_channels = self.require_integer(
            valid_value,
            layer,
            "materialized_input_valid_channels",
            minimum=1,
            code="invalid-materialized-input-channels",
        )
        resident_channels = self.require_integer(
            resident_value,
            layer,
            "materialized_input_resident_channels",
            minimum=1,
            code="invalid-materialized-input-channels",
        )
        if valid_channels > resident_channels:
            self.fail(
                "invalid-materialized-input-channels",
                layer,
                "materialized input valid channels exceed resident channels",
            )
        return valid_channels, resident_channels

    def resolve_materialized_input_payload(
        self,
        layer: str,
        operation: Any,
        input_tensor: Any,
        semantic_values: np.ndarray,
        precision: int,
    ) -> tuple[Any, np.ndarray]:
        """Validate and select an already-materialized physical input."""
        channel_storage = self.materialized_input_channels(layer, operation)
        padding = self.serialized_value(
            operation,
            "materialized_input_padding",
            required=False,
        )
        payload = self.serialized_tensor_role(
            operation,
            "materialized_input_payload",
            required=False,
            output_only=False,
        )
        if padding is None:
            if channel_storage is not None:
                self.fail(
                    "missing-materialized-input-padding",
                    layer,
                    "materialized input channel storage requires "
                    "materialized_input_padding",
                )
            if payload is not None:
                self.fail(
                    "unexpected-materialized-input-payload",
                    layer,
                    "materialized input requires materialized_input_padding",
                )
            return input_tensor, semantic_values
        if not isinstance(padding, list) or len(padding) != 4:
            self.fail(
                "invalid-materialized-input-padding",
                layer,
                "materialized_input_padding: expected four non-negative "
                "integers",
            )
        try:
            left, right, top, bottom = (
                _exact_integer(value) for value in padding
            )
        except ValueError:
            left = right = top = bottom = -1
        if min(left, right, top, bottom) < 0:
            self.fail(
                "invalid-materialized-input-padding",
                layer,
                "materialized_input_padding: expected four exact "
                "non-negative integers",
            )
        if payload is None:
            self.fail(
                "missing-materialized-input-payload",
                layer,
                "materialized_input_padding set but payload missing",
            )
        if semantic_values.ndim != 4:
            self.fail(
                "invalid-materialized-input-payload",
                layer,
                "materialized input: expected rank 4",
            )
        valid_channels = int(semantic_values.shape[1])
        resident_channels = valid_channels
        if channel_storage is not None:
            valid_channels, resident_channels = channel_storage
            if valid_channels != int(semantic_values.shape[1]):
                self.fail(
                    "invalid-materialized-input-channels",
                    layer,
                    "materialized input valid channels do not match the "
                    "semantic input channels",
                )
        payload_precision = self.require_precision(
            payload,
            layer,
            "materialized input payload",
        )
        if payload_precision != precision:
            self.fail(
                "invalid-materialized-input-payload",
                layer,
                "semantic/materialized input precision mismatch",
            )
        expected_shape = (
            semantic_values.shape[0],
            resident_channels,
            semantic_values.shape[2] + top + bottom,
            semantic_values.shape[3] + left + right,
        )
        physical_values = self.load_exact_payload(
            payload,
            layer,
            "materialized input payload",
            precision,
        )
        self.require_shape(
            physical_values,
            expected_shape,
            layer,
            "materialized input payload",
        )
        semantic_address = self.require_address(
            input_tensor,
            layer,
            "semantic input",
            "activation",
            "matrix",
        )
        physical_address = self.require_address(
            payload,
            layer,
            "materialized input payload",
            "activation",
            "matrix",
        )
        expected_values = np.zeros_like(physical_values)
        expected_values[
            :,
            :valid_channels,
            top : top + semantic_values.shape[2],
            left : left + semantic_values.shape[3],
        ] = semantic_values
        if (
            semantic_address != physical_address
            or input_tensor.scratchpad.instance != payload.scratchpad.instance
            or not np.array_equal(physical_values, expected_values)
        ):
            self.fail(
                "invalid-materialized-input-payload",
                layer,
                "materialized input differs from semantic storage with "
                "zero-filled spatial/channel padding",
            )
        return payload, physical_values

    def convolution_descriptor(
        self,
        operation: Any,
        route: str,
    ) -> Any:
        descriptor = self.inner_by_target(
            operation,
            f"{route} descriptor",
            targets=cmod_utils.CMOD_CONVOLUTION_TARGETS,
            required=True,
        )
        return descriptor

    def build_gemm(
        self,
        operation: Any,
        execution_kind: str,
    ) -> LayerResult:
        name = cmod_utils.operation_name(operation)
        conv = self.convolution_descriptor(operation, "gemm")
        weight = cmod_utils.operation_kwarg(conv, "weight")
        input_tensor = cmod_utils.operation_kwarg(conv, "input")
        bias = cmod_utils.operation_kwarg(conv, "bias")
        if weight is None or input_tensor is None:
            missing = [
                role
                for role, tensor in (
                    ("input", input_tensor),
                    ("weight", weight),
                )
                if tensor is None
            ]
            self.fail(
                "missing-executable-operand",
                name,
                f"missing={','.join(missing)}",
            )
        groups = self.require_integer(
            cmod_utils.operation_kwarg(conv, "groups", 1),
            name,
            "groups",
            minimum=1,
        )
        if groups != 1:
            self.fail(
                "invalid-gemm-descriptor",
                name,
                "The Alpha GEMM route requires groups=1.",
            )

        result = LayerResult(
            name=name,
            kind=execution_kind,
            schedule={},
            inputs=self.operation_inputs(operation),
        )
        weight_precision = self.require_precision(
            weight,
            name,
            "GEMM weight",
        )
        weight_payload = self.serialized_tensor_role(
            operation,
            "weight_payload",
            required=True,
            output_only=False,
        )
        valid_weight_c = self.serialized_value(
            operation,
            "weight_valid_input_channels",
        )
        valid_weight_k = self.serialized_value(
            operation,
            "weight_valid_output_channels",
        )
        materialized_channels = self.materialized_input_channels(
            name,
            operation,
        )
        weight_shape = tuple(int(item) for item in weight.shape)
        payload_shape = tuple(int(item) for item in weight_payload.shape)
        if len(weight_shape) != 4 or len(payload_shape) != 4:
            self.fail(
                "invalid-weight-storage-extents",
                name,
                "The semantic and physical GEMM weights must be rank 4.",
            )
        resident_weight_k, resident_weight_c = weight_shape[:2]
        physical_weight_k, physical_weight_c = payload_shape[:2]
        expected_physical_weight_c = (
            materialized_channels[1]
            if materialized_channels is not None
            else (
                ((resident_weight_c + 3) // 4) * 4
                if weight_precision == 8
                else resident_weight_c
            )
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in (
                    valid_weight_c,
                    valid_weight_k,
                )
            )
            or not (valid_weight_c <= resident_weight_c <= physical_weight_c)
            or valid_weight_k > resident_weight_k
            or physical_weight_k != resident_weight_k
            or payload_shape[2:] != weight_shape[2:]
            or physical_weight_c != expected_physical_weight_c
        ):
            self.fail(
                "invalid-weight-storage-extents",
                name,
                "The serialized GEMM weight payload and valid/physical "
                "channel extents are inconsistent.",
            )
        payload_weight_precision = self.require_precision(
            weight_payload,
            name,
            "GEMM physical weight payload",
        )
        if payload_weight_precision != weight_precision:
            self.fail(
                "inconsistent-weight-storage-precision",
                name,
                "Semantic and physical GEMM weights have different "
                "precisions.",
            )
        weight_values = self.load_exact_payload(
            weight_payload,
            name,
            "GEMM weight",
            weight_precision,
            weight=True,
        )
        resident_weight_values = self.load_exact_payload(
            weight,
            name,
            "resident GEMM weight",
            weight_precision,
            weight=True,
        )
        expected_weight_values = np.zeros_like(weight_values)
        expected_weight_values[
            :resident_weight_k,
            :resident_weight_c,
            :,
            :,
        ] = resident_weight_values
        if not np.array_equal(weight_values, expected_weight_values):
            self.fail(
                "inconsistent-weight-storage-payload",
                name,
                "The physical GEMM weight payload must equal the resident "
                "semantic weight with only a zero-filled C tail.",
            )
        resident_k = resident_weight_k
        resident_c = resident_weight_c * groups
        ppu_parameters = self.explicit_ppu_parameters(
            operation,
            name,
        )
        if ppu_parameters is None:
            self.fail(
                "missing-explicit-pe-fixed-point",
                name,
                "Alpha GEMM conversion requires a compiler-finalized PPU "
                "epilogue in the same serialized executable operation; "
                "standalone downstream epilogues are unsupported.",
            )
        quantizer = self.output_quantizer(operation)
        payload = self.output_payload(operation)
        eadd_operand = self.explicit_eadd_operand(
            operation,
            name,
            payload,
            ppu_parameters,
        )
        input_precision = self.require_precision(
            input_tensor,
            name,
            "GEMM input",
        )
        input_values = self.load_exact_payload(
            input_tensor,
            name,
            "GEMM input",
            input_precision,
        )
        if input_values.ndim != 4 or int(input_values.shape[1]) != resident_c:
            self.fail(
                "invalid-gemm-input-channels",
                name,
                f"GEMM input shape {input_values.shape} does not carry "
                f"the compiler-declared resident C={resident_c}.",
            )
        input_storage, input_payload_values = (
            self.resolve_materialized_input_payload(
                name,
                operation,
                input_tensor,
                input_values,
                input_precision,
            )
        )
        input_storage_c = resident_c
        if materialized_channels is not None:
            valid_input_c, input_storage_c = materialized_channels
            if (
                valid_input_c != valid_weight_c
                or input_storage_c != physical_weight_c * groups
            ):
                self.fail(
                    "invalid-materialized-input-channels",
                    name,
                    "materialized input valid/resident channels do not "
                    "match the serialized valid/physical weight channels",
                )
        output_precision = ppu_parameters["output_precision"]
        output_vector = self.output_vector_scale(operation)
        if output_precision == 4:
            if quantizer is None or output_vector is None:
                self.fail(
                    "missing-explicit-pvs-output",
                    name,
                    "Alpha O4 GEMM requires a PVS quantizer, payload, and "
                    "companion scale in the same serialized executable "
                    "operation.",
                )
        elif output_precision == 8:
            if output_vector is not None:
                self.fail(
                    "unexpected-o8-companion-scale",
                    name,
                    "Alpha O8 GEMM output must not carry a PVS scale.",
                )
        else:
            self.fail(
                "invalid-pe-output-precision",
                name,
                f"Unsupported Alpha PE output precision {output_precision}.",
            )

        schedule = self.base_schedule(name)
        factors = self.schedule_factors(
            operation,
            cmod_utils.GEMM_SCHEDULE_FACTORS,
        )
        schedule.update(
            {
                "c1": factors["c1"],
                "c2": factors["c2"],
                "c3": factors["c3"],
                "k1": factors["k1"],
                "k2": factors["k2"],
                "k3": factors["k3"],
                "p0": factors["p0"],
                "p1": factors["p1"],
                "p2": factors["p2"],
                "p2s": factors["p2s"],
                "p3": factors["p3"],
                "q0": factors["q0"],
                "q1": factors["q1"],
                "q2": factors["q2"],
                "q2s": factors["q2s"],
                "q3": factors["q3"],
            }
        )
        kernel = int(weight.shape[2]), int(weight.shape[3])
        expected_factors = {
            "batch_size": int(input_tensor.shape[0]),
            "c0": self.matrix_input_lanes(input_precision),
            "k0": self.matrix_output_lanes(input_precision),
            "r": kernel[0],
            "s": kernel[1],
        }
        self.validate_schedule_semantics(
            name,
            execution_kind,
            factors,
            expected_factors,
        )
        self.validate_gemm_schedule_limits(
            name,
            factors,
            input_precision,
        )
        stride = self.require_pair(
            cmod_utils.operation_kwarg(conv, "stride"),
            name,
            "stride",
            default=1,
            minimum=1,
        )
        dilation = self.require_pair(
            cmod_utils.operation_kwarg(conv, "dilation"),
            name,
            "dilation",
            default=1,
            minimum=1,
        )
        if stride[0] != stride[1] or dilation != (1, 1):
            self.fail(
                "unsupported-gemm-spatial-mode",
                name,
                "Alpha CMOD requires symmetric stride and unit dilation.",
            )
        schedule.update(
            {
                "r": factors["r"],
                "s": factors["s"],
                "batch_size": factors["batch_size"],
                "stride": stride[0],
                "wgt_precision": weight_precision,
                "ia_precision": input_precision,
                "oa_precision": output_precision,
                "wgt_scale_precision": 8 if weight_precision == 4 else 0,
                "ia_scale_precision": 8 if input_precision == 4 else 0,
                "oa_scale_precision": 8 if output_vector is not None else 0,
            }
        )

        explicit_bias = None
        scale_int = ppu_parameters["scale_int"]
        if scale_int.size != resident_k:
            self.fail(
                "target-operand-shape-mismatch",
                name,
                "Alpha PE primary multiplier count "
                f"{scale_int.size} does not match resident K={resident_k}.",
            )
        binary_point = ppu_parameters["binary_point"]
        nonlinear_multiplier = ppu_parameters["nonlinear_multiplier"]
        nonlinear_shift = ppu_parameters["nonlinear_shift"]
        explicit_bias = ppu_parameters["bias"]
        if bias is not None and explicit_bias is None:
            self.fail(
                "missing-explicit-pe-bias",
                name,
                "A biased GEMM requires a compiler-finalized integer bias "
                "payload.",
            )
        schedule.update(
            {
                "scale_length": math.ceil(
                    resident_k / self.matrix_output_lanes(input_precision)
                ),
                "bias_length": (
                    math.ceil(
                        resident_k / self.matrix_output_lanes(input_precision)
                    )
                    if bias is not None or explicit_bias is not None
                    else 0
                ),
                "scale_factor": 0,
                "bias_factor": 0,
                "wt_binary_pt_loc": binary_point,
                "nonlinear_scale_factor": nonlinear_multiplier,
                "nonlinear_scale_shift": nonlinear_shift,
            }
        )
        if ppu_parameters["relu"]:
            schedule["is_relu"] = 1
        if eadd_operand is not None:
            schedule.update(
                {
                    "is_eadd": 1,
                    "num_eop": 1,
                }
            )
        padding = self.padding_fields(operation, spatial=True)
        schedule.update(padding)
        output_spatial = (
            factors["p1"] * factors["p2"] * factors["p3"],
            factors["q1"] * factors["q2"] * factors["q3"],
        )
        schedule.update(
            self.spatial_input_geometry_fields(
                name,
                input_payload_values.shape,
                kernel,
                output_spatial,
                stride,
                dilation,
                padding,
            )
        )
        self.validate_matrix_input_storage(
            name,
            input_storage_c,
            input_precision,
            schedule,
        )
        self.validate_matrix_output_storage(
            name,
            resident_k,
            input_precision,
            schedule,
        )
        schedule.update(
            {
                "inputs0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        input_storage,
                        name,
                        "input",
                        "activation",
                        "matrix",
                    )
                ),
                "outputs0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        payload,
                        name,
                        "output",
                        "output",
                        "matrix",
                    )
                ),
                "weights0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        weight_payload,
                        name,
                        "physical weight payload",
                        "weight",
                        "matrix",
                    )
                ),
            }
        )
        if eadd_operand is not None:
            schedule["inputs1_baseaddr"] = schedule["outputs0_baseaddr"]
        schedule.update(self.concat_fields([operation]))
        schedule.update(self.nmp_storage_fields(operation))
        schedule.update(self.nmp_group_fields(operation))
        result.schedule = schedule

        output_values = self.load_exact_integer_tensor(
            payload,
            name,
            "GEMM output",
            f"int{output_precision}",
            -8 if output_precision == 4 else -128,
            7 if output_precision == 4 else 127,
        )
        expected_output_shape = (
            factors["batch_size"],
            resident_k,
            output_spatial[0],
            output_spatial[1],
        )
        if output_values.shape != expected_output_shape:
            self.fail(
                "invalid-gemm-output-shape",
                name,
                f"GEMM output shape {output_values.shape} does not match "
                "compiler-declared N/K/P/Q coverage "
                f"{expected_output_shape}.",
            )
        self.write_payload(
            result,
            "1_inputs.txt",
            input_payload_values,
            input_precision,
        )
        self.write_payload(
            result,
            "1_outputs.txt",
            output_values,
            output_precision,
        )
        if eadd_operand is not None:
            eadd_values = self.load_exact_payload(
                eadd_operand,
                name,
                "EADD operand2",
                8,
            )
            self.write_payload(
                result,
                "1_inputs2.txt",
                eadd_values,
                8,
            )

        self.write_payload(
            result,
            "1_weights.txt",
            weight_values,
            weight_precision,
            weight=True,
        )
        self.write_values(
            result,
            "1_scale.txt",
            scale_int,
            0,
            65535,
        )

        if explicit_bias is not None:
            bias_values = self.load_exact_integer_values(
                explicit_bias,
                name,
                "Alpha PE bias",
                -128,
                127,
            )
            if bias_values.size != resident_k:
                self.fail(
                    "target-operand-shape-mismatch",
                    name,
                    "Alpha PE bias count "
                    f"{bias_values.size} does not match resident K="
                    f"{resident_k}.",
                )
            self.write_values(
                result,
                "1_bias.txt",
                bias_values,
                -128,
                127,
            )
        input_vector = cmod_utils.operation_kwarg(conv, "input_scale")
        conv_block_size = None
        weight_scale = cmod_utils.operation_kwarg(conv, "weight_scale")
        if input_vector is not None or weight_scale is not None:
            conv_block_size = self.require_block_size(
                conv,
                name,
                "GEMM VSQ operands",
            )
        if input_vector is not None:
            semantic_values = self.load_exact_integer_tensor(
                input_vector,
                name,
                "GEMM input companion scale",
                "uint8",
                0,
                255,
            )
            self.require_shape(
                semantic_values,
                (
                    int(input_tensor.shape[0]),
                    math.ceil(int(input_tensor.shape[1]) / conv_block_size),
                    int(input_tensor.shape[2]),
                    int(input_tensor.shape[3]),
                ),
                name,
                "GEMM input companion scale",
            )
            values = self.load_cmod_vector_scale(
                operation,
                "input_vector_scale_payload",
                name,
                "GEMM input companion scale",
                semantic_values,
                kind="input",
            )
            self.write_values(
                result,
                "1_inputs_vector_scale.txt",
                values,
                0,
                255,
            )
        elif (
            self.serialized_tensor_role(
                operation,
                "input_vector_scale_payload",
                required=False,
                output_only=False,
            )
            is not None
        ):
            self.fail(
                "unexpected-serialized-artifact",
                name,
                "An input vector-scale artifact requires a semantic "
                "input_scale operand.",
            )
        weight_vector = weight_scale
        if weight_precision == 4 and weight_vector is None:
            self.fail(
                "missing-gemm-weight-scale",
                name,
                "INT4 GEMM weights require an explicit companion scale.",
            )
        if weight_vector is not None:
            valid_scale_k = self.require_integer(
                self.serialized_value(
                    operation,
                    "weight_scale_valid_output_channels",
                ),
                name,
                "weight_scale_valid_output_channels",
                minimum=1,
                code="invalid-weight-scale-storage-extents",
            )
            semantic_scale = self.load_exact_integer_tensor(
                weight_vector,
                name,
                "GEMM weight companion scale",
                "uint8",
                0,
                255,
            )
            output_lanes = self.matrix_output_lanes(input_precision)
            expected_scale_shape = (
                resident_weight_k,
                math.ceil(resident_weight_c / conv_block_size),
                int(weight.shape[2]),
                int(weight.shape[3]),
            )
            if (
                valid_scale_k != valid_weight_k
                or semantic_scale.shape != expected_scale_shape
                or physical_weight_k % output_lanes
            ):
                self.fail(
                    "invalid-weight-scale-storage-extents",
                    name,
                    "The semantic GEMM weight scale and valid/physical "
                    "extents are inconsistent.",
                )
            raw_scale = np.zeros(
                (
                    physical_weight_k,
                    *semantic_scale.shape[1:],
                ),
                dtype=semantic_scale.dtype,
            )
            raw_scale[:resident_weight_k] = semantic_scale
            values = self.load_cmod_vector_scale(
                operation,
                "weight_vector_scale_payload",
                name,
                "GEMM weight companion scale",
                raw_scale,
                kind="weight",
                output_lanes=output_lanes,
            )
            self.write_values(
                result,
                "1_weights_vector_scale.txt",
                values,
                0,
                255,
            )
        elif (
            self.serialized_tensor_role(
                operation,
                "weight_vector_scale_payload",
                required=False,
                output_only=False,
            )
            is not None
        ):
            self.fail(
                "unexpected-serialized-artifact",
                name,
                "A weight vector-scale artifact requires a semantic "
                "weight_scale operand.",
            )
        if output_vector is not None:
            output_block_size = self.require_block_size(
                quantizer[1],
                name,
                "GEMM output companion scale",
            )
            semantic_values = self.load_exact_integer_tensor(
                output_vector,
                name,
                "GEMM output companion scale",
                "uint8",
                0,
                255,
            )
            self.require_shape(
                semantic_values,
                (
                    int(payload.shape[0]),
                    math.ceil(resident_k / output_block_size),
                    int(payload.shape[2]),
                    int(payload.shape[3]),
                ),
                name,
                "GEMM output companion scale",
            )
            values = self.load_cmod_vector_scale(
                operation,
                "output_vector_scale_payload",
                name,
                "GEMM output companion scale",
                semantic_values,
                kind="output",
            )
            self.write_values(
                result,
                "1_outputs_vector_scale.txt",
                values,
                0,
                255,
            )
        elif (
            self.serialized_tensor_role(
                operation,
                "output_vector_scale_payload",
                required=False,
                output_only=False,
            )
            is not None
        ):
            self.fail(
                "unexpected-serialized-artifact",
                name,
                "An output vector-scale artifact requires a semantic "
                "output_vector_scale.",
            )
        return result

    def build_depthwise(
        self,
        operation: Any,
        execution_kind: str,
    ) -> LayerResult:
        name = cmod_utils.operation_name(operation)
        conv = self.convolution_descriptor(operation, "depthwise")
        result = LayerResult(
            name=name,
            kind=execution_kind,
            schedule={},
            inputs=self.operation_inputs(operation),
        )
        input_tensor = cmod_utils.operation_kwarg(conv, "input")
        weight = cmod_utils.operation_kwarg(conv, "weight")
        if input_tensor is None or weight is None:
            missing = [
                role
                for role, tensor in (
                    ("input", input_tensor),
                    ("weight", weight),
                )
                if tensor is None
            ]
            self.fail(
                "missing-executable-operand",
                name,
                f"missing={','.join(missing)}",
            )
        input_shape = self.require_nmp_tensor_shape(
            input_tensor,
            name,
            "DWC input",
        )
        weight_shape = self.require_nmp_tensor_shape(
            weight,
            name,
            "DWC weight",
        )
        groups = self.require_integer(
            cmod_utils.operation_kwarg(conv, "groups", 1),
            name,
            "groups",
            minimum=1,
        )
        if (
            groups <= 1
            or weight_shape[1] != 1
            or groups != weight_shape[0]
            or groups != input_shape[1]
        ):
            self.fail(
                "invalid-depthwise-descriptor",
                name,
                "A depthwise route requires one input/output channel per "
                "group.",
            )
        operand_bits = self.require_nmp_input_bits(
            name,
            input=input_tensor,
            weight=weight,
        )
        fixed_point = self.explicit_nmp_parameters(operation, name)
        output_precision = fixed_point["output_precision"]
        quantizer = self.output_quantizer(operation)
        payload = self.output_payload(operation)
        output_shape = self.require_nmp_tensor_shape(
            payload,
            name,
            "DWC output",
        )
        if output_shape[0] != input_shape[0] or output_shape[1] != groups:
            self.reject_nmp_tensor_extents(
                name,
                execution_kind,
                "input/output batch must match and output channels must "
                "equal the depthwise group count",
            )
        output_vector = self.output_vector_scale(operation)
        if output_precision == 4:
            if quantizer is None or output_vector is None:
                self.fail(
                    "missing-explicit-pvs-output",
                    name,
                    "Alpha O4 DWC requires a PVS quantizer, payload, and "
                    "companion scale in the same serialized executable "
                    "operation.",
                )
        elif output_precision == 8:
            if output_vector is not None:
                self.fail(
                    "unexpected-o8-companion-scale",
                    name,
                    "Alpha O8 DWC output must not carry a PVS scale.",
                )
        else:
            self.fail(
                "invalid-nmp-output-precision",
                name,
                f"Unsupported Alpha NMP output precision "
                f"{output_precision}.",
            )
        factors = self.schedule_factors(
            operation,
            cmod_utils.NMP_SCHEDULE_FACTORS,
        )
        lanes = self.nmp_lanes(operand_bits["input"])
        self.validate_schedule_semantics(
            name,
            execution_kind,
            factors,
            {
                "p1": output_shape[2],
                "q1": output_shape[3],
                "c1": (input_shape[1] + lanes - 1) // lanes,
                "k1": (output_shape[1] + lanes - 1) // lanes,
            },
        )
        kernel = weight_shape[2], weight_shape[3]
        stride = self.require_pair(
            cmod_utils.operation_kwarg(conv, "stride"),
            name,
            "stride",
            default=1,
            minimum=1,
        )
        dilation = self.require_pair(
            cmod_utils.operation_kwarg(conv, "dilation"),
            name,
            "dilation",
            default=1,
            minimum=1,
        )
        if stride[0] != stride[1] or dilation != (1, 1):
            self.fail(
                "unsupported-depthwise-spatial-mode",
                name,
                "Alpha CMOD requires symmetric stride and unit dilation.",
            )
        scale_factor = fixed_point["scale_factor"]
        binary_point = fixed_point["binary_point"]
        scalar_bias = fixed_point["bias_factor"]
        nmp_groups = self.nmp_group_fields(operation)
        schedule_groups = nmp_groups.pop("groups", groups)
        schedule = self.base_schedule(name)
        schedule.update(
            {
                "is_dwconv": 1,
                "nmp_scale_length": 0,
                "nmp_scale_factor": scale_factor,
                "p1": factors["p1"],
                "q1": factors["q1"],
                "c1": factors["c1"],
                "k1": factors["k1"],
                "batch_size": input_shape[0],
                "r": kernel[0],
                "s": kernel[1],
                "stride": stride[0],
                "groups": schedule_groups,
                "inputs0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        input_tensor,
                        name,
                        "input",
                        "activation",
                        "nmp",
                    )
                ),
                "outputs0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        payload,
                        name,
                        "output",
                        "output",
                        "nmp",
                    )
                ),
                "inputs1_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        weight,
                        name,
                        "weight",
                        "weight",
                        "nmp",
                    )
                ),
                "bias_length": 0,
                "nmp_bias_factor": scalar_bias,
                "oa_precision": output_precision,
                "oa_scale_precision": 8 if output_vector is not None else 0,
                "ia_precision": operand_bits["input"],
                "wgt_precision": operand_bits["weight"],
                "wt_binary_pt_loc": binary_point,
                # Alpha NMP does not consume the PE nonlinear multiplier.
                # The schedule parser still requires these inactive fields.
                "nonlinear_scale_factor": 1,
                "nonlinear_scale_shift": 0,
            }
        )
        if fixed_point["relu"]:
            schedule["is_relu"] = 1
        padding = self.padding_fields(operation, spatial=True)
        schedule.update(padding)
        schedule.update(
            self.spatial_input_geometry_fields(
                name,
                input_shape,
                kernel,
                (output_shape[2], output_shape[3]),
                stride,
                dilation,
                padding,
                failure_code="invalid-dwc-spatial-coverage",
            )
        )
        schedule.update(self.concat_fields([operation]))
        schedule.update(self.nmp_storage_fields(operation))
        schedule.update(nmp_groups)
        result.schedule = schedule

        input_precision = operand_bits["input"]
        weight_precision = operand_bits["weight"]
        input_values = self.load_exact_payload(
            input_tensor,
            name,
            "DWC input",
            input_precision,
        )
        weight_values = self.load_exact_payload(
            weight,
            name,
            "DWC weight",
            weight_precision,
            weight=True,
        )
        output_values = self.load_exact_integer_tensor(
            payload,
            name,
            "DWC output",
            f"int{output_precision}",
            -8 if output_precision == 4 else -128,
            7 if output_precision == 4 else 127,
        )

        for filename, values, precision, weight_payload in (
            (
                "1_inputs.txt",
                input_values,
                input_precision,
                False,
            ),
            (
                "1_weights.txt",
                weight_values,
                weight_precision,
                True,
            ),
            (
                "1_outputs.txt",
                output_values,
                output_precision,
                False,
            ),
        ):
            self.write_payload(
                result,
                filename,
                values,
                precision,
                weight=weight_payload,
            )
        if output_vector is not None:
            output_block_size = self.require_block_size(
                quantizer[1],
                name,
                "DWC output companion scale",
            )
            semantic_values = self.load_exact_integer_tensor(
                output_vector,
                name,
                "DWC output companion scale",
                "uint8",
                0,
                255,
            )
            self.require_shape(
                semantic_values,
                (
                    int(payload.shape[0]),
                    math.ceil(int(payload.shape[1]) / output_block_size),
                    int(payload.shape[2]),
                    int(payload.shape[3]),
                ),
                name,
                "DWC output companion scale",
            )
            output_vector_values = self.load_cmod_vector_scale(
                operation,
                "output_vector_scale_payload",
                name,
                "DWC output companion scale",
                semantic_values,
                kind="output",
            )
            self.write_values(
                result,
                "1_outputs_vector_scale.txt",
                output_vector_values,
                0,
                255,
            )
        elif (
            self.serialized_tensor_role(
                operation,
                "output_vector_scale_payload",
                required=False,
                output_only=False,
            )
            is not None
        ):
            self.fail(
                "unexpected-serialized-artifact",
                name,
                "An output vector-scale artifact requires a semantic "
                "output_vector_scale.",
            )
        return result

    def build_shuffle(
        self,
        operation: Any,
        execution_kind: str,
    ) -> LayerResult:
        name = cmod_utils.operation_name(operation)
        op = cmod_utils.inner_operations(operation)[0]
        input1 = cmod_utils.operation_kwarg(op, "input1")
        input2 = cmod_utils.operation_kwarg(op, "input2")
        output = self.output_payload(operation)
        addresses = cmod_utils.operation_kwarg(op, "output_addresses", [])
        if input1 is None or input2 is None or len(addresses) < 2:
            self.fail(
                "incomplete-shuffle-descriptor",
                name,
                "Shuffle needs two inputs and two physical output addresses.",
            )
        input1_shape = self.require_nmp_tensor_shape(
            input1,
            name,
            "first shuffle input",
        )
        input2_shape = self.require_nmp_tensor_shape(
            input2,
            name,
            "second shuffle input",
        )
        output_shape = self.require_nmp_tensor_shape(
            output,
            name,
            "shuffle output",
        )
        operand_bits = self.require_nmp_input_bits(
            name,
            input1=input1,
            input2=input2,
        )
        output_bits = self.require_precision(
            output,
            name,
            "shuffle output",
        )
        if len({*operand_bits.values(), output_bits}) != 1:
            self.fail(
                "inconsistent-shuffle-precision",
                name,
                "Shuffle input and output precisions must match.",
            )
        groups = self.require_integer(
            cmod_utils.operation_kwarg(op, "groups"),
            name,
            "groups",
            minimum=1,
        )
        if (
            groups != 2
            or input1_shape != input2_shape
            or input1_shape[0] != output_shape[0]
            or input1_shape[2:] != output_shape[2:]
            or 2 * input1_shape[1] != output_shape[1]
        ):
            self.reject_nmp_tensor_extents(
                name,
                execution_kind,
                "two equal inputs and groups=2 must concatenate exactly "
                "into the shuffle output",
            )
        declared_output_address = self.require_address(
            output,
            name,
            "shuffle output payload",
            "output",
            "nmp",
        )
        output_instance = output.scratchpad.instance
        output_instances = tuple(
            self.serialized_value(
                operation,
                f"output{index}_memory_instance",
            )
            for index in (0, 1)
        )
        if output_instances[0] != output_instance:
            self.fail(
                "invalid-shuffle-output-memory-instance",
                name,
                "The primary shuffle output address must identify the "
                "compiler-selected output payload instance "
                f"{output_instance!r}; found {output_instances[0]!r}.",
            )
        for index, instance in enumerate(output_instances):
            self.validate_memory_instance_access(
                instance,
                "output",
                "nmp",
                name,
                f"shuffle output {index}",
            )
        primary_output_address = self.require_integer(
            addresses[0],
            name,
            "output_addresses[0]",
            minimum=0,
        )
        result = LayerResult(
            name=name,
            kind=execution_kind,
            schedule={},
            inputs=self.operation_inputs(operation),
        )
        factors = self.schedule_factors(
            operation,
            cmod_utils.NMP_SCHEDULE_FACTORS,
        )
        lanes = self.nmp_lanes(output_bits)
        if input1_shape[1] % lanes or output_shape[1] % lanes:
            self.reject_nmp_tensor_extents(
                name,
                execution_kind,
                "shuffle input/output channels must cover complete NMP "
                "lane rows",
            )
        self.validate_schedule_semantics(
            name,
            execution_kind,
            factors,
            {
                "p1": output_shape[2],
                "q1": output_shape[3],
                "c1": input1_shape[1] // lanes,
                "k1": output_shape[1] // lanes,
            },
        )
        schedule = self.base_schedule(name)
        schedule.update(
            {
                "is_shuffling": 1,
                "p1": factors["p1"],
                "q1": factors["q1"],
                "c1": factors["c1"],
                "k1": factors["k1"],
                "batch_size": output_shape[0],
                "channels_per_group": output_shape[1] // groups,
                "groups": groups,
                "inputs0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        input1,
                        name,
                        "first input",
                        "activation",
                        "nmp",
                    )
                ),
                "inputs1_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        input2,
                        name,
                        "second input",
                        "activation",
                        "nmp",
                    )
                ),
                "outputs0_baseaddr": cmod_utils.format_address(
                    primary_output_address
                ),
                "outputs1_baseaddr": cmod_utils.format_address(
                    self.require_integer(
                        addresses[1],
                        name,
                        "output_addresses[1]",
                        minimum=0,
                    )
                ),
                "bias_length": 0,
                "nmp_scale_length": 0,
                "nmp_scale_factor": 1,
                "oa_precision": output_bits,
                "oa_scale_precision": 0,
                "ia_precision": operand_bits["input1"],
                "wt_binary_pt_loc": 0,
                "nonlinear_scale_factor": 1,
                "nonlinear_scale_shift": 0,
            }
        )
        schedule.update(self.padding_fields(operation, spatial=False))
        schedule.update(self.concat_fields([operation]))
        schedule.update(self.nmp_storage_fields(operation))
        effective_output_address = primary_output_address + (
            cmod_utils.cmod_concat_output_offset(
                name,
                is_to_concat=int(schedule.get("is_to_concat", 0)),
                concat_index=int(schedule.get("concat_idx_0", 0)),
                k1=int(schedule["k1"]) if "k1" in schedule else None,
            )
        )
        if effective_output_address != declared_output_address:
            self.fail(
                "inconsistent-shuffle-output-address",
                name,
                "CMOD interprets the primary shuffle output base as "
                f"{effective_output_address:#x}, but finalized storage "
                f"places output_payload at {declared_output_address:#x}.",
            )
        result.schedule = schedule
        for filename, tensor in (
            ("1_inputs.txt", input1),
            ("1_inputs2.txt", input2),
            ("1_outputs.txt", output),
        ):
            values = self.load_exact_payload(
                tensor,
                name,
                filename,
                output_bits,
            )
            self.write_payload(
                result,
                filename,
                values,
                output_bits,
            )
        return result

    def build_copy(
        self,
        operation: Any,
        execution_kind: str,
    ) -> LayerResult:
        name = cmod_utils.operation_name(operation)
        op = cmod_utils.inner_operations(operation)[0]
        input_tensor = cmod_utils.operation_kwarg(op, "input")
        output = self.output_payload(operation)
        source_payload = self.serialized_tensor_role(
            operation,
            "copy_source_payload",
            required=True,
            output_only=False,
        )
        destination_payload = self.serialized_tensor_role(
            operation,
            "copy_destination_payload",
            required=True,
            output_only=False,
        )
        block_channels = self.require_integer(
            self.serialized_value(
                operation,
                "block_channels",
            ),
            name,
            "block_channels",
            minimum=1,
            code="invalid-shuffle-copy",
        )
        movement = {
            field: self.require_integer(
                self.serialized_value(operation, field),
                name,
                field,
                minimum=0,
                code="invalid-shuffle-copy",
            )
            for field in (
                "copy_source_output",
                "copy_source_entry",
                "copy_destination_entry",
            )
        }
        if input_tensor is None:
            self.fail(
                "invalid-shuffle-copy",
                name,
                "A physical copy requires an explicit input endpoint.",
            )
        input_shape = self.require_nmp_tensor_shape(
            input_tensor,
            name,
            "copy input",
        )
        output_shape = self.require_nmp_tensor_shape(
            output,
            name,
            "copy output",
        )
        if movement["copy_source_output"] not in (0, 1):
            self.fail(
                "invalid-shuffle-copy",
                name,
                "copy_source_output must select one of two shuffle outputs.",
            )
        input_bits = self.require_nmp_input_bits(
            name,
            input=input_tensor,
        )["input"]
        output_bits = self.require_precision(
            output,
            name,
            "copy output",
        )
        source_bits = self.require_precision(
            source_payload,
            name,
            "copy source backing",
        )
        destination_bits = self.require_precision(
            destination_payload,
            name,
            "copy destination backing",
        )
        if {
            input_bits,
            output_bits,
            source_bits,
            destination_bits,
        } != {8}:
            self.fail(
                "unsupported-copy-precision",
                name,
                "Alpha shuffle copies require bit-preserving INT8 payloads.",
            )
        source_shape = self.require_nmp_tensor_shape(
            source_payload,
            name,
            "copy source backing",
        )
        destination_shape = self.require_nmp_tensor_shape(
            destination_payload,
            name,
            "copy destination backing",
        )
        if (
            source_shape[0] != destination_shape[0]
            or source_shape[2:] != destination_shape[2:]
            or input_shape[0] != source_shape[0]
            or input_shape[2:] != source_shape[2:]
            or output_shape[0] != destination_shape[0]
            or output_shape[2:] != destination_shape[2:]
            or (movement["copy_source_entry"] + 1) * block_channels
            > source_shape[1]
            or (movement["copy_destination_entry"] + 1) * block_channels
            > destination_shape[1]
        ):
            self.fail(
                "invalid-shuffle-copy-payload-view",
                name,
                "The explicit copy entry views are outside their resident "
                "source or destination backings.",
            )
        lanes = self.nmp_lanes(input_bits)
        if block_channels != lanes or output_shape[0] != 1:
            self.reject_nmp_tensor_extents(
                name,
                execution_kind,
                "copy block channels must equal one NMP lane row and "
                "shuffle copies require batch one",
            )
        self.require_address(
            source_payload,
            name,
            "copy source backing",
            "activation",
            "nmp",
        )
        self.require_address(
            destination_payload,
            name,
            "copy destination backing",
            "output",
            "nmp",
        )
        if (
            input_tensor.scratchpad.instance
            != source_payload.scratchpad.instance
            or output.scratchpad.instance
            != destination_payload.scratchpad.instance
        ):
            self.fail(
                "invalid-shuffle-copy-payload-view",
                name,
                "The copy input/output views must use the same "
                "compiler-selected memory instances as their "
                "source/destination backings.",
            )
        block_values = self.load_serialized_artifact(
            operation,
            "copy_block_payload",
            name,
            "encoded shuffle-copy block",
            (
                destination_shape[0],
                block_channels,
                destination_shape[2],
                destination_shape[3],
            ),
            dtype="int8",
            low=-128,
            high=127,
        )
        result = LayerResult(
            name=name,
            kind=execution_kind,
            schedule={},
            inputs=self.operation_inputs(operation),
        )
        factors = self.schedule_factors(
            operation,
            cmod_utils.NMP_SCHEDULE_FACTORS,
        )
        self.validate_schedule_semantics(
            name,
            execution_kind,
            factors,
            {
                "p1": output_shape[2],
                "q1": output_shape[3],
                "c1": 1,
                "k1": 1,
            },
        )
        schedule = self.base_schedule(name)
        schedule.update(
            {
                "c1": factors["c1"],
                # These higher NMP loop fields are identity-valued entries in
                # CMOD's shared schedule schema, not selected mapping factors.
                # The selected NMP factors are the compiler-owned c1/k1/p1/q1
                # values above and below.  nmp_c2/nmp_k2 are separate
                # compiler-owned address-generator fields.
                "c2": 1,
                "c3": 1,
                "k1": factors["k1"],
                "k2": 1,
                "k3": 1,
                "p1": factors["p1"],
                "p2": 1,
                "p3": 1,
                "q1": factors["q1"],
                "q2": 1,
                "q3": 1,
                "stride": 1,
                "is_copy": 1,
                "bias_length": 0,
                "scale_length": 0,
                "wgt_precision": 8,
                "ia_precision": input_bits,
                "oa_precision": 8,
                "wgt_scale_precision": 0,
                "ia_scale_precision": 0,
                "oa_scale_precision": 0,
                "nmp_bias_factor": 0,
                "inputs0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        input_tensor,
                        name,
                        "copy source",
                        "activation",
                        "nmp",
                    )
                ),
                "outputs0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        output,
                        name,
                        "copy destination",
                        "output",
                        "nmp",
                    )
                ),
            }
        )
        schedule.update(self.concat_fields([operation], copy=True))
        schedule.update(self.nmp_storage_fields(operation))
        result.schedule = schedule
        self.write_payload(result, "1_inputs.txt", block_values, 8)
        self.write_payload(result, "1_outputs.txt", block_values, 8)
        return result

    def build_pool(
        self,
        operation: Any,
        execution_kind: str,
    ) -> LayerResult:
        name = cmod_utils.operation_name(operation)
        op = cmod_utils.inner_operations(operation)[0]
        input_tensor = cmod_utils.operation_kwarg(op, "input")
        output = self.output_payload(operation)
        if input_tensor is None:
            self.fail(
                "missing-executable-operand",
                name,
                "missing=input",
            )
        input_shape = self.require_nmp_tensor_shape(
            input_tensor,
            name,
            "pool input",
        )
        output_shape = self.require_nmp_tensor_shape(
            output,
            name,
            "pool output",
        )
        if input_shape[:2] != output_shape[:2]:
            self.reject_nmp_tensor_extents(
                name,
                execution_kind,
                "pooling must preserve batch and channel extents",
            )
        input_bits = self.require_nmp_input_bits(
            name,
            input=input_tensor,
        )["input"]
        output_bits = self.require_precision(
            output,
            name,
            "pool output",
        )
        if output_bits != input_bits:
            self.fail(
                "inconsistent-pool-precision",
                name,
                "Pool input and output precisions must match.",
            )
        kernel = self.require_pair(
            cmod_utils.operation_kwarg(op, "kernel_size"),
            name,
            "kernel_size",
            minimum=1,
        )
        stride = self.require_pair(
            cmod_utils.operation_kwarg(op, "stride"),
            name,
            "stride",
            minimum=1,
        )
        dilation = self.require_pair(
            cmod_utils.operation_kwarg(op, "dilation"),
            name,
            "dilation",
            default=1,
            minimum=1,
        )
        if stride[0] != stride[1] or dilation != (1, 1):
            self.fail(
                "unsupported-pool-spatial-mode",
                name,
                "Alpha CMOD requires symmetric stride and unit dilation.",
            )
        result = LayerResult(
            name=name,
            kind=execution_kind,
            schedule={},
            inputs=self.operation_inputs(operation),
        )
        factors = self.schedule_factors(
            operation,
            cmod_utils.NMP_SCHEDULE_FACTORS,
        )
        lanes = self.nmp_lanes(input_bits)
        self.validate_schedule_semantics(
            name,
            execution_kind,
            factors,
            {
                "p1": input_shape[2],
                "q1": input_shape[3],
                "c1": (input_shape[1] + lanes - 1) // lanes,
                "k1": (output_shape[1] + lanes - 1) // lanes,
            },
        )
        input_address = self.require_address(
            input_tensor,
            name,
            "input",
            "activation",
            "nmp",
        )
        output_address = self.require_address(
            output,
            name,
            "output",
            "output",
            "nmp",
        )
        one_bank = self.require_boolean(
            self.serialized_value(operation, "is_pool_1bank"),
            name,
            "is_pool_1bank",
            code="invalid-serialized-storage-field",
        )
        schedule = self.base_schedule(name)
        schedule.update(
            {
                "is_pool": 1,
                "is_maxpool": 1,
                "is_pool_1bank": int(one_bank),
                "pooling_kernel": kernel[0],
                "pooling_kernel_r": kernel[0],
                "pooling_kernel_s": kernel[1],
                "pooling_stride": stride[0],
                "p_pooled": output_shape[2],
                "q_pooled": output_shape[3],
                "p1": factors["p1"],
                "q1": factors["q1"],
                "c1": factors["c1"],
                "k1": factors["k1"],
                "batch_size": input_shape[0],
                "inputs0_baseaddr": cmod_utils.format_address(input_address),
                "outputs0_baseaddr": cmod_utils.format_address(output_address),
                "bias_length": 0,
                "nmp_scale_length": 0,
                "nmp_scale_factor": 1,
                "nmp_bias_factor": 0,
                "oa_precision": output_bits,
                "oa_scale_precision": 0,
                "ia_precision": input_bits,
                "wt_binary_pt_loc": 0,
                "nonlinear_scale_factor": 1,
                "nonlinear_scale_shift": 0,
            }
        )
        schedule.update(self.concat_fields([operation]))
        schedule.update(self.nmp_storage_fields(operation))
        result.schedule = schedule
        for filename, tensor in (
            ("1_inputs.txt", input_tensor),
            ("1_outputs.txt", output),
        ):
            values = self.load_exact_payload(
                tensor,
                name,
                filename,
                input_bits,
            )
            self.write_payload(
                result,
                filename,
                values,
                input_bits,
            )
        return result

    def build_upsample(
        self,
        operation: Any,
        execution_kind: str,
    ) -> LayerResult:
        name = cmod_utils.operation_name(operation)
        op = cmod_utils.inner_operations(operation)[0]
        input_tensor = cmod_utils.operation_kwarg(op, "input")
        output = self.output_payload(operation)
        if input_tensor is None:
            self.fail(
                "missing-executable-operand",
                name,
                "missing=input",
            )
        input_shape = self.require_nmp_tensor_shape(
            input_tensor,
            name,
            "upsample input",
        )
        output_shape = self.require_nmp_tensor_shape(
            output,
            name,
            "upsample output",
        )
        if input_tensor.dtype != "int8" or output.dtype != "int8":
            self.fail(
                "unsupported-upsample-precision",
                name,
                "Alpha upsample is bit-preserving INT8; INT4 and floating "
                "payloads are unsupported.",
            )
        input_bits = self.require_nmp_input_bits(
            name,
            input=input_tensor,
        )["input"]
        scale_factors = cmod_utils.operation_kwarg(op, "scale_factors")
        if scale_factors is None:
            output_size = cmod_utils.operation_kwarg(op, "output_size")
            try:
                serialized_output_size = tuple(
                    int(value) for value in output_size
                )
            except (TypeError, ValueError):
                serialized_output_size = ()
            if serialized_output_size != tuple(output_shape[2:]):
                self.fail(
                    "unsupported-upsample-factor",
                    name,
                    "Alpha CMOD requires output_size to match the serialized "
                    "output tensor when scale_factors are omitted.",
                )
            scale_factors = tuple(
                output_extent / input_extent
                for input_extent, output_extent in zip(
                    input_shape[2:],
                    output_shape[2:],
                )
            )
        try:
            factors_float = tuple(float(value) for value in scale_factors)
        except (TypeError, ValueError):
            factors_float = ()
        if (
            len(factors_float) != 2
            or factors_float[0] != factors_float[1]
            or not math.isfinite(factors_float[0])
            or factors_float[0] <= 0.0
            or not factors_float[0].is_integer()
        ):
            self.fail(
                "unsupported-upsample-factor",
                name,
                "Alpha CMOD requires one positive integer nearest-neighbor "
                "factor for both spatial dimensions.",
            )
        factor = int(factors_float[0])
        if (
            input_shape[:2] != output_shape[:2]
            or output_shape[2] != input_shape[2] * factor
            or output_shape[3] != input_shape[3] * factor
        ):
            self.reject_nmp_tensor_extents(
                name,
                execution_kind,
                "nearest upsample must preserve N/C and multiply both "
                "spatial extents by the serialized scale factor",
            )
        result = LayerResult(
            name=name,
            kind=execution_kind,
            schedule={},
            inputs=self.operation_inputs(operation),
        )
        factors = self.schedule_factors(
            operation,
            cmod_utils.NMP_SCHEDULE_FACTORS,
        )
        lanes = self.nmp_lanes(input_bits)
        channel_groups = (output_shape[1] + lanes - 1) // lanes
        self.validate_schedule_semantics(
            name,
            execution_kind,
            factors,
            {
                "p1": output_shape[2],
                "q1": output_shape[3],
                "c1": channel_groups,
                "k1": channel_groups,
            },
        )
        schedule = self.base_schedule(name)
        schedule.update(
            {
                "is_upsampling": 1,
                "r": factor,
                "s": factor,
                "upsample_kernel": factor,
                "p1": factors["p1"],
                "q1": factors["q1"],
                "c1": factors["c1"],
                "k1": factors["k1"],
                "batch_size": input_shape[0],
                "inputs0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        input_tensor,
                        name,
                        "input",
                        "activation",
                        "nmp",
                    )
                ),
                "outputs0_baseaddr": cmod_utils.format_address(
                    self.require_address(
                        output,
                        name,
                        "output",
                        "output",
                        "nmp",
                    )
                ),
                "bias_length": 0,
                "nmp_scale_length": 0,
                "nmp_scale_factor": 1,
                "oa_precision": 8,
                "oa_scale_precision": 0,
                "ia_precision": input_bits,
                "wt_binary_pt_loc": 0,
                "nonlinear_scale_factor": 1,
                "nonlinear_scale_shift": 0,
            }
        )
        schedule.update(self.padding_fields(operation, spatial=False))
        schedule.update(self.concat_fields([operation]))
        schedule.update(self.nmp_storage_fields(operation))
        result.schedule = schedule
        input_values = self.load_exact_payload(
            input_tensor,
            name,
            "upsample input",
            8,
        )
        output_values = self.load_exact_payload(
            output,
            name,
            "upsample output",
            8,
        )
        expected = np.repeat(
            np.repeat(input_values, factor, axis=2),
            factor,
            axis=3,
        )
        if not np.array_equal(output_values, expected):
            self.fail(
                "non-bit-preserving-upsample",
                name,
                "The serialized upsample output is not the exact nearest "
                "neighbor repetition of its INT8 input.",
            )
        self.write_payload(result, "1_inputs.txt", input_values, 8)
        self.write_payload(result, "1_outputs.txt", output_values, 8)
        return result

    def _has_legacy_copy_contract(self, operation: Any) -> bool:
        """Recognize old ``add`` copies only from complete physical facts."""
        fields = {
            field
            for inner in cmod_utils.inner_operations(operation)
            for field in inner.kwargs
        }
        return _LEGACY_COPY_FIELDS <= fields

    def _execution_kind_from_targets(
        self,
        operation: Any,
    ) -> str | None:
        """Select one CMOD builder from the canonical target sequence."""
        targets = tuple(
            inner.target for inner in cmod_utils.inner_operations(operation)
        )
        if len(targets) == 1:
            target = targets[0]
            kind = _CMOD_KIND_BY_SINGLE_TARGET.get(target)
            if kind is not None:
                return kind
            if (
                target == cmod_utils.CMOD_LEGACY_COPY_TARGET
                and self._has_legacy_copy_contract(operation)
            ):
                return "nmp_copy"
            return None

        if not targets or targets[0] not in (
            cmod_utils.CMOD_CONVOLUTION_TARGETS
        ):
            return None
        suffix = targets[1:]
        if suffix == ("ppu_fixed_relu", "saturating_add"):
            return (
                "gemm_convolution"
                if targets[0] == "conv2d_int8_accumulator"
                else None
            )
        if suffix in {
            ("ppu_fixed_relu",),
            ("ppu_fixed_relu", "pvs_quantize"),
        }:
            return "gemm_convolution"
        if targets[0] == "conv2d_int8_accumulator" and suffix in {
            ("nmp_fixed_requantize",),
            ("nmp_fixed_requantize", "pvs_quantize"),
        }:
            return "nmp_depthwise_convolution"
        return None

    @staticmethod
    def _is_serialized_nop(operation: Any) -> bool:
        """Return whether the compiler explicitly emitted a NOP record."""
        inner = cmod_utils.inner_operations(operation)
        return bool(inner) and all(op.op == "nop" for op in inner)

    @staticmethod
    def _nop_skip_reason(target: str) -> str:
        """Project one diagnostic CMOD skip reason from a canonical NOP."""
        if target in cmod_utils.CMOD_HOST_OUTPUT_LAYOUT_TARGETS:
            return cmod_utils.CMOD_SKIP_REASON_HOST_OUTPUT_LAYOUT
        if target == "view":
            return "shared_address_only"
        return "target_address_only"

    def _parse_execution_record(
        self,
        operation: Any,
    ) -> _ExecutionRecord:
        """Derive one route without resolving cross-record Q/DQ ownership."""
        name = cmod_utils.operation_name(operation)
        target = cmod_utils.operation_target(operation)
        inner_operations = cmod_utils.inner_operations(operation)
        if not inner_operations:
            self.fail(
                "unsupported-cmod-record",
                name,
                "CMOD conversion requires an ordinary or fused operation.",
            )
        unsupported_markers = tuple(
            inner.op == _SERIALIZED_UNSUPPORTED_OPERATION
            for inner in inner_operations
        )
        if any(unsupported_markers):
            if not all(unsupported_markers):
                self.fail(
                    "invalid-serialized-unsupported-record",
                    name,
                    "A fused unsupported owner must mark every executable "
                    "member as unsupported.",
                )
            kind = None
            disposition = "unsupported"
            skip_reason = "serialized_unsupported"
        elif self._is_serialized_nop(operation):
            kind = None
            disposition = "non_executable"
            skip_reason = self._nop_skip_reason(target)
        else:
            kind = self._execution_kind_from_targets(operation)
            if kind is not None:
                disposition = "supported"
                skip_reason = None
            elif (
                operation.WhichOneof("op_type") == "op"
                and len(inner_operations) == 1
                and target in cmod_utils.CMOD_FUSIBLE_TARGETS
            ):
                disposition = _FUSION_CANDIDATE
                skip_reason = None
            elif (
                operation.WhichOneof("op_type") == "op"
                and len(inner_operations) == 1
                and target in cmod_utils.CMOD_HOST_OUTPUT_LAYOUT_TARGETS
            ):
                disposition = _HOST_LAYOUT_CANDIDATE
                skip_reason = None
            elif (
                operation.WhichOneof("op_type") == "op"
                and len(inner_operations) == 1
                and target == cmod_utils.CMOD_LEGACY_COPY_TARGET
            ):
                disposition = "unsupported"
                skip_reason = "unsupported_cmod_target"
            else:
                disposition = "unsupported"
                skip_reason = "unsupported_cmod_target"

        return _ExecutionRecord(
            operation=operation,
            name=name,
            target=target,
            disposition=disposition,
            kind=kind,
            skip_reason=skip_reason,
            fusion_owners=(),
        )

    @staticmethod
    def _record_outputs(record: _ExecutionRecord) -> tuple[str, ...]:
        return tuple(
            tensor.node
            for tensor in cmod_utils.output_tensors(record.operation)
            if tensor.node
        )

    def _record_inputs(self, record: _ExecutionRecord) -> tuple[str, ...]:
        return tuple(self.operation_inputs(record.operation))

    @staticmethod
    def _primary_input_node(record: _ExecutionRecord) -> str | None:
        inner = cmod_utils.inner_operations(record.operation)
        if len(inner) != 1:
            return None
        value = cmod_utils.operation_kwarg(inner[0], "input")
        if hasattr(value, "node") and value.node:
            return value.node
        for argument in inner[0].args:
            value = cmod_utils.argument_value(argument)
            if hasattr(value, "node") and value.node:
                return value.node
        return None

    def _fusion_owners_for_record(
        self,
        candidate: _ExecutionRecord,
        records: Sequence[_ExecutionRecord],
    ) -> tuple[_ExecutionRecord, ...]:
        """Derive every tiled public-boundary Q/DQ execution owner."""
        outputs_by_node: dict[str, list[_ExecutionRecord]] = defaultdict(list)
        consumers_by_node: dict[str, list[_ExecutionRecord]] = defaultdict(list)
        for record in records:
            for node in self._record_outputs(record):
                outputs_by_node[node].append(record)
            for node in self._record_inputs(record):
                consumers_by_node[node].append(record)

        primary_input = self._primary_input_node(candidate)
        output_nodes = self._record_outputs(candidate)
        if primary_input is None or len(output_nodes) != 1:
            return ()

        def ordered_owners(found: dict[int, _ExecutionRecord]):
            return tuple(found.values())

        if candidate.target == "quantize":
            public_inputs = {
                tensor.node
                for tensor in getattr(self.model, "inputs", ())
                if tensor.node
            }
            pending_inputs = [primary_input]
            seen_inputs = set()
            reaches_public_input = False
            while pending_inputs:
                node = pending_inputs.pop()
                if node in seen_inputs:
                    continue
                seen_inputs.add(node)
                if node in public_inputs:
                    reaches_public_input = True
                    continue
                producers = outputs_by_node.get(node, ())
                if len(producers) != 1:
                    return ()
                producer = producers[0]
                if producer.disposition != "non_executable":
                    return ()
                pending_inputs.extend(self._record_inputs(producer))
            if not reaches_public_input:
                return ()

            owners: dict[int, _ExecutionRecord] = {}
            pending_outputs = list(output_nodes)
            seen_outputs = set()
            while pending_outputs:
                node = pending_outputs.pop()
                if node in seen_outputs:
                    continue
                seen_outputs.add(node)
                consumers = consumers_by_node.get(node, ())
                if not consumers:
                    return ()
                for consumer in consumers:
                    if consumer.disposition == "supported":
                        owners[id(consumer.operation)] = consumer
                    elif consumer.disposition == "non_executable":
                        pending_outputs.extend(self._record_outputs(consumer))
                    else:
                        return ()
            return ordered_owners(owners)

        if candidate.target != "dequantize":
            return ()
        owners = {}
        pending_inputs = [primary_input]
        seen_inputs = set()
        while pending_inputs:
            node = pending_inputs.pop()
            if node in seen_inputs:
                continue
            seen_inputs.add(node)
            producers = outputs_by_node.get(node, ())
            if len(producers) != 1:
                return ()
            producer = producers[0]
            if producer.disposition == "supported":
                owners[id(producer.operation)] = producer
            elif producer.disposition == "non_executable":
                pending_inputs.extend(self._record_inputs(producer))
            else:
                return ()

        pending_outputs = list(output_nodes)
        seen_outputs = set()
        while pending_outputs:
            node = pending_outputs.pop()
            if node in seen_outputs:
                continue
            seen_outputs.add(node)
            for consumer in consumers_by_node.get(node, ()):
                if consumer.disposition not in {
                    "non_executable",
                    _HOST_LAYOUT_CANDIDATE,
                }:
                    return ()
                pending_outputs.extend(self._record_outputs(consumer))
        return ordered_owners(owners)

    def _resolve_fusion_owners(
        self,
        records: Sequence[_ExecutionRecord],
    ) -> list[_ExecutionRecord]:
        """Resolve public-boundary Q/DQ against its supported tile owners."""
        names = Counter(record.name for record in records)
        resolved = []
        for record in records:
            if record.disposition != _FUSION_CANDIDATE:
                resolved.append(record)
                continue
            owners = self._fusion_owners_for_record(record, records)
            if not owners or any(names[owner.name] != 1 for owner in owners):
                resolved.append(record)
                continue
            resolved.append(
                replace(
                    record,
                    disposition="fused",
                    fusion_owners=tuple(owner.name for owner in owners),
                )
            )
        return resolved

    def _resolve_host_output_layouts(
        self,
        records: Sequence[_ExecutionRecord],
    ) -> list[_ExecutionRecord]:
        """Prove host layout records form a terminal fused-DQ suffix."""
        outputs_by_node: dict[str, list[_ExecutionRecord]] = defaultdict(list)
        consumers_by_node: dict[str, list[_ExecutionRecord]] = defaultdict(list)
        for record in records:
            for node in self._record_outputs(record):
                outputs_by_node[node].append(record)
            for node in self._record_inputs(record):
                consumers_by_node[node].append(record)

        def is_terminal_suffix(candidate: _ExecutionRecord) -> bool:
            pending = list(self._record_outputs(candidate))
            seen = set()
            if not pending:
                return False
            while pending:
                node = pending.pop()
                if node in seen:
                    continue
                seen.add(node)
                for consumer in consumers_by_node.get(node, ()):
                    if consumer.disposition not in {
                        "non_executable",
                        _HOST_LAYOUT_CANDIDATE,
                    }:
                        return False
                    pending.extend(self._record_outputs(consumer))
            return True

        def has_fused_dequantize_root(candidate: _ExecutionRecord) -> bool:
            primary_input = self._primary_input_node(candidate)
            if primary_input is None:
                return False
            pending = [primary_input]
            seen = set()
            roots: dict[int, _ExecutionRecord] = {}
            while pending:
                node = pending.pop()
                if node in seen:
                    continue
                seen.add(node)
                producers = outputs_by_node.get(node, ())
                if len(producers) != 1:
                    return False
                producer = producers[0]
                if (
                    producer.disposition == "fused"
                    and producer.target == "dequantize"
                ):
                    roots[id(producer.operation)] = producer
                elif producer.disposition in {
                    "non_executable",
                    _HOST_LAYOUT_CANDIDATE,
                }:
                    pending.extend(self._record_inputs(producer))
                else:
                    return False
            return len(roots) == 1

        resolved = []
        for record in records:
            if record.disposition != _HOST_LAYOUT_CANDIDATE:
                resolved.append(record)
                continue
            if is_terminal_suffix(record) and has_fused_dequantize_root(record):
                resolved.append(
                    replace(
                        record,
                        disposition="non_executable",
                        skip_reason=(
                            cmod_utils.CMOD_SKIP_REASON_HOST_OUTPUT_LAYOUT
                        ),
                    )
                )
            else:
                resolved.append(record)
        return resolved

    def _execution_kind_for_record(
        self,
        record: _ExecutionRecord,
    ) -> str | None:
        """Return one builder kind or report a conversion coverage gap."""
        if record.disposition == _FUSION_CANDIDATE:
            self.fail(
                "unresolved-cmod-fusion-owner",
                record.name,
                f"{record.target}: no unique public-boundary execution "
                "owner was derivable from tensor dataflow.",
            )
        if record.disposition == _HOST_LAYOUT_CANDIDATE:
            self.fail(
                "unsupported-cmod-host-layout",
                record.name,
                f"{record.target}: layout is not a terminal suffix after "
                "a fused public dequantizer.",
            )
        if record.disposition == "unsupported":
            if record.skip_reason == "serialized_unsupported":
                return None
            self.fail(
                "unsupported-cmod-target",
                record.name,
                f"{record.target}: no registered Alpha CMOD translation",
            )
        return record.kind if record.disposition == "supported" else None

    def _execution_scope(
        self,
        records: Sequence[_ExecutionRecord],
    ) -> dict[str, Any]:
        """Describe supported islands and explicit unsupported boundaries."""
        outputs_by_node: dict[str, list[_ExecutionRecord]] = defaultdict(list)
        consumers_by_node: dict[str, list[_ExecutionRecord]] = defaultdict(list)
        order = {id(record): index for index, record in enumerate(records)}
        supported = [
            record for record in records if record.disposition == "supported"
        ]
        unsupported = [
            record for record in records if record.disposition == "unsupported"
        ]
        for record in records:
            for node in self._record_outputs(record):
                outputs_by_node[node].append(record)
            for node in self._record_inputs(record):
                consumers_by_node[node].append(record)

        transparent = frozenset({"fused", "non_executable"})

        def upstream(
            node: str,
            seen: set[tuple[str, int]],
        ) -> dict[int, _ExecutionRecord]:
            found: dict[int, _ExecutionRecord] = {}
            for producer in outputs_by_node.get(node, ()):
                key = (node, id(producer))
                if key in seen:
                    continue
                next_seen = {*seen, key}
                if producer.disposition in transparent:
                    for input_node in self._record_inputs(producer):
                        found.update(upstream(input_node, next_seen))
                else:
                    found[id(producer)] = producer
            return found

        def downstream(
            node: str,
            seen: set[tuple[str, int]],
        ) -> dict[int, _ExecutionRecord]:
            found: dict[int, _ExecutionRecord] = {}
            for consumer in consumers_by_node.get(node, ()):
                key = (node, id(consumer))
                if key in seen:
                    continue
                next_seen = {*seen, key}
                if consumer.disposition in transparent:
                    for output_node in self._record_outputs(consumer):
                        found.update(downstream(output_node, next_seen))
                else:
                    found[id(consumer)] = consumer
            return found

        adjacency = {id(record): set() for record in supported}
        external_inputs: dict[tuple[str, str, str], dict[str, str]] = {}
        external_outputs: dict[tuple[str, str, str], dict[str, str]] = {}
        for record in supported:
            record_id = id(record)
            for node in self._record_inputs(record):
                for producer_id, producer in upstream(node, set()).items():
                    if producer.disposition == "supported":
                        adjacency[record_id].add(producer_id)
                        adjacency[producer_id].add(record_id)
                    elif producer.disposition == "unsupported":
                        key = (node, producer.name, record.name)
                        external_inputs[key] = {
                            "tensor": node,
                            "producer": producer.name,
                            "producer_target": producer.target,
                            "consumer": record.name,
                        }
            for node in self._record_outputs(record):
                for consumer_id, consumer in downstream(node, set()).items():
                    if consumer.disposition == "supported":
                        adjacency[record_id].add(consumer_id)
                        adjacency[consumer_id].add(record_id)
                    elif consumer.disposition == "unsupported":
                        key = (node, record.name, consumer.name)
                        external_outputs[key] = {
                            "tensor": node,
                            "producer": record.name,
                            "consumer": consumer.name,
                            "consumer_target": consumer.target,
                        }

        islands = []
        pending = set(adjacency)
        while pending:
            root = min(pending, key=order.__getitem__)
            component = set()
            stack = [root]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(adjacency[current] - component)
            pending -= component
            islands.append(
                [
                    records[order[record_id]].name
                    for record_id in sorted(component, key=order.__getitem__)
                ]
            )

        return {
            "unsupported": [
                {
                    "name": record.name,
                    "target": record.target,
                    "inputs": list(self._record_inputs(record)),
                    "outputs": list(self._record_outputs(record)),
                }
                for record in unsupported
            ],
            "execution_islands": [
                {"index": index, "layers": layers}
                for index, layers in enumerate(islands)
            ],
            "external_inputs": [
                external_inputs[key] for key in sorted(external_inputs)
            ],
            "external_outputs": [
                external_outputs[key] for key in sorted(external_outputs)
            ],
        }

    def execution_kind(
        self,
        operation: Any,
    ) -> str | None:
        """Return one independently derivable target kind."""
        record = self._parse_execution_record(operation)
        return self._execution_kind_for_record(record)

    def validate_result(self, result: LayerResult) -> None:
        """Check the emitted directory against the C-model file contract."""
        directory = self.output_dir / result.name
        required = {
            "schedule0.txt",
            "schedule1.txt",
            "schedule10.txt",
            "1_inputs.txt",
            "1_outputs.txt",
        }
        if result.kind in {
            "gemm_convolution",
            "nmp_depthwise_convolution",
        }:
            required.add("1_weights.txt")
        if result.kind == "nmp_channel_shuffle":
            required.add("1_inputs2.txt")
        if result.schedule.get("is_eadd") == 1:
            if result.schedule.get("num_eop") != 1:
                self.error(
                    "invalid-eadd-output-count",
                    result.name,
                    "Alpha PE EADD requires num_eop=1",
                )
            input_address = result.schedule.get("inputs1_baseaddr")
            output_address = result.schedule.get("outputs0_baseaddr")
            try:
                input_address_value = int(str(input_address), 0)
                output_address_value = int(str(output_address), 0)
            except (TypeError, ValueError):
                input_address_value = output_address_value = None
            if (
                input_address_value is None
                or input_address_value != output_address_value
            ):
                self.error(
                    "invalid-eadd-output-alias",
                    result.name,
                    "Alpha PE EADD requires inputs1_baseaddr to equal "
                    "outputs0_baseaddr exactly",
                )
            required.add("1_inputs2.txt")
        if int(result.schedule.get("scale_length", 0)) > 0:
            required.add("1_scale.txt")
        if int(result.schedule.get("nmp_scale_length", 0)) > 0:
            required.add("1_scale.txt")
        if int(result.schedule.get("bias_length", 0)) > 0:
            required.add("1_bias.txt")
        if int(result.schedule.get("wgt_scale_precision", 0)) > 0:
            required.add("1_weights_vector_scale.txt")
        if int(result.schedule.get("ia_scale_precision", 0)) > 0:
            required.add("1_inputs_vector_scale.txt")
        if int(result.schedule.get("oa_scale_precision", 0)) > 0:
            required.add("1_outputs_vector_scale.txt")
        if not self.write_tensors:
            required = {
                name for name in required if name.startswith("schedule")
            }
        missing = sorted(
            name for name in required if not (directory / name).is_file()
        )
        if missing:
            self.error(
                "incomplete-cmodel-directory",
                result.name,
                f"missing={','.join(missing)}",
            )

        address_fields = (
            key for key in result.schedule if key.endswith("_baseaddr")
        )
        for key in address_fields:
            value = str(result.schedule[key])
            try:
                address = int(value, 0)
            except ValueError:
                address = -1
            if address < 0x800000:
                self.error(
                    "invalid-alpha-address",
                    result.name,
                    f"{key}={value}; expected encoded global address",
                )

    def convert(self) -> dict[str, Any]:
        builders = {
            kind: getattr(self, method)
            for kind, method in CMOD_EXECUTION_BUILDERS.items()
        }
        records = []
        for operation in self.operations:
            try:
                records.append(self._parse_execution_record(operation))
            except SkipLayer:
                continue
        records = self._resolve_fusion_owners(records)
        records = self._resolve_host_output_layouts(records)

        skipped: Counter[str] = Counter()
        skipped_by_reason: Counter[str] = Counter()
        skipped_by_reason_and_target: dict[
            str,
            Counter[str],
        ] = defaultdict(Counter)
        fused: Counter[str] = Counter()
        fused_by_owner: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            try:
                kind = self._execution_kind_for_record(record)
                if record.disposition == "fused":
                    fused[record.target] += 1
                    for owner in record.fusion_owners:
                        fused_by_owner[owner][record.target] += 1
                    continue
                if kind is None:
                    target = record.target
                    reason = record.skip_reason
                    skipped[target] += 1
                    skipped_by_reason[reason] += 1
                    skipped_by_reason_and_target[reason][target] += 1
                    continue
                result = builders[kind](record.operation, kind)
            except SkipLayer:
                continue
            self.emit_schedule(result)
            self.results.append(result)

        supported_record_count = sum(
            record.disposition == "supported" for record in records
        )
        unsupported_count = sum(
            record.disposition == "unsupported" for record in records
        )
        if not self.results:
            self.error(
                "no-executable-cmod-layers",
                "model",
                "no supported executable layer was emitted",
            )
        for result in self.results:
            self.validate_result(result)

        counts = Counter(result.kind for result in self.results)
        scope = self._execution_scope(records)
        has_errors = any(issue.severity == "error" for issue in self.issues)
        all_supported_emitted = (
            not has_errors and len(self.results) == supported_record_count
        )
        all_model_operations_supported = (
            all_supported_emitted and unsupported_count == 0
        )
        connected_alpha_program = (
            all_model_operations_supported
            and len(scope["execution_islands"]) <= 1
        )
        report = {
            "schema_version": CONVERSION_REPORT_SCHEMA_VERSION,
            "status": "failed" if has_errors else "complete",
            "source": str(self.model_dir),
            "output": str(self.output_dir),
            "network": self.network,
            "claims": {
                "all_supported_operations_emitted": all_supported_emitted,
                "all_model_operations_supported": (
                    all_model_operations_supported
                ),
                "connected_alpha_program": connected_alpha_program,
            },
            "execution_scope": scope,
            "summary": {
                "emitted_layers": len(self.results),
                "emitted_by_kind": dict(sorted(counts.items())),
                "supported_execution_records": supported_record_count,
                "unsupported_operations": unsupported_count,
                "unsupported_by_target": dict(
                    sorted(
                        Counter(
                            record.target
                            for record in records
                            if record.disposition == "unsupported"
                        ).items()
                    )
                ),
                "execution_island_count": len(scope["execution_islands"]),
                "external_input_count": len(scope["external_inputs"]),
                "external_output_count": len(scope["external_outputs"]),
                "fused_graph_ops": dict(sorted(fused.items())),
                "fused_graph_ops_by_owner": {
                    owner: dict(sorted(targets.items()))
                    for owner, targets in sorted(fused_by_owner.items())
                },
                "skipped_graph_ops": dict(sorted(skipped.items())),
                "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
                "skipped_by_reason_and_target": {
                    reason: dict(sorted(targets.items()))
                    for reason, targets in sorted(
                        skipped_by_reason_and_target.items()
                    )
                },
                "issues_by_severity": dict(
                    sorted(Counter(i.severity for i in self.issues).items())
                ),
            },
            "layers": [asdict(result) for result in self.results],
            "issues": [asdict(issue) for issue in self.issues],
        }
        (self.output_dir / "conversion_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        (self.output_dir / "layers.txt").write_text(
            "".join(f"{result.name}\n" for result in self.results)
        )
        return report


def _resolve_model_dir(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        path = repo_root / "models" / value
    if path.is_file() and path.name == "model.txt":
        path = path.parent
    if not (path / "model.txt").is_file():
        raise FileNotFoundError(f"No model.txt in {path}")
    return path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        help="Voyager model directory, model.txt, or name under models/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <model>/cmod).",
    )
    parser.add_argument("--network", help="Schedule network name.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--no-tensors",
        action="store_true",
        help="Emit schedules and diagnostics without text tensor files.",
    )
    parser.add_argument(
        "--require-all-supported",
        action="store_true",
        help="Fail unless every executable model operation is Alpha-supported.",
    )
    parser.add_argument(
        "--require-connected-program",
        action="store_true",
        help=(
            "Fail unless supported operations form one complete Alpha "
            "program."
        ),
    )
    return parser.parse_args()


def _append_report_error(
    report: dict[str, Any],
    code: str,
    message: str,
) -> None:
    """Add one command-line acceptance error to a conversion report."""
    report["issues"].append(
        {
            "severity": "error",
            "code": code,
            "layer": "model",
            "message": message,
        }
    )
    severities = report["summary"]["issues_by_severity"]
    severities["error"] = int(severities.get("error", 0)) + 1
    report["status"] = "failed"


def _remove_path(path: Path) -> None:
    """Remove a known stage or backup without following directory links."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _write_failure_report(output: Path, report: dict[str, Any]) -> Path:
    """Publish diagnostics separately from runnable converted collateral."""
    failure_path = output.with_name(f"{output.name}.failure.json")
    temporary = failure_path.with_name(
        f".{failure_path.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(failure_path)
    return failure_path


def _publish_staged_output(
    staging: Path,
    output: Path,
    *,
    replace: bool,
) -> Path | None:
    """Atomically publish a complete stage while preserving old output."""
    if (output.exists() or output.is_symlink()) and not replace:
        raise FileExistsError(f"output exists: {output}")

    backup = None
    if output.exists() or output.is_symlink():
        backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
        output.replace(backup)
    try:
        staging.replace(output)
    except BaseException:
        if backup is not None and not output.exists():
            backup.replace(output)
        raise
    if backup is not None:
        try:
            _remove_path(backup)
        except OSError:
            # Publication has already completed atomically. Keep the
            # recoverable old tree and report it to the caller rather than
            # incorrectly marking the new complete output as failed.
            return backup
    return None


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    try:
        model_dir = _resolve_model_dir(args.model, repo_root)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else model_dir / "cmod"
    )
    if (output.exists() or output.is_symlink()) and not args.force:
        print(
            f"error: output exists: {output}; pass --force to replace it",
            file=sys.stderr,
        )
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.stage-",
            dir=output.parent,
        )
    )
    try:
        converter = _Converter(
            model_dir=model_dir,
            output_dir=staging,
            network=args.network or model_dir.name,
            write_tensors=not args.no_tensors,
        )
        report = converter.convert()
    except Exception as error:
        report = {
            "schema_version": CONVERSION_REPORT_SCHEMA_VERSION,
            "status": "failed",
            "source": str(model_dir),
            "output": str(output),
            "claims": {
                "all_supported_operations_emitted": False,
                "all_model_operations_supported": False,
                "connected_alpha_program": False,
            },
            "execution_scope": {
                "unsupported": [],
                "execution_islands": [],
                "external_inputs": [],
                "external_outputs": [],
            },
            "summary": {
                "emitted_layers": 0,
                "emitted_by_kind": {},
                "supported_execution_records": 0,
                "unsupported_operations": 0,
                "unsupported_by_target": {},
                "execution_island_count": 0,
                "external_input_count": 0,
                "external_output_count": 0,
                "fused_graph_ops": {},
                "fused_graph_ops_by_owner": {},
                "skipped_graph_ops": {},
                "skipped_by_reason": {},
                "skipped_by_reason_and_target": {},
                "issues_by_severity": {"error": 1},
            },
            "layers": [],
            "issues": [
                {
                    "severity": "error",
                    "code": "conversion-exception",
                    "layer": "",
                    "message": f"{type(error).__name__}: {error}",
                }
            ],
        }
        failure_path = _write_failure_report(output, report)
        _remove_path(staging)
        print(
            f"error: conversion failed; diagnostics: {failure_path}",
            file=sys.stderr,
        )
        return 1

    claims = report["claims"]
    if (
        getattr(args, "require_all_supported", False)
        and not claims["all_model_operations_supported"]
    ):
        _append_report_error(
            report,
            "incomplete-alpha-operation-coverage",
            "The command requires every executable model operation to be "
            "Alpha-supported.",
        )
    if (
        getattr(args, "require_connected_program", False)
        and not claims["connected_alpha_program"]
    ):
        _append_report_error(
            report,
            "disconnected-alpha-program",
            "The command requires one complete connected Alpha program.",
        )

    summary = report["summary"]
    errors = summary["issues_by_severity"].get("error", 0)
    report["output"] = str(output)
    report["status"] = "failed" if errors else "complete"
    if errors:
        failure_path = _write_failure_report(output, report)
        _remove_path(staging)
        print(
            f"error: conversion produced {errors} error(s); "
            f"diagnostics: {failure_path}",
            file=sys.stderr,
        )
        return 1

    (staging / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    try:
        retained_backup = _publish_staged_output(
            staging,
            output,
            replace=args.force,
        )
    except Exception as error:
        report["status"] = "failed"
        report["summary"]["issues_by_severity"]["error"] = 1
        report["issues"].append(
            {
                "severity": "error",
                "code": "publication-failed",
                "layer": "",
                "message": f"{type(error).__name__}: {error}",
            }
        )
        failure_path = _write_failure_report(output, report)
        _remove_path(staging)
        print(
            f"error: conversion publication failed; "
            f"diagnostics: {failure_path}",
            file=sys.stderr,
        )
        return 1

    output.with_name(f"{output.name}.failure.json").unlink(missing_ok=True)
    if retained_backup is not None:
        print(
            "warning: conversion was published, but the previous output "
            f"could not be removed: {retained_backup}",
            file=sys.stderr,
        )
    print(f"Emitted {summary['emitted_layers']} layers into {output}")
    print(f"Layer kinds: {summary['emitted_by_kind']}")
    print(f"Issues: {summary['issues_by_severity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
