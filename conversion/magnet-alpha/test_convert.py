#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Focused tests for dump-only Magnet Alpha conversion."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import convert
import numpy as np
import utils as cmod_utils
from google.protobuf import text_format


def _hardware_config():
    return {
        "schema_version": 1,
        "config_type": "MagnetAlphaHardwareConfig",
        "target": "magnet-alpha",
        "compute_unit_bindings": {
            "matrix": "vector_mac",
            "nmp": "nmp",
        },
        "compute_units": [
            {
                "name": "vector_mac",
                "accepted_input_bits": [4, 8],
                "unroll_dimensions": [
                    {
                        "name": "IC",
                        "extent": 32,
                        "dtype_extents": [{"input_bits": 8, "extent": 16}],
                    },
                    {
                        "name": "OC",
                        "extent": 32,
                        "dtype_extents": [],
                    },
                ],
            },
            {
                "name": "nmp",
                "accepted_input_bits": [8],
                "unroll_dimensions": [
                    {
                        "name": "CHANNEL",
                        "extent": 32,
                        "dtype_extents": [],
                    }
                ],
            },
        ],
        "global_memory": {
            "name": "global",
            "instances": [
                {
                    "name": "global_activation",
                    "bank_count": 64,
                    "depth_per_bank": 1024,
                    "bits_per_bank_entry": 128,
                    "accepted_accesses": [
                        {
                            "role": "activation",
                            "compute_unit": "vector_mac",
                        },
                        {
                            "role": "activation",
                            "compute_unit": "nmp",
                        },
                        {
                            "role": "output",
                            "compute_unit": None,
                        },
                        {
                            "role": "weight",
                            "compute_unit": "nmp",
                        },
                    ],
                },
                {
                    "name": "global_weight",
                    "bank_count": 16,
                    "depth_per_bank": 1024,
                    "bits_per_bank_entry": 512,
                    "accepted_accesses": [
                        {
                            "role": "weight",
                            "compute_unit": "vector_mac",
                        },
                    ],
                },
            ],
            "input_memory": "global_activation",
            "output_memory": "global_activation",
            "weight_memory": "global_weight",
        },
    }


def _set_schedule_factors(operation, factors):
    owner = cmod_utils.inner_operations(operation)[0]
    for name, value in factors.items():
        argument = owner.kwargs[f"schedule_factor_{name}"]
        if isinstance(value, bool):
            argument.bool_value = value
        elif isinstance(value, int):
            argument.int_value = value
        else:
            argument.float_value = value


def _set_legacy_cmod_metadata(
    operation,
    disposition,
    *,
    kind=None,
    skip_reason=None,
    fusion_owner=None,
):
    owner = cmod_utils.inner_operations(operation)[0]
    owner.kwargs["cmod_execution_disposition"].str_value = disposition
    if kind is not None:
        owner.kwargs["cmod_execution_kind"].str_value = kind
    if disposition == "non_executable" and skip_reason is None:
        skip_reason = "shared_address_only"
    if skip_reason is not None:
        owner.kwargs["cmod_skip_reason"].str_value = skip_reason
    if fusion_owner is not None:
        owner.kwargs["cmod_fusion_owner"].str_value = fusion_owner


def _gemm_schedule_factors():
    return {
        "batch_size": 1,
        "c0": 16,
        "c1": 1,
        "c2": 1,
        "c3": 1,
        "k0": 32,
        "k1": 1,
        "k2": 1,
        "k3": 1,
        "p0": 1,
        "p1": 1,
        "p2": 2,
        "p2s": 1,
        "p3": 1,
        "q0": 1,
        "q1": 1,
        "q2": 2,
        "q2s": 1,
        "q3": 1,
        "r": 1,
        "s": 1,
    }


def _nmp_schedule_factors(*, channels=1, outputs=1, height, width):
    return {
        "c1": channels,
        "k1": outputs,
        "p1": height,
        "q1": width,
    }


class TargetHardwareTest(unittest.TestCase):
    def setUp(self):
        self.converter = convert._Converter.__new__(convert._Converter)
        self.converter.issues = []
        self.converter.hardware_config = _hardware_config()

    def test_dtype_specific_matrix_extent(self):
        self.assertEqual(self.converter.matrix_input_lanes(4), 32)
        self.assertEqual(self.converter.matrix_input_lanes(8), 16)
        self.assertEqual(self.converter.matrix_output_lanes(8), 32)

    def test_hardware_widths_reject_bool_and_fractional_values(self):
        matrix = self.converter.compute_unit("matrix")
        for invalid in (True, 8.5):
            with self.subTest(value=invalid):
                matrix["accepted_input_bits"] = [invalid]
                with self.assertRaises(ValueError):
                    self.converter.accepted_input_bits("matrix")
        matrix["accepted_input_bits"] = [4, 8]

    def test_hardware_extents_reject_bool_and_fractional_values(self):
        matrix = self.converter.compute_unit("matrix")
        descriptor = matrix["unroll_dimensions"][0]
        for invalid in (True, 16.5):
            with self.subTest(value=invalid):
                descriptor["extent"] = invalid
                descriptor["dtype_extents"] = []
                with self.assertRaises(ValueError):
                    self.converter.unroll_extent("matrix", "IC", 4)

    def test_vsq_block_size_is_read_from_serialized_operation(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation().op
        operation.name = "vsq"
        operation.kwargs["block_size"].int_value = 16

        self.assertEqual(
            self.converter.require_block_size(
                operation,
                "layer",
                "VSQ operand",
            ),
            16,
        )

    def test_vsq_block_size_is_required(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation().op
        operation.name = "vsq"

        with self.assertRaises(convert.SkipLayer):
            self.converter.require_block_size(
                operation,
                "layer",
                "VSQ operand",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-vsq-block-size",
        )


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.converter = convert._Converter.__new__(convert._Converter)
        self.converter.issues = []
        self.converter.hardware_config = _hardware_config()

    @staticmethod
    def _args(
        model_dir,
        output_dir,
        *,
        force,
        require_all_supported=False,
        require_connected_program=False,
    ):
        return SimpleNamespace(
            model=str(model_dir),
            output=output_dir,
            network=None,
            force=force,
            no_tensors=False,
            require_all_supported=require_all_supported,
            require_connected_program=require_connected_program,
        )

    @staticmethod
    def _report(errors, *, all_supported=True, connected=True):
        return {
            "schema_version": convert.CONVERSION_REPORT_SCHEMA_VERSION,
            "status": "failed" if errors else "complete",
            "source": "source",
            "output": "stage",
            "claims": {
                "all_supported_operations_emitted": not errors,
                "all_model_operations_supported": all_supported,
                "connected_alpha_program": connected,
            },
            "execution_scope": {
                "unsupported": [],
                "execution_islands": [],
                "external_inputs": [],
                "external_outputs": [],
            },
            "summary": {
                "emitted_layers": 1,
                "emitted_by_kind": {"gemm_convolution": 1},
                "fused_graph_ops": {},
                "fused_graph_ops_by_owner": {},
                "skipped_graph_ops": {},
                "issues_by_severity": ({"error": errors} if errors else {}),
            },
            "layers": [],
            "issues": [],
        }

    def _run_fake(
        self,
        root,
        *,
        errors=0,
        force=False,
        fail=False,
        all_supported=True,
        connected=True,
        require_all_supported=False,
        require_connected_program=False,
    ):
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "model.txt").write_text("")
        output = root / "converted"

        report = self._report(
            errors,
            all_supported=all_supported,
            connected=connected,
        )

        class FakeConverter:
            def __init__(self, **kwargs):
                self.output_dir = kwargs["output_dir"]

            def convert(self):
                layer = self.output_dir / "valid_layer"
                layer.mkdir()
                (layer / "schedule0.txt").write_text("runnable\n")
                if fail:
                    raise OSError("synthetic write failure")
                return report

        arguments = self._args(
            model_dir,
            output,
            force=force,
            require_all_supported=require_all_supported,
            require_connected_program=require_connected_program,
        )
        with (
            mock.patch.object(convert, "_parse_args", return_value=arguments),
            mock.patch.object(convert, "_Converter", FakeConverter),
        ):
            result = convert.main()
        return result, output

    def test_integer_parser_accepts_only_exact_non_boolean_scalars(self):
        self.assertEqual(
            self.converter.require_integer(17.0, "layer", "factor"),
            17,
        )
        for value in (True, 1.5, float("nan"), "1"):
            with self.subTest(value=value):
                self.converter.issues = []
                with self.assertRaises(convert.SkipLayer):
                    self.converter.require_integer(
                        value,
                        "layer",
                        "factor",
                    )
                self.assertEqual(
                    self.converter.issues[-1].code,
                    "invalid-serialized-integer",
                )

    def test_address_validates_compiler_selected_memory_capability(self):
        proto = convert._load_proto_module(convert._repo_root())
        tensor = proto.Tensor()
        tensor.node = "input"
        tensor.scratchpad.address = 0x1080000
        tensor.scratchpad.instance = "global_activation"

        self.assertEqual(
            self.converter.require_address(
                tensor,
                "layer",
                "input",
                "activation",
                "matrix",
            ),
            0x1080000,
        )
        # A role-wide ``compute_unit=None`` access is a wildcard, matching
        # MemoryInstance.accepts() in the compiler.
        self.assertEqual(
            self.converter.require_address(
                tensor,
                "layer",
                "output",
                "output",
                "matrix",
            ),
            0x1080000,
        )

        tensor.scratchpad.instance = "global_weight"
        with self.assertRaises(convert.SkipLayer):
            self.converter.require_address(
                tensor,
                "layer",
                "input",
                "activation",
                "nmp",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-hardware-memory-access",
        )

        global_weight = next(
            instance
            for instance in self.converter.hardware_config["global_memory"][
                "instances"
            ]
            if instance["name"] == "global_weight"
        )
        global_weight["accepted_accesses"].append(
            {
                "role": "activation",
                "compute_unit": "nmp",
            }
        )
        # The converter accepts an alternate compiler-selected instance once
        # the serialized hardware says that exact access is legal.
        self.assertEqual(
            self.converter.require_address(
                tensor,
                "layer",
                "input",
                "activation",
                "nmp",
            ),
            0x1080000,
        )

    def test_nmp_weight_instance_validates_compiler_selection(self):
        proto = convert._load_proto_module(convert._repo_root())
        tensor = proto.Tensor()
        tensor.node = "weight"
        tensor.scratchpad.instance = "global_weight"
        global_weight = next(
            instance
            for instance in self.converter.hardware_config["global_memory"][
                "instances"
            ]
            if instance["name"] == "global_weight"
        )
        with self.assertRaises(convert.SkipLayer):
            self.converter.access_memory_instance(
                tensor,
                "weight",
                "nmp",
                "depthwise",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-hardware-memory-access",
        )
        global_weight["accepted_accesses"].append(
            {
                "role": "weight",
                "compute_unit": "nmp",
            }
        )

        self.assertEqual(
            self.converter.access_memory_instance(
                tensor,
                "weight",
                "nmp",
                "depthwise",
            ),
            "global_weight",
        )

    def test_error_does_not_publish_partial_collateral(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, output = self._run_fake(root, errors=1)

            self.assertEqual(result, 1)
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".converted.stage-*")))
            failure = root / "converted.failure.json"
            self.assertEqual(
                json.loads(failure.read_text())["status"],
                "failed",
            )

    def test_failed_force_preserves_previous_complete_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "converted"
            output.mkdir()
            (output / "known_good").write_text("keep")

            result, published = self._run_fake(
                root,
                errors=1,
                force=True,
            )

            self.assertEqual(result, 1)
            self.assertEqual(published, output)
            self.assertEqual((output / "known_good").read_text(), "keep")
            self.assertFalse((output / "valid_layer").exists())

    def test_success_replaces_old_output_only_after_complete_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "converted"
            output.mkdir()
            (output / "old").write_text("old")
            (root / "converted.failure.json").write_text("{}")

            result, published = self._run_fake(root, force=True)

            self.assertEqual(result, 0)
            self.assertEqual(published, output)
            self.assertFalse((output / "old").exists())
            self.assertTrue((output / "valid_layer/schedule0.txt").is_file())
            report = json.loads((output / "conversion_report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertFalse((root / "converted.failure.json").exists())

    def test_strict_coverage_rejects_partial_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, output = self._run_fake(
                root,
                all_supported=False,
                connected=False,
                require_all_supported=True,
            )

            self.assertEqual(result, 1)
            self.assertFalse(output.exists())
            failure = json.loads((root / "converted.failure.json").read_text())
            self.assertIn(
                "incomplete-alpha-operation-coverage",
                {issue["code"] for issue in failure["issues"]},
            )

    def test_backup_cleanup_failure_does_not_invalidate_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "converted"
            output.mkdir()
            (output / "old").write_text("old")
            staging = root / ".converted.stage-test"
            staging.mkdir()
            (staging / "complete").write_text("new")

            with mock.patch.object(
                convert,
                "_remove_path",
                side_effect=PermissionError("retained"),
            ):
                retained = convert._publish_staged_output(
                    staging,
                    output,
                    replace=True,
                )

            self.assertEqual((output / "complete").read_text(), "new")
            self.assertIsNotNone(retained)
            self.assertEqual((retained / "old").read_text(), "old")

    def test_exception_never_publishes_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, output = self._run_fake(root, fail=True)

            self.assertEqual(result, 1)
            self.assertFalse(output.exists())
            failure = json.loads((root / "converted.failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(
                failure["issues"][0]["code"],
                "conversion-exception",
            )

    def test_gemm_factors_are_only_compiler_serialized_fields(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "matrix"
        expected = {
            name: index
            for index, name in enumerate(
                cmod_utils.GEMM_SCHEDULE_FACTORS,
                start=1,
            )
        }
        _set_schedule_factors(operation, expected)

        self.assertEqual(
            self.converter.schedule_factors(
                operation,
                cmod_utils.GEMM_SCHEDULE_FACTORS,
            ),
            expected,
        )

    def test_nmp_input_widths_follow_target_hardware(self):
        input_tensor = type(
            "Tensor",
            (),
            {"dtype": "int8", "node": "input"},
        )()
        weight = type(
            "Tensor",
            (),
            {"dtype": "int8", "node": "weight"},
        )()

        self.assertEqual(
            self.converter.require_nmp_input_bits(
                "depthwise",
                input=input_tensor,
                weight=weight,
            ),
            {"input": 8, "weight": 8},
        )

    def test_nmp_tensor_extents_require_positive_rank_four(self):
        proto = convert._load_proto_module(convert._repo_root())
        for shape in ((1, 32, 4), (1, 0, 4, 4)):
            with self.subTest(shape=shape):
                tensor = proto.Tensor()
                tensor.node = "malformed"
                tensor.shape.extend(shape)
                self.converter.issues = []
                with self.assertRaises(convert.SkipLayer):
                    self.converter.require_nmp_tensor_shape(
                        tensor,
                        "nmp",
                        "input",
                    )
                self.assertEqual(
                    self.converter.issues[-1].code,
                    "invalid-nmp-tensor-extents",
                )

    def test_nmp_rejects_unsupported_physical_weight_width(self):
        weight = type(
            "Tensor",
            (),
            {"dtype": "int4", "node": "weight"},
        )()

        with self.assertRaises(convert.SkipLayer):
            self.converter.require_nmp_input_bits(
                "depthwise",
                weight=weight,
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "unsupported-nmp-input-precision",
        )

    def test_nmp_width_is_read_from_serialized_compute_unit(self):
        nmp = next(
            unit
            for unit in self.converter.hardware_config["compute_units"]
            if unit["name"] == "nmp"
        )
        nmp["accepted_input_bits"] = [4, 8]
        weight = type(
            "Tensor",
            (),
            {"dtype": "int4", "node": "weight"},
        )()

        self.assertEqual(
            self.converter.require_nmp_input_bits(
                "depthwise",
                weight=weight,
            ),
            {"weight": 4},
        )

    def test_gemm_rejects_missing_schedule_factor(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "matrix"
        factors = _gemm_schedule_factors()
        factors.pop("q3")
        _set_schedule_factors(operation, factors)

        with self.assertRaises(convert.SkipLayer):
            self.converter.schedule_factors(
                operation,
                cmod_utils.GEMM_SCHEDULE_FACTORS,
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "missing-target-schedule-factor",
        )

    def test_gemm_schedule_factors_pass_cmod_limit_validation(self):
        factors = _gemm_schedule_factors()
        self.converter.validate_gemm_schedule_limits(
            "matrix",
            factors,
            8,
        )

    def test_gemm_schedule_rejects_derived_counter_overflow(self):
        factors = _gemm_schedule_factors()
        factors["p2"] = 1 << 16

        with self.assertRaises(convert.SkipLayer):
            self.converter.validate_gemm_schedule_limits(
                "matrix",
                factors,
                8,
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "alpha-schedule-limit-exceeded",
        )

    def test_schedule_factor_rejects_noninteger_or_nonpositive_value(self):
        proto = convert._load_proto_module(convert._repo_root())
        for bad_value in (True, 1.5, 0, -1):
            with self.subTest(value=bad_value):
                operation = proto.Operation()
                operation.op.name = "nmp"
                factors = _nmp_schedule_factors(
                    height=2,
                    width=3,
                )
                factors["c1"] = bad_value
                _set_schedule_factors(operation, factors)
                self.converter.issues = []

                with self.assertRaises(convert.SkipLayer):
                    self.converter.schedule_factors(
                        operation,
                        cmod_utils.NMP_SCHEDULE_FACTORS,
                    )

                self.assertEqual(
                    self.converter.issues[-1].code,
                    "invalid-target-schedule-factor",
                )

    def test_nmp_uses_only_compiler_serialized_factors(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "nmp"
        expected = _nmp_schedule_factors(
            channels=4,
            outputs=5,
            height=2,
            width=3,
        )
        _set_schedule_factors(operation, expected)

        self.assertEqual(
            self.converter.schedule_factors(
                operation,
                cmod_utils.NMP_SCHEDULE_FACTORS,
            ),
            expected,
        )

    def test_canonical_target_sequences_select_builders(self):
        proto = convert._load_proto_module(convert._repo_root())
        cases = (
            (("max_pool2d",), "nmp_max_pool"),
            (("upsample_nearest2d",), "nmp_nearest_upsample"),
            (("channel_shuffle",), "nmp_channel_shuffle"),
            (("channel_shuffle_copy",), "nmp_copy"),
            (
                ("conv2d_vsq", "ppu_fixed_relu"),
                "gemm_convolution",
            ),
            (
                (
                    "conv2d_int8_accumulator",
                    "ppu_fixed_relu",
                    "saturating_add",
                ),
                "gemm_convolution",
            ),
            (
                (
                    "conv2d_int8_accumulator",
                    "nmp_fixed_requantize",
                    "pvs_quantize",
                ),
                "nmp_depthwise_convolution",
            ),
        )
        for targets, expected in cases:
            with self.subTest(targets=targets):
                operation = proto.Operation()
                if len(targets) == 1:
                    inner = operation.op
                    inner.name = "operation"
                    inner.op = "call_function"
                    inner.target = targets[0]
                else:
                    operation.fused_op.name = "operation"
                    for index, target in enumerate(targets):
                        inner = operation.fused_op.op_list.add()
                        inner.name = f"inner_{index}"
                        inner.op = "call_function"
                        inner.target = target

                self.assertEqual(
                    self.converter.execution_kind(operation),
                    expected,
                )

    def test_eadd_target_sequence_is_exact(self):
        proto = convert._load_proto_module(convert._repo_root())
        for targets in (
            ("conv2d_int8_accumulator", "saturating_add"),
            (
                "conv2d_int8_accumulator",
                "saturating_add",
                "ppu_fixed_relu",
            ),
            (
                "conv2d_int8_accumulator",
                "ppu_fixed_relu",
                "saturating_add",
                "pvs_quantize",
            ),
            ("conv2d_vsq", "ppu_fixed_relu", "saturating_add"),
        ):
            with self.subTest(targets=targets):
                operation = proto.Operation()
                operation.fused_op.name = "invalid_eadd"
                for index, target in enumerate(targets):
                    inner = operation.fused_op.op_list.add()
                    inner.name = f"inner_{index}"
                    inner.op = "call_function"
                    inner.target = target
                self.converter.issues = []

                with self.assertRaises(convert.SkipLayer):
                    self.converter.execution_kind(operation)
                self.assertEqual(
                    self.converter.issues[-1].code,
                    "unsupported-cmod-target",
                )

    def test_partial_program_emits_supported_island_and_boundaries(self):
        proto = convert._load_proto_module(convert._repo_root())
        model = proto.Model()
        model.inputs.add(node="public_input")

        producer = model.ops.add()
        producer.op.name = "external_producer"
        producer.op.op = "unsupported"
        producer.op.target = "external_preprocess"
        producer.op.kwargs["input"].tensor.node = "public_input"
        producer.output.node = "materialized_input"

        supported = model.ops.add()
        supported.op.name = "pool"
        supported.op.op = "call_function"
        supported.op.target = "max_pool2d"
        supported.op.kwargs["input"].tensor.node = "materialized_input"
        supported.output.node = "pool_output"

        consumer = model.ops.add()
        consumer.op.name = "external_consumer"
        consumer.op.op = "unsupported"
        consumer.op.target = "external_postprocess"
        consumer.op.kwargs["input"].tensor.node = "pool_output"
        consumer.output.node = "public_output"

        self.converter.model = model
        self.converter.operations = list(model.ops)
        self.converter.results = []
        self.converter.model_dir = Path("/compiler")
        self.converter.network = "partial"
        self.converter.write_tensors = False
        self.converter.build_pool = lambda operation, kind: (
            convert.LayerResult(
                name=cmod_utils.operation_name(operation),
                kind=kind,
                schedule={},
                inputs=self.converter.operation_inputs(operation),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            self.converter.output_dir = Path(directory)
            report = self.converter.convert()

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["summary"]["emitted_layers"], 1)
        self.assertEqual(
            report["summary"]["unsupported_operations"],
            2,
        )
        self.assertEqual(
            report["summary"]["skipped_by_reason"],
            {"serialized_unsupported": 2},
        )
        self.assertEqual(
            report["claims"],
            {
                "all_supported_operations_emitted": True,
                "all_model_operations_supported": False,
                "connected_alpha_program": False,
            },
        )
        self.assertEqual(
            report["execution_scope"]["execution_islands"],
            [{"index": 0, "layers": ["pool"]}],
        )
        self.assertEqual(
            report["execution_scope"]["external_inputs"],
            [
                {
                    "tensor": "materialized_input",
                    "producer": "external_producer",
                    "producer_target": "external_preprocess",
                    "consumer": "pool",
                }
            ],
        )
        self.assertEqual(
            report["execution_scope"]["external_outputs"],
            [
                {
                    "tensor": "pool_output",
                    "producer": "pool",
                    "consumer": "external_consumer",
                    "consumer_target": "external_postprocess",
                }
            ],
        )

    def test_legacy_execution_metadata_is_not_authoritative(self):
        proto = convert._load_proto_module(convert._repo_root())
        pool = proto.Operation()
        pool.op.name = "pool"
        pool.op.op = "call_function"
        pool.op.target = "max_pool2d"
        _set_legacy_cmod_metadata(
            pool,
            "unsupported",
            kind="gemm_convolution",
            skip_reason="legacy_wrong_reason",
        )
        self.assertEqual(
            self.converter.execution_kind(pool),
            "nmp_max_pool",
        )

        opaque = proto.Operation()
        opaque.op.name = "opaque"
        opaque.op.op = "call_function"
        opaque.op.target = "opaque_target"
        _set_legacy_cmod_metadata(
            opaque,
            "supported",
            kind="nmp_max_pool",
        )
        with self.assertRaises(convert.SkipLayer):
            self.converter.execution_kind(opaque)
        self.assertEqual(
            self.converter.issues[-1].code,
            "unsupported-cmod-target",
        )

    def test_legacy_inner_references_are_not_authoritative(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.fused_op.name = "matrix"
        convolution = operation.fused_op.op_list.add()
        convolution.name = "actual_convolution"
        convolution.target = "conv2d_vsq"
        epilogue = operation.fused_op.op_list.add()
        epilogue.name = "actual_epilogue"
        epilogue.target = "ppu_fixed_relu"
        convolution.kwargs["cmod_descriptor_name"].str_value = (
            "invented_convolution"
        )
        convolution.kwargs["cmod_ppu_epilogue_name"].str_value = (
            "invented_epilogue"
        )

        self.assertIs(
            self.converter.convolution_descriptor(operation, "gemm"),
            convolution,
        )
        self.assertIs(
            self.converter.inner_by_target(
                operation,
                "PPU epilogue",
                targets=frozenset({"ppu_fixed_relu"}),
                required=True,
            ),
            epilogue,
        )

    def test_unconverted_compiler_targets_are_rejected(self):
        proto = convert._load_proto_module(convert._repo_root())
        for target in ("group_norm", "softmax"):
            with self.subTest(target=target):
                operation = proto.Operation()
                operation.op.name = target
                operation.op.op = "call_function"
                operation.op.target = target
                self.converter.issues = []

                with self.assertRaises(convert.SkipLayer):
                    self.converter.execution_kind(operation)
                self.assertEqual(
                    self.converter.issues[-1].code,
                    "unsupported-cmod-target",
                )

    def test_compiler_nop_is_non_executable_without_cmod_metadata(self):
        proto = convert._load_proto_module(convert._repo_root())
        for target in ("reshape", "max_pool2d"):
            with self.subTest(target=target):
                operation = proto.Operation()
                operation.op.name = target
                operation.op.op = "nop"
                operation.op.target = target

                record = self.converter._parse_execution_record(operation)

                self.assertEqual(record.disposition, "non_executable")
                self.assertEqual(record.skip_reason, "target_address_only")
                self.assertIsNone(self.converter.execution_kind(operation))

    def test_legacy_metadata_cannot_turn_work_into_a_nop(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "opaque"
        operation.op.op = "call_function"
        operation.op.target = "opaque_target"
        _set_legacy_cmod_metadata(operation, "non_executable")

        with self.assertRaises(convert.SkipLayer):
            self.converter.execution_kind(operation)
        self.assertEqual(
            self.converter.issues[-1].code,
            "unsupported-cmod-target",
        )

    def test_quantizer_needs_a_dataflow_derived_owner(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "quantizer"
        operation.op.op = "call_function"
        operation.op.target = "quantize"
        operation.op.kwargs["input"].tensor.node = "input"
        operation.output.node = "quantized"
        _set_legacy_cmod_metadata(
            operation,
            "fused",
            fusion_owner="invented_owner",
        )

        with self.assertRaises(convert.SkipLayer):
            self.converter.execution_kind(operation)
        self.assertEqual(
            self.converter.issues[-1].code,
            "unresolved-cmod-fusion-owner",
        )

    def test_host_layout_needs_a_terminal_fused_dequantizer(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "permute"
        operation.op.op = "call_function"
        operation.op.target = "permute"

        with self.assertRaises(convert.SkipLayer):
            self.converter.execution_kind(operation)
        self.assertEqual(
            self.converter.issues[-1].code,
            "unsupported-cmod-host-layout",
        )

    def test_legacy_add_maps_only_with_complete_physical_copy_facts(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "legacy_copy"
        operation.op.op = "call_function"
        operation.op.target = "add"
        with self.assertRaises(convert.SkipLayer):
            self.converter.execution_kind(operation)
        self.assertEqual(
            self.converter.issues[-1].code,
            "unsupported-cmod-target",
        )

        for field in (
            "block_channels",
            "copy_destination_entry",
            "copy_source_entry",
            "copy_source_output",
        ):
            operation.op.kwargs[field].int_value = 1
        for field in (
            "copy_block_payload",
            "copy_destination_payload",
            "copy_source_payload",
        ):
            operation.op.kwargs[field].tensor.node = field

        self.assertEqual(
            self.converter.execution_kind(operation),
            "nmp_copy",
        )

    def test_padding_is_read_from_serialized_operation(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "stem"
        fields = {
            "padding_c": 15,
            "padding_k": 16,
            "pad_h_up": 1,
            "pad_h_down": 1,
            "pad_w_left": 1,
            "pad_w_right": 1,
        }
        for name, value in fields.items():
            operation.op.kwargs[name].int_value = value

        self.assertEqual(
            self.converter.padding_fields(operation, spatial=True),
            fields,
        )

    def test_matrix_output_storage_accepts_partial_vector(self):
        self.converter.validate_matrix_output_storage(
            "stem",
            16,
            8,
            {
                "k1": 1,
                "k2": 1,
                "k3": 1,
                "padding_k": 16,
            },
        )

    def test_matrix_output_storage_accepts_aligned_vector(self):
        self.converter.validate_matrix_output_storage(
            "layer",
            32,
            8,
            {
                "k1": 1,
                "k2": 1,
                "k3": 1,
                "padding_k": 0,
            },
        )

    def test_matrix_output_storage_rejects_unsupported_projection(self):
        with self.assertRaises(convert.SkipLayer):
            self.converter.validate_matrix_output_storage(
                "layer",
                48,
                8,
                {
                    "k1": 2,
                    "k2": 1,
                    "k3": 1,
                    "padding_k": 16,
                },
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "target-output-storage-mismatch",
        )

    def test_materialized_input_payload_is_consumed_verbatim(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "stem"
        input_tensor = operation.op.kwargs["input"].tensor
        input_tensor.node = "semantic_input"
        input_tensor.shape.extend((1, 1, 2, 2))
        input_tensor.dtype = "int8"
        input_tensor.scratchpad.address = 0x1080000
        input_tensor.scratchpad.instance = "global_activation"
        payload = operation.op.kwargs["materialized_input_payload"].tensor
        payload.node = "physical_input"
        payload.shape.extend((1, 1, 2, 3))
        payload.dtype = "int8"
        payload.scratchpad.address = 0x1080000
        payload.scratchpad.instance = "global_activation"
        operation.op.kwargs[
            "materialized_input_padding"
        ].int_list.values.extend((1, 0, 0, 0))
        semantic = np.asarray([[[[1, 2], [3, 4]]]], dtype=np.float32)
        physical = np.asarray(
            [[[[0, 1, 2], [0, 3, 4]]]],
            dtype=np.float32,
        )
        self.converter.array_cache = {
            "semantic_input": semantic,
            "physical_input": physical,
        }

        selected, actual = self.converter.resolve_materialized_input_payload(
            "stem",
            operation,
            input_tensor,
            semantic,
            8,
        )

        self.assertEqual(selected.node, "physical_input")
        np.testing.assert_array_equal(actual, physical)
        self.assertFalse(
            hasattr(convert._Converter, "apply_materialized_input_padding")
        )

    def test_materialized_padding_requires_physical_payload(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "stem"
        operation.op.kwargs[
            "materialized_input_padding"
        ].int_list.values.extend((1, 0, 0, 0))
        input_tensor = operation.op.kwargs["input"].tensor
        input_tensor.node = "semantic_input"
        input_tensor.shape.extend((1, 1, 2, 2))
        input_tensor.dtype = "int8"

        with self.assertRaises(convert.SkipLayer):
            self.converter.resolve_materialized_input_payload(
                "stem",
                operation,
                input_tensor,
                np.ones((1, 1, 2, 2), dtype=np.float32),
                8,
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "missing-materialized-input-payload",
        )

    def test_missing_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "matrix XY"):
            self.converter.unroll_extent("matrix", "XY", 8)

    def test_explicit_nmp_operands_are_read_without_rederivation(self):
        proto = convert._load_proto_module(convert._repo_root())
        epilogue = proto.Operation()
        epilogue.op.name = "dwc_fixed"
        epilogue.op.target = "nmp_fixed_requantize"
        epilogue.op.kwargs["scale_factor"].int_value = 39847
        epilogue.op.kwargs["binary_point"].int_value = 23
        epilogue.op.kwargs["bias_factor"].int_value = -3
        epilogue.op.kwargs["output_precision"].int_value = 8
        epilogue.op.kwargs["relu"].bool_value = False

        actual = self.converter.explicit_nmp_parameters(
            epilogue,
            "depthwise",
        )

        self.assertEqual(
            actual,
            {
                "scale_factor": 39847,
                "binary_point": 23,
                "bias_factor": -3,
                "output_precision": 8,
                "relu": False,
            },
        )

    def test_output_quantizer_only_reads_same_operation_record(self):
        proto = convert._load_proto_module(convert._repo_root())
        producer = proto.Operation()
        producer.op.name = "producer"
        producer.op.target = "upsample_nearest2d"
        producer.output.node = "pre_pvs"

        standalone = proto.Operation()
        standalone.op.name = "pvs"
        standalone.op.target = "pvs_quantize"
        standalone.op.kwargs["input"].tensor.node = "pre_pvs"

        self.assertIsNone(self.converter.output_quantizer(producer))

        fused = proto.Operation()
        fused.fused_op.name = "fused"
        fused.fused_op.op_list.add().CopyFrom(producer.op)
        fused.fused_op.op_list.add().CopyFrom(standalone.op)

        owner, quantizer = self.converter.output_quantizer(fused)
        self.assertIs(owner, fused)
        self.assertEqual(quantizer.target, "pvs_quantize")

    def test_standalone_unknown_target_is_rejected(self):
        proto = convert._load_proto_module(convert._repo_root())
        pvs = proto.Operation()
        pvs.op.name = "pvs"
        pvs.op.target = "pvs_quantize"
        self.converter.operations = [pvs]
        self.converter.results = []
        self.converter.model_dir = Path("/compiler")
        self.converter.network = "tiny"

        with tempfile.TemporaryDirectory() as directory:
            self.converter.output_dir = Path(directory)
            report = self.converter.convert()

        self.assertEqual(report["summary"]["emitted_layers"], 0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["summary"]["unsupported_operations"], 1)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("unsupported-cmod-target", codes)
        self.assertIn("no-executable-cmod-layers", codes)

    def test_all_serialized_unsupported_records_are_rejected_as_empty(self):
        proto = convert._load_proto_module(convert._repo_root())
        external = proto.Operation()
        external.op.name = "external"
        external.op.op = "unsupported"
        external.op.target = "external_preprocess"
        self.converter.operations = [external]
        self.converter.results = []
        self.converter.model_dir = Path("/compiler")
        self.converter.network = "tiny"

        with tempfile.TemporaryDirectory() as directory:
            self.converter.output_dir = Path(directory)
            report = self.converter.convert()

        self.assertEqual(report["summary"]["emitted_layers"], 0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["summary"]["skipped_by_reason"],
            {"serialized_unsupported": 1},
        )
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("no-executable-cmod-layers", codes)

    def test_fused_records_are_reported_against_supported_owner(self):
        proto = convert._load_proto_module(convert._repo_root())

        model = proto.Model()
        model.inputs.add(node="public_input")
        input_quantizer = model.ops.add()
        input_quantizer.op.name = "input_quantizer"
        input_quantizer.op.op = "call_function"
        input_quantizer.op.target = "quantize"
        input_quantizer.op.kwargs["input"].tensor.node = "public_input"
        input_quantizer.output.node = "quantized_input"

        owner = model.ops.add()
        owner.op.name = "matrix"
        owner.op.op = "call_function"
        owner.op.target = "max_pool2d"
        owner.op.kwargs["input"].tensor.node = "quantized_input"
        owner.output.node = "owner_output"

        output_dequantizer = model.ops.add()
        output_dequantizer.op.name = "output_dequantizer"
        output_dequantizer.op.op = "call_function"
        output_dequantizer.op.target = "dequantize"
        output_dequantizer.op.kwargs["input"].tensor.node = "owner_output"
        output_dequantizer.output.node = "dequantized_output"

        permute = model.ops.add()
        permute.op.name = "output_permute"
        permute.op.op = "call_function"
        permute.op.target = "permute"
        permute.op.kwargs["input"].tensor.node = "dequantized_output"
        permute.output.node = "permuted_output"

        contiguous = model.ops.add()
        contiguous.op.name = "output_contiguous"
        contiguous.op.op = "call_function"
        contiguous.op.target = "contiguous"
        contiguous.op.kwargs["input"].tensor.node = "permuted_output"
        contiguous.output.node = "public_output"

        self.converter.model = model
        self.converter.operations = list(model.ops)
        self.converter.results = []
        self.converter.model_dir = Path("/compiler")
        self.converter.network = "tiny"
        self.converter.write_tensors = False
        self.converter.build_pool = lambda operation, kind: (
            convert.LayerResult(
                name=cmod_utils.operation_name(operation),
                kind=kind,
                schedule={},
                inputs=[],
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            self.converter.output_dir = Path(directory)
            report = self.converter.convert()

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["summary"]["emitted_layers"], 1)
        self.assertEqual(
            report["summary"]["fused_graph_ops"],
            {
                "dequantize": 1,
                "quantize": 1,
            },
        )
        self.assertEqual(
            report["summary"]["fused_graph_ops_by_owner"],
            {
                "matrix": {
                    "dequantize": 1,
                    "quantize": 1,
                }
            },
        )
        self.assertEqual(
            report["summary"]["skipped_graph_ops"],
            {
                "contiguous": 1,
                "permute": 1,
            },
        )
        self.assertEqual(
            report["summary"]["skipped_by_reason"],
            {"host_output_layout": 2},
        )

    def test_fused_owner_requires_at_least_one_supported_record(self):
        proto = convert._load_proto_module(convert._repo_root())
        for owner_count in (0, 2):
            with self.subTest(owner_count=owner_count):
                model = proto.Model()
                model.inputs.add(node="public_input")
                quantizer = model.ops.add()
                quantizer.op.name = "input_quantizer"
                quantizer.op.op = "call_function"
                quantizer.op.target = "quantize"
                quantizer.op.kwargs["input"].tensor.node = "public_input"
                quantizer.output.node = "quantized_input"
                for index in range(owner_count):
                    owner = model.ops.add()
                    owner.op.name = f"owner_{index}"
                    owner.op.op = "call_function"
                    owner.op.target = "max_pool2d"
                    owner.op.kwargs["input"].tensor.node = "quantized_input"
                    owner.output.node = f"owner_output_{index}"

                self.converter.model = model
                records = [
                    self.converter._parse_execution_record(operation)
                    for operation in model.ops
                ]
                records = self.converter._resolve_fusion_owners(records)
                quantizer_record = records[0]

                if owner_count == 0:
                    self.assertEqual(
                        quantizer_record.disposition,
                        convert._FUSION_CANDIDATE,
                    )
                    self.converter.issues = []
                    with self.assertRaises(convert.SkipLayer):
                        self.converter._execution_kind_for_record(
                            quantizer_record
                        )
                    self.assertEqual(
                        self.converter.issues[-1].code,
                        "unresolved-cmod-fusion-owner",
                    )
                else:
                    self.assertEqual(quantizer_record.disposition, "fused")
                    self.assertEqual(
                        quantizer_record.fusion_owners,
                        ("owner_0", "owner_1"),
                    )

    def test_all_non_executable_records_are_rejected_as_empty_model(self):
        proto = convert._load_proto_module(convert._repo_root())
        metadata = proto.Operation()
        metadata.op.name = "metadata"
        metadata.op.op = "nop"
        metadata.op.target = "reshape"
        self.converter.operations = [metadata]
        self.converter.results = []
        self.converter.model_dir = Path("/compiler")
        self.converter.network = "tiny"

        with tempfile.TemporaryDirectory() as directory:
            self.converter.output_dir = Path(directory)
            report = self.converter.convert()

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["summary"]["emitted_layers"], 0)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("no-executable-cmod-layers", codes)

    def test_serialized_depthwise_contract_emits_cmodel_collateral(self):
        proto = convert._load_proto_module(convert._repo_root())
        model = proto.Model()

        depthwise = model.ops.add()
        depthwise.fused_op.name = "depthwise"
        conv = depthwise.fused_op.op_list.add()
        conv.name = "depthwise_accumulator"
        conv.target = "conv2d_int8_accumulator"
        input_tensor = conv.kwargs["input"].tensor
        self._set_tensor(
            input_tensor,
            "input",
            (1, 32, 5, 5),
            "int8",
            0x1080000,
        )
        weight = conv.kwargs["weight"].tensor
        self._set_tensor(
            weight,
            "weight",
            (32, 1, 3, 3),
            "int8",
            0x10A0000,
        )
        conv.kwargs["stride"].int_list.values.extend((1, 1))
        conv.kwargs["padding"].int_list.values.extend((0, 0))
        conv.kwargs["dilation"].int_list.values.extend((1, 1))
        conv.kwargs["groups"].int_value = 32

        epilogue = depthwise.fused_op.op_list.add()
        epilogue.name = "depthwise_requantize"
        epilogue.target = "nmp_fixed_requantize"
        self._set_tensor(
            epilogue.kwargs["accumulator"].tensor,
            "depthwise_accumulator",
            (1, 32, 3, 3),
            "int32",
            0x1080800,
        )
        self._set_tensor(
            epilogue.kwargs["semantic_accumulator_scale"].tensor,
            "accumulator_scale",
            (1,),
            "float32",
            0x1080C00,
        )
        epilogue.kwargs["scale_factor"].int_value = 39847
        epilogue.kwargs["binary_point"].int_value = 23
        epilogue.kwargs["bias_factor"].int_value = -3
        epilogue.kwargs["output_precision"].int_value = 4
        epilogue.kwargs["relu"].bool_value = False
        pvs = depthwise.fused_op.op_list.add()
        pvs.name = "depthwise_pvs"
        pvs.target = "pvs_quantize"
        pvs.kwargs["block_size"].int_value = 32
        coarse = depthwise.outputs.tensors.add()
        self._set_tensor(
            coarse,
            "coarse",
            (1,),
            "float32",
            0x1081000,
        )
        fine = depthwise.outputs.tensors.add()
        self._set_tensor(
            fine,
            "fine",
            (1, 1, 3, 3),
            "uint8",
            0x1081000,
        )
        output = depthwise.outputs.tensors.add()
        self._set_tensor(
            output,
            "output",
            (1, 32, 3, 3),
            "int4",
            0x1081000,
        )
        conv.kwargs["output_payload"].tensor.CopyFrom(output)
        conv.kwargs["output_vector_scale"].tensor.CopyFrom(fine)
        fine_artifact = conv.kwargs["output_vector_scale_payload"].tensor
        fine_artifact.node = "fine_artifact"
        fine_artifact.shape.extend((1, 1, 3, 3))
        fine_artifact.dtype = "uint8"
        conv.kwargs["nmp_c2"].int_value = 0
        conv.kwargs["nmp_k2"].int_value = 0
        conv.kwargs["is_from_split"].bool_value = False
        for field in (
            "padding_c",
            "padding_k",
            "pad_h_up",
            "pad_h_down",
            "pad_w_left",
            "pad_w_right",
        ):
            conv.kwargs[field].int_value = 0
        _set_schedule_factors(
            depthwise,
            _nmp_schedule_factors(
                height=3,
                width=3,
            ),
        )
        semantic_scale = np.arange(
            1,
            10,
            dtype=np.float32,
        ).reshape((1, 1, 3, 3))
        encoded_scale = semantic_scale.transpose(0, 3, 2, 1).copy()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compiler_dir = root / "compiler"
            tensor_dir = compiler_dir / "tensor_files"
            tensor_dir.mkdir(parents=True)
            (compiler_dir / "model.txt").write_text(
                text_format.MessageToString(model)
            )
            (compiler_dir / "hardware_config.json").write_text(
                json.dumps(_hardware_config())
            )
            np.zeros((1, 32, 5, 5), dtype=np.float32).tofile(
                tensor_dir / "input.bin"
            )
            np.ones((32, 1, 3, 3), dtype=np.float32).tofile(
                tensor_dir / "weight.bin"
            )
            np.zeros((1, 32, 3, 3), dtype=np.float32).tofile(
                tensor_dir / "output.bin"
            )
            semantic_scale.tofile(tensor_dir / "fine.bin")
            semantic_scale.tofile(tensor_dir / "fine_artifact.bin")
            np.ones((1,), dtype=np.float32).tofile(tensor_dir / "coarse.bin")

            output_dir = root / "cmodel"
            report = convert._Converter(
                compiler_dir,
                output_dir,
                "tiny_depthwise",
                write_tensors=True,
            ).convert()

            self.assertEqual(report["summary"]["emitted_layers"], 1)
            self.assertNotIn(
                "error",
                report["summary"]["issues_by_severity"],
            )
            layer = report["layers"][0]
            self.assertEqual(
                layer["kind"],
                "nmp_depthwise_convolution",
            )
            self.assertEqual(layer["schedule"]["nmp_scale_factor"], 39847)
            self.assertEqual(layer["schedule"]["wt_binary_pt_loc"], 23)
            self.assertEqual(layer["schedule"]["nmp_bias_factor"], -3)
            self.assertEqual(
                layer["schedule"]["outputs0_baseaddr"],
                "0x1081000",
            )
            self.assertEqual(layer["schedule"]["p1"], 3)
            self.assertEqual(layer["schedule"]["q1"], 3)
            self.assertEqual(layer["schedule"]["input_height"], 5)
            self.assertEqual(layer["schedule"]["input_width"], 5)
            self.assertEqual(
                set(layer["files"]),
                {
                    "1_inputs.txt",
                    "1_weights.txt",
                    "1_outputs.txt",
                    "1_outputs_vector_scale.txt",
                },
            )
            layer_dir = output_dir / "depthwise"
            for filename in (
                "schedule0.txt",
                "schedule1.txt",
                "schedule10.txt",
                "1_inputs.txt",
                "1_weights.txt",
                "1_outputs.txt",
                "1_outputs_vector_scale.txt",
            ):
                self.assertTrue((layer_dir / filename).is_file())
            outputs = np.fromstring(
                (layer_dir / "1_outputs.txt").read_text(),
                sep=",",
                dtype=np.int64,
            )
            scales = np.fromstring(
                (layer_dir / "1_outputs_vector_scale.txt").read_text(),
                sep=",",
                dtype=np.int64,
            )
            np.testing.assert_array_equal(outputs, 0)
            np.testing.assert_array_equal(scales, encoded_scale.ravel())

            conv.kwargs["schedule_factor_c1"].int_value = 2
            (compiler_dir / "model.txt").write_text(
                text_format.MessageToString(model)
            )
            rejected_dir = root / "rejected"
            rejected_dir.mkdir()
            rejected = convert._Converter(
                compiler_dir,
                rejected_dir,
                "tiny_depthwise",
                write_tensors=False,
            ).convert()
            self.assertEqual(rejected["status"], "failed")
            self.assertIn(
                "target-schedule-semantic-mismatch",
                {issue["code"] for issue in rejected["issues"]},
            )

    @staticmethod
    def _set_tensor(
        tensor,
        node,
        shape,
        dtype,
        address,
        instance="global_activation",
    ):
        tensor.node = node
        tensor.shape.extend(shape)
        tensor.dtype = dtype
        tensor.scratchpad.partition = 0
        tensor.scratchpad.address = address
        tensor.scratchpad.instance = instance

    def _run_matrix_conversion(
        self,
        root,
        *,
        contain_epilogue,
        include_factors=True,
        resident_weight_c=16,
        physical_weight_c=None,
        physical_weight_k=None,
        weight_precision=8,
        output_shape=(1, 32, 2, 2),
        input_hw=(2, 2),
        stride=(1, 1),
        schedule_p1=1,
        schedule_p2=2,
        schedule_q1=1,
        schedule_q2=2,
        schedule_padding_c=None,
        materialized_input_valid_channels=None,
        materialized_input_resident_channels=None,
        materialized_input_padding=None,
        corrupt_weight_payload=None,
        write_tensors=True,
        eadd=False,
        ppu_relu=True,
        eadd_input2_address=0x1082000,
        output_address=0x1082000,
    ):
        if physical_weight_c is None:
            physical_weight_c = resident_weight_c
        if physical_weight_k is None:
            physical_weight_k = 32
        proto = convert._load_proto_module(convert._repo_root())
        model = proto.Model()
        matrix = model.ops.add()
        matrix.fused_op.name = "matrix"
        conv = matrix.fused_op.op_list.add()
        conv.name = "matrix_accumulator"
        conv.target = "conv2d_int8_accumulator"
        self._set_tensor(
            conv.kwargs["input"].tensor,
            "input",
            (1, resident_weight_c, *input_hw),
            "int8",
            0x1080000,
        )
        if materialized_input_valid_channels is not None:
            conv.kwargs["materialized_input_valid_channels"].int_value = (
                materialized_input_valid_channels
            )
        if materialized_input_resident_channels is not None:
            conv.kwargs["materialized_input_resident_channels"].int_value = (
                materialized_input_resident_channels
            )
        if materialized_input_padding is not None:
            conv.kwargs["materialized_input_padding"].int_list.values.extend(
                materialized_input_padding
            )
        has_materialized_input = any(
            value is not None
            for value in (
                materialized_input_valid_channels,
                materialized_input_resident_channels,
                materialized_input_padding,
            )
        )
        materialized_payload_shape = None
        if has_materialized_input:
            payload_channels = (
                materialized_input_resident_channels
                if materialized_input_resident_channels is not None
                else resident_weight_c
            )
            left, right, top, bottom = (
                materialized_input_padding
                if materialized_input_padding is not None
                else (0, 0, 0, 0)
            )
            materialized_payload_shape = (
                1,
                payload_channels,
                input_hw[0] + top + bottom,
                input_hw[1] + left + right,
            )
            self._set_tensor(
                conv.kwargs["materialized_input_payload"].tensor,
                "materialized_input",
                materialized_payload_shape,
                "int8",
                0x1080000,
            )
        weight = conv.kwargs["weight"].tensor
        self._set_tensor(
            weight,
            "weight",
            (32, resident_weight_c, 1, 1),
            f"int{weight_precision}",
            0x10A0000,
            "global_weight",
        )
        weight_payload = conv.kwargs["weight_payload"].tensor
        self._set_tensor(
            weight_payload,
            "weight_payload",
            (physical_weight_k, physical_weight_c, 1, 1),
            f"int{weight_precision}",
            0x10A0000,
            "global_weight",
        )
        conv.kwargs["weight_valid_input_channels"].int_value = resident_weight_c
        conv.kwargs["weight_valid_output_channels"].int_value = 32
        conv.kwargs["groups"].int_value = 1
        conv.kwargs["stride"].int_list.values.extend(stride)
        if weight_precision == 4:
            weight_scale = conv.kwargs["weight_scale"].tensor
            weight_scale.node = "weight_scale"
            weight_scale.shape.extend((32, 1, 1, 1))
            weight_scale.dtype = "uint8"
            encoded_scale = conv.kwargs["weight_vector_scale_payload"].tensor
            encoded_scale.node = "weight_scale_artifact"
            encoded_scale.shape.extend((32, 1, 1, 1))
            encoded_scale.dtype = "uint8"
            conv.kwargs["block_size"].int_value = 32
            conv.kwargs["weight_scale_valid_output_channels"].int_value = 32
        conv.kwargs["padding_c"].int_value = (
            (-resident_weight_c) % 16
            if schedule_padding_c is None
            else schedule_padding_c
        )
        for field in (
            "padding_k",
            "pad_h_up",
            "pad_h_down",
            "pad_w_left",
            "pad_w_right",
        ):
            conv.kwargs[field].int_value = 0

        if contain_epilogue:
            ppu = matrix.fused_op.op_list.add()
        else:
            standalone = model.ops.add()
            standalone.op.name = "matrix_ppu"
            ppu = standalone.op
        ppu.name = "matrix_ppu"
        ppu.target = "ppu_fixed_relu"
        self._set_tensor(
            ppu.kwargs["input"].tensor,
            "matrix_accumulator",
            (1, 32, 2, 2),
            "int32",
            0x1081000,
        )
        multiplier = ppu.kwargs["primary_multiplier"].tensor
        multiplier.node = "primary_multiplier"
        multiplier.shape.extend((32,))
        multiplier.dtype = "int32"
        ppu.kwargs["primary_shift"].int_value = 16
        ppu.kwargs["secondary_multiplier"].int_value = 1
        ppu.kwargs["secondary_shift"].int_value = 0
        ppu.kwargs["integer_accumulator"].bool_value = True
        ppu.kwargs["output_precision"].int_value = 8
        ppu.kwargs["relu"].bool_value = ppu_relu

        output = matrix.output if contain_epilogue else standalone.output
        self._set_tensor(
            output,
            "output",
            output_shape,
            "int8",
            output_address,
        )
        if contain_epilogue:
            ppu.kwargs["output_payload"].tensor.CopyFrom(output)
        if eadd:
            eadd_op = matrix.fused_op.op_list.add()
            eadd_op.name = "matrix_eadd"
            eadd_op.target = "saturating_add"
            self._set_tensor(
                eadd_op.kwargs["input1"].tensor,
                "matrix_ppu",
                output_shape,
                "int8",
                0x1081800,
            )
            self._set_tensor(
                eadd_op.kwargs["input2"].tensor,
                "skip_input",
                output_shape,
                "int8",
                eadd_input2_address,
            )
        if not contain_epilogue:
            self._set_tensor(
                matrix.output,
                "matrix_accumulator",
                (1, 32, 2, 2),
                "int32",
                0x1081000,
            )
        if include_factors:
            factors = _gemm_schedule_factors()
            factors["p1"] = schedule_p1
            factors["p2"] = schedule_p2
            factors["q1"] = schedule_q1
            factors["q2"] = schedule_q2
            _set_schedule_factors(
                matrix,
                factors,
            )

        compiler_dir = root / "compiler"
        tensor_dir = compiler_dir / "tensor_files"
        tensor_dir.mkdir(parents=True)
        (compiler_dir / "model.txt").write_text(
            text_format.MessageToString(model)
        )
        (compiler_dir / "hardware_config.json").write_text(
            json.dumps(_hardware_config())
        )
        input_values = np.zeros(
            (1, resident_weight_c, *input_hw),
            dtype=np.float32,
        )
        if materialized_payload_shape is not None:
            input_values = (
                np.arange(input_values.size, dtype=np.float32).reshape(
                    input_values.shape
                )
                % 31
            ) - 15
        arrays = {
            "input": input_values,
            "weight": np.ones(
                (32, resident_weight_c, 1, 1),
                dtype=np.float32,
            ),
            "weight_payload": np.pad(
                np.ones(
                    (32, resident_weight_c, 1, 1),
                    dtype=np.float32,
                ),
                (
                    (0, physical_weight_k - 32),
                    (0, physical_weight_c - resident_weight_c),
                    (0, 0),
                    (0, 0),
                ),
            ),
            "primary_multiplier": np.ones((32,), dtype=np.float32),
            "output": np.zeros(output_shape, dtype=np.float32),
        }
        if materialized_payload_shape is not None:
            left, right, top, bottom = (
                materialized_input_padding
                if materialized_input_padding is not None
                else (0, 0, 0, 0)
            )
            materialized_values = np.zeros(
                materialized_payload_shape,
                dtype=np.float32,
            )
            materialized_values[
                :,
                :resident_weight_c,
                top : top + input_hw[0],
                left : left + input_hw[1],
            ] = input_values
            arrays["materialized_input"] = materialized_values
        if eadd:
            skip_values = np.arange(
                math.prod(output_shape),
                dtype=np.float32,
            ).reshape(output_shape)
            arrays["skip_input"] = (skip_values % 256) - 128
            arrays["output"] = arrays["skip_input"].copy()
        if corrupt_weight_payload is not None:
            index, value = corrupt_weight_payload
            arrays["weight_payload"][index] = value
        if weight_precision == 4:
            arrays["weight_scale"] = np.arange(
                32,
                dtype=np.float32,
            ).reshape((32, 1, 1, 1))
            arrays["weight_scale_artifact"] = np.arange(
                32,
                dtype=np.float32,
            ).reshape((32, 1, 1, 1))
        for name, values in arrays.items():
            values.tofile(tensor_dir / f"{name}.bin")

        output_dir = root / "cmodel"
        output_dir.mkdir()
        return convert._Converter(
            compiler_dir,
            output_dir,
            "tiny_matrix",
            write_tensors=write_tensors,
        ).convert()

    def test_self_contained_matrix_epilogue_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run_matrix_conversion(
                Path(directory),
                contain_epilogue=True,
            )

        self.assertEqual(report["summary"]["emitted_layers"], 1)
        self.assertNotIn("error", report["summary"]["issues_by_severity"])

    def test_matrix_preserves_stride2_resident_input_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._run_matrix_conversion(
                root,
                contain_epilogue=True,
                input_hw=(14, 14),
                output_shape=(1, 32, 7, 7),
                stride=(2, 2),
                schedule_p1=1,
                schedule_p2=7,
                schedule_q1=7,
                schedule_q2=1,
            )
            emitted_input = np.fromstring(
                (root / "cmodel/matrix/1_inputs.txt").read_text(),
                sep=",",
                dtype=np.int64,
            )

        self.assertEqual(report["status"], "complete")
        schedule = report["layers"][0]["schedule"]
        self.assertEqual(schedule["input_height"], 14)
        self.assertEqual(schedule["input_width"], 14)
        self.assertEqual(emitted_input.size, 16 * 14 * 14)

    def test_matrix_consumes_declared_c8_input_for_semantic_c3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._run_matrix_conversion(
                root,
                contain_epilogue=True,
                resident_weight_c=3,
                physical_weight_c=8,
                input_hw=(2, 8),
                output_shape=(1, 32, 2, 8),
                schedule_q1=8,
                schedule_q2=1,
                schedule_padding_c=8,
                materialized_input_valid_channels=3,
                materialized_input_resident_channels=8,
                materialized_input_padding=(0, 0, 0, 0),
            )
            semantic = np.fromfile(
                root / "compiler/tensor_files/input.bin",
                dtype=np.float32,
            ).reshape(1, 3, 2, 8)
            emitted = np.fromstring(
                (root / "cmodel/matrix/1_inputs.txt").read_text(),
                sep=",",
                dtype=np.int64,
            ).reshape(1, 8, 2, 8)

        self.assertEqual(report["status"], "complete")
        layer = report["layers"][0]
        self.assertEqual(layer["schedule"]["q1"], 8)
        self.assertEqual(layer["schedule"]["padding_c"], 8)
        np.testing.assert_array_equal(emitted[:, :3], semantic)
        np.testing.assert_array_equal(emitted[:, 3:], 0)

    def test_matrix_eadd_emits_exact_operand2_collateral(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._run_matrix_conversion(
                root,
                contain_epilogue=True,
                eadd=True,
                ppu_relu=False,
            )
            layer = report["layers"][0]
            operand2 = np.fromstring(
                (root / "cmodel/matrix/1_inputs2.txt").read_text(),
                sep=",",
                dtype=np.int64,
            )

            self.assertEqual(report["status"], "complete")
            self.assertEqual(layer["kind"], "gemm_convolution")
            self.assertEqual(layer["schedule"]["is_eadd"], 1)
            self.assertEqual(layer["schedule"]["num_eop"], 1)
            self.assertEqual(
                layer["schedule"]["inputs1_baseaddr"],
                layer["schedule"]["outputs0_baseaddr"],
            )
            self.assertIn("1_inputs2.txt", layer["files"])
            np.testing.assert_array_equal(
                operand2,
                np.arange(128, dtype=np.int64) - 128,
            )

    def test_matrix_eadd_requires_output_to_alias_operand2(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run_matrix_conversion(
                Path(directory),
                contain_epilogue=True,
                eadd=True,
                ppu_relu=False,
                eadd_input2_address=0x1083000,
            )

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "invalid-eadd-output-alias",
            {issue["code"] for issue in report["issues"]},
        )

    def test_matrix_writes_only_canonical_int8_weight_padding(self):
        for resident_weight_c in (1, 3):
            with self.subTest(resident_weight_c=resident_weight_c):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    report = self._run_matrix_conversion(
                        root,
                        contain_epilogue=True,
                        resident_weight_c=resident_weight_c,
                        physical_weight_c=4,
                    )
                    weights = np.fromstring(
                        (root / "cmodel/matrix/1_weights.txt").read_text(),
                        sep=",",
                        dtype=np.int64,
                    ).reshape(32, 4, 1, 1)

                self.assertNotIn(
                    "error",
                    report["summary"]["issues_by_severity"],
                )
                np.testing.assert_array_equal(
                    weights[:, :resident_weight_c],
                    1,
                )
                np.testing.assert_array_equal(
                    weights[:, resident_weight_c:],
                    0,
                )

    def test_matrix_rejects_noncanonical_int8_weight_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._run_matrix_conversion(
                root,
                contain_epilogue=True,
                resident_weight_c=16,
                physical_weight_c=20,
            )

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "invalid-weight-storage-extents",
            {issue["code"] for issue in report["issues"]},
        )

    def test_matrix_rejects_noncanonical_int4_weight_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run_matrix_conversion(
                Path(directory),
                contain_epilogue=True,
                physical_weight_c=64,
                weight_precision=4,
            )

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "invalid-weight-storage-extents",
            {issue["code"] for issue in report["issues"]},
        )

    def test_matrix_rejects_noncanonical_weight_output_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run_matrix_conversion(
                Path(directory),
                contain_epilogue=True,
                physical_weight_k=64,
            )

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "invalid-weight-storage-extents",
            {issue["code"] for issue in report["issues"]},
        )

    def test_matrix_rejects_weight_payload_semantic_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run_matrix_conversion(
                Path(directory),
                contain_epilogue=True,
                corrupt_weight_payload=((0, 0, 0, 0), 2),
            )

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "inconsistent-weight-storage-payload",
            {issue["code"] for issue in report["issues"]},
        )

    def test_matrix_rejects_nonzero_weight_c_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run_matrix_conversion(
                Path(directory),
                contain_epilogue=True,
                resident_weight_c=1,
                physical_weight_c=4,
                corrupt_weight_payload=((0, 1, 0, 0), 1),
            )

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "inconsistent-weight-storage-payload",
            {issue["code"] for issue in report["issues"]},
        )

    def test_matrix_rejects_output_extent_mismatch(self):
        for output_shape in (
            (32, 2, 2),
            (2, 32, 2, 2),
            (1, 32, 3, 2),
            (1, 32, 2, 3),
        ):
            with self.subTest(output_shape=output_shape):
                with tempfile.TemporaryDirectory() as directory:
                    report = self._run_matrix_conversion(
                        Path(directory),
                        contain_epilogue=True,
                        output_shape=output_shape,
                        write_tensors=False,
                    )

                self.assertEqual(report["status"], "failed")
                self.assertIn(
                    "invalid-gemm-output-shape",
                    {issue["code"] for issue in report["issues"]},
                )

    def test_matrix_encodes_raw_weight_scale_for_cmod(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._run_matrix_conversion(
                root,
                contain_epilogue=True,
                weight_precision=4,
            )
            scales = np.fromstring(
                (root / "cmodel/matrix/1_weights_vector_scale.txt").read_text(),
                sep=",",
                dtype=np.int64,
            )

        self.assertNotIn("error", report["summary"]["issues_by_severity"])
        np.testing.assert_array_equal(scales, np.arange(32))

    def test_incomplete_matrix_target_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run_matrix_conversion(
                Path(directory),
                contain_epilogue=False,
            )

        self.assertEqual(report["summary"]["emitted_layers"], 0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["summary"]["unsupported_operations"], 2)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("unsupported-cmod-target", codes)
        self.assertIn("no-executable-cmod-layers", codes)

    def test_conversion_requires_compiler_schedule_factors(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run_matrix_conversion(
                Path(directory),
                contain_epilogue=True,
                include_factors=False,
            )

        self.assertEqual(report["summary"]["emitted_layers"], 0)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("missing-target-schedule-factor", codes)
        self.assertNotIn("layer-skipped", codes)
        self.assertIn("no-executable-cmod-layers", codes)

    def test_float_target_payload_is_rejected_without_conversion(self):
        proto = convert._load_proto_module(convert._repo_root())
        tensor = proto.Tensor()
        tensor.node = "float_output"
        tensor.dtype = "float32"

        with self.assertRaises(convert.SkipLayer):
            self.converter.load_exact_integer_tensor(
                tensor,
                "conv",
                "GEMM output",
                "int8",
                -128,
                127,
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "unfinalized-target-payload",
        )

    def test_fractional_target_payload_is_rejected_without_rounding(self):
        proto = convert._load_proto_module(convert._repo_root())
        tensor = proto.Tensor()
        tensor.node = "fractional_output"
        tensor.dtype = "int4"
        self.converter.load_tensor = lambda tensor, layer: np.asarray([0.5])

        with self.assertRaises(convert.SkipLayer):
            self.converter.load_exact_integer_tensor(
                tensor,
                "dwc",
                "DWC output",
                "int4",
                -8,
                7,
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-target-payload",
        )

    def test_text_writer_rejects_fractional_values_without_rounding(self):
        result = convert.LayerResult("layer", "test", {}, [])
        self.converter.write_tensors = False

        with self.assertRaises(convert.SkipLayer):
            self.converter.write_values(
                result,
                "1_inputs.txt",
                np.asarray([0.0, 1.5], dtype=np.float32),
                -128,
                127,
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-cmodel-payload",
        )

    def test_text_writer_rejects_out_of_range_values_without_clipping(self):
        result = convert.LayerResult("layer", "test", {}, [])
        self.converter.write_tensors = False

        with self.assertRaises(convert.SkipLayer):
            self.converter.write_payload(
                result,
                "1_weights.txt",
                np.asarray([-8, 7], dtype=np.float32),
                4,
                weight=True,
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-cmodel-payload",
        )

    def test_text_writer_preserves_exact_integer_payload(self):
        result = convert.LayerResult("layer", "test", {}, [])
        values = np.asarray([-128, -1, 0, 127], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            self.converter.output_dir = Path(directory)
            self.converter.write_tensors = True
            self.converter.write_payload(
                result,
                "1_inputs.txt",
                values,
                8,
            )
            text = (Path(directory) / "layer/1_inputs.txt").read_text()

        self.assertEqual(text, "-128,-1,0,127\n")
        self.assertEqual(result.files, ["1_inputs.txt"])

    def test_canonical_artifact_is_consumed_without_layout_rewrite(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "artifact_owner"
        artifact = operation.op.kwargs["encoded_payload"].tensor
        artifact.node = "encoded_payload"
        artifact.shape.extend((1, 2, 3, 4, 5))
        artifact.dtype = "uint8"
        expected = np.arange(120, dtype=np.float32).reshape((1, 2, 3, 4, 5))
        self.converter.array_cache = {"encoded_payload": expected}

        actual = self.converter.load_serialized_artifact(
            operation,
            "encoded_payload",
            "layer",
            "encoded payload",
            expected.shape,
        )

        np.testing.assert_array_equal(actual, expected)

    def test_vector_scale_lane_order_is_converter_owned(self):
        activation = np.arange(12).reshape((1, 2, 2, 3))
        weight = np.arange(48).reshape((4, 2, 2, 3))

        np.testing.assert_array_equal(
            convert.encode_cmod_vector_scale(
                activation,
                kind="output",
            ),
            activation.transpose(0, 3, 2, 1),
        )
        np.testing.assert_array_equal(
            convert.encode_cmod_vector_scale(
                weight,
                kind="weight",
                output_lanes=2,
            ),
            weight.reshape(2, 2, 2, 2, 3).transpose(0, 1, 4, 3, 2),
        )

    def test_vector_scale_loader_accepts_legacy_encoded_artifact(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "scale_owner"
        raw = np.arange(12, dtype=np.float32).reshape((1, 2, 2, 3))
        encoded = raw.transpose(0, 3, 2, 1).copy()
        artifact = operation.op.kwargs["output_vector_scale_payload"].tensor
        artifact.node = "legacy_scale"
        artifact.shape.extend(encoded.shape)
        artifact.dtype = "uint8"
        self.converter.array_cache = {"legacy_scale": encoded}

        actual = self.converter.load_cmod_vector_scale(
            operation,
            "output_vector_scale_payload",
            "layer",
            "output vector scale",
            raw,
            kind="output",
        )

        np.testing.assert_array_equal(actual, encoded)

    def test_serialized_artifact_rejects_runtime_residence(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "artifact_owner"
        artifact = operation.op.kwargs["encoded_payload"].tensor
        artifact.node = "encoded_payload"
        artifact.shape.extend((1,))
        artifact.dtype = "uint8"
        artifact.scratchpad.address = 0x1080000
        artifact.scratchpad.instance = "global_activation"

        with self.assertRaises(convert.SkipLayer):
            self.converter.load_serialized_artifact(
                operation,
                "encoded_payload",
                "layer",
                "encoded payload",
                (1,),
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "resident-serialized-artifact",
        )

    def test_output_payload_requires_an_explicit_serialized_role(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "ambiguous"
        for node in ("small", "large"):
            tensor = operation.outputs.tensors.add()
            tensor.node = node
            tensor.shape.extend((1, 1))
            tensor.dtype = "int8"

        with self.assertRaises(convert.SkipLayer):
            self.converter.output_payload(operation)

        self.assertEqual(
            self.converter.issues[-1].code,
            "missing-serialized-tensor-role",
        )

    def test_missing_builder_operands_report_root_error(self):
        proto = convert._load_proto_module(convert._repo_root())
        for builder_name, execution_kind, needs_descriptor in (
            ("build_gemm", "gemm_convolution", True),
            ("build_depthwise", "nmp_depthwise_convolution", True),
            ("build_pool", "nmp_max_pool", False),
            ("build_upsample", "nmp_nearest_upsample", False),
        ):
            with self.subTest(builder=builder_name):
                operation = proto.Operation()
                operation.op.name = builder_name
                if needs_descriptor:
                    operation.op.target = (
                        "conv2d_vsq"
                        if builder_name == "build_gemm"
                        else "conv2d_int8_accumulator"
                    )
                else:
                    output = operation.output
                    output.node = f"{builder_name}_output"
                    output.shape.extend((1, 1, 1, 1))
                    output.dtype = "int8"
                    operation.op.kwargs["output_payload"].tensor.CopyFrom(
                        output
                    )

                self.converter.issues = []
                with self.assertRaises(convert.SkipLayer):
                    getattr(self.converter, builder_name)(
                        operation,
                        execution_kind,
                    )
                self.assertEqual(
                    self.converter.issues[-1].code,
                    "missing-executable-operand",
                )

    def test_missing_nmp_operands_are_rejected_without_derivation(self):
        proto = convert._load_proto_module(convert._repo_root())
        depthwise = proto.Operation()
        depthwise.op.name = "conv"
        depthwise.op.target = "conv2d"

        with self.assertRaises(convert.SkipLayer):
            self.converter.explicit_nmp_parameters(
                depthwise,
                "depthwise",
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "missing-depthwise-fixed-point-operands",
        )

    def test_out_of_range_nmp_operands_are_rejected(self):
        proto = convert._load_proto_module(convert._repo_root())
        epilogue = proto.Operation()
        epilogue.op.name = "dwc_fixed"
        epilogue.op.target = "nmp_fixed_requantize"
        epilogue.op.kwargs["scale_factor"].int_value = 65536
        epilogue.op.kwargs["binary_point"].int_value = 23
        epilogue.op.kwargs["bias_factor"].int_value = -3
        epilogue.op.kwargs["output_precision"].int_value = 8
        epilogue.op.kwargs["relu"].bool_value = False

        with self.assertRaises(convert.SkipLayer):
            self.converter.explicit_nmp_parameters(epilogue, "depthwise")

        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-depthwise-fixed-point-operands",
        )

    def test_nmp_storage_view_is_consumed_without_graph_traversal(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "shuffle"
        operation.op.kwargs["nmp_c2"].int_value = 1
        operation.op.kwargs["nmp_k2"].int_value = 4
        operation.op.kwargs["is_from_split"].bool_value = True
        operation.op.kwargs["split_channel_idx"].int_value = 1
        operation.op.kwargs["split_cur_size"].int_value = 32
        operation.op.kwargs["split_idx"].int_value = 0
        operation.op.kwargs["split_scale"].float_value = 2.0
        operation.op.kwargs["split_total_size"].int_value = 64

        self.assertEqual(
            self.converter.nmp_storage_fields(operation),
            {
                "nmp_c2": 1,
                "nmp_k2": 4,
                "is_from_split": 1,
                "split_channel_idx": 1,
                "split_cur_size": 32,
                "split_idx": 0,
                "split_scale": 2.0,
                "split_total_size": 64,
            },
        )

    def test_explicit_ppu_operands_are_read_without_rederivation(self):
        proto = convert._load_proto_module(convert._repo_root())
        ppu = proto.Operation()
        ppu.op.name = "ppu"
        ppu.op.target = "ppu_fixed_relu"
        multiplier = ppu.op.kwargs["primary_multiplier"].tensor
        multiplier.node = "primary_multiplier"
        multiplier.shape.extend((2,))
        bias = ppu.op.kwargs["bias_payload"].tensor
        bias.node = "bias_payload"
        bias.shape.extend((2,))
        ppu.op.kwargs["primary_shift"].int_value = 21
        ppu.op.kwargs["secondary_multiplier"].int_value = 49104
        ppu.op.kwargs["secondary_shift"].int_value = 18
        ppu.op.kwargs["integer_accumulator"].bool_value = True
        ppu.op.kwargs["output_precision"].int_value = 8
        ppu.op.kwargs["relu"].bool_value = True
        self.converter.array_cache = {
            "primary_multiplier": np.asarray([17668, 11070]),
        }

        actual = self.converter.explicit_ppu_parameters(ppu, "layer")

        self.assertEqual(actual["scale_int"].tolist(), [17668, 11070])
        self.assertEqual(actual["binary_point"], 21)
        self.assertEqual(actual["nonlinear_multiplier"], 49104)
        self.assertEqual(actual["nonlinear_shift"], 18)
        self.assertEqual(actual["bias"].node, "bias_payload")
        self.assertEqual(actual["output_precision"], 8)

    def test_semantic_ppu_is_rejected_without_rederivation(self):
        proto = convert._load_proto_module(convert._repo_root())
        ppu = proto.Operation()
        ppu.op.name = "ppu"
        ppu.op.target = "ppu_fixed_relu"

        with self.assertRaises(convert.SkipLayer):
            self.converter.explicit_ppu_parameters(ppu, "layer")

        self.assertEqual(
            self.converter.issues[-1].code,
            "noninteger-pe-output",
        )

    def test_incomplete_ppu_operands_report_structured_error(self):
        proto = convert._load_proto_module(convert._repo_root())
        ppu = proto.Operation()
        ppu.op.name = "ppu"
        ppu.op.target = "ppu_fixed_relu"
        ppu.op.kwargs["integer_accumulator"].bool_value = True
        multiplier = ppu.op.kwargs["primary_multiplier"].tensor
        multiplier.node = "primary_multiplier"
        multiplier.shape.extend((1,))

        with self.assertRaises(convert.SkipLayer):
            self.converter.explicit_ppu_parameters(ppu, "layer")

        self.assertEqual(
            self.converter.issues[-1].code,
            "missing-pe-fixed-point-operands",
        )

    def _pool_operation(self):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "pool"
        operation.op.target = "max_pool2d"
        self._set_tensor(
            operation.op.kwargs["input"].tensor,
            "pool_input",
            (1, 16, 4, 4),
            "int8",
            0x1080000,
        )
        operation.op.kwargs["kernel_size"].int_list.values.extend((2, 2))
        operation.op.kwargs["stride"].int_list.values.extend((2, 2))
        operation.op.kwargs["dilation"].int_list.values.extend((1, 1))
        output = operation.output
        self._set_tensor(
            output,
            "pool_output",
            (1, 16, 2, 2),
            "int8",
            0x1081000,
        )
        operation.op.kwargs["output_payload"].tensor.CopyFrom(output)
        operation.op.kwargs["is_pool_1bank"].bool_value = False
        operation.op.kwargs["nmp_c2"].int_value = 0
        operation.op.kwargs["nmp_k2"].int_value = 0
        operation.op.kwargs["is_from_split"].bool_value = False
        operation.op.kwargs["padding_c"].int_value = 0
        operation.op.kwargs["padding_k"].int_value = 0
        _set_schedule_factors(
            operation,
            _nmp_schedule_factors(
                height=4,
                width=4,
            ),
        )
        return operation

    def test_pool_uses_input_spatial_factor_coverage(self):
        operation = self._pool_operation()
        self.converter.network = "tiny"
        self.converter.write_tensors = False
        self.converter.array_cache = {
            "pool_input": np.zeros(
                (1, 16, 4, 4),
                dtype=np.float32,
            ),
            "pool_output": np.zeros(
                (1, 16, 2, 2),
                dtype=np.float32,
            ),
        }

        result = self.converter.build_pool(
            operation,
            "nmp_max_pool",
        )
        self.assertEqual(result.schedule["p1"], 4)
        self.assertEqual(result.schedule["q1"], 4)

        operation.op.kwargs["schedule_factor_p1"].int_value = 2
        with self.assertRaises(convert.SkipLayer):
            self.converter.build_pool(
                operation,
                "nmp_max_pool",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "target-schedule-semantic-mismatch",
        )

    def _upsample_operation(self, dtype="int8"):
        proto = convert._load_proto_module(convert._repo_root())
        operation = proto.Operation()
        operation.op.name = "upsample"
        operation.op.target = "upsample_nearest2d"
        input_tensor = operation.op.kwargs["input"].tensor
        input_tensor.node = "upsample_input"
        input_tensor.shape.extend((1, 2, 2, 2))
        input_tensor.dtype = dtype
        input_tensor.scratchpad.address = 0x1080000
        input_tensor.scratchpad.instance = "global_activation"
        operation.op.kwargs["scale_factors"].scalar_list.values.extend(
            (2.0, 2.0)
        )
        output = operation.output
        output.node = "upsample_output"
        output.shape.extend((1, 2, 4, 4))
        output.dtype = dtype
        output.scratchpad.address = 0x1081000
        output.scratchpad.instance = "global_activation"
        operation.op.kwargs["output_payload"].tensor.CopyFrom(output)
        operation.op.kwargs["nmp_c2"].int_value = 0
        operation.op.kwargs["nmp_k2"].int_value = 0
        operation.op.kwargs["is_from_split"].bool_value = False
        operation.op.kwargs["padding_c"].int_value = 0
        operation.op.kwargs["padding_k"].int_value = 0
        _set_schedule_factors(
            operation,
            _nmp_schedule_factors(
                height=4,
                width=4,
            ),
        )
        return operation

    def test_upsample_preserves_int8_payload_without_rescaling(self):
        operation = self._upsample_operation()
        input_values = np.asarray(
            [
                [
                    [[-128, -1], [0, 127]],
                    [[3, 4], [5, 6]],
                ]
            ],
            dtype=np.float32,
        )
        output_values = np.repeat(
            np.repeat(input_values, 2, axis=2),
            2,
            axis=3,
        )
        self.converter.array_cache = {
            "upsample_input": input_values,
            "upsample_output": output_values,
        }
        self.converter.network = "tiny"

        with tempfile.TemporaryDirectory() as directory:
            self.converter.output_dir = Path(directory)
            self.converter.write_tensors = True
            result = self.converter.build_upsample(
                operation,
                "nmp_nearest_upsample",
            )
            input_text = (Path(directory) / "upsample/1_inputs.txt").read_text()
            output_text = (
                Path(directory) / "upsample/1_outputs.txt"
            ).read_text()

        serialized_input = np.fromstring(
            input_text,
            sep=",",
            dtype=np.int64,
        )
        serialized_output = np.fromstring(
            output_text,
            sep=",",
            dtype=np.int64,
        )
        np.testing.assert_array_equal(
            serialized_input,
            input_values.astype(np.int64).ravel(),
        )
        np.testing.assert_array_equal(
            serialized_output,
            output_values.astype(np.int64).ravel(),
        )
        self.assertEqual(result.schedule["ia_precision"], 8)
        self.assertEqual(result.schedule["oa_precision"], 8)

    def test_upsample_accepts_equivalent_output_size_encoding(self):
        operation = self._upsample_operation()
        del operation.op.kwargs["scale_factors"]
        operation.op.kwargs["output_size"].int_list.values.extend((4, 4))
        input_values = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
        self.converter.array_cache = {
            "upsample_input": input_values,
            "upsample_output": np.repeat(
                np.repeat(input_values, 2, axis=2),
                2,
                axis=3,
            ),
        }
        self.converter.network = "tiny"
        self.converter.write_tensors = False

        result = self.converter.build_upsample(
            operation,
            "nmp_nearest_upsample",
        )

        self.assertEqual(result.schedule["r"], 2)
        self.assertEqual(result.schedule["s"], 2)

    def test_upsample_rejects_int4_payload(self):
        operation = self._upsample_operation(dtype="int4")
        self.converter.network = "tiny"

        with self.assertRaises(convert.SkipLayer):
            self.converter.build_upsample(
                operation,
                "nmp_nearest_upsample",
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "unsupported-upsample-precision",
        )

    def test_upsample_rejects_rescaled_int8_output(self):
        operation = self._upsample_operation()
        input_values = np.ones((1, 2, 2, 2), dtype=np.float32)
        self.converter.array_cache = {
            "upsample_input": input_values,
            "upsample_output": np.full(
                (1, 2, 4, 4),
                2,
                dtype=np.float32,
            ),
        }
        self.converter.network = "tiny"
        self.converter.write_tensors = False

        with self.assertRaises(convert.SkipLayer):
            self.converter.build_upsample(
                operation,
                "nmp_nearest_upsample",
            )

        self.assertEqual(
            self.converter.issues[-1].code,
            "non-bit-preserving-upsample",
        )

    def test_upsample_rejects_output_factor_mismatch(self):
        operation = self._upsample_operation()
        operation.op.kwargs["schedule_factor_q1"].int_value = 2
        input_values = np.ones((1, 2, 2, 2), dtype=np.float32)
        self.converter.array_cache = {
            "upsample_input": input_values,
            "upsample_output": np.repeat(
                np.repeat(input_values, 2, axis=2),
                2,
                axis=3,
            ),
        }
        self.converter.network = "tiny"
        self.converter.write_tensors = False

        with self.assertRaises(convert.SkipLayer):
            self.converter.build_upsample(
                operation,
                "nmp_nearest_upsample",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "target-schedule-semantic-mismatch",
        )

    def test_shuffle_requires_both_output_memory_instances(self):
        proto = convert._load_proto_module(convert._repo_root())
        shuffle = proto.Operation()
        shuffle.op.name = "shuffle"
        shuffle.op.target = "channel_shuffle"
        for field, node, address in (
            ("input1", "shuffle_input1", 0x1080000),
            ("input2", "shuffle_input2", 0x1081000),
        ):
            tensor = shuffle.op.kwargs[field].tensor
            tensor.node = node
            tensor.shape.extend((1, 64, 2, 2))
            tensor.dtype = "int8"
            tensor.scratchpad.address = address
            tensor.scratchpad.instance = "global_activation"
        output = shuffle.output
        output.node = "shuffle_output"
        output.shape.extend((1, 128, 2, 2))
        output.dtype = "int8"
        output.scratchpad.address = 0x1082000
        output.scratchpad.instance = "global_activation"
        shuffle.op.kwargs["output_payload"].tensor.CopyFrom(output)
        shuffle.op.kwargs["groups"].int_value = 2
        shuffle.op.kwargs["output_addresses"].int_list.values.extend(
            (0x1082000, 0x1083000)
        )
        for index in (0, 1):
            shuffle.op.kwargs[f"output{index}_memory_instance"].str_value = (
                "global_activation"
            )
        shuffle.op.kwargs["nmp_c2"].int_value = 0
        shuffle.op.kwargs["nmp_k2"].int_value = 0
        shuffle.op.kwargs["is_from_split"].bool_value = False
        shuffle.op.kwargs["padding_c"].int_value = 0
        shuffle.op.kwargs["padding_k"].int_value = 0
        _set_schedule_factors(
            shuffle,
            _nmp_schedule_factors(
                channels=2,
                outputs=4,
                height=2,
                width=2,
            ),
        )
        self.converter.network = "network"
        self.converter.write_tensors = False
        self.converter.array_cache = {
            "shuffle_input1": np.zeros(
                (1, 64, 2, 2),
                dtype=np.float32,
            ),
            "shuffle_input2": np.zeros(
                (1, 64, 2, 2),
                dtype=np.float32,
            ),
            "shuffle_output": np.zeros(
                (1, 128, 2, 2),
                dtype=np.float32,
            ),
        }

        result = self.converter.build_shuffle(
            shuffle,
            "nmp_channel_shuffle",
        )

        self.assertEqual(result.schedule["outputs0_baseaddr"], "0x1082000")
        self.assertEqual(result.schedule["outputs1_baseaddr"], "0x1083000")
        shuffle.op.kwargs["output1_memory_instance"].str_value = "global_weight"
        with self.assertRaises(convert.SkipLayer):
            self.converter.build_shuffle(
                shuffle,
                "nmp_channel_shuffle",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-hardware-memory-access",
        )
        global_weight = next(
            instance
            for instance in self.converter.hardware_config["global_memory"][
                "instances"
            ]
            if instance["name"] == "global_weight"
        )
        global_weight["accepted_accesses"].append(
            {
                "role": "output",
                "compute_unit": "nmp",
            }
        )
        result = self.converter.build_shuffle(
            shuffle,
            "nmp_channel_shuffle",
        )
        self.assertEqual(result.schedule["outputs1_baseaddr"], "0x1083000")

        shuffle.op.kwargs["output0_memory_instance"].str_value = "global_weight"
        with self.assertRaises(convert.SkipLayer):
            self.converter.build_shuffle(
                shuffle,
                "nmp_channel_shuffle",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-shuffle-output-memory-instance",
        )
        shuffle.op.kwargs["output0_memory_instance"].str_value = (
            "global_activation"
        )
        shuffle.op.kwargs["output1_memory_instance"].str_value = (
            "global_activation"
        )

        output.scratchpad.address = 0x1082002
        shuffle.op.kwargs["output_payload"].tensor.CopyFrom(output)
        for field, value in (
            ("concat_channel_index_0", 0),
            ("concat_idx_0", 1),
            ("concat_size_0", 2),
            ("concat_total_size_0", 4),
        ):
            shuffle.op.kwargs[field].int_value = value
        shuffle.op.kwargs["is_to_concat"].bool_value = True
        shuffle.op.name = "model_4_6_shuffle"
        result = self.converter.build_shuffle(
            shuffle,
            "nmp_channel_shuffle",
        )
        self.assertEqual(result.schedule["outputs0_baseaddr"], "0x1082000")

        shuffle.op.name = "channel_shuffle_default_11"
        with self.assertRaises(convert.SkipLayer):
            self.converter.build_shuffle(
                shuffle,
                "nmp_channel_shuffle",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "inconsistent-shuffle-output-address",
        )

        shuffle.op.name = "shuffle"
        output.scratchpad.address = 0x1082000
        shuffle.op.kwargs["output_payload"].tensor.CopyFrom(output)
        for field in convert.cmod_utils.CONCAT_FIELDS:
            del shuffle.op.kwargs[field]
        shuffle.op.kwargs["schedule_factor_k1"].int_value = 3
        with self.assertRaises(convert.SkipLayer):
            self.converter.build_shuffle(
                shuffle,
                "nmp_channel_shuffle",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "target-schedule-semantic-mismatch",
        )

    def test_copy_consumes_compiler_schedule_factors(self):
        proto = convert._load_proto_module(convert._repo_root())
        copy = proto.Operation()
        copy.op.name = "shuffle_cpy1"
        copy.op.target = "channel_shuffle_copy"
        input_tensor = copy.op.kwargs["input"].tensor
        input_tensor.node = "copy_input"
        input_tensor.shape.extend((1, 128, 2, 2))
        input_tensor.dtype = "int8"
        input_tensor.scratchpad.address = 0x10A0100
        input_tensor.scratchpad.instance = "global_activation"
        copy.op.kwargs["block_channels"].int_value = 32
        output = copy.output
        output.node = "copy_output"
        output.shape.extend((1, 128, 2, 2))
        output.dtype = "int8"
        output.scratchpad.address = 0x10A0200
        output.scratchpad.instance = "global_activation"
        copy.op.kwargs["output_payload"].tensor.CopyFrom(output)
        output_payload = copy.op.kwargs["output_payload"].tensor
        source_payload = copy.op.kwargs["copy_source_payload"].tensor
        source_payload.node = "copy_source_backing"
        source_payload.shape.extend((1, 128, 2, 2))
        source_payload.dtype = "int8"
        source_payload.scratchpad.address = 0x10A0000
        source_payload.scratchpad.instance = "global_activation"
        destination_payload = copy.op.kwargs["copy_destination_payload"].tensor
        destination_payload.node = "copy_destination_backing"
        destination_payload.shape.extend((1, 128, 2, 2))
        destination_payload.dtype = "int8"
        destination_payload.scratchpad.address = 0x10A0080
        destination_payload.scratchpad.instance = "global_activation"
        block_payload = copy.op.kwargs["copy_block_payload"].tensor
        block_payload.node = "copy_block"
        block_payload.shape.extend((1, 32, 2, 2))
        block_payload.dtype = "int8"
        copy.op.kwargs["copy_source_output"].int_value = 0
        copy.op.kwargs["copy_source_entry"].int_value = 1
        copy.op.kwargs["copy_destination_entry"].int_value = 2
        copy.op.kwargs["nmp_c2"].int_value = 3
        copy.op.kwargs["nmp_k2"].int_value = 3
        copy.op.kwargs["is_to_concat"].bool_value = True
        copy.op.kwargs["concat_total_size_0"].int_value = 128

        self.converter.network = "network"
        block_values = (
            np.arange(
                1 * 32 * 2 * 2,
                dtype=np.float32,
            ).reshape((1, 32, 2, 2))
            - 64
        )
        self.converter.array_cache = {"copy_block": block_values}
        _set_schedule_factors(
            copy,
            _nmp_schedule_factors(
                height=2,
                width=2,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            self.converter.output_dir = Path(directory)
            self.converter.write_tensors = True
            result = self.converter.build_copy(
                copy,
                "nmp_copy",
            )
            inputs = np.fromstring(
                (Path(directory) / "shuffle_cpy1/1_inputs.txt").read_text(),
                sep=",",
                dtype=np.int64,
            )
            outputs = np.fromstring(
                (Path(directory) / "shuffle_cpy1/1_outputs.txt").read_text(),
                sep=",",
                dtype=np.int64,
            )

        self.assertEqual(result.kind, "nmp_copy")
        self.assertEqual(result.schedule["p1"], 2)
        self.assertEqual(result.schedule["q1"], 2)
        self.assertEqual(result.schedule["is_copy"], 1)
        self.assertNotIn("padding_c", result.schedule)
        self.assertNotIn("padding_k", result.schedule)
        self.assertNotIn("is_from_split", result.schedule)
        self.assertNotIn("concat_channel_index_0", result.schedule)
        self.assertNotIn("concat_idx_0", result.schedule)
        self.assertNotIn("concat_size_0", result.schedule)
        np.testing.assert_array_equal(inputs, block_values.ravel())
        np.testing.assert_array_equal(outputs, block_values.ravel())

        global_weight = next(
            instance
            for instance in self.converter.hardware_config["global_memory"][
                "instances"
            ]
            if instance["name"] == "global_weight"
        )
        global_weight["accepted_accesses"].extend(
            (
                {
                    "role": "activation",
                    "compute_unit": "nmp",
                },
                {
                    "role": "output",
                    "compute_unit": "nmp",
                },
            )
        )
        for tensor in (
            input_tensor,
            output_payload,
            source_payload,
            destination_payload,
        ):
            tensor.scratchpad.instance = "global_weight"
        self.converter.write_tensors = False
        result = self.converter.build_copy(
            copy,
            "nmp_copy",
        )
        self.assertEqual(result.schedule["inputs0_baseaddr"], "0x10a0100")
        self.assertEqual(result.schedule["outputs0_baseaddr"], "0x10a0200")

        input_tensor.scratchpad.instance = "global_activation"
        with self.assertRaises(convert.SkipLayer):
            self.converter.build_copy(
                copy,
                "nmp_copy",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "invalid-shuffle-copy-payload-view",
        )
        input_tensor.scratchpad.instance = "global_weight"

        copy.op.kwargs["schedule_factor_c1"].int_value = 2
        with self.assertRaises(convert.SkipLayer):
            self.converter.build_copy(
                copy,
                "nmp_copy",
            )
        self.assertEqual(
            self.converter.issues[-1].code,
            "target-schedule-semantic-mismatch",
        )


if __name__ == "__main__":
    unittest.main()
