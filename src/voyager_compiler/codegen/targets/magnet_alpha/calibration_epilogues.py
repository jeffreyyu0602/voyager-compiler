# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Temporary Alpha PE/NMP execution references used only by PTQ.

These modules own executable calibration behavior. Graph placement, observer
isolation, PVS carrier edges, and lifecycle ordering remain in
``calibration.py``. Nothing here participates in target lowering or training.
"""

from typing import Optional, Tuple

import torch

from voyager_compiler.codegen.targets.magnet_alpha.fixed_point import (
    ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    ALPHA_Q8_8_FRACTIONAL_BITS,
    ALPHA_Q8_8_NORMALIZATION,
    alpha_int8_conv2d_accumulator,
    alpha_nmp_epilogue,
    alpha_output_domain,
    alpha_pe_postprocess,
    alpha_pvs_quantize,
    alpha_truncate_payload,
    alpha_vsq_conv2d_accumulator,
    encode_alpha_nmp_pipeline,
    encode_alpha_pe_pipeline,
    encode_alpha_vsq_scales,
    rebalance_alpha_vsq_coarse_scale,
)
from voyager_compiler.fixed_point import (
    FixedPointFormat,
    quantize_fixed_point,
    reconstruct_fixed_point,
)

__all__ = [
    "ALPHA_Q8_8_REFERENCE_FORMAT",
    "AlphaCalibrationEpilogue",
    "AlphaNMPCalibrationEpilogue",
]


# This is the normalized PE reference, not either wrapping int16 arithmetic
# stage implemented in ``fixed_point.py``.
ALPHA_Q8_8_REFERENCE_FORMAT = FixedPointFormat(
    storage_dtype=torch.int16,
    bit_width=16,
    fractional_bits=ALPHA_Q8_8_FRACTIONAL_BITS,
    value_scale=ALPHA_Q8_8_NORMALIZATION,
    signed=True,
    saturate=True,
)


class AlphaCalibrationEpilogue(torch.nn.Module):
    """Calibration-only PE reference with an optional exact PTQ mode."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "semantic_conv_output_amax",
            torch.tensor(0.0, dtype=torch.float64),
        )
        for name, dtype in (
            ("accumulator_scale", torch.float64),
            ("real_bias", torch.float64),
            ("primary_multiplier", torch.int64),
            ("bias_payload", torch.int64),
            ("input_coarse_scale", torch.float32),
            ("input_coarse_qmap", torch.float32),
            ("input_fine_qmap", torch.float32),
            ("input_scale_qmap", torch.float32),
            ("weight_coarse_scale", torch.float32),
            ("weight_coarse_qmap", torch.float32),
            ("weight_fine_qmap", torch.float32),
            ("weight_scale_qmap", torch.float32),
        ):
            self.register_buffer(name, torch.empty(0, dtype=dtype))
        self.register_buffer(
            "exact_output_scale", torch.tensor(1.0, dtype=torch.float64)
        )
        self.exact_enabled = False
        self.exact_mode = "none"
        self.primary_shift = 0
        self.secondary_multiplier = 1
        self.secondary_shift = 0
        self.output_precision = 8
        self.uses_pvs = False
        self.relu = False
        self.input_axes = (1,)
        self.weight_axes = (1,)
        self.input_block_size = 32
        self.weight_block_size = 32
        self.input_fine_quant_max = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT
        self.weight_fine_quant_max = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT
        self.input_coarse_scale_quant_max = (
            ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
        )
        self.weight_coarse_scale_quant_max = (
            ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
        )
        self.input_force_scale_power_of_two = False
        self.weight_force_scale_power_of_two = False
        self.input_has_scale_qmap = False
        self.weight_has_scale_qmap = False
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

    @staticmethod
    def _pair(value) -> Tuple[int, int]:
        if isinstance(value, int):
            return (value, value)
        pair = tuple(int(item) for item in value)
        if len(pair) != 2:
            raise ValueError("Alpha Conv2d arguments must have two values")
        return pair

    @staticmethod
    def _channel_operand(
        value: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        value = value.to(device=values.device).reshape(-1)
        if value.numel() == 1:
            return value[0]
        channels = values.shape[1] if values.ndim > 1 else values.shape[0]
        if value.numel() != channels:
            raise ValueError(
                "Alpha PE operand must be scalar or per output channel"
            )
        shape = [1] * values.ndim
        shape[1 if values.ndim > 1 else 0] = channels
        return value.reshape(shape)

    @staticmethod
    def _observer_endpoints(observer) -> Tuple[float, float]:
        fine = getattr(observer, "scale_quant_max", None)
        if fine is None:
            fine = getattr(observer, "quant_max", None)
        coarse = getattr(observer, "coarse_scale_quant_max", None)
        if coarse is None:
            coarse = getattr(observer, "coarse_quant_max", None)
        return (
            float(
                ALPHA_DEFAULT_VSQ_FINE_ENDPOINT if fine is None else fine
            ),
            float(
                ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
                if coarse is None
                else coarse
            ),
        )

    @staticmethod
    def _axes(observer) -> Tuple[int, ...]:
        axes = observer.ch_axis
        return (axes,) if isinstance(axes, int) else tuple(axes)

    def _copy_buffer(self, name: str, value: torch.Tensor) -> None:
        setattr(self, name, value.detach().clone())

    def _configure_pipeline(
        self,
        accumulator_scale: torch.Tensor,
        real_bias: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        output_precision: int,
        uses_pvs: bool,
        relu: bool,
        vsq_operands: bool,
        accumulator_bound=None,
        fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ) -> None:
        output = alpha_output_domain(
            output_scale,
            output_precision,
            uses_pvs,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )
        pipeline = encode_alpha_pe_pipeline(
            accumulator_scale=accumulator_scale.reshape(-1),
            output=output,
            bias=real_bias.reshape(-1),
            relu=relu,
            vsq_operands=vsq_operands,
            accumulator_bound=accumulator_bound,
        )
        self._copy_buffer(
            "accumulator_scale",
            accumulator_scale.detach().to(torch.float64).reshape(-1),
        )
        self._copy_buffer(
            "real_bias", real_bias.detach().to(torch.float64).reshape(-1)
        )
        self._copy_buffer(
            "primary_multiplier",
            torch.as_tensor(pipeline.primary.values, dtype=torch.int64),
        )
        self._copy_buffer(
            "bias_payload",
            torch.as_tensor(pipeline.bias, dtype=torch.int64),
        )
        self.exact_output_scale.fill_(output.scale)
        self.primary_shift = int(pipeline.primary.shift)
        self.secondary_multiplier = int(pipeline.secondary.multiplier)
        self.secondary_shift = int(pipeline.secondary.shift)
        self.output_precision = int(output_precision)
        self.uses_pvs = bool(uses_pvs)
        self.relu = bool(relu)
        self.exact_enabled = True

    def configure_exact(
        self,
        accumulator_scale: torch.Tensor,
        real_bias: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        accumulator_real_amax: torch.Tensor,
        output_precision: int,
        uses_pvs: bool,
        relu: bool,
        fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ) -> None:
        """Configure a finalized W8 PE boundary."""
        self._configure_pipeline(
            accumulator_scale,
            real_bias,
            output_scale,
            output_precision=output_precision,
            uses_pvs=uses_pvs,
            relu=relu,
            vsq_operands=False,
            accumulator_bound=accumulator_real_amax,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )
        self.exact_mode = "w8_o4" if uses_pvs else "w8_o8"

    def _configure_vsq_quantizer(self, role: str, observer) -> None:
        self._copy_buffer(
            f"{role}_coarse_scale", observer.calculate_qparams()
        )
        self._copy_buffer(f"{role}_coarse_qmap", observer.coarse_qmap)
        self._copy_buffer(f"{role}_fine_qmap", observer.qmap)
        scale_qmap = observer.scale_qmap
        setattr(self, f"{role}_has_scale_qmap", scale_qmap is not None)
        if scale_qmap is not None:
            self._copy_buffer(f"{role}_scale_qmap", scale_qmap)
        setattr(self, f"{role}_axes", self._axes(observer))
        setattr(self, f"{role}_block_size", int(observer.block_size))
        fine, coarse = self._observer_endpoints(observer)
        setattr(self, f"{role}_fine_quant_max", fine)
        setattr(self, f"{role}_coarse_scale_quant_max", coarse)
        setattr(
            self,
            f"{role}_force_scale_power_of_two",
            bool(observer.force_scale_power_of_two),
        )

    def configure_exact_vsq(
        self,
        input_observer,
        weight_observer,
        real_bias: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        output_precision: int,
        uses_pvs: bool,
        stride,
        padding,
        dilation,
        groups: int,
        relu: bool,
        fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ) -> None:
        """Configure a finalized W4/A4 VSQ PE boundary."""
        self._configure_vsq_quantizer("input", input_observer)
        self._configure_vsq_quantizer("weight", weight_observer)
        input_coarse = rebalance_alpha_vsq_coarse_scale(
            self.input_coarse_scale,
            fine_quant_max=self.input_fine_quant_max,
            coarse_scale_quant_max=self.input_coarse_scale_quant_max,
        )
        weight_coarse = rebalance_alpha_vsq_coarse_scale(
            self.weight_coarse_scale,
            fine_quant_max=self.weight_fine_quant_max,
            coarse_scale_quant_max=self.weight_coarse_scale_quant_max,
        )
        if input_coarse.numel() != 1:
            raise ValueError("Alpha VSQ activation coarse scale must be scalar")
        folded = input_coarse.reshape(-1)[0] * weight_coarse.reshape(-1)
        self._configure_pipeline(
            folded,
            real_bias,
            output_scale,
            output_precision=output_precision,
            uses_pvs=uses_pvs,
            relu=relu,
            vsq_operands=True,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.dilation = self._pair(dilation)
        self.groups = int(groups)
        self.exact_mode = "vsq_o4" if uses_pvs else "vsq_o8"

    def _quantize_vsq(self, values: torch.Tensor, role: str):
        coarse = getattr(self, f"{role}_coarse_scale").to(
            device=values.device, dtype=values.dtype
        )
        scale_qmap = (
            getattr(self, f"{role}_scale_qmap").to(values.device)
            if getattr(self, f"{role}_has_scale_qmap")
            else None
        )
        _, fine, payload = torch.ops.quantized_ops.quantize_vsq.default(
            values,
            coarse,
            getattr(self, f"{role}_coarse_qmap").to(values.device),
            getattr(self, f"{role}_fine_qmap").to(values.device),
            getattr(self, f"{role}_axes"),
            getattr(self, f"{role}_block_size"),
            getattr(self, f"{role}_fine_quant_max"),
            getattr(self, f"{role}_force_scale_power_of_two"),
            scale_qmap,
            None,
            getattr(self, f"{role}_coarse_scale_quant_max"),
        )
        encoded = encode_alpha_vsq_scales(
            fine,
            coarse,
            fine_quant_max=getattr(self, f"{role}_fine_quant_max"),
            coarse_scale_quant_max=getattr(
                self, f"{role}_coarse_scale_quant_max"
            ),
        )
        return payload, encoded.mantissa, encoded.coarse_scale

    def exact_components(
        self,
        values: torch.Tensor,
        input_values: Optional[torch.Tensor] = None,
        weight_values: Optional[torch.Tensor] = None,
        input_components=None,
    ):
        if not self.exact_enabled:
            raise RuntimeError("Alpha PE exact calibration is not configured")
        if self.exact_mode.startswith("w8_"):
            accumulator_scale = self._channel_operand(
                self.accumulator_scale, values
            )
            real_bias = self._channel_operand(self.real_bias, values)
            accumulator = torch.round(
                (values.to(torch.float64) - real_bias) / accumulator_scale
            ).to(torch.int64)
        else:
            if input_values is None or weight_values is None:
                raise ValueError("Alpha VSQ exact calibration needs operands")
            if input_components is None:
                input_payload, input_mantissa, _ = self._quantize_vsq(
                    input_values, "input"
                )
            else:
                _coarse, input_mantissa, input_payload = input_components
            weight_payload, weight_mantissa, _ = self._quantize_vsq(
                weight_values, "weight"
            )
            accumulator = alpha_vsq_conv2d_accumulator(
                input_payload,
                weight_payload,
                input_mantissa,
                weight_mantissa,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                block_size=self.input_block_size,
            )

        primary = self._channel_operand(self.primary_multiplier, values)
        bias = self._channel_operand(self.bias_payload, values)
        pre_output = alpha_pe_postprocess(
            accumulator,
            primary,
            self.primary_shift,
            bias,
            self.secondary_multiplier,
            self.secondary_shift,
            relu=self.relu,
        )
        if self.uses_pvs:
            encoded = alpha_pvs_quantize(
                pre_output,
                output_precision=self.output_precision,
                vector_dim=1,
            )
            return (
                self.exact_output_scale.to(
                    device=values.device, dtype=values.dtype
                ),
                encoded.mantissa,
                encoded.payload,
            )
        return alpha_truncate_payload(pre_output, self.output_precision)

    def forward(
        self,
        values: torch.Tensor,
        input_values: Optional[torch.Tensor] = None,
        weight_values: Optional[torch.Tensor] = None,
        *,
        input_components=None,
        return_components: bool = False,
    ):
        if not values.is_floating_point():
            raise ValueError(
                "the Alpha calibration epilogue requires floating input"
            )
        if not self.exact_enabled:
            current = torch.amax(torch.abs(values.detach())).to(
                self.semantic_conv_output_amax
            )
            self.semantic_conv_output_amax.copy_(
                torch.maximum(self.semantic_conv_output_amax, current)
            )
            if return_components:
                raise RuntimeError("PVS components require exact calibration")
            return reconstruct_fixed_point(
                quantize_fixed_point(values, ALPHA_Q8_8_REFERENCE_FORMAT),
                ALPHA_Q8_8_REFERENCE_FORMAT,
                values.dtype,
            )

        components = self.exact_components(
            values, input_values, weight_values, input_components
        )
        if return_components:
            if not isinstance(components, tuple):
                raise RuntimeError("only an O4/PVS output has components")
            return components
        if isinstance(components, tuple):
            from voyager_compiler.ops.quantized import expand

            coarse, fine, payload = components
            return (
                payload.to(values.dtype)
                * expand(fine, payload.shape, 32).to(values.dtype)
                * coarse
            )
        return components.to(values.dtype) * self.exact_output_scale.to(
            device=values.device, dtype=values.dtype
        )


class AlphaNMPCalibrationEpilogue(torch.nn.Module):
    """Calibration-only exact reference for one W8 depthwise boundary."""

    def __init__(
        self,
        *,
        lane_width: int,
        stride,
        padding,
        dilation,
        groups: int,
    ) -> None:
        super().__init__()
        if lane_width <= 0:
            raise ValueError("Alpha NMP lane width must be positive")
        self.lane_width = int(lane_width)
        self.stride = AlphaCalibrationEpilogue._pair(stride)
        self.padding = AlphaCalibrationEpilogue._pair(padding)
        self.dilation = AlphaCalibrationEpilogue._pair(dilation)
        self.groups = int(groups)
        self.register_buffer("input_scale", torch.ones(1))
        self.register_buffer("weight_scale", torch.ones(1))
        self.register_buffer("input_qmap", torch.empty(0))
        self.register_buffer("weight_qmap", torch.empty(0))
        self.register_buffer("output_scale", torch.ones(1))
        self.exact_enabled = False
        self.output_precision = 8
        self.uses_pvs = False
        self.scale_factor = 1
        self.binary_point = 0
        self.bias_factor = 0
        self.relu = False

    def configure_exact(
        self,
        input_observer,
        weight_observer,
        real_bias: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        output_precision: int,
        uses_pvs: bool,
        relu: bool,
        fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ) -> None:
        input_scale = input_observer.calculate_qparams().detach().reshape(-1)
        weight_scale = weight_observer.calculate_qparams().detach().reshape(-1)
        if input_scale.numel() != 1 or weight_scale.numel() != 1:
            raise ValueError("Alpha NMP requires scalar W8 operand scales")
        output = alpha_output_domain(
            output_scale,
            output_precision,
            uses_pvs,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )
        multiplier, bias = encode_alpha_nmp_pipeline(
            accumulator_scale=float(input_scale[0] * weight_scale[0]),
            output_scale=output.scale,
            output_bits=output_precision,
            bias=real_bias,
            relu=relu,
        )
        self.input_scale = input_scale.clone()
        self.weight_scale = weight_scale.clone()
        self.input_qmap = input_observer.qmap.detach().clone()
        self.weight_qmap = weight_observer.qmap.detach().clone()
        self.output_scale = torch.tensor(
            [output.scale], dtype=output_scale.dtype, device=output_scale.device
        )
        self.scale_factor = int(multiplier.multiplier)
        self.binary_point = int(multiplier.shift)
        self.bias_factor = int(bias)
        self.output_precision = int(output_precision)
        self.uses_pvs = bool(uses_pvs)
        self.relu = bool(relu)
        self.exact_enabled = True

    @staticmethod
    def _quantize(
        values: torch.Tensor,
        scale: torch.Tensor,
        qmap: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.quantized_ops.quantize.default(
            values,
            scale.to(device=values.device, dtype=values.dtype),
            None,
            None,
            None,
            qmap.to(values.device),
        )

    def exact_components(
        self,
        input_values: torch.Tensor,
        weight_values: torch.Tensor,
    ):
        if not self.exact_enabled:
            raise RuntimeError("Alpha NMP exact calibration is not configured")
        input_payload = self._quantize(
            input_values, self.input_scale, self.input_qmap
        )
        weight_payload = self._quantize(
            weight_values, self.weight_scale, self.weight_qmap
        )
        logical_channels = int(input_payload.shape[1])
        if (
            logical_channels != self.groups
            or int(weight_payload.shape[0]) != logical_channels
            or int(weight_payload.shape[1]) != 1
        ):
            raise ValueError("Alpha NMP requires multiplier-one depthwise")
        physical_channels = (
            (logical_channels + self.lane_width - 1) // self.lane_width
        ) * self.lane_width
        channel_padding = physical_channels - logical_channels
        if channel_padding:
            input_payload = torch.nn.functional.pad(
                input_payload, (0, 0, 0, 0, 0, channel_padding)
            )
            weight_payload = torch.nn.functional.pad(
                weight_payload,
                (0, 0, 0, 0, 0, 0, 0, channel_padding),
            )
        accumulator = alpha_int8_conv2d_accumulator(
            input_payload,
            weight_payload,
            self.stride,
            self.padding,
            self.dilation,
            physical_channels,
        )
        pre_output = alpha_nmp_epilogue(
            accumulator,
            self.scale_factor,
            self.binary_point,
            self.bias_factor,
            relu=self.relu,
        )
        if self.uses_pvs:
            encoded = alpha_pvs_quantize(
                pre_output, output_precision=4, vector_dim=1
            )
            return (
                self.output_scale.to(
                    device=input_values.device, dtype=input_values.dtype
                ),
                encoded.mantissa,
                encoded.payload[:, :logical_channels],
            )
        return alpha_truncate_payload(
            pre_output[:, :logical_channels], output_precision=8
        )

    def forward(
        self,
        semantic_output: torch.Tensor,
        input_values: torch.Tensor,
        weight_values: torch.Tensor,
        *,
        return_components: bool = False,
    ):
        if not self.exact_enabled:
            if return_components:
                raise RuntimeError("PVS components require exact calibration")
            return semantic_output
        components = self.exact_components(input_values, weight_values)
        if return_components:
            if not isinstance(components, tuple):
                raise RuntimeError("only an O4/PVS output has components")
            return components
        if isinstance(components, tuple):
            from voyager_compiler.ops.quantized import expand

            coarse, fine, payload = components
            return (
                payload.to(semantic_output.dtype)
                * expand(fine, payload.shape, self.lane_width).to(
                    semantic_output.dtype
                )
                * coarse
            )
        return components.to(semantic_output.dtype) * self.output_scale.to(
            device=semantic_output.device, dtype=semantic_output.dtype
        )
