import logging
import torch
from voyager_compiler import QuantizationSpec, QuantizationConfig
from voyager_compiler.fake_quantize import FusedAmaxObsFakeQuantize


logger = logging.getLogger(__name__)


QUANTIZATION_CONFIGS = {
    "w4a4_attn6": {
        torch.nn.Linear: [
            "nf4,qs=microscaling,bs=64,ax=-1",
            "nf4,qs=microscaling,bs=64,ax=-1",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1",
            "int6,qs=microscaling,bs=64,ax=-2",
        ],
    },
    "w4a4_attn6_s8": {
        torch.nn.Linear: [
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "int6,qs=microscaling,bs=64,ax=-2,scale=fp8_e5m3",
        ],
    },
    "w4a4_attn6_heada6": {
        torch.nn.Linear: [
            "nf4_6,qs=microscaling,bs=64,ax=-1",
            "nf4_6,qs=microscaling,bs=64,ax=-1",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1",
            "int6,qs=microscaling,bs=64,ax=-2",
        ],
        ("lm_head", torch.ops.aten.linear.default, 0): [
            "int6,qs=microscaling,bs=64,ax=-1",
            "nf4_6,qs=microscaling,bs=64,ax=-1",
        ],
    },
    "w4a4_attn6_heada6_s8": {
        torch.nn.Linear: [
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "int6,qs=microscaling,bs=64,ax=-2,scale=fp8_e5m3",
        ],
        ("lm_head", torch.ops.aten.linear.default, 0): [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        ],
    },
    "w4a4_of": {
        torch.nn.Linear: [
            "nf4_6,qs=microscaling,bs=64,ax=-1,opct=0.01",
            "nf4_6,qs=microscaling,bs=64,ax=-1",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1",
            "nf4_6,qs=microscaling,bs=64,ax=-2,othr=6.0",
        ],
    },
    "w4a4_of_s8": {
        torch.nn.Linear: [
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3,opct=0.01",
            "nf4_6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
        ],
        torch.ops.aten.matmul.default: [
            "int6,qs=microscaling,bs=64,ax=-1,scale=fp8_e5m3",
            "nf4_6,qs=microscaling,bs=64,ax=-2,scale=fp8_e5m3,othr=6.0",
        ],
    },
}


def set_qconfig(quantizer, qconfigs, force_scale_power_of_two=False):
    def make_qspec(spec):
        if spec is None:
            return None
        quant_spec = QuantizationSpec.from_str(spec)
        quant_spec.observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize.with_args(
            force_scale_power_of_two=force_scale_power_of_two,
        )
        return quant_spec

    for key, qspec in qconfigs.items():
        if qspec is None:
            qconfig = None
        elif isinstance(qspec, str):
            quant_spec = make_qspec(qspec)
            qconfig = QuantizationConfig(quant_spec, None, quant_spec, None)
        else:
            num_specs = len(qspec)

            if num_specs not in (2, 3):
                raise ValueError(f"Invalid qspec: {qspec}")

            activation = make_qspec(qspec[0])
            weight = make_qspec(qspec[1])
            bias = make_qspec(qspec[2]) if num_specs == 3 else None

            qconfig = QuantizationConfig(activation, None, weight, bias)

        if isinstance(key, tuple):
            logger.info(f"Setting qconfig for module name, object type and order: {key}")
            quantizer.set_module_name_object_type_order(*key, qconfig)
        elif isinstance(key, str):
            logger.info(f"Setting qconfig for module name: {key}")
            quantizer.set_module_name(key, qconfig)
        elif isinstance(key, type) and issubclass(key, torch.nn.Module):
            logger.info(f"Setting qconfig for module type: {key}")
            quantizer.set_module_type(key, qconfig)
        elif isinstance(key, torch._ops.OpOverload):
            logger.info(f"Setting qconfig for op overload: {key}")
            quantizer.set_object_type(key, qconfig)
        else:
            raise ValueError(f"Invalid module name or type: {key}")

    return quantizer
