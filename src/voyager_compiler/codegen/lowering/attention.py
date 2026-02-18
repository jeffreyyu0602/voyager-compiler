import math
import operator
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.fx.node import map_arg
from torch.utils._pytree import tree_flatten
from torch._higher_order_ops import while_loop
from google.protobuf import text_format

import voyager_compiler.decomposed
from voyager_compiler.quantize_pt2e import export_model
from voyager_compiler.codegen import ShapeProp, is_nop
from voyager_compiler.codegen.lowering.ir import (
    IndexValue,
    Module,
    Loops,
    NameGenerator,
    Operation,
    IRNode,
    _propagate_dtype,
)
from voyager_compiler.codegen.lowering.allocator import run_memory_pass
from voyager_compiler.codegen.lowering.codegen import generate_proto


aten = torch.ops.aten
quantized_ops = torch.ops.quantized_ops


def _create_causal_mask(i: int, j: int, Br: int, Bc: int, device=None):
    """
    Returns a boolean mask of shape [Bm, Bn] where True means "allowed".
    """
    i = torch.arange(Br, device=device).unsqueeze(1) + i  # [Br, 1]
    j = torch.arange(Bc, device=device).unsqueeze(0) + j  # [1, Bc]
    return (j <= i)  # [Br, Bc]


class FlashAttention(torch.nn.Module):
    """
    torch.export friendly Flash Attention implementation for compiler lowering.

    The auxiliary quantization tensors (qmap/codebooks/etc.) are passed in as
    arguments and registered as buffers for quantizing attention probs on the fly.
    QKV scales and codebooks are are passed as forward args during runtime.
    """

    def __init__(
        self,
        *,
        qmap: Optional[torch.Tensor] = None,
        axes: Optional[List[int]] = None,
        block_size: int = None,
        quant_max: Optional[float] = None,
        force_scale_power_of_two: bool = False,
        scale_qmap: Optional[torch.Tensor] = None,
        output_code: Optional[torch.Tensor] = None,
        input_code: Optional[torch.Tensor] = None,
        accumulate_fp32: bool = False,
    ):
        super().__init__()
        self.register_buffer("qmap", qmap)
        self.register_buffer("input_code", input_code)
        self.register_buffer("scale_qmap", scale_qmap)
        self.register_buffer("output_code", output_code)

        self.axes = axes
        self.block_size = block_size
        self.quant_max = quant_max
        self.force_scale_power_of_two = force_scale_power_of_two
        self.accumulate_fp32 = accumulate_fp32

    def _inner_loop_body(
        self,
        indices,
        O_i: torch.Tensor,
        l_i: torch.Tensor,
        m_i: torch.Tensor,
        Qi: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        d = key.shape[-1]
        b, h, i, j = indices

        Kj = quantized_ops.load_tile(key, [b, h, j], [1, 1, Bc, d], [0, 1, 2])
        Vj = quantized_ops.load_tile(value, [b, h, j], [1, 1, Bc, d], [0, 1, 2])

        use_quant = (
            query_scale is not None
            and key_scale is not None
            and value_scale is not None
        )

        if use_quant:
            # Load per-block scales for K and V
            Kj_scale = quantized_ops.load_tile(
                key_scale,
                [b, h, j],
                [1, 1, Bc, d // self.block_size],
                [0, 1, 2],
            )
            Vj_scale = quantized_ops.load_tile(
                value_scale,
                [b, h, j],
                [1, 1, Bc // self.block_size, d],
                [0, 1, 2],
            )

            scores = quantized_ops.matmul_mx(
                Qi,
                Kj.transpose(-1, -2),
                input_scale=query_scale,
                weight_scale=Kj_scale.transpose(-1, -2),
                block_size=self.block_size,
                input_code=query_code,
                weight_code=key_code,
            )
        else:
            scores = torch.matmul(Qi, Kj.transpose(-1, -2))

        scores = scores * (1 / math.sqrt(d))
        # mask = _create_causal_mask(i * Br, j * Bc, Br, Bc, device=key.device)
        # scores = scores.masked_fill(~mask, -float("inf"))

        # Numerically stable accumulation
        if self.accumulate_fp32 and scores.dtype != torch.float32:
            scores = scores.to(torch.float32)

        m_block = torch.maximum(m_i, scores.amax(dim=-1, keepdim=True))

        alpha = torch.exp(m_i - m_block)
        P = torch.exp(scores - m_block)
        l_i = alpha * l_i + P.sum(dim=-1, keepdim=True)

        if use_quant:
            P_scale, P_q = quantized_ops.quantize_mx(
                P,
                qmap=self.qmap,
                axes=self.axes,
                block_size=self.block_size,
                quant_max=self.quant_max,
                force_scale_power_of_two=self.force_scale_power_of_two,
                scale_qmap=self.scale_qmap,
                output_code=self.output_code,
            )

            attention_probs = quantized_ops.matmul_mx(
                P_q,
                Vj,
                input_scale=P_scale,
                weight_scale=Vj_scale,
                block_size=self.block_size,
                input_code=self.input_code,
                weight_code=value_code,
            )
        else:
            attention_probs = torch.matmul(P, Vj)

        O_i = alpha * O_i + attention_probs

        # Perform a nop to copy new maximum to the correct memory location
        m_i = m_block.add(0)

        return O_i, l_i, m_i

    def _outer_loop_body(
        self,
        indices,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        **kwargs,
    ):
        B, H, N, d = query.shape
        b, h, i, init_j = indices

        Qi = quantized_ops.load_tile(query, [b, h, i], [1, 1, Br, d], [0, 1, 2])

        Qi_scale = None
        if (query_scale := kwargs.get("query_scale")) is not None:
            Qi_scale = quantized_ops.load_tile(
                query_scale,
                [b, h, i],
                [1, 1, Br, d // self.block_size],
                [0, 1, 2],
            )

        # Accumulator dtype choice
        acc_dtype = torch.float32 if self.accumulate_fp32 else query.dtype
        O_i = torch.zeros((1, 1, Br, d), device=query.device, dtype=acc_dtype)
        l_i = torch.zeros((1, 1, Br, 1), device=query.device, dtype=acc_dtype)
        m_i = torch.full((1, 1, Br, 1), -float("inf"), device=query.device, dtype=acc_dtype)

        def cond_fn(b, h, i, j, *args):
            return j < N // Bc

        def body_fn(b, h, i, j, *args):
            additional_inputs = (Qi, key, value, Br, Bc)
            additional_kwargs = {
                **kwargs,
                "query_scale": Qi_scale,
            }
            next_state = self._inner_loop_body(
                [b, h, i, j], *args, *additional_inputs, **additional_kwargs
            )
            return (b, h, i, j + 1) + next_state

        *_, O_i, l_i, m_i = while_loop(
            cond_fn,
            body_fn,
            (b, h, i, init_j, O_i, l_i, m_i),
        )

        out = O_i / l_i
        if self.accumulate_fp32 and out.dtype != query.dtype:
            out = out.to(query.dtype)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int = 64,
        Bc: int = 128,
        **kwargs,
    ):
        B, H, N, d = query.shape
        Tr = N // Br

        # Allocate output buffer
        O = torch.empty((B, H, N, d), device=query.device, dtype=query.dtype)

        def cond_fn(b, h, i, j, O_buf):
            return b < B

        def body_fn(b, h, i, j, O_buf):
            indices = [b, h, i, j]
            O_tile = self._outer_loop_body(
                indices,
                query=query,
                key=key,
                value=value,
                Br=Br,
                Bc=Bc,
                **kwargs,
            )
            O_buf = quantized_ops.store_tile(
                O_tile, O_buf, [b, h, i], [1, 1, Br, d], [0, 1, 2]
            )
            new_indices = quantized_ops.increment_indices(
                indices, [B + 1, H, Tr, 1]
            )
            return (*new_indices, O_buf)

        factory_kwargs = {"device": query.device, "dtype": torch.int32}
        initial_state = (
            torch.tensor(0, **factory_kwargs),  # b
            torch.tensor(0, **factory_kwargs),  # h
            torch.tensor(0, **factory_kwargs),  # i
            torch.tensor(0, **factory_kwargs),  # j
            O,
        )

        *_, final_O = while_loop(cond_fn, body_fn, initial_state)
        return final_O


class FlashAttentionFlat(FlashAttention):
    """
    Flattened version of FlashAttention where both the outer and inner loops are
    unrolled for testing purpose.
    """

    def _inner_loop_body(
        self,
        indices,
        O_i: torch.Tensor,
        l_i: torch.Tensor,
        m_i: torch.Tensor,
        Qi: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        query_scale: Optional[torch.Tensor] = None,
        key_scale: Optional[torch.Tensor] = None,
        value_scale: Optional[torch.Tensor] = None,
        query_code: Optional[torch.Tensor] = None,
        key_code: Optional[torch.Tensor] = None,
        value_code: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        d = key.shape[-1]
        b, h, i, j = indices

        Kj = quantized_ops.load_tile(
            key, [], [1, 1, Bc, d], [], static_indices=[b, h, j, 0]
        )
        Vj = quantized_ops.load_tile(
            value, [], [1, 1, Bc, d], [], static_indices=[b, h, j, 0]
        )

        use_quant = (
            query_scale is not None
            and key_scale is not None
            and value_scale is not None
        )

        if use_quant:
            # Load per-block scales for K and V
            Kj_scale = quantized_ops.load_tile(
                key_scale,
                [],
                [1, 1, Bc, d // self.block_size],
                [],
                static_indices=[b, h, j, 0],
            )
            Vj_scale = quantized_ops.load_tile(
                value_scale,
                [],
                [1, 1, Bc // self.block_size, d],
                [],
                static_indices=[b, h, j, 0],
            )

            scores = quantized_ops.matmul_mx(
                Qi,
                Kj.transpose(-1, -2),
                input_scale=query_scale,
                weight_scale=Kj_scale.transpose(-1, -2),
                block_size=self.block_size,
                input_code=query_code,
                weight_code=key_code,
            )
        else:
            scores = torch.matmul(Qi, Kj.transpose(-1, -2))

        scores = scores * (1 / math.sqrt(d))
        # mask = _create_causal_mask(i * Br, j * Bc, Br, Bc, device=key.device)
        # scores = scores.masked_fill(~mask, -float("inf"))

        # Numerically stable accumulation
        if self.accumulate_fp32 and scores.dtype != torch.float32:
            scores = scores.to(torch.float32)

        m_block = torch.maximum(m_i, scores.amax(dim=-1, keepdim=True))

        alpha = torch.exp(m_i - m_block)
        P = torch.exp(scores - m_block)
        l_i = alpha * l_i + P.sum(dim=-1, keepdim=True)

        if use_quant:
            P_scale, P_q = quantized_ops.quantize_mx(
                P,
                qmap=self.qmap,
                axes=self.axes,
                block_size=self.block_size,
                quant_max=self.quant_max,
                force_scale_power_of_two=self.force_scale_power_of_two,
                scale_qmap=self.scale_qmap,
                output_code=self.output_code,
            )

            attention_probs = quantized_ops.matmul_mx(
                P_q,
                Vj,
                input_scale=P_scale,
                weight_scale=Vj_scale,
                block_size=self.block_size,
                input_code=self.input_code,
                weight_code=value_code,
            )
        else:
            attention_probs = torch.matmul(P, Vj)

        O_i = alpha * O_i + attention_probs

        # Perform a nop to copy new maximum to the correct memory location
        m_i = m_block.add(0)

        return O_i, l_i, m_i

    def _outer_loop_body(
        self,
        indices,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int,
        Bc: int,
        **kwargs,
    ):
        B, H, N, d = query.shape
        b, h, i = indices

        Qi = quantized_ops.load_tile(
            query, [], [1, 1, Br, d], [], static_indices=[b, h, i, 0]
        )

        Qi_scale = None
        if (query_scale := kwargs.get("query_scale")) is not None:
            Qi_scale = quantized_ops.load_tile(
                query_scale,
                [],
                [1, 1, Br, d // self.block_size],
                [],
                static_indices=[b, h, i, 0],
            )

        # Accumulator dtype choice
        acc_dtype = torch.float32 if self.accumulate_fp32 else query.dtype
        O_i = torch.zeros((1, 1, Br, d), device=query.device, dtype=acc_dtype)
        l_i = torch.zeros((1, 1, Br, 1), device=query.device, dtype=acc_dtype)
        m_i = torch.full((1, 1, Br, 1), -float("inf"), device=query.device, dtype=acc_dtype)

        for j in range(N // Bc):
            additional_inputs = (Qi, key, value, Br, Bc)
            additional_kwargs = {
                **kwargs,
                "query_scale": Qi_scale,
            }
            O_i, l_i, m_i = self._inner_loop_body(
                [b, h, i, j], O_i, l_i, m_i, *additional_inputs, **additional_kwargs
            )

        out = O_i / l_i
        if self.accumulate_fp32 and out.dtype != query.dtype:
            out = out.to(query.dtype)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        Br: int = 64,
        Bc: int = 128,
        **kwargs,
    ):
        B, H, N, d = query.shape
        Tr = N // Br

        # Allocate output buffer
        O = torch.empty((B, H, N, d), device=query.device, dtype=query.dtype)

        import itertools
        for b, h, i in itertools.product(range(B), range(H), range(Tr)):
            O_tile = self._outer_loop_body(
                [b, h, i],
                query=query,
                key=key,
                value=value,
                Br=Br,
                Bc=Bc,
                **kwargs,
            )
            quantized_ops.store_tile(
                O_tile, O, [], [1, 1, Br, d], [], static_indices=[b, h, i, 0]
            )

        return O


def _is_reading_from_dram(node, named_modules):
    if node.target == quantized_ops.store_tile.default:
        return True

    for user in node.users:
        if user.target == quantized_ops.load_tile.default:
            return True
        elif user.target == quantized_ops.store_tile.default:
            if user.args[1] == node:
                return True
        elif user.target == torch.ops.higher_order.while_loop:
            loop_body = named_modules[user.args[1].target]
            loop_inputs = user.args[2] + user.args[3]

            placeholders = [
                n for n in loop_body.graph.nodes if n.op == "placeholder"
            ]
            assert len(placeholders) == len(loop_inputs)

            index = loop_inputs.index(node)

            if _is_reading_from_dram(placeholders[index], named_modules):
                return True

    return False


def _lower_nested_loops(
    model: torch.fx.GraphModule,
    loop_bounds,
    env,
    namer,
    index_values=None,
    depth=0,
):
    named_modules = dict(model.named_modules())

    output_node = next(n for n in model.graph.nodes if n.op == "output")
    outputs_order = {n: i for i, n in enumerate(output_node.args[0])}

    input_nodes = [n for n in model.graph.nodes if n.op == "placeholder"]

    model.graph.print_tabular()

    def load_arg(a):
        return map_arg(a, lambda n: n.value)

    body: List[IRNode] = []

    for node in model.graph.nodes:
        if node.op == "placeholder":
            sn = node.meta.get("source_node", None)
            if sn is not None:
                env[node] = env[sn]
        elif node.target == torch.ops.higher_order.while_loop:
            loop_body = named_modules[node.args[1].target]
            loop_body_inputs = node.args[2] + node.args[3]

            placeholders = [
                n for n in loop_body.graph.nodes if n.op == "placeholder"
            ]
            assert len(placeholders) == len(loop_body_inputs)

            # Link iter_args to variables in the outer scope
            for placeholder, input_node in zip(placeholders, loop_body_inputs):
                placeholder.meta["source_node"] = input_node

            example_inputs = load_arg(loop_body_inputs)
            ShapeProp(loop_body).propagate(*example_inputs)

            stmts, outputs = _lower_nested_loops(
                loop_body,
                loop_bounds,
                env=env,
                namer=namer,
                index_values=index_values,
                depth=depth + 1
            )

            indices_count = sum(len(bounds) for bounds in loop_bounds[:depth])
            bounds = loop_bounds[depth]

            for i in reversed(range(len(bounds))):
                stmts = Loops(
                    index=index_values[indices_count + i],
                    start=0,
                    end=bounds[i],
                    step=1,
                    body=stmts if isinstance(stmts, list) else [stmts],
                )

            body.append(stmts)
            env[node] = outputs
        elif node.op == "call_function":
            # Skip emitting code for loop indices increment
            index = outputs_order.get(node, None)
            if (
                depth > 0
                and index is not None
                and index < len(index_values)
                or node.target == quantized_ops.increment_indices.default
            ):
                continue

            if is_nop(node):
                env[node] = env[node.all_input_nodes[0]]
            elif node.target == operator.getitem:
                idx = node.args[1]
                env[node] = env[node.all_input_nodes[0]][idx]
            else:
                is_dma_op = _is_reading_from_dram(node, named_modules)
                mem_loc = "DRAM" if is_dma_op else "Scratchpad"

                # If the node is a loop state, accumulate into the buffer
                output_ir = (
                    env[input_nodes[index]]
                    if depth > 0 and index is not None else None
                )

                op = Operation.from_fx_node(
                    node, env, namer, mem_space=mem_loc, outputs=output_ir
                )
                body.append(op)
                print(op.format())

    outputs = []
    for i, n in enumerate(output_node.args[0]):
        if depth > 0 and i < len(index_values):
            outputs.append(env[input_nodes[i]])
        else:
            outputs.append(env[n])

    return body, outputs


def lower_flash_attention(pattern, query, key, value, Br=64, Bc=128, kwargs=None):
    gm = export_model(
        pattern,
        (query, key, value, Br, Bc),
        kwargs=kwargs,
    )

    # Remove Br and Bc from the graph since they're compile-time constants
    for n in list(gm.graph.nodes):
        if n.op == "placeholder" and not n.users:
            gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()

    example_args = (query, key, value)
    flatten_args, spec = tree_flatten((example_args, kwargs))

    ShapeProp(gm).propagate(*flatten_args)

    B, H, N, d = query.shape
    loop_bounds = [[B, H, N // Br], [N // Bc]]

    namer = NameGenerator()
    env = {}
    placeholders = []
    parameters = []
    index_values = []

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ir_node = Operation.from_fx_node(node, env, namer)
            placeholders.extend(ir_node.outputs)
        elif node.op == "get_attr":
            if node.target.startswith("lifted_tensor"):
                ssa_index = IndexValue(name=namer.new_index(), expr=node.target)
                env[node] = ssa_index
                index_values.append(ssa_index)
            elif isinstance(getattr(gm, node.target), torch.Tensor):
                ir_node = Operation.from_fx_node(node, env, namer)
                parameters.extend(ir_node.outputs)

    args = placeholders + parameters

    print("Initial environment mapping:")
    for k, v in env.items():
        print(f"{k} -> {v}")

    body, outputs = _lower_nested_loops(
        gm,
        loop_bounds,
        env=env,
        namer=namer,
        index_values=index_values,
    )

    module = Module("main", args=args, body=body, results=outputs)
    print("\nFinal IR:")
    print(module.format())

    return module, gm


if __name__ == "__main__":
    """
    Usage:
    python voyager-compiler/src/voyager_compiler/codegen/lowering/attention.py \
        --output_dir test/compiler/networks/flash_attention/MXNF4
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="A script to process data and save results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory where the output files will be saved",
        default="test/compiler/networks/flash_attention"
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    B, H, N, D = 1, 32, 1024, 64
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    flash_attention_pattern = FlashAttention()

    out_ref = F.scaled_dot_product_attention(q, k, v)
    out_gold = flash_attention_pattern(q, k, v, Br=512, Bc=128)

    print(out_ref)
    print(out_gold)

    # Compare in fp32 for a fair error metric
    max_err = (out_gold.float() - out_ref.float()).abs().max().item()
    rms_err = torch.mean((out_gold.float() - out_ref.float()) ** 2).sqrt().item()
    print(f"max_err={max_err:.6g}  rms_err={rms_err:.6g}\n\n")

    example_inputs = (q.cpu(), k.cpu(), v.cpu(), 512, 128)
    module, gm = lower_flash_attention(flash_attention_pattern, *example_inputs)
    run_memory_pass(module)

    os.makedirs(args.output_dir, exist_ok=True)

    proto = generate_proto(module, gm, example_inputs)
    with open(os.path.join(args.output_dir, 'fp16_model.txt'), "w") as f:
        f.write(text_format.MessageToString(proto))


    from voyager_compiler.fake_quantize import get_quantization_map

    nf4_qmap, nf4_code = get_quantization_map("nf4_6", device=device)
    midpoints = (nf4_code[:-1] + nf4_code[1:]) / 2

    scale_qmap = get_quantization_map("fp8_e5m3", device=device)

    quant_kwargs = {
        "qmap": nf4_qmap,
        "axes": [-1],
        "block_size": 64,
        "quant_max": 31,
        "force_scale_power_of_two": False,
        "scale_qmap": scale_qmap,
        "output_code": midpoints,
    }

    qs, qq = quantized_ops.quantize_mx(q, **quant_kwargs)
    ks, kq = quantized_ops.quantize_mx(k, **quant_kwargs)
    vs, vq = quantized_ops.quantize_mx(v, **quant_kwargs)

    flash_attention_quantized_pattern = FlashAttentionFlat(
        **quant_kwargs,
        input_code=nf4_code.clone(),
    )

    example_inputs = (qq, kq, vq, 512, 128)
    example_kwargs = {
        "query_scale": qs,
        "key_scale": ks,
        "value_scale": vs,
        "query_code": nf4_code.clone(),
        "key_code": nf4_code.clone(),
        "value_code": nf4_code.clone(),
    }
    input_dtypes = [
        "int4", "int4", "int4", "fp8_e5m3", "fp8_e5m3", "fp8_e5m3",
        "int6", "int6", "int6", "int4", "fp8_e5m3", None, None
    ]

    out_gold = flash_attention_quantized_pattern(
        *example_inputs, **example_kwargs
    )

    print(out_ref)
    print(out_gold)

    # Compare in fp32 for a fair error metric
    max_err = (out_gold.float() - out_ref.float()).abs().max().item()
    rms_err = torch.mean((out_gold.float() - out_ref.float()) ** 2).sqrt().item()
    print(f"max_err={max_err:.6g}  rms_err={rms_err:.6g}\n\n")

    # Move inputs to CPU for export
    example_inputs = tuple(
        x.cpu() if isinstance(x, torch.Tensor) else x for x in example_inputs
    )
    example_kwargs = {k: v.cpu() for k, v in example_kwargs.items()}

    module, gm = lower_flash_attention(
        flash_attention_quantized_pattern.cpu(),
        *example_inputs,
        kwargs=example_kwargs,
    )
    _propagate_dtype(module, input_dtypes=input_dtypes)
    run_memory_pass(module)

    flatten_args, spec = tree_flatten((example_inputs[:3], example_kwargs))
    output_dir = os.path.join(args.output_dir, "tensor_files")
    proto = generate_proto(module, gm, flatten_args, output_dir=output_dir)

    with open(os.path.join(args.output_dir, 'model.txt'), "w") as f:
        f.write(text_format.MessageToString(proto))
