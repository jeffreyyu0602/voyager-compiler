import math
import operator
import os
from typing import List

import torch
import torch.nn.functional as F
from torch.fx.node import map_arg
from torch._higher_order_ops import while_loop
from torchao.quantization.pt2e.export_utils import WrapperModule
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
    _resolve_fx_graph_input,
    _resolve_fx_graph_outputs,
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


def inner_loop_body(
    indices: List[torch.Tensor],
    O_i: torch.Tensor,
    l_i: torch.Tensor,
    m_i: torch.Tensor,
    Qi: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    Br: int,
    Bc: int,
):
    d = key.shape[-1]
    b, h, i, j = indices

    Kj = quantized_ops.load_tile(key, [b, h, j], [1, 1, Bc, d], [0, 1, 2])
    Vj = quantized_ops.load_tile(value, [b, h, j], [1, 1, Bc, d], [0, 1, 2])

    scaling = 1.0 / math.sqrt(d)

    scores = torch.matmul(Qi, Kj.transpose(-1, -2)) * scaling
    # mask = _create_causal_mask(i * Br, j * Bc, Br, Bc, device=key.device)
    # scores = scores.masked_fill(~mask, -float("inf"))

    m_block = torch.maximum(m_i, scores.amax(dim=-1, keepdim=True))

    alpha = torch.exp(m_i - m_block)
    P = torch.exp(scores - m_block)
    l_i = alpha * l_i + P.sum(dim=-1, keepdim=True)
    O_i = alpha * O_i + torch.matmul(P, Vj)

    # Perform a add nop to copy new maximum to the correct memory location
    m_i = m_block.add(0)

    return O_i, l_i, m_i


def outer_loop_body(indices, query, key, value, Br, Bc):
    B, H, N, d = query.shape
    b, h, i, init_j = indices
    Qi = quantized_ops.load_tile(query, [b, h, i], [1, 1, Br, d], [0, 1, 2])

    kwargs = {"device": query.device, "dtype": query.dtype}

    # Running max and sum for this block of queries
    O_i = torch.zeros((1, 1, Br, d), **kwargs)
    l_i = torch.zeros((1, 1, Br, 1), **kwargs)
    m_i = torch.full((1, 1, Br, 1), -float("inf"), **kwargs)

    def cond_fn(b, h, i, j, *args):
        return j < N // Bc

    def body_fn(b, h, i, j, *args):
        next_state = inner_loop_body([b, h, i, j], *args, Qi, key, value, Br, Bc)
        return (b.clone(), h.clone(), i.clone(), j + 1) + next_state

    *_, O_i, l_i, m_i = while_loop(
        cond_fn,
        body_fn,
        (b, h, i, init_j, O_i, l_i, m_i),
    )

    return O_i / l_i


def flash_attention_pattern(query, key, value, Br=64, Bc=128):
    B, H, N, d = query.shape
    Tr = N // Br

    # Output buffer
    O = torch.zeros((B, H, N, d), device=query.device, dtype=query.dtype)

    def cond_fn(b, h, i, j, O_buf):
        return b < B

    def body_fn(b, h, i, j, O_buf):
        indices = [b, h, i, j]
        O_tile = outer_loop_body(indices, query, key, value, Br, Bc)
        O_buf = quantized_ops.store_tile(
            O_tile, O_buf, [b, h, i], [1, 1, Br, d], [0, 1, 2]
        )
        new_indices = quantized_ops.increment_indices(indices, [B + 1, H, Tr, 1])
        return (*new_indices, O_buf)

    kwargs = {"device": query.device, "dtype": torch.int32}
    initial_state = (
        torch.tensor(0, **kwargs),  # b
        torch.tensor(0, **kwargs),  # h
        torch.tensor(0, **kwargs),  # i
        torch.tensor(0, **kwargs),  # j
        O,
    )

    *_, final_O = while_loop(cond_fn, body_fn, initial_state)

    return final_O


def _lower_nested_loops(
    model: torch.fx.GraphModule,
    loop_bounds,
    env,
    namer,
    ssa_indices=None,
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
                ssa_indices=ssa_indices,
                depth=depth + 1
            )

            indices_count = sum(len(bounds) for bounds in loop_bounds[:depth])
            bounds = loop_bounds[depth]

            for i in reversed(range(len(bounds))):
                stmts = Loops(
                    index=ssa_indices[indices_count + i],
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
                and index < len(ssa_indices)
                or node.target == quantized_ops.increment_indices.default
            ):
                continue

            if is_nop(node):
                env[node] = env[node.all_input_nodes[0]]
            elif node.target == operator.getitem:
                idx = node.args[1]
                env[node] = env[node.all_input_nodes[0]][idx]
            else:
                mem_loc = "Scratchpad"
                if (
                    node.target == quantized_ops.store_tile.default
                    or depth == 0
                    and node.target == aten.zeros.default
                ):
                    mem_loc = "DRAM"

                # If the node is an output of the loop, accumulate into the input buffer
                output_ir = env[input_nodes[index]] if index is not None else None

                op = Operation.from_fx_node(
                    node, env, namer, mem_space=mem_loc, outputs=output_ir
                )
                body.append(op)
                print(op.format())

    outputs = []
    for i, n in enumerate(output_node.args[0]):
        if depth > 0 and i < len(ssa_indices):
            outputs.append(env[input_nodes[i]])
        else:
            outputs.append(env[n])

    return body, outputs


def lower_flash_attention(query, key, value, Br=64, Bc=128):
    gm = export_model(
        WrapperModule(flash_attention_pattern),
        (query, key, value),
        kwargs={"Br": Br, "Bc": Bc}
    )

    for n in list(gm.graph.nodes):
        if n.op == "placeholder" and not n.users:
            gm.graph.erase_node(n)

    gm.graph.lint()
    gm.recompile()

    ShapeProp(gm).propagate(query, key, value)

    B, H, N, d = query.shape
    loop_bounds = [[B, H, N // Br], [N // Bc]]

    namer = NameGenerator()
    env = {}
    args = []
    ssa_indices = []

    i = 0

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            v = _resolve_fx_graph_input(node, namer)
            v.producer_op = Operation.from_fx_node(node, env, namer)
            env[node] = v
            args.append(v)
        elif node.op == "get_attr" and node.target.startswith("lifted_tensor"):
            ssa_index = IndexValue(name=namer.new_index(), expr=f"{node}_{i}")
            env[node] = ssa_index
            ssa_indices.append(ssa_index)
            i += 1

    for k, v in env.items():
        print(f"{k} -> {v}")

    body, outputs = _lower_nested_loops(
        gm,
        loop_bounds,
        env=env,
        namer=namer,
        ssa_indices=ssa_indices,
    )

    module = Module("main", args=args, body=body, results=outputs)
    print("\nFinal IR:")
    print(module.format())

    return module, gm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="A script to process data and save results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory where the output files will be saved",
        default="./output"  # Optional: provides a fallback
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    B, H, N, D = 1, 32, 1024, 64
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    out_ref = F.scaled_dot_product_attention(q, k, v)
    out_gold = flash_attention_pattern(q, k, v, Br=512, Bc=128)

    print(out_ref)
    print(out_gold)

    # Compare in fp32 for a fair error metric
    max_err = (out_gold.float() - out_ref.float()).abs().max().item()
    rms_err = torch.mean((out_gold.float() - out_ref.float()) ** 2).sqrt().item()
    print(f"max_err={max_err:.6g}  rms_err={rms_err:.6g}\n\n")

    example_inputs = (q.cpu(), k.cpu(), v.cpu(), 512, 128)
    module, gm = lower_flash_attention(*example_inputs)
    run_memory_pass(module)

    proto = generate_proto(module, gm, example_inputs, output_dir=args.output_dir)
    with open(os.path.join(args.output_dir, 'module.txt'), "w") as f:
        f.write(text_format.MessageToString(proto))
