"""
Shared lowering utilities for Voyager compiler loop-nest IR construction.

Used by both attention.py and b2bgemm.py.
"""

from __future__ import annotations

import operator
from typing import Dict, List, Optional, Tuple

import torch
from torch.fx.node import map_arg
from torch.utils._pytree import tree_flatten

from voyager_compiler.codegen import ShapeProp, is_nop
from voyager_compiler.codegen.lowering.ir import (
    IndexValue,
    IRNode,
    Loops,
    NameGenerator,
    Operation,
    TensorBox,
)

quantized_ops = torch.ops.quantized_ops


def _prepare_ir_graph(
    gm: torch.fx.GraphModule,
    example_args: tuple,
    namer: NameGenerator,
    env: Dict,
    kwargs: Optional[dict] = None,
) -> Tuple[List, List, List[IndexValue], List[float]]:
    """
    Shared setup for lowering an FX GraphModule to Voyager IR.

    Runs ShapeProp, then walks the graph to collect:
      - placeholders : graph input nodes → IR args
      - parameters   : get_attr tensor nodes → IR params
      - index_values : lifted_tensor scalar nodes → IR IndexValues (loop indices)
      - starts       : scalar .item() values of each lifted_tensor node (loop start bounds)

    Args:
        namer: SSA name generator (created by the caller).
        env:   FX-node → IR-Value mapping (created by the caller).

    Returns:
        placeholders, parameters, index_values, starts
    """
    flatten_args, _ = tree_flatten((example_args, kwargs))
    tensor_args = [a for a in flatten_args if isinstance(a, torch.Tensor)]
    ShapeProp(gm).propagate(*tensor_args)

    placeholders = []
    parameters = []
    index_values = []
    starts = []

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ir_node = Operation.from_fx_node(node, env, namer)
            placeholders.extend(ir_node.outputs)
        elif node.op == "get_attr":
            target = node.target
            if target.startswith("lifted_tensor"):
                ssa_index = IndexValue(name=namer.new_index(), expr=target)
                env[node] = ssa_index
                index_values.append(ssa_index)
                starts.append(getattr(gm, target).item())
            elif isinstance(getattr(gm, target), torch.Tensor):
                ir_node = Operation.from_fx_node(node, env, namer)
                parameters.extend(ir_node.outputs)

    return placeholders, parameters, index_values, starts


def _should_use_dram(node: torch.fx.Node, named_modules) -> bool:
    """
    Returns True if the IR value for `node` should live in DRAM.

    Heuristic: a buffer lives in DRAM if it is ever passed to load_tile /
    store_tile, or if it is a carried-state tensor that transitively feeds a
    DRAM buffer inside a nested while_loop.
    """
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
            if _should_use_dram(placeholders[index], named_modules):
                return True

    return False


def _lower_nested_loops(
    model: torch.fx.GraphModule,
    loop_bounds,
    env,
    namer: NameGenerator,
    index_values=None,
    depth: int = 0,
):
    """
    Recursively lower a while_loop-based FX graph into a nested Loops IR.

    SSA names are allocated in textual (top-down) order so that the numeric
    suffix matches the position in the formatted IR — matching MLIR's scf.for
    convention where loop results and iter_args are defined at the loop header,
    before any body operations.

    For a depth-d group with n loop levels the allocation order is:
        output[0], iter_var[0],
        output[1], iter_var[1],
        …
        output[n-1], iter_var[n-1],    ← innermost of this group
        <body ops via recursion>

    Args:
        model:        The FX GraphModule for this loop body.
        loop_bounds:  Sequence of level groups, each group is a sequence of
                      (start, end) pairs (one per loop level in that group).
        env:          FX-node → IR-Value mapping (mutated in place).
        namer:        SSA name generator.
        index_values: All IndexValue objects created for the entire nest.
        depth:        Current recursion depth (0 = outermost group).

    Returns:
        (body, outputs): body is a list of IRNodes for this scope;
                         outputs is a list of Values produced by this scope.
    """
    named_modules = dict(model.named_modules())

    output_node = next(n for n in model.graph.nodes if n.op == "output")
    outputs_order = {n: i for i, n in enumerate(output_node.args[0])}

    num_indices = len(index_values) if index_values is not None else 0

    def load_arg(a):
        return map_arg(a, lambda n: n.value)

    body: List[IRNode] = []

    for node in model.graph.nodes:
        if node.target == torch.ops.higher_order.while_loop:
            loop_body     = named_modules[node.args[1].target]
            carried_inputs  = node.args[2]
            loop_body_inputs = node.args[2] + node.args[3]

            example_inputs = load_arg(loop_body_inputs)
            ShapeProp(loop_body).propagate(*example_inputs)

            # Indices are pre-created — exclude them from the carried state.
            current_init_args = [env[n] for n in carried_inputs[num_indices:]]

            indices_count = sum(len(bounds) for bounds in loop_bounds[:depth])
            curr_loop_bound = loop_bounds[depth]
            num_loop_lvl = len(curr_loop_bound)

            placeholders = [
                n for n in loop_body.graph.nodes if n.op == "placeholder"
            ]
            assert len(placeholders) == len(loop_body_inputs)

            # ----------------------------------------------------------------
            # Pre-allocate SSA names for loop outputs and iter_vars in
            # textual (outermost-first, interleaved) order so that numeric
            # suffixes match the position in the formatted IR:
            #
            #   %out[0] = for %i[0] … iter_args(%iv[0]=%init):   <- out[0], iv[0]
            #     %out[1] = for %i[1] … iter_args(%iv[1]=%iv[0]):  <- out[1], iv[1]
            #       …
            #         <body ops>                                    <- higher numbers
            # ----------------------------------------------------------------
            output_concrete_vals = node.value[num_indices:]

            all_loop_outputs: list[list[TensorBox]] = []
            all_iter_vars:    list[list[TensorBox]] = []

            for lvl in range(num_loop_lvl):
                # Loop outputs for this level (defined at the for-header line)
                level_outs = [
                    TensorBox(
                        name=namer.new_tensor(),
                        shape=tuple(v.shape),
                        dtype=v.dtype,
                        space=init.space,
                    )
                    for v, init in zip(output_concrete_vals, current_init_args)
                ]
                all_loop_outputs.append(level_outs)

                # Iter-vars for this level (defined as block arguments, right
                # after the loop header in the textual IR)
                level_vars = [
                    TensorBox(
                        name=namer.new_tensor(),
                        shape=tuple(v.shape),
                        dtype=v.dtype,
                        space=init.space,
                    )
                    for v, init in zip(output_concrete_vals, current_init_args)
                ]
                all_iter_vars.append(level_vars)

            # ----------------------------------------------------------------
            # Wire the innermost iter_vars to the while_loop body placeholders.
            # Pass the pre-allocated TensorBoxes as `outputs` so from_fx_node
            # does not call namer.new_tensor() again.
            # ----------------------------------------------------------------
            inner_iter_vars = all_iter_vars[-1]
            inner_env = env.copy()
            iter_var_idx = 0

            for idx, p in enumerate(placeholders):
                if idx < num_indices:
                    inner_env[p] = index_values[idx]
                elif idx < len(carried_inputs):
                    pre_alloc = inner_iter_vars[iter_var_idx]
                    iter_var_idx += 1
                    # from_fx_node sets inner_env[p] = pre_alloc
                    Operation.from_fx_node(p, inner_env, namer, outputs=pre_alloc)
                else:
                    inner_env[p] = env[loop_body_inputs[idx]]

            # ----------------------------------------------------------------
            # Recurse — body ops are allocated here, after all loop-level
            # values, so their numeric suffixes are larger (textually later).
            # ----------------------------------------------------------------
            stmts, yields = _lower_nested_loops(
                loop_body,
                loop_bounds,
                env=inner_env,
                namer=namer,
                index_values=index_values,
                depth=depth + 1,
            )

            # ----------------------------------------------------------------
            # Build Loops IR from innermost to outermost (reversed), using
            # the pre-allocated outputs and iter_vars.
            # ----------------------------------------------------------------
            for i in reversed(range(num_loop_lvl)):
                is_outermost = (i == 0)
                is_innermost = (i == num_loop_lvl - 1)

                stmts = Loops(
                    index=index_values[indices_count + i],
                    start=curr_loop_bound[i][0],
                    end=curr_loop_bound[i][1],
                    step=1,
                    body=stmts if isinstance(stmts, list) else [stmts],
                    init_args=current_init_args if is_outermost else all_iter_vars[i - 1],
                    iter_vars=all_iter_vars[i],
                    yields=yields if is_innermost else all_loop_outputs[i + 1],
                    outputs=all_loop_outputs[i],
                )

            body.append(stmts)
            # Outermost loop outputs become the env entry for this while_loop node
            env[node] = index_values + all_loop_outputs[0]

        elif node.op == "call_function":
            # Skip index-increment ops and index outputs at depth > 0
            index = outputs_order.get(node)
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
                use_dram = _should_use_dram(node, named_modules)
                mem_loc  = "DRAM" if use_dram else "Scratchpad"
                op = Operation.from_fx_node(node, env, namer, mem_space=mem_loc)
                body.append(op)

    outputs = []
    for i, n in enumerate(output_node.args[0]):
        if depth == 0 or i >= len(index_values):
            outputs.append(env[n])

    return body, outputs
