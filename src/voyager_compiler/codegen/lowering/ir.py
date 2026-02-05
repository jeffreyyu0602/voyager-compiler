from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.fx.node import map_arg
from torch.fx.operator_schemas import normalize_function
from torchao.quantization.pt2e.export_utils import WrapperModule
from torchao.quantization.pt2e.utils import _get_aten_graph_module_for_pattern

from ..mapping import get_tiled_tensor, get_reference_node
from ..mapping_utils import is_gemm_op, is_bmm
from ..shape_prop import ShapeProp


logger = logging.getLogger(__name__)

quantized_lib = torch.ops.quantized_ops


""" [Note: Voyager IR]

Voyager IR is an overly-simplified ML compiler IR that is meant to have minimal
level of abstraction to represent loops for BMM and tiling. Each torch.fx.Node
have a one-to-one mapping to an Operation in Voyager IR, and each Operation have
SSA inputs/outputs represented as TensorBox.

Loops are used to provide control flow for BMM and tiled computation.

"""

@dataclass
class Value:
    """Base class for Compiler IR SSA values."""
    name: str
    users: Dict["Stmt", None] = field(
        default_factory=dict, init=False, repr=False
    )
    producer_op: "Stmt" = field(default=None, init=False, repr=False)

    def replace_all_uses_with(self, new: Value):
        for user in self.users:
            for i, inp in enumerate(user.inputs):
                if inp == self:
                    user.replace_input_with(self, new)
                    new.users[user] = None
        self.users.clear()


@dataclass
class TensorBox(Value):
    """
    SSA value for tensors (Inductor-like spirit).
    Each produced tensor should be a fresh TensorBox instance.
    """
    shape: tuple[int, ...]
    dtype: torch.dtype
    space: str = "DRAM"

    def short_type(self) -> str:
        shape = ",".join(str(d) for d in self.shape)
        dtype = _dtype_name(self.dtype)
        return f"tensor<{shape}x{dtype}>"

    def __str__(self) -> str:
        return f"{self.name}:{self.short_type()}@{self.space}"

    def __hash__(self):
        # Uses the unique memory address; safe and fast for IR nodes
        return id(self)


@dataclass
class IndexValue(Value):
    """
    SSA value for loop indices / scalar-ish IR values.
    For minimalism, we carry a string expression and an optional debug name.
    """
    expr: Optional[str] = None  # e.g. "0", "N", "i*128", etc.

    def __str__(self) -> str:
        if self.expr is None:
            return f"{self.name}:index"
        return f"{self.name}:index={self.expr}"

    def __hash__(self):
        # Uses the unique memory address; safe and fast for IR nodes
        return id(self)


def lift_arg(a: Any, env, inputs) -> Any:
    if isinstance(a, torch.fx.Node):
        sn = a.meta.get("source_node", a)
        v = env[sn]
        if isinstance(v, (list, tuple)):
            inputs.extend(list(v))
            return [vv.name for vv in v]
        inputs.append(v)
        return v.name
    if isinstance(a, (list, tuple)):
        return [lift_arg(x, env, inputs) for x in a]
    if isinstance(a, dict):
        return {k: lift_arg(v, env, inputs) for k, v in a.items()}
    return a


def link_operation(op: Stmt):
    if isinstance(op, Loops):
        for body_op in op.body:
            link_operation(body_op)
    else:
        for input_val in op.inputs:
            if op not in input_val.users:
                input_val.users[op] = None

        for output_val in op.outputs:
            output_val.producer_op = op


def _get_gemm_tiled_input_pattern(node: torch.fx.Node, gemm_node, tiling):
    """
    We support GEMM tiling on X, K, C dimensions. Input and input scale are
    tiled on X and C dimensions, weight and weight scale are tiled on C and K
    dimensions, bias is tiled on K dimension.
    """
    logger.debug(f"Lowering GEMM for {node.op} node {node} with tiling: {tiling}")

    kwargs = _get_node_args_and_kwargs(gemm_node)[1]
    node_to_key = {
        n.meta.get("source_node", n): k for k, n in kwargs.items()
        if isinstance(n, torch.fx.Node)
    }
    input_nodes = node.all_input_nodes

    assert len(tiling) == 3, "GEMM tiling should have 3 dimensions (X, K, C)"

    output_shape = gemm_node.value.shape
    num_batch_dims = len(output_shape) - 2 if is_bmm(gemm_node) else 0
    idx_X = num_batch_dims + 0
    idx_K = num_batch_dims + 1
    idx_C = num_batch_dims + 2

    logical_to_active_map = {}
    active_idx_count = 0

    for b in range(num_batch_dims):
        if output_shape[b] > 1:
            logical_to_active_map[b] = active_idx_count
            active_idx_count += 1

    for num_tiles, idx in zip(tiling, [idx_X, idx_K, idx_C]):
        if num_tiles > 1:
            logical_to_active_map[idx] = active_idx_count
            active_idx_count += 1

    tiled_shapes = node.meta.get('tiled_shapes')
    tile_strides = node.meta.get('tile_strides', {})
    input_sizes = tuple(tiled_shapes.get(a) for a in input_nodes)
    input_strides = tuple(tile_strides.get(a) for a in input_nodes)

    SPATIAL_MAP = {
        "input":        ([idx_X, idx_C], lambda ndim: [ndim - 2, ndim - 1]),
        "input_scale":  ([idx_X, idx_C], lambda ndim: [ndim - 2, ndim - 1]),
        "weight":       ([idx_C, idx_K], lambda ndim: [ndim - 2, ndim - 1]),
        "weight_scale": ([idx_C, idx_K], lambda ndim: [ndim - 2, ndim - 1]),
        "bias":         ([idx_K],        lambda _: [0]),
        "output":       ([idx_X, idx_K], lambda ndim: [ndim - 2, ndim - 1]),
    }

    tile_index_maps = []
    tile_dim_axes = []

    for n in input_nodes:
        key = node_to_key.get(n)
        indices, axis_func = SPATIAL_MAP.get(key, SPATIAL_MAP["output"])

        if n.value.ndim > 2 and num_batch_dims > 0:
            idx_batch = list(range(num_batch_dims))
            tile_index_maps.append(idx_batch + indices)
            tile_dim_axes.append(idx_batch + axis_func(n.value.ndim))
        else:
            tile_index_maps.append(indices)
            tile_dim_axes.append(axis_func(n.value.ndim))

    def _tile_inputs_pattern(active_indices, *args):
        tiled_args = []
        for arg, idx_map, axes, shape, stride in zip(
            args, tile_index_maps, tile_dim_axes, input_sizes, input_strides
        ):
            if isinstance(arg, torch.Tensor) and shape is not None:
                # Map logical IDs to the actual tensor values we prepared above
                mapped_indices = []
                mapped_axes = []

                logger.debug(
                    f"Tiling arg with idx_map: {idx_map}, axes: {axes}, shape: "
                    f"{shape}, stride: {stride}"
                )
                for i, idx in enumerate(idx_map):
                    if (active_idx := logical_to_active_map.get(idx)) is not None:
                        mapped_indices.append(active_indices[active_idx])
                        mapped_axes.append(axes[i])
                logger.debug(f"  Mapped indices: {mapped_indices}, axes: {mapped_axes}")

                arg = quantized_lib.copy_tile(
                    arg, mapped_indices, shape, mapped_axes, stride
                )
            tiled_args.append(arg)
        return tuple(tiled_args)

    _example_inputs = (
        tuple(torch.tensor([i], dtype=torch.int32) for i in range(active_idx_count)),
        *map_arg(input_nodes, lambda n: n.value),
    )

    match_pattern = WrapperModule(_tile_inputs_pattern)
    match_pattern = _get_aten_graph_module_for_pattern(
        match_pattern,
        _example_inputs,
    )
    match_pattern.graph.print_tabular()

    flatten_args, _ = torch.utils._pytree.tree_flatten(_example_inputs)
    ShapeProp(match_pattern).propagate(*flatten_args)

    return match_pattern, logical_to_active_map


def _get_vector_tiled_input_pattern(node, tiling):
    logger.debug(f"Lowering vector for {node.op} node {node} with tiling: {tiling}")

    dims = [i for i, t in enumerate(tiling) if t > 1]
    num_indices = len(dims)

    tiled_shapes = node.meta.get('tiled_shapes')
    tile_strides = node.meta.get('tile_strides', {})

    input_nodes = node.all_input_nodes
    input_sizes = tuple(tiled_shapes.get(a) for a in input_nodes)
    input_strides = tuple(tile_strides.get(a) for a in input_nodes)

    # TODO: handle broadcasting

    def _tile_inputs_pattern(indices, *args):
        tiled_args = []
        for arg, shape, stride in zip(args, input_sizes, input_strides):
            if isinstance(arg, torch.Tensor) and shape is not None:
                arg = quantized_lib.copy_tile(arg, indices, shape, dims, stride)
            tiled_args.append(arg)
        return tuple(tiled_args)

    _example_inputs = (
        tuple(torch.tensor([i], dtype=torch.int32) for i in range(num_indices)),
        *map_arg(input_nodes, lambda n: n.value),
    )

    match_pattern = WrapperModule(_tile_inputs_pattern)
    match_pattern = _get_aten_graph_module_for_pattern(
        match_pattern,
        _example_inputs,
    )

    flatten_args, _ = torch.utils._pytree.tree_flatten(_example_inputs)
    ShapeProp(match_pattern).propagate(*flatten_args)

    # Remove unused placeholder nodes
    for n in match_pattern.graph.nodes:
        if n.op == 'placeholder' and len(n.users) == 0:
            match_pattern.graph.erase_node(n)

    match_pattern.graph.lint()
    match_pattern.graph.print_tabular()
    return match_pattern


def _generate_tiling_wrapper(node: torch.fx.Node, namer, env, stmt):
    tiling = node.meta.get('l2_tiling')

    if node.op == "call_module":
        mod = node.meta["submodule"]
        ref_node = get_reference_node(mod.graph.nodes)
        tiling = ref_node.meta.get('l2_tiling')
    else:
        ref_node = node

    if tiling is None:
        logger.info("No tiling found for node:", node)
        return stmt

    if is_gemm_op(ref_node):
        # Expand 2D tiling to 3D (X, K, C)
        if len(tiling) == 2:
            tiling = tiling + (1,)
        match_pattern, idx_map = _get_gemm_tiled_input_pattern(node, ref_node, tiling)

        ndim = ref_node.value.ndim
        num_batch_dims = ndim - 2 if is_bmm(ref_node) else 0
        loop_bounds = ref_node.shape[:num_batch_dims] + tiling
        output_dims = [*range(num_batch_dims), ndim - 2, ndim - 1]

        num_indices = len(idx_map)
        num_output_dims = num_indices - 1 if tiling[-1] > 1 else num_indices
        output_dims = [d for i, d in enumerate(output_dims) if i in idx_map]
        loop_bounds = [b for i, b in enumerate(loop_bounds) if i in idx_map]
    else:
        match_pattern = _get_vector_tiled_input_pattern(node, tiling)
        output_dims = [i for i, t in enumerate(tiling) if t > 1]
        num_indices = len(output_dims)
        num_output_dims = num_indices
        loop_bounds = tuple(tiling[d] for d in output_dims)

    # ==========================================================================
    # Input tiling pattern
    # ==========================================================================

    input_nodes = node.all_input_nodes
    placeholders = [
        n for n in match_pattern.graph.nodes if n.op == 'placeholder'
    ]
    assert len(placeholders) == len(input_nodes) + num_indices

    # Create new Operation from tiled op pattern
    ssa_indices = [
        IndexValue(name=namer.new_index(), expr=f"{node.name}_tiling_{i}")
        for i in range(num_indices)
    ]

    ops: List[Operation] = []
    arg_idx = 0
    env_copy = env.copy()

    for n in list(match_pattern.graph.nodes):
        if n.op == "placeholder":
            if arg_idx < num_indices:
                env_copy[n] = ssa_indices[arg_idx]
            else:
                env_copy[n] = env[input_nodes[arg_idx - num_indices]]
            arg_idx += 1
        elif n.op == "call_function" and n.target == quantized_lib.copy_tile.default:
            ops.append(Operation.from_fx_node(
                n, env_copy, namer, "Scratchpad", is_fused=True
            ))
        elif n.op == "output":
            new_inputs = [env_copy[n] for n in n.args[0]]

    # Update inputs to new tiled inputs
    for old_input, new_input in zip(stmt.inputs, new_inputs):
        stmt.replace_input_with(old_input, new_input)

    ops.append(stmt)

    # ==========================================================================
    # Output tiling pattern
    # ==========================================================================

    tiled_shapes = node.meta.get('tiled_shapes')
    output_size = tiled_shapes.get(node)

    def _tile_output_pattern(indices, outputs):
        if isinstance(outputs, torch.Tensor):
            if output_size is not None:
                return quantized_lib.copy_tile(
                    outputs, indices, output_size, output_dims
                )
            return outputs

        new_outputs = []
        for output, tile_size in zip(outputs, output_size):
            if tile_size is not None:
                output = quantized_lib.copy_tile(
                    output, indices, tile_size, output_dims
                )
            new_outputs.append(output)
        return tuple(new_outputs)

    _example_inputs = (
        tuple(torch.tensor([i], dtype=torch.int32) for i in range(num_output_dims)),
        node.value,
    )

    match_pattern = WrapperModule(_tile_output_pattern)
    match_pattern = _get_aten_graph_module_for_pattern(
        match_pattern,
        _example_inputs,
    )
    match_pattern.graph.print_tabular()

    flatten_args, _ = torch.utils._pytree.tree_flatten(_example_inputs)
    ShapeProp(match_pattern).propagate(*flatten_args)

    outputs = env[node]
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    arg_idx = 0

    for n in list(match_pattern.graph.nodes):
        if n.op == "placeholder":
            if arg_idx < num_output_dims:
                env_copy[n] = ssa_indices[arg_idx]
            else:
                env_copy[n] = outputs[arg_idx - num_output_dims]
            arg_idx += 1
        elif n.op == "call_function" and n.target == quantized_lib.copy_tile.default:
            ops.append(Operation.from_fx_node(n, env_copy, namer, is_fused=True))
        elif n.op == "output":
            outputs = [env_copy[n] for n in n.args[0]]

    # Update stmt users
    env[node] = outputs[0] if len(outputs) == 1 else tuple(outputs)

    value_to_depth = {}
    ops_at_depth = defaultdict(list)

    for d, idx_val in enumerate(ssa_indices):
        value_to_depth[idx_val] = d

    for op in ops:
        depths = [value_to_depth.get(inp, -1) for inp in op.inputs]
        op_depth = max(depths) if depths else -1
        ops_at_depth[op_depth].append(op)
        for output in op.outputs:
            value_to_depth[output] = op_depth

    loop_op = None
    # Iterate from innermost loop to outermost
    for d in reversed(range(len(ssa_indices))):
        body = ops_at_depth[d]

        loop_op = Loops(
            index=ssa_indices[d],
            start=0,
            end=loop_bounds[d],
            step=1,
            # TODO where should loop_op be placed at?
            body=body if loop_op is None else body + [loop_op],
        )

    print("Final tiled op:")
    print(loop_op.format())
    return loop_op


@dataclass(eq=False)
class Operation:
    """
    Generic operation node that can represent any torch.fx.Node.
    - Keeps FX "op" (call_function/call_method/call_module/placeholder/get_attr/output)
    - Keeps "target" as a stable string
    - Carries arbitrary kwargs (escape hatch)
    - Inputs/outputs are SSA Values (TensorBox / IndexValue)
    """
    op_kind: str
    target: Union[torch._ops.OpOverload, str]
    inputs: List[Value]
    outputs: List[Value]
    kwargs: Dict[str, Any]
    origin_node: torch.fx.Node = None
    annotations: dict[str, Any] = field(default_factory=dict)

    def format(self, indent: int = 0) -> str:
        pad = " " * indent
        outs = ", ".join(v.name for v in self.outputs) if self.outputs else ""
        ins = ", ".join(v.name for v in self.inputs) if self.inputs else ""
        target = _stringify_target(self.target)
        op = str(self.op_kind)

        hdr = f"{pad}{outs} = {op} {target}({ins})" if outs else f"{pad}{op} {target}({ins})"

        # Keep kwargs compact and readable
        if self.kwargs:
            kw_parts = []
            for k, v in self.kwargs.items():
                if v is not None:
                    kw_parts.append(f"{k}={v}")
            hdr += "  {" + ", ".join(kw_parts) + "}"

        # Attach brief type info for outputs (useful for debugging)
        if self.outputs:
            type_ann = []
            for o in self.outputs:
                if isinstance(o, TensorBox):
                    type_ann.append(f"{o.name}:{o.short_type()}@{o.space}")
                else:
                    type_ann.append(str(o))
            hdr += "  :: " + ", ".join(type_ann)

        return hdr

    @staticmethod
    def from_fx_node(
        node: torch.fx.Node,
        env: Dict[torch.fx.Node, Union[Value, Tuple[Value, ...]]],
        namer: NameGenerator,
        mem_space: str = "DRAM",
        is_fused: bool = False,
    ) -> Operation:
        """
        Converts a single FX node into an Operation + freshly created SSA outputs.
        Stores outputs into env[node].

        Notes:
        - Constants that appear inside args/kwargs are encoded into Operation.kwargs.
        - Tensor outputs become TensorBox.
        - Non-tensor outputs become IndexValue with expr='unknown' (minimal).
        - Tuple outputs become a tuple of Values and are stored as such in env.
        """
        inputs: List[Value] = []

        args, kwargs = _get_node_args_and_kwargs(node)
        lift_arg(args, env, inputs)
        kwargs = lift_arg(kwargs, env, inputs)

        orig_val = node.value

        if (tiled_shapes := node.meta.get('tiled_shapes')):
            def load_arg(a):
                return map_arg(a, lambda n: get_tiled_tensor(n, tiled_shapes))
            node.value = node.target(*load_arg(node.args), **load_arg(node.kwargs))

        outputs = _resolve_fx_graph_input(node, namer, mem_space)
        env[node] = outputs

        node.value = orig_val  # restore

        op = Operation(
            op_kind=node.op,
            target=node.target,
            inputs=inputs,
            outputs=outputs if isinstance(outputs, (list, tuple)) else [outputs],
            kwargs=kwargs,
            origin_node=node,
            annotations=dict(getattr(node, "meta", {}) or {}),
        )

        if not is_fused and node.op == "call_function":
            op = _generate_tiling_wrapper(node, namer, env, op)

        link_operation(op)

        return op

    def replace_input_with(self, old: Value, new: Value):
        new_inputs = []
        success = False
        for input in self.inputs:
            if input is old:
                new_inputs.append(new)
                old.users.pop(self, None)
                new.users[self] = None
                success = True
            else:
                new_inputs.append(input)

        if not success:
            raise ValueError(f"Input {old} not found in operation inputs.")

        self.inputs = new_inputs

        new_kwargs = {}
        for k, v in self.kwargs.items():
            if v is old:
                new_kwargs[k] = new.name
            else:
                new_kwargs[k] = v
        self.kwargs = new_kwargs


@dataclass(eq=False)
class FusedOp:
    inputs: List[Value]
    outputs: List[Value]
    ops: List[Operation]
    origin_node: torch.fx.Node
    annotations: dict[str, Any]

    def format(self, indent: int = 0) -> str:
        pad = " " * indent
        ins = ", ".join(f"{v.name}:{v.short_type()}" for v in self.inputs)
        outs = ", ".join(f"{v.name}:{v.short_type()}" for v in self.outputs)
        hdr = f"{pad}{outs} = fused({ins})" if outs else f"{pad}fused({ins})"

        lines = [hdr + " {"]
        for op in self.ops:
            lines.append(op.format(indent + 2))
        lines.append(pad + "}")

        return "\n".join(lines)

    @staticmethod
    def from_fx_node(
        node: torch.fx.Node,
        env: Dict[torch.fx.Node, Union[Value, Tuple[Value, ...]]],
        namer: NameGenerator,
        mem_space: str = "DRAM",
    ) -> FusedOp:
        gm = node.meta["submodule"]
        assert isinstance(gm, torch.fx.GraphModule)

        # Apply tiled shapes if available
        if (tiled_shapes := node.meta.get('tiled_shapes')):
            args = map_arg(
                node.args, lambda n: get_tiled_tensor(n, tiled_shapes)
            )
            ShapeProp(gm).propagate(*args)

        ops: List[Operation] = []

        for n in list(gm.graph.nodes):
            if n.op == "call_function":
                ops.append(Operation.from_fx_node(n, env, namer, is_fused=True))
            if n.op == "output":
                outputs = _resolve_fx_graph_outputs(n.args[0], env)
                env[node] = outputs

        inputs = [env[n] for n in node.all_input_nodes]

        fused_op = FusedOp(
            inputs=inputs,
            outputs=outputs if isinstance(outputs, (list, tuple)) else [outputs],
            ops=ops,
            origin_node=node,
            annotations=dict(getattr(node, "meta", {}) or {}),
        )

        fused_op = _generate_tiling_wrapper(node, namer, env, fused_op)

        link_operation(fused_op)

        return fused_op

    def replace_input_with(self, old: Value, new: Value):
        new_inputs = []
        success = False
        for input in self.inputs:
            if input is old:
                new_inputs.append(new)
                old.users.pop(self, None)
                new.users[self] = None
                success = True
            else:
                new_inputs.append(input)

        if not success:
            raise ValueError(f"Input {old} not found in operation inputs.")

        self.inputs = new_inputs

        for op in self.ops:
            if old in op.inputs:
                op.replace_input_with(old, new)


@dataclass
class Loops:
    """
    Minimal for-loop control structure.

    scf::for-like shape:
      for (index = start; index < end; index += step) {
        body...
      }

    This does not enforce loop-carried SSA (iter_args/yield) to keep it minimal,
    but you can add them later if needed.
    """
    index: IndexValue
    start: Union[IndexValue, int]
    end: Union[IndexValue, int]
    step: Union[IndexValue, int]
    body: List["Stmt"]
    origin_node: Optional[torch.fx.Node] = None
    annotations: dict[str, Any] = field(default_factory=dict)

    def format(self, indent: int = 0) -> str:
        pad = " " * indent
        hdr = f"{pad}for {self.index.name} in range({self.start}, {self.end}, {self.step}):"
        lines = [hdr]
        for s in self.body:
            lines.append(s.format(indent=indent + 2))
        return "\n".join(lines)


Stmt = Union[Operation, FusedOp, Loops]


@dataclass
class FunctionIR:
    name: str
    args: List[Value]
    body: List[Stmt]
    results: List[Value]

    def format(self) -> str:
        args_s = ", ".join(str(a) for a in self.args)
        res_s = ", ".join(str(r) for r in self.results)
        lines = [f"func @{self.name}({args_s}) -> ({res_s}) {{"]

        for s in self.body:
            lines.append(s.format(indent=2))

        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


# =============================================================================
# FX -> IR conversion
# =============================================================================

class NameGenerator:
    def __init__(self, prefix_tensor: str = "%t", prefix_index: str = "%i"):
        self._t = 0
        self._i = 0
        self._pt = prefix_tensor
        self._pi = prefix_index

    def new_tensor(self) -> str:
        self._t += 1
        return f"{self._pt}{self._t}"

    def new_index(self) -> str:
        self._i += 1
        return f"{self._pi}{self._i}"


class FXToIR:
    """
    Perform lowering from torch.fx.Graph (or GraphModule) to FunctionIR.

    Key properties:
    - Each FX node output becomes a fresh SSA TensorBox/IndexValue.
    - Operation nodes retain FX args/kwargs in a debug-friendly form in Operation.kwargs.
    - Real dependencies are expressed via Operation.inputs (SSA values).
    """

    @staticmethod
    def convert(
        gm: torch.fx.GraphModule,
        *,
        func_name: str = "main",
    ) -> FunctionIR:
        namer = NameGenerator()
        env: Dict[torch.fx.Node, Union[Value, Tuple[Value, ...]]] = {}

        args: List[Value] = []
        body: List[Stmt] = []
        results: List[Value] = []

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                v = _resolve_fx_graph_input(node, namer)
                env[node] = v
                args.append(v)
                # Still emit an Operation for traceability
                body.append(Operation.from_fx_node(node, env, namer))
            elif node.op == "output":
                results = _resolve_fx_graph_outputs(node.args[0], env)
                body.append(Operation.from_fx_node(node, env, namer))
            elif node.op == "call_module":
                body.append(FusedOp.from_fx_node(node, env, namer))
            else:
                body.append(Operation.from_fx_node(node, env, namer))

        return FunctionIR(name=func_name, args=args, body=body, results=results)


# =============================================================================
# Helpers (type inference, formatting)
# =============================================================================

def _get_node_args_and_kwargs(node: torch.fx.Node):
    if node.op in ["call_function", "call_method"]:
        args_and_kwargs = normalize_function(
            node.target,
            node.args,
            node.kwargs,
            normalize_to_only_use_kwargs=True
        )

        if args_and_kwargs is not None:
            return args_and_kwargs.args, args_and_kwargs.kwargs

    return node.args, node.kwargs


def _stringify_target(target: Any) -> str:
    # Prefer stable string representations over Python object reprs
    if isinstance(target, str):
        return target
    if target is None:
        return "None"
    # torch.ops.* often stringifies nicely
    try:
        return str(target)
    except Exception:
        return repr(target)


def _pretty_atom(x: Any) -> str:
    if isinstance(x, (int, float, bool)) or x is None:
        return repr(x)
    if isinstance(x, str):
        if len(x) > 80:
            return repr(x[:77] + "...")
        return repr(x)
    if isinstance(x, list):
        if len(x) > 8:
            return "[" + ", ".join(_pretty_atom(v) for v in x[:8]) + ", ...]"
        return "[" + ", ".join(_pretty_atom(v) for v in x) + "]"
    if isinstance(x, dict):
        items = list(x.items())
        if len(items) > 8:
            items = items[:8] + [("...", "...")]
        return "{" + ", ".join(f"{_pretty_atom(k)}: {_pretty_atom(v)}" for k, v in items) + "}"
    return repr(x)


def _dtype_name(dtype_obj: Any) -> str:
    # torch dtype -> readable name
    s = str(dtype_obj)
    # e.g. "torch.float16" -> "float16"
    if s.startswith("torch."):
        return s.split(".", 1)[1]
    return s


def _resolve_fx_graph_input(
    node: torch.fx.Node,
    namer: NameGenerator,
    mem_space: str = "DRAM",
) -> Union[Value, Tuple[Value, ...]]:
    """
    Attempts to infer whether node output is tensor / tuple[tensor] / scalar.
    Uses node.meta['tensor_meta'] when available. Falls back to unknown.
    """
    val = getattr(node, "value", None)

    if isinstance(val, torch.Tensor):
        return TensorBox(
            name=namer.new_tensor(),
            shape=tuple(val.shape),
            dtype=val.dtype,
            space=mem_space
        )

    if isinstance(val, (list, tuple)):
        outputs: List[Value] = []
        for v in val:
            outputs.append(
                TensorBox(
                    name=namer.new_tensor(),
                    shape=tuple(v.shape),
                    dtype=v.dtype,
                    space=mem_space
                )
            )
        return tuple(outputs)

    return IndexValue(name=namer.new_index(), expr="unknown")


def _resolve_fx_graph_outputs(
    node: Union[torch.fx.Node, List[torch.fx.Node]],
    env: Dict[torch.fx.Node, Union[Value, Tuple[Value, ...]]],
) -> List[Value]:
    """
    FX output arg can be:
      - a Node
      - a tuple/list/dict containing Nodes
      - constants
    We only collect Values corresponding to Nodes.
    """
    if isinstance(node, torch.fx.Node):
        return env[node]

    if isinstance(node, (list, tuple)):
        outputs: List[Value] = []
        for x in node:
            v = _resolve_fx_graph_outputs(x, env)
            outputs.extend(v if isinstance(v, list) else [v])
        return outputs

    return []  # constants or unsupported types
