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
from ..mapping_utils import is_gemm_op, is_bmm, is_elementwise_op
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


@dataclass(kw_only=True)
class Value:
    """Base class for Compiler IR SSA values."""
    name: str
    users: Dict["IRNode", None] = field(default_factory=dict, repr=False)
    producer_op: "IRNode" = field(default=None, repr=False)

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
    address: Optional[int] = None

    def short_type(self) -> str:
        shape = ",".join(str(d) for d in self.shape)
        dtype = _dtype_name(self.dtype)
        return f"tensor<{shape}x{dtype}>"

    def __str__(self) -> str:
        if self.address is None:
            return f"{self.name}:{self.short_type()}@{self.space}"
        return f"{self.name}:{self.short_type()}@{self.space}[0x{self.address:x}]"

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


@dataclass(eq=False, kw_only=True)
class IRNode:
    origin_node: torch.fx.Node = None
    annotations: dict[str, Any] = field(default_factory=dict)
    block: Optional["Loops" | "Module"] = field(default=None, init=False, repr=False)

    def get_parent(self) -> Optional["Loops"]:
        """Climb one level to find the immediate parent loop."""
        return self.block


@dataclass(eq=False)
class Operation(IRNode):
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

    def format(self, indent: int = 0) -> str:
        pad = " " * indent
        outs = ", ".join(v.name for v in self.outputs) if self.outputs else ""
        ins = ", ".join(v.name for v in self.inputs) if self.inputs else ""
        target = _stringify_target(self.target)

        hdr = f"{pad}{outs} = {target}({ins})" if outs else f"{pad}{target}({ins})"

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
        outputs: Optional[List[Value]] = None,
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

        if outputs is None:
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
class FusedOp(IRNode):
    inputs: List[Value]
    outputs: List[Value]
    ops: List[Operation]

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
class Loops(IRNode):
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
    body: List[IRNode]

    init_args: List[Value] = field(default_factory=list) # Values coming from outside
    iter_vars: List[Value] = field(default_factory=list) # Block args inside the loop
    yields: List[Value] = field(default_factory=list)    # Values yielded to next iter
    outputs: List[Value] = field(default_factory=list)   # Final results of the loop

    def format(self, indent: int = 0) -> str:
        pad = " " * indent

        # 1. Format the outputs of the loop
        out_str = ", ".join(v.name for v in self.outputs)
        out_prefix = f"{out_str} = " if out_str else ""

        # 2. Format the iter_args (mapping block vars to incoming values)
        iter_str = ""
        if self.init_args:
            args_map = ", ".join(
                f"{v.name}={i.name}" for v, i in zip(self.iter_vars, self.init_args)
            )
            iter_str = f" iter_args({args_map})"

        # 3. Build the loop header
        hdr = f"{pad}{out_prefix}for {self.index.name} in range({self.start}, {self.end}, {self.step}){iter_str}:"
        lines = [hdr]

        # 4. Format the body
        for s in self.body:
            lines.append(s.format(indent=indent + 2))

        # 5. Format the yield statement (acts as the end of the block)
        if self.yields:
            yield_str = ", ".join(y.name for y in self.yields)
            lines.append(f"{pad}  yield({yield_str})")

        return "\n".join(lines)

    def __post_init__(self):
        # When a Loop is created, it immediately claims
        # ownership of everything in its body.
        for stmt in self.body:
            link_operation(stmt, parent=self)

        # Also ensure the index knows this loop is its producer
        self.index.producer_op = self

        # TODO: outputs need to point to correct producer op so that codegen
        # can get the right FX node.
        # # Iter_vars and Outputs are produced by this loop
        # for ivar in self.iter_vars:
        #     ivar.producer_op = self
        # for out in self.outputs:
        #     out.producer_op = self

    def __hash__(self):
        return id(self)


@dataclass
class Module:
    name: str
    args: List[Value]
    params: List[Value]
    body: List[IRNode]
    results: List[Value]

    def format(self) -> str:
        inputs = ", ".join(str(a) for a in self.args + self.params)
        outputs = ", ".join(str(r) for r in self.results)
        lines = [f"func @{self.name}({inputs}) -> ({outputs}) {{"]
        for s in self.body:
            lines.append(s.format(indent=2))
        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()

    def __post_init__(self):
        # When a Module is created, it immediately claims
        # ownership of everything in its body.
        for stmt in self.body:
            link_operation(stmt, parent=self)

    @staticmethod
    def convert(
        gm: torch.fx.GraphModule,
        name: str = "main",
    ) -> Module:
        namer = NameGenerator()
        env: Dict[torch.fx.Node, Union[Value, Tuple[Value, ...]]] = {}

        body: List[IRNode] = []
        args: List[Value] = []
        params: List[Value] = []
        results: List[Value] = []

        for node in gm.graph.nodes:
            if node.op == "call_module":
                stmt = FusedOp.from_fx_node(node, env, namer)
            else:
                stmt = Operation.from_fx_node(node, env, namer)

            if node.op == "placeholder":
                args.append(env[node])
            elif node.op == "get_attr":
                params.append(env[node])
            elif node.op == "output":
                results = _resolve_fx_graph_outputs(node.args[0], env)
            else:
                body.append(stmt)

        return Module(name, args, params, body, results)


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


# =============================================================================
# Helpers (type inference, formatting)
# =============================================================================

def _get_node_args_and_kwargs(node: torch.fx.Node):
    args, kwargs = node.args, node.kwargs

    if node.op in ("call_function", "call_method"):
        args_and_kwargs = normalize_function(
            node.target,
            args,
            kwargs,
            normalize_to_only_use_kwargs=True
        )

        if args_and_kwargs is not None:
            args, kwargs = args_and_kwargs.args, args_and_kwargs.kwargs

    kwargs = {
        k: v for k, v in kwargs.items() if v is not None and "qmap" not in k
    }

    return args, kwargs


def _stringify_target(target: Any) -> str:
    if isinstance(target, torch._ops.OpOverload):
        return target._schema.name.split('::')[1]
    else:
        return str(target)


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


def _get_output_dtype(ir_node: Operation, dtype=None):
    node = ir_node.origin_node
    input_dtypes = [getattr(i, 'dtype', None) for i in ir_node.inputs]
    input_dtypes = [dt for dt in input_dtypes if dt is not None]

    def get_list_item(array, index):
        return array[index] if index < len(array) else None

    if node.target == quantized_lib.quantize_mx.default:
        out_dtype = get_list_item(input_dtypes, 1)
        scale_dtype = get_list_item(input_dtypes, 6)
        dtype = [out_dtype, scale_dtype]
    elif node.target == quantized_lib.quantize.default:
        dtype = get_list_item(input_dtypes, 5)
    elif node.target == quantized_lib.calculate_mx_qparam.default:
        dtype = get_list_item(input_dtypes, 5)
    elif not is_gemm_op(node) and not is_elementwise_op(node) and len(input_dtypes) == 1:
        dtype = input_dtypes[0]

    return dtype if isinstance(dtype, list) else [dtype]


def _propagate_dtype(module: Module, input_dtypes: List[Any]):
    """
    Propagates data types through the IR module.
    """
    all_input_nodes = module.args + module.params
    if len(all_input_nodes) != len(input_dtypes):
        raise ValueError(
            f"Expected {len(all_input_nodes)} input dtypes, got {len(input_dtypes)}"
        )

    for arg, dtype in zip(all_input_nodes, input_dtypes):
        if isinstance(arg, TensorBox) and dtype is not None:
            arg.dtype = dtype

    def visit_block(body: List[IRNode]):
        for node in body:
            if isinstance(node, Loops):
                visit_block(node.body)
            else:
                if isinstance(node, FusedOp):
                    visit_block(node.ops)
                output_dtypes = _get_output_dtype(node)
                for output, dtype in zip(node.outputs, output_dtypes):
                    if isinstance(output, TensorBox) and dtype is not None:
                        output.dtype = dtype

    visit_block(module.body)


def _resolve_fx_graph_input(
    node: torch.fx.Node,
    namer: NameGenerator,
    mem_space: str = "DRAM",
    ir_node: Optional[IRNode] = None
) -> Union[Value, Tuple[Value, ...]]:
    """
    Attempts to infer whether node output is tensor / tuple[tensor] / scalar.
    Uses node.meta['tensor_meta'] when available. Falls back to unknown.
    """
    val = getattr(node, "value", None)
    dtype = getattr(node, "dtype", None)

    if isinstance(val, torch.Tensor):
        return TensorBox(
            name=namer.new_tensor(),
            shape=tuple(val.shape),
            dtype=dtype or val.dtype,
            space=mem_space,
            producer_op=ir_node,
        )

    if isinstance(val, (list, tuple)):
        outputs: List[Value] = [
            TensorBox(
                name=namer.new_tensor(),
                shape=tuple(v.shape),
                dtype=dtype[i] if dtype is not None else v.dtype,
                space=mem_space,
                producer_op=ir_node,
            )
            for i, v in enumerate(val)
        ]
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


def link_operation(node: IRNode, parent: Union["Loops", "Module"] = None):
    """
    Sets the structural block link and updates SSA pointers.
    """
    node.block = parent

    # 1. Handle SSA value linking for Operations
    if isinstance(node, (Operation, FusedOp)):
        for input_val in node.inputs:
            input_val.users[node] = None
        for output_val in node.outputs:
            output_val.producer_op = node

        # If FusedOp has internal ops, they share the same block/parent
        if isinstance(node, FusedOp):
            for internal_op in node.ops:
                link_operation(internal_op, parent=parent)

    # 2. Handle Recursive Linking for Nested Loops
    elif isinstance(node, Loops):
        for body_stmt in node.body:
            link_operation(body_stmt, parent=node)


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

                arg = quantized_lib.load_tile(
                    arg, mapped_indices, shape, mapped_axes, stride
                )
            tiled_args.append(arg)
        return tuple(tiled_args)

    _example_inputs = (
        tuple(torch.tensor(i, dtype=torch.int32) for i in range(active_idx_count)),
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
                arg = quantized_lib.load_tile(arg, indices, shape, dims, stride)
            tiled_args.append(arg)
        return tuple(tiled_args)

    _example_inputs = (
        tuple(torch.tensor(i, dtype=torch.int32) for i in range(num_indices)),
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
        elif n.op == "call_function" and n.target == quantized_lib.load_tile.default:
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
                return quantized_lib.load_tile(
                    outputs, indices, output_size, output_dims
                )
            return outputs

        new_outputs = []
        for output, tile_size in zip(outputs, output_size):
            if tile_size is not None:
                output = quantized_lib.load_tile(
                    output, indices, tile_size, output_dims
                )
            new_outputs.append(output)
        return tuple(new_outputs)

    _example_inputs = (
        tuple(torch.tensor(i, dtype=torch.int32) for i in range(num_output_dims)),
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
        elif n.op == "call_function" and n.target == quantized_lib.load_tile.default:
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
