from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.fx.operator_schemas import normalize_function

from ..mapping_utils import is_gemm_op, is_elementwise_op


logger = logging.getLogger(__name__)

quantized_lib = torch.ops.quantized_ops


""" [Note: Voyager IR]

Voyager IR is a minimal compiler IR for the Voyager accelerator. It models
tiled computation with explicit loop nesting and buffer semantics.

Core design decisions:
- Each FX node maps 1:1 to an Operation, preserving FX OpOverload targets so
  the backend can recognize them directly.
- SSA values are TensorBox (tensors with shape/dtype/memory-space) or
  IndexValue (loop indices / scalars).
- Loops follow scf.for semantics: init_args flow in, iter_vars shadow them
  inside the body, yields thread the carried state per iteration, and outputs
  hold the final results after the loop exits.
- Operations carry origin_node links for verification-data propagation.

"""


@dataclass(kw_only=True)
class Value:
    """Base class for Compiler IR SSA values."""
    name: str
    users: Dict["IRNode", None] = field(default_factory=dict, repr=False)
    producer_op: "IRNode" = field(default=None, repr=False)

    def replace_all_uses_with(self, new: Value):
        for user in list(self.users):  # copy: replace_input_with mutates self.users
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

        if outputs is None:
            outputs = _resolve_fx_graph_input(node, namer, mem_space)
        env[node] = outputs

        op = Operation(
            op_kind=node.op,
            target=node.target,
            inputs=inputs,
            outputs=outputs if isinstance(outputs, (list, tuple)) else [outputs],
            kwargs=kwargs,
            origin_node=node,
            annotations=dict(getattr(node, "meta", {}) or {}),
        )

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

        ops: List[Operation] = []

        for n in list(gm.graph.nodes):
            if n.op == "call_function":
                ops.append(Operation.from_fx_node(n, env, namer))
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

        # Gap 2: iter_vars and outputs are produced by this loop —
        # set producer_op so callers can walk up the def chain.
        for ivar in self.iter_vars:
            ivar.producer_op = self
        for out in self.outputs:
            out.producer_op = self

        # init_args flow into the loop from outside — register as users
        for arg in self.init_args:
            arg.users[self] = None

        # Gap 5: yields are the last-use of carried values each iteration —
        # register this loop as a user so def-use traversal is complete.
        for y in self.yields:
            y.users[self] = None

    def replace_input_with(self, old: Value, new: Value):
        """Update init_args / yields when an upstream Value is replaced."""
        self.init_args = [new if a is old else a for a in self.init_args]
        self.yields    = [new if y is old else y for y in self.yields]
        old.users.pop(self, None)
        new.users[self] = None

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
    # Gap 1: Value objects may appear directly (e.g. IndexValue loop indices
    # threaded through lowering passes rather than via FX Node env lookup).
    # Track them in `inputs` so they appear in the SSA def-use graph.
    if isinstance(a, Value):
        inputs.append(a)
        return a.name
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


def canonicalize(module: Module) -> None:
    """
    In-place canonicalization pass: eliminates degenerate loops with statically
    known trip counts.

    - Dead loops (0 trips):   replace outputs with init_args, drop the loop.
    - Single-trip loops (1 trip): inline the body into the parent scope,
      substitute iter_vars → init_args and outputs → yields.

    The loop index of a single-trip loop is turned into a constant IndexValue
    (expr = str(start)) so downstream ops that reference it remain valid.

    Module.results is updated to reflect any substituted output values.
    """
    # subst[old] = new — for values that are unreachable via users (e.g. the
    # top-level module results list, which isn't registered as a user).
    subst: Dict[Value, Value] = {}

    def resolve(v: Value) -> Value:
        while v in subst:
            v = subst[v]
        return v

    def _replace(old: Value, new: Value) -> None:
        subst[old] = new
        old.replace_all_uses_with(new)

    def _process_body(
        body: List[IRNode], parent: Union[Loops, Module]
    ) -> List[IRNode]:
        new_body: List[IRNode] = []
        for stmt in body:
            if isinstance(stmt, Loops):
                result = _process_loops(stmt, parent)
                if result is None:
                    pass  # dead loop removed
                elif isinstance(result, list):
                    new_body.extend(result)  # inlined single-trip body
                else:
                    new_body.append(result)
            else:
                new_body.append(stmt)
        return new_body

    def _process_loops(
        loop: Loops, parent: Union[Loops, Module]
    ) -> Optional[Union[Loops, List[IRNode]]]:
        # Recurse first so inner degenerate loops are folded before outer ones.
        loop.body = _process_body(loop.body, loop)

        start, end, step = loop.start, loop.end, loop.step
        if not (isinstance(start, int) and isinstance(end, int)
                and isinstance(step, int) and step > 0):
            return loop  # dynamic or zero-step bounds — cannot fold

        trips = max((end - start + step - 1) // step, 0)

        if trips == 0:
            # Body never executes: outputs equal the init_args unchanged.
            for out, init in zip(loop.outputs, loop.init_args):
                _replace(out, init)
                init.users.pop(loop, None)
            for y in loop.yields:
                y.users.pop(loop, None)
            return None

        if trips == 1:
            # Body runs exactly once.

            # 1. Substitute iter_vars → init_args (same value at iteration 0).
            for iv, init in zip(loop.iter_vars, loop.init_args):
                _replace(iv, init)
                init.users.pop(loop, None)

            # 2. Replace the loop index with a constant IndexValue.
            #    Keep the same SSA name so kwargs string references stay valid.
            const_idx = IndexValue(name=loop.index.name, expr=str(start))
            _replace(loop.index, const_idx)

            # 3. Re-parent body ops to the enclosing scope.
            for s in loop.body:
                s.block = parent

            # 4. Substitute outputs → yields.
            for out, y in zip(loop.outputs, loop.yields):
                _replace(out, y)
                y.users.pop(loop, None)

            return loop.body  # caller splices these into the parent body

        return loop  # multi-trip: leave unchanged

    module.body = _process_body(module.body, module)

    # Fix up module results that may have been substituted.
    module.results = [resolve(r) for r in module.results]
