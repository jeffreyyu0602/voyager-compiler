from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.fx as fx
from torch.fx.operator_schemas import normalize_function


Dim = Union[int, str]  # int for static, str for symbolic ("N", "K", ...)


@dataclass(frozen=True)
class DType:
    name: str  # "fp16", "bf16", "fp32", "int32", ...


@dataclass(frozen=True)
class MemSpace:
    name: str  # "HBM", "SRAM", "ACC", ...


@dataclass(frozen=True)
class TensorType:
    shape: Tuple[Dim, ...]
    dtype: DType


@dataclass
class Layout:
    """
    Bare-minimum layout metadata; extend when needed.
    """
    strides: Optional[Tuple[int, ...]] = None
    permutation: Optional[Tuple[int, ...]] = None


@dataclass
class TensorBox:
    """
    SSA value for tensors (Inductor-like spirit).
    Each produced tensor should be a fresh TensorBox instance.
    """
    name: str
    ttype: TensorType
    space: MemSpace = field(default_factory=lambda: MemSpace("HBM"))
    layout: Layout = field(default_factory=Layout)
    storage_id: Optional[str] = None  # optional identity for aliasing/storage

    def short_type(self) -> str:
        shp = ",".join(str(d) for d in self.ttype.shape)
        return f"tensor<{shp}x{self.ttype.dtype.name}>"

    def __str__(self) -> str:
        return f"{self.name}:{self.short_type()}@{self.space.name}"


@dataclass(frozen=True)
class IndexValue:
    """
    SSA value for loop indices / scalar-ish IR values.
    For minimalism, we carry a string expression and an optional debug name.
    """
    name: str
    expr: Optional[str] = None  # e.g. "0", "N", "i*128", etc.

    def __str__(self) -> str:
        if self.expr is None:
            return f"{self.name}:index"
        return f"{self.name}:index={self.expr}"


IRValue = Union[TensorBox, IndexValue]


@dataclass
class Operation:
    """
    Generic operation node that can represent any torch.fx.Node.
    - Keeps FX "op" (call_function/call_method/call_module/placeholder/get_attr/output)
    - Keeps "target" as a stable string
    - Carries arbitrary kwargs (escape hatch)
    - Inputs/outputs are SSA IRValues (TensorBox / IndexValue)
    """
    op_kind: str
    target: torch._ops.OpOverload
    inputs: List[IRValue]
    outputs: List[IRValue]
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # Optional debug mapping back to FX:
    fx_node_name: Optional[str] = None
    fx_node_meta: Dict[str, Any] = field(default_factory=dict)

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
                kw_parts.append(f"{k}={_pretty_atom(v)}")
            hdr += "  {" + ", ".join(kw_parts) + "}"

        # Attach brief type info for outputs (useful for debugging)
        if self.outputs:
            type_ann = []
            for o in self.outputs:
                if isinstance(o, TensorBox):
                    type_ann.append(f"{o.name}:{o.short_type()}@{o.space.name}")
                else:
                    type_ann.append(str(o))
            hdr += "  :: " + ", ".join(type_ann)

        if self.fx_node_name:
            hdr += f"  ; fx={self.fx_node_name}"
        return hdr

    @staticmethod
    def from_fx_node(
        node: "fx.Node",
        env: Dict["fx.Node", Union[IRValue, Tuple[IRValue, ...]]],
        namer: "NameGenerator",
    ) -> "Operation":
        """
        Converts a single FX node into an Operation + freshly created SSA outputs.
        Stores outputs into env[node].

        Notes:
        - Constants that appear inside args/kwargs are encoded into Operation.kwargs.
        - Tensor outputs become TensorBox.
        - Non-tensor outputs become IndexValue with expr='unknown' (minimal).
        - Tuple outputs become a tuple of IRValues and are stored as such in env.
        """
        op_kind = node.op
        target = node.target

        # Convert FX args into SSA inputs when they refer to other nodes.
        inputs: List[IRValue] = []
        extra_kwargs: Dict[str, Any] = {}

        def lift_arg(a: Any) -> Any:
            # Node references become IRValue inputs
            if fx is not None and isinstance(a, fx.Node):
                v = env[a]
                if isinstance(v, tuple):
                    # If a node produced multiple outputs, you can decide how to reference it.
                    # Minimal choice: include each output as an input in-order.
                    inputs.extend(list(v))
                    return [vv.name for vv in v]
                inputs.append(cast(IRValue, v))
                return cast(IRValue, v).name
            # Containers recurse; embed constants into kwargs for debug.
            if isinstance(a, (list, tuple)):
                return [lift_arg(x) for x in a]
            if isinstance(a, dict):
                return {k: lift_arg(v) for k, v in a.items()}
            # Keep constants as-is (will appear in kwargs)
            return a

        args = node.args
        kwargs = node.kwargs

        if node.op == "call_function" or node.op == "call_method":
            args_and_kwargs = normalize_function(
                node.target,
                node.args,
                node.kwargs,
                normalize_to_only_use_kwargs=True
            )

            if args_and_kwargs is not None:
                args = args_and_kwargs.args
                kwargs = args_and_kwargs.kwargs

        # We keep a debug-friendly mirror of args in kwargs, without turning
        # everything into SSA. The SSA inputs list captures dependencies.
        extra_kwargs["fx_args"] = lift_arg(args)
        extra_kwargs["fx_kwargs"] = lift_arg(kwargs)

        # Determine outputs (SSA)
        outputs: List[IRValue] = []
        produced = _infer_fx_outputs_to_ir(node, namer)

        if isinstance(produced, tuple):
            outputs.extend(list(produced))
            env[node] = produced
        else:
            outputs.append(produced)
            env[node] = produced

        # Special-case placeholder/output/get_attr: keep target meaningful
        # but the generic mechanism works regardless.
        op = Operation(
            op_kind=op_kind,
            target=target,
            inputs=inputs,
            outputs=outputs,
            kwargs=extra_kwargs,
            fx_node_name=getattr(node, "name", None),
            fx_node_meta=dict(getattr(node, "meta", {}) or {}),
        )
        return op


@dataclass
class Loop:
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
    start: IndexValue
    end: IndexValue
    step: IndexValue
    body: List["Stmt"]

    def format(self, indent: int = 0) -> str:
        pad = " " * indent
        hdr = f"{pad}for {self.index.name} in range({self.start.expr}, {self.end.expr}, {self.step.expr}):"
        lines = [hdr]
        for s in self.body:
            lines.append(s.format(indent=indent + 2))
        return "\n".join(lines)


Stmt = Union[Operation, Loop]


@dataclass
class FunctionIR:
    name: str
    args: List[IRValue]
    body: List[Stmt]
    results: List[IRValue]

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
    Bare-minimum lowering from torch.fx.Graph (or GraphModule) to FunctionIR.

    Key properties:
    - Each FX node output becomes a fresh SSA TensorBox/IndexValue.
    - Operation nodes retain FX args/kwargs in a debug-friendly form in Operation.kwargs.
    - Real dependencies are expressed via Operation.inputs (SSA values).
    """

    @staticmethod
    def convert(
        graph_or_gm: Union["fx.Graph", "fx.GraphModule"],
        *,
        func_name: str = "main",
        default_space: str = "HBM",
    ) -> FunctionIR:
        if fx is None:
            raise RuntimeError("torch.fx is not available in this environment.")

        graph: fx.Graph = graph_or_gm.graph if isinstance(graph_or_gm, fx.GraphModule) else graph_or_gm

        namer = NameGenerator()
        env: Dict[fx.Node, Union[IRValue, Tuple[IRValue, ...]]] = {}

        args: List[IRValue] = []
        body: List[Stmt] = []
        results: List[IRValue] = []

        for node in graph.nodes:
            if node.op == "placeholder":
                # Represent placeholders as TensorBox if we can infer tensor_meta; else IndexValue.
                v = _infer_fx_outputs_to_ir(node, namer, default_space=default_space)
                env[node] = v if not isinstance(v, tuple) else v
                # record as function arg(s)
                if isinstance(v, tuple):
                    args.extend(list(v))
                else:
                    args.append(v)
                # Still emit an Operation for traceability
                body.append(Operation.from_fx_node(node, env, namer))

            elif node.op == "output":
                # FX output node's args[0] is the returned value(s).
                # We do not create a new SSA value; we reference existing ones.
                out_ir_vals = _resolve_fx_output_arg(node.args[0], env)
                results = list(out_ir_vals)
                body.append(Operation.from_fx_node(node, env, namer))

            else:
                body.append(Operation.from_fx_node(node, env, namer))

        return FunctionIR(name=func_name, args=args, body=body, results=results)


# =============================================================================
# Helpers (type inference, formatting)
# =============================================================================

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
    # Pretty-print atoms without exploding
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


def _infer_fx_outputs_to_ir(
    node: "fx.Node",
    namer: NameGenerator,
    default_space: str = "HBM",
) -> Union[IRValue, Tuple[IRValue, ...]]:
    """
    Attempts to infer whether node output is tensor / tuple[tensor] / scalar.
    Uses node.meta['tensor_meta'] when available. Falls back to unknown.
    """
    meta = getattr(node, "meta", {}) or {}
    val = getattr(node, "value", None)

    # Case 1: Single tensor
    if isinstance(val, torch.Tensor):
        tt = TensorType(shape=tuple(int(d) for d in val.shape), dtype=DType(_dtype_name(val.dtype)))
        return TensorBox(name=namer.new_tensor(), ttype=tt, space=MemSpace(default_space))

    # Case 2: tuple/list of tensors
    if isinstance(val, (list, tuple)):
        outs: List[IRValue] = []
        for v in val:
            tt = TensorType(shape=tuple(int(d) for d in v.shape), dtype=DType(_dtype_name(v.dtype)))
            outs.append(TensorBox(name=namer.new_tensor(), ttype=tt, space=MemSpace(default_space)))
        return tuple(outs)

    # Default: unknown scalar / index-like
    return IndexValue(name=namer.new_index(), expr="unknown")


def _tensor_type_from_tensor_meta(node: torch.fx.Node) -> TensorType:
    shape = tuple(int(d) if isinstance(d, (int,)) else str(d) for d in getattr(node, "shape"))
    dtype = DType(_dtype_name(getattr(node, "dtype")))
    return TensorType(shape=shape, dtype=dtype)


def _dtype_name(dtype_obj: Any) -> str:
    # torch dtype -> readable name
    s = str(dtype_obj)
    # e.g. "torch.float16" -> "float16"
    if s.startswith("torch."):
        return s.split(".", 1)[1]
    return s


def _resolve_fx_output_arg(
    out_arg: Any,
    env: Dict["fx.Node", Union[IRValue, Tuple[IRValue, ...]]],
) -> List[IRValue]:
    """
    FX output arg can be:
      - a Node
      - a tuple/list/dict containing Nodes
      - constants
    We only collect IRValues corresponding to Nodes.
    """
    vals: List[IRValue] = []

    def walk(x: Any) -> None:
        if fx is not None and isinstance(x, fx.Node):
            v = env[x]
            if isinstance(v, tuple):
                vals.extend(list(v))
            else:
                vals.append(v)
            return
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
            return
        if isinstance(x, dict):
            for y in x.values():
                walk(y)
            return
        # constants ignored (minimal). You can model them as Constant ops if needed.

    walk(out_arg)
    return vals
