from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Callable,
    Set,
)
import itertools


# -----------------------------
# Utilities
# -----------------------------

_uid = itertools.count(0)


def fresh(prefix: str) -> str:
    return f"{prefix}_{next(_uid)}"


class VerifyError(RuntimeError):
    pass


# -----------------------------
# Types / Shapes / DTypes
# -----------------------------

@dataclass(frozen=True)
class DType:
    name: str  # e.g., "float32", "bfloat16", "int8", "int4", "fp8_e5m3"


@dataclass(frozen=True)
class TensorType:
    shape: Tuple[int, ...]
    dtype: DType

    def rank(self) -> int:
        return len(self.shape)


# -----------------------------
# Memory: Buffer + TensorView
# -----------------------------

@dataclass
class Buffer:
    """
    Storage identity. Addresses can be assigned late (allocator pass).
    Mirrors your protobuf Memory/partition/address concept but makes it explicit.
    """
    name: str
    partition: int = 0
    address: Optional[int] = None
    bytes: Optional[int] = None
    scope: str = "global"  # e.g. "global", "scratchpad", "l2"
    alignment: int = 64


@dataclass
class Layout:
    """
    Optional tiling + stride metadata (kept close to your protobuf fields).
    """
    tiled_shape: Optional[Tuple[int, ...]] = None
    tile_strides: Optional[Tuple[int, ...]] = None


@dataclass
class TensorView:
    """
    A value-level reference to storage.
    - buffer: optional: may be None for purely functional tensors.
    - byte_offset/strides: for views.
    """
    ttype: TensorType
    buffer: Optional[Buffer] = None
    byte_offset: int = 0
    strides: Optional[Tuple[int, ...]] = None
    layout: Layout = field(default_factory=Layout)
    scratchpad: Optional[Buffer] = None  # optional staging buffer
    is_none: bool = False

    # Optional fused decorations (matches your protobuf style)
    dequant: Optional["FXOp"] = None
    reshape: Optional["FXOp"] = None


# -----------------------------
# SSA Values
# -----------------------------

@dataclass(eq=False)
class Value:
    """
    SSA-ish Value produced by an IRNode.
    """
    name: str
    tview: TensorView
    producer: Optional["IRNode"] = None
    users: List["IRNode"] = field(default_factory=list)

    def add_user(self, node: "IRNode") -> None:
        self.users.append(node)

    def __hash__(self) -> int:
        return id(self)


# -----------------------------
# Base IR Nodes
# -----------------------------

class IRNode:
    """
    Base class for all IR nodes. TorchInductor-like: nodes are objects with
    verify() and explicit inputs/outputs. Keep it minimal.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or fresh(self.__class__.__name__.lower())
        self.parent_region: Optional["Region"] = None

    def inputs(self) -> Sequence[Union[Value, "Region"]]:
        return ()

    def outputs(self) -> Sequence[Value]:
        return ()

    def verify(self) -> None:
        # Default: nothing
        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


# -----------------------------
# FX-like Op Node
# -----------------------------

Arg = Union[
    Value,
    int,
    float,
    bool,
    str,
    None,
    Tuple["Arg", ...],
    List["Arg"],
    Dict[str, "Arg"],
]


class FXOp(IRNode):
    """
    FX-shaped op node: (op, target, args, kwargs).
    - op: "call_function" | "call_module" | "call_method" | "placeholder" | "output"
      (You can add custom op kinds as needed.)
    - target: function/module name or identifier
    """

    def __init__(
        self,
        op: str,
        target: str,
        args: Sequence[Arg] = (),
        kwargs: Optional[Dict[str, Arg]] = None,
        *,
        name: Optional[str] = None,
        out_types: Optional[Union[TensorType, Sequence[TensorType]]] = None,
        out_layouts: Optional[Union[Layout, Sequence[Layout]]] = None,
    ):
        super().__init__(name=name)
        self.op = op
        self.target = target
        self.args: Tuple[Arg, ...] = tuple(args)
        self.kwargs: Dict[str, Arg] = dict(kwargs or {})

        self._outputs: List[Value] = []

        # Create outputs if type info provided
        if out_types is not None:
            if isinstance(out_types, TensorType):
                out_types_seq = [out_types]
            else:
                out_types_seq = list(out_types)

            if out_layouts is None:
                out_layouts_seq = [Layout() for _ in out_types_seq]
            elif isinstance(out_layouts, Layout):
                out_layouts_seq = [out_layouts]
            else:
                out_layouts_seq = list(out_layouts)

            if len(out_layouts_seq) != len(out_types_seq):
                raise ValueError("out_layouts length must match out_types length")

            for i, (tt, ly) in enumerate(zip(out_types_seq, out_layouts_seq)):
                vname = f"{self.name}:{i}" if len(out_types_seq) > 1 else self.name
                v = Value(
                    name=vname,
                    tview=TensorView(ttype=tt, layout=ly),
                    producer=self,
                )
                self._outputs.append(v)

        # Record users for Value args
        for v in iter_values(self.args, self.kwargs):
            v.add_user(self)

    def inputs(self) -> Sequence[Value]:
        return list(iter_values(self.args, self.kwargs))

    def outputs(self) -> Sequence[Value]:
        return self._outputs

    def verify(self) -> None:
        # Minimal checks
        if self.op not in {
            "call_function",
            "call_module",
            "call_method",
            "placeholder",
            "get_attr",
            "output",
        }:
            raise VerifyError(f"{self.name}: unknown FX op kind '{self.op}'")
        if self.op == "output" and len(self._outputs) != 0:
            raise VerifyError(f"{self.name}: output node should not produce outputs")


def iter_values(args: Sequence[Arg], kwargs: Dict[str, Arg]) -> Iterable[Value]:
    """
    Recursively yield Value objects from nested args/kwargs.
    """
    def walk(x: Arg) -> Iterable[Value]:
        if isinstance(x, Value):
            yield x
        elif isinstance(x, (tuple, list)):
            for y in x:
                yield from walk(y)  # type: ignore[arg-type]
        elif isinstance(x, dict):
            for y in x.values():
                yield from walk(y)  # type: ignore[arg-type]
        else:
            return

    for a in args:
        yield from walk(a)
    for a in kwargs.values():
        yield from walk(a)


# -----------------------------
# Regions (Structured Control Flow)
# -----------------------------

@dataclass
class BlockArg:
    """
    Region argument: like MLIR block argument.
    Used for induction variables and loop-carried values.
    """
    value: Value


class Region:
    """
    A region is a sequence of IRNodes plus explicit inputs and yields.
    This is the minimal structured construct you need for If/For.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        inputs: Optional[List[BlockArg]] = None,
    ):
        self.name = name or fresh("region")
        self.inputs: List[BlockArg] = list(inputs or [])
        self.nodes: List[IRNode] = []
        self.yields: List[Value] = []

    def append(self, node: IRNode) -> IRNode:
        node.parent_region = self
        self.nodes.append(node)
        return node

    def set_yield(self, *values: Value) -> None:
        self.yields = list(values)

    def verify(self) -> None:
        # Verify node placement and basic well-formedness
        seen: Set[IRNode] = set()
        for n in self.nodes:
            if n in seen:
                raise VerifyError(f"{self.name}: node appears twice: {n}")
            seen.add(n)
            n.verify()
        # yields must be Values produced in this region or passed via inputs
        allowed: Set[Value] = {ba.value for ba in self.inputs}
        for n in self.nodes:
            allowed.update(n.outputs())
        for v in self.yields:
            if v not in allowed:
                raise VerifyError(
                    f"{self.name}: yielded value '{v.name}' not defined in region"
                )


class If(IRNode):
    """
    Structured If:
      - cond: Value (bool or 0d tensor)
      - then_region, else_region: regions
      - results are explicit via yields from regions
    """

    def __init__(
        self,
        cond: Value,
        then_region: Region,
        else_region: Region,
        *,
        name: Optional[str] = None,
        out_types: Optional[Union[TensorType, Sequence[TensorType]]] = None,
    ):
        super().__init__(name=name)
        self.cond = cond
        self.then_region = then_region
        self.else_region = else_region
        self._outputs: List[Value] = []

        # user tracking
        cond.add_user(self)

        if out_types is not None:
            out_types_seq = [out_types] if isinstance(out_types, TensorType) else list(out_types)
            for i, tt in enumerate(out_types_seq):
                vname = f"{self.name}:{i}" if len(out_types_seq) > 1 else self.name
                self._outputs.append(Value(name=vname, tview=TensorView(ttype=tt), producer=self))

    def inputs(self) -> Sequence[Union[Value, Region]]:
        return [self.cond, self.then_region, self.else_region]

    def outputs(self) -> Sequence[Value]:
        return self._outputs

    def verify(self) -> None:
        self.then_region.verify()
        self.else_region.verify()
        if len(self.then_region.yields) != len(self.else_region.yields):
            raise VerifyError(
                f"{self.name}: then/else yields arity mismatch: "
                f"{len(self.then_region.yields)} vs {len(self.else_region.yields)}"
            )
        if self._outputs and len(self._outputs) != len(self.then_region.yields):
            raise VerifyError(
                f"{self.name}: output arity mismatch with yields: "
                f"{len(self._outputs)} vs {len(self.then_region.yields)}"
            )


class For(IRNode):
    """
    Structured For loop:
      - static bounds for now (start/end/step ints)
      - body region gets:
          inputs: [iv] + loop-carried values
        and yields:
          next loop-carried values
      - results are the final carried values
    """

    def __init__(
        self,
        start: int,
        end: int,
        step: int,
        body: Region,
        *,
        name: Optional[str] = None,
        out_types: Optional[Union[TensorType, Sequence[TensorType]]] = None,
    ):
        super().__init__(name=name)
        self.start = int(start)
        self.end = int(end)
        self.step = int(step)
        self.body = body
        self._outputs: List[Value] = []

        if out_types is not None:
            out_types_seq = [out_types] if isinstance(out_types, TensorType) else list(out_types)
            for i, tt in enumerate(out_types_seq):
                vname = f"{self.name}:{i}" if len(out_types_seq) > 1 else self.name
                self._outputs.append(Value(name=vname, tview=TensorView(ttype=tt), producer=self))

    def inputs(self) -> Sequence[Union[Value, Region]]:
        return [self.body]

    def outputs(self) -> Sequence[Value]:
        return self._outputs

    def verify(self) -> None:
        if self.step == 0:
            raise VerifyError(f"{self.name}: step cannot be 0")
        if (self.end - self.start) * self.step < 0:
            raise VerifyError(f"{self.name}: loop bounds are inconsistent")
        self.body.verify()
        # Require body.inputs[0] be induction variable (convention)
        if len(self.body.inputs) < 1:
            raise VerifyError(f"{self.name}: body region must have at least the induction var input")
        if self._outputs and len(self._outputs) != len(self.body.yields):
            raise VerifyError(
                f"{self.name}: output arity mismatch with body yields: "
                f"{len(self._outputs)} vs {len(self.body.yields)}"
            )


# -----------------------------
# Module / Function
# -----------------------------

class Function:
    """
    A function owns a top-level region.
    Keeps FX-like placeholders and outputs possible, but you can also represent
    structured control flow via If/For nodes in the region.
    """

    def __init__(self, name: str):
        self.name = name
        self.body = Region(name=f"{name}_body")
        self.inputs: List[Value] = []
        self.parameters: List[Value] = []
        self.outputs: List[Value] = []

    def add_input(self, name: str, ttype: TensorType) -> Value:
        v = Value(name=name, tview=TensorView(ttype=ttype), producer=None)
        self.inputs.append(v)
        return v

    def add_param(self, name: str, ttype: TensorType, buffer: Optional[Buffer] = None) -> Value:
        v = Value(name=name, tview=TensorView(ttype=ttype, buffer=buffer), producer=None)
        self.parameters.append(v)
        return v

    def set_outputs(self, *vals: Value) -> None:
        self.outputs = list(vals)

    def verify(self) -> None:
        self.body.verify()
        # Ensure outputs exist in function scope
        allowed: Set[Value] = set(self.inputs) | set(self.parameters)
        for n in self.body.nodes:
            allowed.update(n.outputs())
        for v in self.outputs:
            if v not in allowed:
                raise VerifyError(f"{self.name}: output '{v.name}' not defined in function")


class Module:
    def __init__(self):
        self.functions: Dict[str, Function] = {}

    def add_function(self, fn: Function) -> None:
        if fn.name in self.functions:
            raise ValueError(f"Duplicate function {fn.name}")
        self.functions[fn.name] = fn

    def verify(self) -> None:
        for fn in self.functions.values():
            fn.verify()
