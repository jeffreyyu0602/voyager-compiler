from typing import Callable, Dict, Tuple, Optional, Union, List, Any

import torch

from .ir import (
    Stmt,
    Loop,
    Operation,
    IRValue,
    TensorBox,
    IndexValue,
    Dim,
    MemSpace,
    TensorType,
    NameGenerator,
    FunctionIR,
)


LoweringFn = Callable[..., Union[Stmt, List[Stmt]]]

# target → lowering fn
_LOWERING_REGISTRY: Dict[Any, LoweringFn] = {}


def register_lowering(target: Any):
    """
    Register a lowering function for a given FX target.

    Usage:
        @register_lowering(aten.matmul.default)
        def lower_matmul(op: Operation, *, namer: NameGenerator, **kwargs):
            ...
    """
    def decorator(fn: LoweringFn) -> LoweringFn:
        if target in _LOWERING_REGISTRY:
            raise RuntimeError(f"Lowering already registered for {target}")
        _LOWERING_REGISTRY[target] = fn
        return fn
    return decorator


def expect_tensor(v: IRValue) -> TensorBox:
    assert isinstance(v, TensorBox), f"Expected TensorBox, got {type(v)}"
    return v


def make_slice_op(
    namer: NameGenerator,
    src: TensorBox,
    batch_idx: IndexValue,
    out_shape: Tuple[Dim, ...],
    *,
    kind: str,
) -> TensorBox:
    """
    Create an op that slices src[b, ...] -> dst
    """
    out = TensorBox(
        name=namer.new_tensor(),
        ttype=TensorType(shape=out_shape, dtype=src.ttype.dtype),
        space=MemSpace("SRAM"),  # load into SRAM
    )

    return out, Operation(
        op_kind="call_function",
        target=f"memory.{kind}",  # "load_slice" or "store_slice"
        inputs=[src, batch_idx],
        outputs=[out] if kind == "load_slice" else [],
        kwargs={
            "dim": 0,
            "index": batch_idx.name,
        },
    )


def split_batched_matmul_shape(A: TensorBox):
    # A.shape = (*batch, X, K)
    shape = A.ttype.shape
    assert len(shape) >= 3
    return shape[:-2], shape[-2], shape[-1]


def build_nested_loops(
    batch_dims: Tuple[Dim, ...],
    namer: NameGenerator,
    inner_body_fn,
) -> Loop:
    """
    Recursively build nested loops over batch dimensions.
    inner_body_fn(batch_indices) -> List[Stmt]
    """
    def build(level: int, indices: List[IndexValue]) -> Loop:
        dim = batch_dims[level]
        idx = IndexValue(name=namer.new_index(), expr=f"b{level}")
        start = IndexValue(name=namer.new_index(), expr="0")
        end = IndexValue(name=namer.new_index(), expr=str(dim))
        step = IndexValue(name=namer.new_index(), expr="1")

        if level == len(batch_dims) - 1:
            body = inner_body_fn(indices + [idx])
        else:
            body = [build(level + 1, indices + [idx])]

        return Loop(
            index=idx,
            start=start,
            end=end,
            step=step,
            body=body,
        )

    return build(0, [])


aten = torch.ops.aten


@register_lowering(aten.matmul.default)
def lower_nd_batched_matmul(op: Operation, namer: NameGenerator) -> Loop:
    A = expect_tensor(op.inputs[0])
    B = expect_tensor(op.inputs[1])
    C = expect_tensor(op.outputs[0])

    batch_dims, X, K = split_batched_matmul_shape(A)
    _, _, N = C.ttype.shape

    def inner_body(batch_indices: List[IndexValue]) -> List[Stmt]:
        body: List[Stmt] = []

        # Load A slice
        A_tile = TensorBox(
            name=namer.new_tensor(),
            ttype=TensorType((X, K), A.ttype.dtype),
            space=MemSpace("SRAM"),
        )
        body.append(Operation(
            op_kind="call_function",
            target="memory.load_slice",
            inputs=[A] + batch_indices,
            outputs=[A_tile],
            kwargs={"batch_rank": len(batch_indices)},
        ))

        # Load B slice
        B_tile = TensorBox(
            name=namer.new_tensor(),
            ttype=TensorType((K, N), B.ttype.dtype),
            space=MemSpace("SRAM"),
        )
        body.append(Operation(
            op_kind="call_function",
            target="memory.load_slice",
            inputs=[B] + batch_indices,
            outputs=[B_tile],
            kwargs={"batch_rank": len(batch_indices)},
        ))

        # Compute
        C_tile = TensorBox(
            name=namer.new_tensor(),
            ttype=TensorType((X, N), C.ttype.dtype),
            space=MemSpace("SRAM"),
        )
        body.append(Operation(
            op_kind="call_function",
            target="aten.matmul",
            inputs=[A_tile, B_tile],
            outputs=[C_tile],
            kwargs={},
        ))

        # Store
        body.append(Operation(
            op_kind="call_function",
            target="memory.store_slice",
            inputs=[C_tile, C] + batch_indices,
            outputs=[],
            kwargs={"batch_rank": len(batch_indices)},
        ))

        return body

    return build_nested_loops(batch_dims, namer, inner_body)


def lower_operations(func: FunctionIR) -> FunctionIR:
    namer = NameGenerator()
    new_body: List[Stmt] = []

    for stmt in func.body:
        if not isinstance(stmt, Operation):
            new_body.append(stmt)
            continue

        lowering_fn = _LOWERING_REGISTRY.get(stmt.target)

        if lowering_fn is None:
            new_body.append(stmt)
            continue

        lowered = lowering_fn(stmt, namer=namer)

        if isinstance(lowered, list):
            new_body.extend(lowered)
        else:
            new_body.append(lowered)

    return FunctionIR(
        name=func.name,
        args=func.args,
        body=new_body,
        results=func.results,
    )
