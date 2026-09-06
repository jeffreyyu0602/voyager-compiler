from __future__ import annotations

import logging
import math
import operator
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch import fx

from voyager_compiler.codegen.node_info import (
    AXES_ARG_INDEX_MAP,
    ancestors,
    get_anchor_node,
    is_aliasing_op,
    is_elementwise_op,
    is_nop,
    is_reshape_op,
    quant_param_arg_nodes,
    reshape_preserves_full_blocks,
)
from voyager_compiler.shape_prop import (
    ShapeProp,
    propagate_shape,
    set_node_value,
)

logger = logging.getLogger(__name__)


Shape = Tuple[int, ...]
# One dimension of an address map: row-major ``(radix, stride)`` digits.
Digits = Tuple[Tuple[int, int], ...]


def _canonical(digits) -> Digits:
    """``digits`` with radix-1 entries dropped and adjacent entries merged
    where the higher one's stride is the lower one's radix times its stride,
    so two equal maps have equal digits.  A size-1 dimension is ``((1, 0),)``.
    """
    out = []
    for radix, stride in digits:
        if radix == 1:
            continue
        if out and out[-1][1] == radix * stride:
            out[-1] = (out[-1][0] * radix, stride)
        else:
            out.append((radix, stride))
    return tuple(out) or ((1, 0),)


def _axis(dim: int, ndim: int) -> int:
    if not -ndim <= dim < ndim:
        raise ValueError(f"dimension {dim} out of range for {ndim} dims")
    return dim % ndim


@dataclass(frozen=True)
class AddressMap:
    """Where every position of a fused subgraph value reads its external
    input: the flat index of that input, as a function of the position.

    The function is held symbolically.  ``dims[d]`` is the row-major list of
    ``(radix, stride)`` digits of dimension ``d``: an index along it
    contributes each of its mixed-radix digits times that digit's stride,
    and ``offset`` is added to the total.  A map over a 32-head ``n x n``
    attention tensor is then a handful of tuples, not an int64 tensor of
    ``32 n^2`` entries, and every shape op the normalizer replays keeps the
    form exact: a permute reorders dimensions, an expand is a zero-stride
    digit, and a reshape regroups digits, splitting one where a new boundary
    falls inside it.  A dimension of one digit is read at a fixed stride;
    one that keeps several (the head axis once a GQA repeat is merged) is
    not, and ``affine_strides`` reports it as such.
    """

    dims: Tuple[Digits, ...]
    offset: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dims", tuple(_canonical(d) for d in self.dims)
        )

    @classmethod
    def arange(cls, shape: Sequence[int]) -> "AddressMap":
        """The identity map over ``shape``: position ``p`` reads element
        ``p`` of a contiguous input of that shape."""
        dims = []
        stride = 1
        for size in reversed([int(s) for s in shape]):
            dims.append(((size, stride),))
            stride *= max(size, 1)
        return cls(tuple(reversed(dims)))

    @property
    def shape(self) -> Shape:
        return tuple(math.prod(r for r, _ in d) for d in self.dims)

    def numel(self) -> int:
        return math.prod(self.shape)

    def affine_strides(self) -> Optional[Shape]:
        """One stride per dimension (0 along a size-1 or broadcast one), or
        ``None`` when some dimension is not read at a single stride."""
        strides = []
        for digits in self.dims:
            if len(digits) != 1:
                return None
            radix, stride = digits[0]
            strides.append(stride if radix > 1 else 0)
        return tuple(strides)

    def reshape(self, shape: Sequence[int]) -> "AddressMap":
        """Regroup the digits into ``shape`` (one ``-1`` allowed).  A
        dimension boundary that falls inside a digit splits it; one that
        falls at no multiple of the digit's radix cannot be expressed."""
        shape = [int(s) for s in shape]
        if shape.count(-1) == 1:
            known = math.prod(s for s in shape if s != -1)
            shape[shape.index(-1)] = self.numel() // max(known, 1)
        if math.prod(shape) != self.numel():
            raise ValueError(f"cannot reshape {self.shape} to {shape}")
        flat = [d for dim in self.dims for d in dim if d[0] != 1]
        dims = []
        for size in shape:
            taken = []
            remaining = size
            while remaining > 1:
                radix, stride = flat.pop(0)
                if radix <= remaining:
                    if remaining % radix:
                        raise ValueError(
                            f"reshape of {self.shape} to {shape} cuts a "
                            "dimension inside a digit"
                        )
                    taken.append((radix, stride))
                    remaining //= radix
                else:
                    if radix % remaining:
                        raise ValueError(
                            f"reshape of {self.shape} to {shape} cuts a "
                            "dimension inside a digit"
                        )
                    taken.append((remaining, stride * (radix // remaining)))
                    flat.insert(0, (radix // remaining, stride))
                    remaining = 1
            dims.append(tuple(taken))
        return AddressMap(tuple(dims), self.offset)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "AddressMap":
        shape = self.shape
        start = _axis(start_dim, len(shape))
        end = _axis(end_dim, len(shape))
        merged = math.prod(shape[start : end + 1])
        return self.reshape(shape[:start] + (merged,) + shape[end + 1 :])

    def permute(self, order: Sequence[int]) -> "AddressMap":
        ndim = len(self.dims)
        return AddressMap(
            tuple(self.dims[_axis(o, ndim)] for o in order), self.offset
        )

    def transpose(self, dim0: int, dim1: int) -> "AddressMap":
        order = list(range(len(self.dims)))
        a, b = _axis(dim0, len(order)), _axis(dim1, len(order))
        order[a], order[b] = order[b], order[a]
        return self.permute(order)

    def unsqueeze(self, dim: int) -> "AddressMap":
        d = _axis(dim, len(self.dims) + 1)
        return AddressMap(
            self.dims[:d] + (((1, 0),),) + self.dims[d:], self.offset
        )

    def squeeze(self, dims=None) -> "AddressMap":
        """Drop the size-1 dimensions, or those of ``dims`` that are size 1."""
        ndim = len(self.dims)
        if dims is None:
            drop = set(range(ndim))
        elif isinstance(dims, int):
            drop = {_axis(dims, ndim)}
        else:
            drop = {_axis(d, ndim) for d in dims}
        shape = self.shape
        kept = tuple(
            digits
            for d, digits in enumerate(self.dims)
            if not (d in drop and shape[d] == 1)
        )
        return AddressMap(kept, self.offset)

    def select(self, dim: int, index: int) -> "AddressMap":
        d = _axis(dim, len(self.dims))
        index = _axis(index, self.shape[d])
        offset = self.offset
        for radix, stride in reversed(self.dims[d]):
            offset += (index % radix) * stride
            index //= radix
        return AddressMap(self.dims[:d] + self.dims[d + 1 :], offset)

    def slice(self, dim=0, start=None, end=None, step=1) -> "AddressMap":
        d = _axis(dim, len(self.dims))
        size = self.shape[d]
        start = 0 if start is None else start
        end = size if end is None else end
        start = max(start + size, 0) if start < 0 else min(start, size)
        end = max(end + size, 0) if end < 0 else min(end, size)
        length = max(0, -(-(end - start) // step))
        if start == 0 and length == size and step == 1:
            return self
        if len(self.dims[d]) != 1:
            raise ValueError("slice of a dimension read at several strides")
        radix, stride = self.dims[d][0]
        dims = (
            self.dims[:d] + (((length, stride * step),),) + self.dims[d + 1 :]
        )
        return AddressMap(dims, self.offset + start * stride)

    def expand(self, sizes: Sequence[int]) -> "AddressMap":
        """Broadcast to ``sizes`` (``-1`` keeps a dimension): a size-1
        dimension, or a new leading one, becomes a zero-stride digit."""
        sizes = [int(s) for s in sizes]
        lead = len(sizes) - len(self.dims)
        if lead < 0:
            raise ValueError(f"cannot expand {self.shape} to {sizes}")
        dims = [((size, 0),) for size in sizes[:lead]]
        for size, digits, current in zip(sizes[lead:], self.dims, self.shape):
            if size in (-1, current):
                dims.append(digits)
            elif current == 1:
                dims.append(((size, 0),))
            else:
                raise ValueError(f"cannot expand {self.shape} to {sizes}")
        return AddressMap(tuple(dims), self.offset)


NodePredicate = Callable[[fx.Node], bool]

# Emit relayouts as aten call_function ops (not call_method): the rest of the
# compiler recognizes these as shape nops and aliases their memory.
_RELAYOUT_OPS = {
    "view": torch.ops.aten.view.default,
    "reshape": torch.ops.aten.reshape.default,
}


def _blocking_axes(node: fx.Node) -> Optional[Sequence[int]]:
    """The axes ``node`` blocks its quantization along, or ``None`` if it
    quantizes per tensor or does not quantize at all."""
    index = AXES_ARG_INDEX_MAP.get(node.target)
    if index is None or index >= len(node.args):
        return None
    return node.args[index] or None


class NormalizationError(RuntimeError):
    """Raised when a fused child cannot use one iteration space."""

    def __init__(self, message: str, *, node: Optional[fx.Node] = None) -> None:
        if node is not None:
            message = (
                f"{message} [node={node.name}, op={node.op}, "
                f"target={node.target!r}]"
            )
        super().__init__(message)
        self.node = node


@dataclass(frozen=True)
class InputPlan:
    placeholder: fx.Node
    original_shape: Shape
    boundary_shape: Shape
    iteration_shape: Shape
    base: int
    strides: Shape
    broadcast_dims: Tuple[int, ...]
    parent_operation: str  # "identity" or "view"
    role: str = "pointwise_operand"


@dataclass(frozen=True)
class OutputPlan:
    internal_shape: Shape
    external_shape: Shape
    parent_operation: str  # "identity" or "view"


@dataclass
class NormalizationResult:
    iteration_shape: Shape
    anchor_name: Optional[str]
    input_plans: Dict[str, InputPlan]
    output_plan: OutputPlan
    parent: fx.GraphModule
    child: fx.GraphModule
    call_node: fx.Node
    anchor_input_names: Tuple[str, ...] = ()
    output_plans: Tuple[OutputPlan, ...] = ()


@dataclass(frozen=True)
class _OutputDescription:
    value_node: fx.Node
    map_target: fx.Node
    iteration_node: fx.Node
    external_shapes: Tuple[Shape, ...]
    output_relayout: Tuple[fx.Node, ...] = ()

    @property
    def is_multi_output(self) -> bool:
        return len(self.external_shapes) > 1


@dataclass
class NormalizationConfig:
    """Configuration for :class:`IterationSpaceNormalizer`.

    The default target detection is intentionally conservative. Projects with
    custom operators should pass explicit predicates or annotate nodes with:

        node.meta["strong_anchor"] = True
        node.meta["pointwise"] = True
    """

    require_contiguous_inputs: bool = True
    parent_shape_method: str = "view"  # "view" or "reshape"
    anchor_predicate: Optional[NodePredicate] = None
    pointwise_predicate: Optional[NodePredicate] = None

    def __post_init__(self) -> None:
        if self.parent_shape_method not in {"view", "reshape"}:
            raise ValueError("parent_shape_method must be 'view' or 'reshape'")


class IterationSpaceNormalizer:
    """Normalize one fused ``call_module`` in a parent FX graph.

    Preconditions
    -------------
    * Static tensor shapes are available in ``node.meta["tensor_meta"]`` or
      ``node.meta["val"]`` for the child graph.
    * The child has one tensor output, or a terminal microscaling quantization
      op whose tuple outputs are consumed through parent ``getitem`` nodes.
    * External inputs are contiguous unless ``require_contiguous_inputs=False``.
    * Non-compute child operations obey the restricted fusion contract.

    The pass performs all analysis first and mutates the graphs only after every
    input has a valid plan.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None) -> None:
        self.config = config or NormalizationConfig()

    def normalize(
        self,
        parent: fx.GraphModule,
        call_node: fx.Node,
    ) -> NormalizationResult:
        if call_node.op != "call_module":
            raise NormalizationError(
                "Expected a call_module node", node=call_node
            )

        child_obj = parent.get_submodule(str(call_node.target))
        if not isinstance(child_obj, fx.GraphModule):
            raise NormalizationError(
                "The call_module target must be an FX GraphModule",
                node=call_node,
            )
        child = child_obj

        placeholders = [n for n in child.graph.nodes if n.op == "placeholder"]
        output_node = next(
            (n for n in child.graph.nodes if n.op == "output"), None
        )
        if output_node is None:
            raise NormalizationError("Child graph has no output node")
        output = self._describe_output(output_node)

        anchor = get_anchor_node(call_node)
        # Keep only a *strong* anchor (gemm/conv/layernorm/softmax); a pointwise
        # "anchor" means the group is anchorless (iteration == output shape).
        if anchor is None or is_elementwise_op(anchor):
            anchor = None

        if anchor is None:
            iteration_shape = output.iteration_node.shape
            anchor_inputs: set[fx.Node] = set()
            normalizable_placeholders = placeholders
        else:
            iteration_shape = anchor.shape
            anchor_inputs = {
                n for n in ancestors(anchor) if n.op == "placeholder"
            }
            self._validate_anchor_stream(
                anchor, output.map_target, iteration_shape
            )
            self._reject_anchor_inputs_reused_in_tail(
                placeholders=placeholders,
                anchor=anchor,
                output_value=output.map_target,
                anchor_inputs=anchor_inputs,
            )
            normalizable_placeholders = [
                p for p in placeholders if p not in anchor_inputs
            ]

        # Quantization lookup tables (qmap/code/...) are indexed by value, not
        # by iteration position, so they have no address map to propagate;
        # they are passed whole to every tile. Skip them.
        quant_tables = set()
        for node in child.graph.nodes:
            quant_tables |= quant_param_arg_nodes(node)

        input_plans: Dict[str, InputPlan] = {}
        for placeholder in normalizable_placeholders:
            if placeholder in quant_tables:
                continue

            original_shape = placeholder.shape
            # A scalar input broadcasts to any iteration shape, so it has no
            # address map to propagate; pass it whole, like the quant tables.
            if math.prod(original_shape) == 1:
                continue
            self._validate_external_layout(placeholder, original_shape)

            propagated = self._propagate_map(
                child=child,
                seed_node=placeholder,
                output_value=output.map_target,
                anchor=anchor,
            )
            if propagated is None:
                # The placeholder does not contribute to the output. Dead
                # code can be eliminated independently; it does not need a
                # boundary plan.
                continue

            if anchor is not None:
                if propagated.numel() != math.prod(iteration_shape):
                    raise NormalizationError(
                        "Tail operand cannot be expressed in the anchor "
                        "iteration space because the external tail changes "
                        "element count",
                        node=placeholder,
                    )
                propagated = propagated.reshape(iteration_shape)
            elif tuple(propagated.shape) != iteration_shape:
                raise NormalizationError(
                    f"Propagated map shape {tuple(propagated.shape)} does not "
                    f"match iteration shape {iteration_shape}",
                    node=placeholder,
                )

            base, strides = self._recover_and_validate_affine_map(
                propagated, iteration_shape, placeholder
            )
            boundary_shape, broadcast_dims = self._derive_boundary_shape(
                original_shape=original_shape,
                iteration_shape=iteration_shape,
                base=base,
                strides=strides,
                placeholder=placeholder,
            )
            input_plans[placeholder.name] = InputPlan(
                placeholder=placeholder,
                original_shape=original_shape,
                boundary_shape=boundary_shape,
                iteration_shape=iteration_shape,
                base=base,
                strides=strides,
                broadcast_dims=broadcast_dims,
                parent_operation=(
                    "identity"
                    if original_shape == boundary_shape
                    else self.config.parent_shape_method
                ),
            )

        placeholder_bindings = self._bind_parent_arguments(
            call_node, placeholders
        )
        self._rewrite_parent_inputs(
            parent=parent,
            call_node=call_node,
            placeholder_bindings=placeholder_bindings,
            input_plans=input_plans,
        )
        self._rewrite_child(child=child, anchor=anchor, output=output)

        child.graph.eliminate_dead_code()
        child.graph.lint()
        child.recompile()

        args = fx.map_arg(call_node.all_input_nodes, lambda n: n.value)
        result = ShapeProp(child).propagate(*args)
        set_node_value(call_node, result)

        output_plans = self._build_output_plans(output)
        output_plan = output_plans[-1]
        self._rewrite_parent_output(parent, call_node, output, output_plans)

        return NormalizationResult(
            iteration_shape=iteration_shape,
            anchor_name=anchor.name if anchor is not None else None,
            input_plans=input_plans,
            output_plan=output_plan,
            parent=parent,
            child=child,
            call_node=call_node,
            anchor_input_names=tuple(
                p.name for p in placeholders if p in anchor_inputs
            ),
            output_plans=output_plans,
        )

    # ------------------------------------------------------------------
    # Address-map propagation
    # ------------------------------------------------------------------

    def _propagate_map(
        self,
        *,
        child: fx.GraphModule,
        seed_node: fx.Node,
        output_value: fx.Node,
        anchor: Optional[fx.Node],
    ) -> Optional[AddressMap]:
        seed = AddressMap.arange(seed_node.shape)

        env: Dict[fx.Node, Optional[AddressMap]] = {}
        for node in child.graph.nodes:
            if node is seed_node:
                env[node] = seed
            elif node.op in {"placeholder", "get_attr"}:
                env[node] = None
            elif anchor is not None and node is anchor:
                env[node] = None
            elif (
                is_aliasing_op(node)
                or is_reshape_op(node)
                or node.target is torch.ops.aten.expand.default
            ):
                source = node.all_input_nodes[0]
                source_map = env.get(source) if source is not None else None
                env[node] = (
                    None
                    if source_map is None
                    else self._apply_shape_noop(node, source_map)
                )
            elif is_nop(node) or node.target == torch.ops.aten.to.dtype:
                # A dtype cast is position-preserving (it reinterprets each
                # element, never moves one), so it is the identity on the
                # address map.
                source = node.args[0]
                env[node] = (
                    env.get(source) if isinstance(source, fx.Node) else None
                )
            elif node.target in {
                torch.ops.quantized_ops.dequantize.default,
                torch.ops.quantized_ops.quantize.default,
            }:
                # Microscaling (de)quantization is not a true elementwise
                # op: its scale/zero_point are per-block parameters indexed
                # by block, not by iteration position, so they carry no
                # address map. Only the data input (args[0]) does.
                source = node.args[0] if node.args else None
                env[node] = (
                    env.get(source) if isinstance(source, fx.Node) else None
                )
            elif is_elementwise_op(node):
                maps = [
                    env[arg]
                    for arg in node.all_input_nodes
                    if env.get(arg) is not None
                ]
                if not maps:
                    env[node] = None
                else:
                    output_shape = node.shape
                    broadcasted = [
                        self._broadcast_map(m, output_shape, node) for m in maps
                    ]
                    first = broadcasted[0]
                    for other in broadcasted[1:]:
                        if first != other:
                            raise NormalizationError(
                                "One external input reaches the same "
                                "pointwise operation through incompatible "
                                "index mappings. Duplicate the argument "
                                "explicitly or reject the fusion",
                                node=node,
                            )
                    env[node] = first
            elif self._is_supported_mx_op(node):
                source = node.args[0] if node.args else None
                env[node] = (
                    env.get(source) if isinstance(source, fx.Node) else None
                )
            elif node.op != "output":
                raise NormalizationError(
                    "Cannot propagate address map through node", node=node
                )

            if node is output_value:
                return env.get(node)

        return env.get(output_value)

    def _validate_anchor_stream(
        self,
        anchor: fx.Node,
        output_value: fx.Node,
        iteration_shape: Shape,
    ) -> None:
        propagated = self._propagate_map(
            child=anchor.graph.owning_module,  # type: ignore[arg-type]
            seed_node=anchor,
            output_value=output_value,
            anchor=anchor,
        )
        # _propagate_map treats anchor as the seed before applying the anchor
        # barrier.
        if propagated is None:
            raise NormalizationError(
                "The fused output does not depend on the strong anchor",
                node=anchor,
            )
        if propagated.numel() != math.prod(iteration_shape):
            raise NormalizationError(
                "The tail expands or contracts the anchor-produced stream",
                node=anchor,
            )
        if propagated.reshape(iteration_shape) != AddressMap.arange(
            iteration_shape
        ):
            raise NormalizationError(
                "The tail changes the order or multiplicity of the "
                "anchor-produced stream",
                node=anchor,
            )

    def _recover_and_validate_affine_map(
        self,
        propagated: AddressMap,
        iteration_shape: Shape,
        placeholder: fx.Node,
    ) -> Tuple[int, Shape]:
        if tuple(propagated.shape) != iteration_shape:
            raise NormalizationError(
                f"Map shape {tuple(propagated.shape)} != iteration shape "
                f"{iteration_shape}",
                node=placeholder,
            )
        strides = propagated.affine_strides()
        if strides is None:
            raise NormalizationError(
                "Input map is not representable by one fixed base and one "
                "stride per iteration dimension",
                node=placeholder,
            )
        return propagated.offset, strides

    def _derive_boundary_shape(
        self,
        *,
        original_shape: Shape,
        iteration_shape: Shape,
        base: int,
        strides: Shape,
        placeholder: fx.Node,
    ) -> Tuple[Shape, Tuple[int, ...]]:
        if base != 0:
            raise NormalizationError(
                "A nonzero source offset cannot be produced by the allowed "
                "shape-only fusion operations",
                node=placeholder,
            )

        boundary = tuple(
            1 if extent > 1 and stride == 0 else extent
            for extent, stride in zip(iteration_shape, strides)
        )
        if math.prod(boundary) != math.prod(original_shape):
            raise NormalizationError(
                f"Derived boundary shape {boundary} has "
                f"{math.prod(boundary)} elements, but input shape "
                f"{original_shape} has {math.prod(original_shape)}",
                node=placeholder,
            )

        # The map must read ``boundary`` in row-major order, broadcast along
        # the dimensions it drops.
        contiguous = self._contiguous_strides(boundary)
        expected = tuple(
            0 if extent <= 1 or boundary_extent == 1 else stride
            for extent, boundary_extent, stride in zip(
                iteration_shape, boundary, contiguous
            )
        )
        if tuple(strides) != expected:
            raise NormalizationError(
                "Affine access is not specifically expressible as an "
                "order-preserving view plus fetch-side broadcast",
                node=placeholder,
            )

        broadcast_dims = tuple(
            dim
            for dim, (boundary_extent, iteration_extent) in enumerate(
                zip(boundary, iteration_shape)
            )
            if boundary_extent == 1 and iteration_extent > 1
        )
        return boundary, broadcast_dims

    # ------------------------------------------------------------------
    # Output interpretation
    # ------------------------------------------------------------------

    def _describe_output(self, output_node: fx.Node) -> _OutputDescription:
        value = output_node.args[0]
        if not isinstance(value, fx.Node):
            raise NormalizationError(
                "Only tensor outputs are supported", node=output_node
            )

        core, relayout = self._strip_output_relayout(value)
        if relayout:
            return _OutputDescription(
                value_node=value,
                map_target=core,
                iteration_node=core,
                external_shapes=(value.shape,),
                output_relayout=relayout,
            )

        if self._is_supported_mx_op(value):
            # The quantize may sit on top of an output relayout, quantizing the
            # tile the fused op is about to store through it.  The relayout is
            # still what ends the iteration space -- the quantize does not move
            # an element -- so peel it from the quantize's input, not from the
            # output.
            core, relayout = self._strip_output_relayout(value.args[0])
            return _OutputDescription(
                value_node=value,
                map_target=core if relayout else value,
                iteration_node=core,
                external_shapes=value.shape,
                output_relayout=relayout,
            )

        return _OutputDescription(
            value_node=value,
            map_target=value,
            iteration_node=value,
            external_shapes=(value.shape,),
        )

    def _strip_output_relayout(
        self, value: fx.Node
    ) -> Tuple[fx.Node, Tuple[fx.Node, ...]]:
        """Peel a trailing single-use chain of relayout ops (view / reshape /
        squeeze / unsqueeze / transpose / permute) off the output, returning
        the core node that feeds it and the chain (output-first). Only honored
        when the chain contains a transpose/permute: that cannot be re-expressed
        as a parent view, so it must stay inside the fused op and the iteration
        space ends at the core."""
        chain = []
        node = value
        while (
            (is_reshape_op(node) or is_aliasing_op(node))
            and len(node.users) == 1
            and node.all_input_nodes
        ):
            chain.append(node)
            node = node.all_input_nodes[0]
        if not any(is_reshape_op(n) for n in chain):
            return value, ()
        return node, tuple(chain)

    def _build_output_plans(
        self,
        output: _OutputDescription,
    ) -> Tuple[OutputPlan, ...]:
        # Internal shapes were just stamped onto the output node by ShapeProp.
        # An MX node's shape is already a tuple of per-output shapes; an
        # ordinary single output is one shape, wrapped to match external_shapes.
        is_mx = self._is_supported_mx_op(output.value_node)
        if is_mx:
            internal_shapes = tuple(tuple(s) for s in output.value_node.shape)
        else:
            internal_shapes = (tuple(output.value_node.shape),)

        if len(internal_shapes) != len(output.external_shapes):
            raise NormalizationError(
                "Internal and external output arity differ",
                node=output.value_node,
            )

        mx_restore_safe = not is_mx or self._mx_restore_safe(
            output.value_node, internal_shapes, output.external_shapes
        )

        plans = []
        for internal_shape, external_shape in zip(
            internal_shapes, output.external_shapes
        ):
            if math.prod(internal_shape) != math.prod(external_shape):
                raise NormalizationError(
                    "Output restore would change element count: "
                    f"internal={internal_shape}, external={external_shape}",
                    node=output.value_node,
                )
            if is_mx and not mx_restore_safe:
                raise NormalizationError(
                    "MX output restore would regroup non-unit dimensions and "
                    "change quantization block semantics",
                    node=output.value_node,
                )
            plans.append(
                OutputPlan(
                    internal_shape=internal_shape,
                    external_shape=external_shape,
                    parent_operation=(
                        "identity"
                        if internal_shape == external_shape
                        else self.config.parent_shape_method
                    ),
                )
            )
        return tuple(plans)

    def _mx_restore_safe(
        self,
        value_node: fx.Node,
        internal_shapes: Tuple[Shape, ...],
        external_shapes: Tuple[Shape, ...],
    ) -> bool:
        """Whether restoring a single-axis ``quantize_mx`` op's outputs to
        ``external_shapes`` keeps every quantization block intact.

        The restore is a pure order-preserving row-major view (guaranteed here:
        a permute would route through the relayout branch and the restore is an
        ``aten.view``), so ``reshape_preserves_full_blocks`` decides it, on any
        quantized axis.  An identity restore passes it by construction: nothing
        moves, and the padding pass has already made the axis a whole number of
        blocks long.
        """
        axes = value_node.args[2]
        block_size = value_node.args[3]
        if not isinstance(block_size, int) or block_size <= 0:
            return False
        if len(axes) != 1:
            return False
        # quantize_mx returns (scale, data); blocks live on the data output.
        data_internal = tuple(internal_shapes[-1])
        data_external = tuple(external_shapes[-1])
        # The axis counts from the end, so the restore hands it to whatever dim
        # sits that far from the end of the restored shape.
        dim = axes[0]
        if dim >= 0:
            dim -= len(data_internal)
        if len(data_external) < -dim:
            return False
        return reshape_preserves_full_blocks(
            data_internal,
            dim + len(data_internal),
            data_external,
            dim + len(data_external),
            block_size,
        )

    # ------------------------------------------------------------------
    # Graph rewriting
    # ------------------------------------------------------------------

    def _rewrite_parent_inputs(
        self,
        *,
        parent: fx.GraphModule,
        call_node: fx.Node,
        placeholder_bindings: Mapping[str, Tuple[str, Any]],
        input_plans: Mapping[str, InputPlan],
    ) -> None:
        args = list(call_node.args)
        kwargs = dict(call_node.kwargs)

        with parent.graph.inserting_before(call_node):
            for name, plan in input_plans.items():
                binding_kind, binding_key = placeholder_bindings[name]
                value = (
                    args[binding_key]
                    if binding_kind == "arg"
                    else kwargs[binding_key]
                )
                if plan.parent_operation == "identity":
                    continue
                if not isinstance(value, fx.Node):
                    raise NormalizationError(
                        f"Cannot insert {plan.parent_operation} for non-node "
                        f"argument {name}"
                    )
                formatted = parent.graph.call_function(
                    _RELAYOUT_OPS[plan.parent_operation],
                    (value, list(plan.boundary_shape)),
                )
                # Shallow copy: a deepcopy would touch a FakeTensor ``val``.
                formatted.meta = dict(value.meta)
                propagate_shape(formatted, parent)
                if binding_kind == "arg":
                    args[binding_key] = formatted
                else:
                    kwargs[binding_key] = formatted

        call_node.args = tuple(args)
        call_node.kwargs = kwargs

    def _rewrite_child(
        self,
        *,
        child: fx.GraphModule,
        anchor: Optional[fx.Node],
        output: _OutputDescription,
    ) -> None:
        # Shape-nops to keep: those building anchor operands (the anchor
        # prelude) and the trailing output relayout kept inside the fused op.
        keep = ancestors(anchor) | set(output.output_relayout)
        removable = [
            node
            for node in child.graph.nodes
            if is_aliasing_op(node) and node not in keep
        ]

        # An axis named from the front is stated against the rank the tensor
        # carries before the shape-nops go, so record that rank per blocking op.
        ranks = {}
        for node in child.graph.nodes:
            if _blocking_axes(node) is None:
                continue
            shape = getattr(node.args[0], "shape", None)
            if shape is not None:
                ranks[node] = len(shape)

        for node in removable:
            source = node.all_input_nodes[0] if node.all_input_nodes else None
            if source is None:
                raise NormalizationError(
                    "Shape no-op has no tensor input", node=node
                )
            node.replace_all_uses_with(source)
            child.graph.erase_node(node)

        for node, rank in ranks.items():
            shape = getattr(node.args[0], "shape", None)
            if shape is None or len(shape) == rank:
                continue
            node.update_arg(
                AXES_ARG_INDEX_MAP[node.target],
                tuple(
                    a + len(shape) - rank if a >= 0 else a
                    for a in _blocking_axes(node)
                ),
            )

    def _rewrite_parent_output(
        self,
        parent: fx.GraphModule,
        call_node: fx.Node,
        output: _OutputDescription,
        output_plans: Tuple[OutputPlan, ...],
    ) -> None:
        if output.is_multi_output:
            self._rewrite_parent_tuple_output(parent, call_node, output_plans)
            return

        plan = output_plans[0]
        if plan.parent_operation == "identity":
            return

        old_users = list(call_node.users)
        with parent.graph.inserting_after(call_node):
            restored = parent.graph.call_function(
                _RELAYOUT_OPS[plan.parent_operation],
                (call_node, list(plan.external_shape)),
            )
        # Shallow copy: a deepcopy would touch a FakeTensor ``val``.
        restored.meta = dict(call_node.meta)
        propagate_shape(restored, parent)
        for user in old_users:
            user.replace_input_with(call_node, restored)

    def _rewrite_parent_tuple_output(
        self,
        parent: fx.GraphModule,
        call_node: fx.Node,
        output_plans: Tuple[OutputPlan, ...],
    ) -> None:
        if all(plan.parent_operation == "identity" for plan in output_plans):
            return

        graph = call_node.graph
        for user in list(call_node.users):
            if user.target != operator.getitem:
                if user.op == "output":
                    user.args = (
                        self._materialize_tuple_outputs(
                            parent, call_node, output_plans
                        ),
                    )
                    continue
                raise NormalizationError(
                    "Tuple output restoration requires parent getitem users",
                    node=call_node,
                )
            raw_index = user.args[1]
            if not isinstance(raw_index, int):
                raise NormalizationError(
                    "Tuple output getitem index must be static",
                    node=user,
                )
            index = raw_index
            if index < 0:
                index += len(output_plans)
            if index < 0 or index >= len(output_plans):
                raise NormalizationError(
                    f"Tuple output index {raw_index} is out of range",
                    node=user,
                )

            plan = output_plans[index]
            self._set_tuple_item_meta(user, call_node, index)
            propagate_shape(user, parent)
            if plan.parent_operation == "identity":
                continue

            old_users = list(user.users)
            with graph.inserting_after(user):
                restored = graph.call_function(
                    _RELAYOUT_OPS[plan.parent_operation],
                    (user, list(plan.external_shape)),
                )
            restored.meta = dict(user.meta)
            propagate_shape(restored, parent)
            for old_user in old_users:
                old_user.replace_input_with(user, restored)

    def _materialize_tuple_outputs(
        self,
        parent: fx.GraphModule,
        call_node: fx.Node,
        output_plans: Tuple[OutputPlan, ...],
    ) -> Tuple[fx.Node, ...]:
        graph = call_node.graph
        outputs = []
        insert_after = call_node
        for index, plan in enumerate(output_plans):
            with graph.inserting_after(insert_after):
                item = graph.call_function(operator.getitem, (call_node, index))
            item.meta = dict(call_node.meta)
            self._set_tuple_item_meta(item, call_node, index)
            propagate_shape(item, parent)
            insert_after = item
            if plan.parent_operation == "identity":
                outputs.append(item)
                continue
            with graph.inserting_after(insert_after):
                restored = graph.call_function(
                    _RELAYOUT_OPS[plan.parent_operation],
                    (item, list(plan.external_shape)),
                )
            restored.meta = dict(item.meta)
            propagate_shape(restored, parent)
            outputs.append(restored)
            insert_after = restored
        return tuple(outputs)

    def _set_tuple_item_meta(
        self,
        item: fx.Node,
        tuple_node: fx.Node,
        index: int,
    ) -> None:
        dtypes = tuple_node.meta.get("dtype")
        if (
            isinstance(dtypes, (tuple, list))
            and index < len(dtypes)
            and dtypes[index] is not None
        ):
            item.meta["dtype"] = dtypes[index]

    # ------------------------------------------------------------------
    # Anchor use analysis
    # ------------------------------------------------------------------

    def _reject_anchor_inputs_reused_in_tail(
        self,
        *,
        placeholders: Sequence[fx.Node],
        anchor: fx.Node,
        output_value: fx.Node,
        anchor_inputs: set[fx.Node],
    ) -> None:
        for placeholder in placeholders:
            if placeholder not in anchor_inputs:
                continue
            side_map = self._propagate_map(
                child=anchor.graph.owning_module,  # type: ignore[arg-type]
                seed_node=placeholder,
                output_value=output_value,
                anchor=anchor,
            )
            if side_map is not None:
                raise NormalizationError(
                    "An anchor operand is also reused by the pointwise tail "
                    "outside the anchor. This implementation requires a "
                    "duplicated/formatted argument for that case",
                    node=placeholder,
                )

    # ------------------------------------------------------------------
    # Shape operation interpretation
    # ------------------------------------------------------------------

    def _apply_shape_noop(self, node: fx.Node, value: AddressMap) -> AddressMap:
        """Replay a shape-only op on the address map, reindexing it exactly as
        the op would the data: ``view`` / ``reshape`` / ``flatten`` regroup
        the digits, ``squeeze`` / ``unsqueeze`` / ``select`` / ``slice`` drop
        or restrict dimensions, ``transpose`` / ``permute`` reorder them and
        ``expand`` broadcasts.  A non-order-preserving op yields a map that
        later fails the fixed-stride check."""
        aten = torch.ops.aten
        args = node.args[1:]
        try:
            if node.target in (
                aten.view.default,
                aten.reshape.default,
                aten._unsafe_view.default,
            ):
                return value.reshape(args[0])
            if node.target is aten.flatten.using_ints:
                return value.flatten(*args)
            if node.target in (aten.squeeze.dim, aten.squeeze.dims):
                return value.squeeze(args[0])
            if node.target is aten.squeeze.default:
                return value.squeeze()
            if node.target is aten.unsqueeze.default:
                return value.unsqueeze(args[0])
            if node.target is aten.select.int:
                return value.select(args[0], args[1])
            if node.target is aten.slice.Tensor:
                return value.slice(*args)
            if node.target is aten.transpose.int:
                return value.transpose(args[0], args[1])
            if node.target is aten.permute.default:
                return value.permute(args[0])
            if node.target is aten.expand.default:
                return value.expand(args[0])
        except ValueError as exc:
            raise NormalizationError(
                "Unsupported shape no-op", node=node
            ) from exc
        raise NormalizationError("Unsupported shape no-op", node=node)

    # ------------------------------------------------------------------
    # Metadata and FX utilities
    # ------------------------------------------------------------------

    def _is_supported_mx_op(self, node: Optional[fx.Node]) -> bool:
        return (
            isinstance(node, fx.Node)
            and node.op == "call_function"
            and node.target
            in {
                torch.ops.quantized_ops.quantize_mx.default,
                torch.ops.quantized_ops.quantize_mx_outlier.default,
                torch.ops.quantized_ops.quantize_affine.default,
            }
        )

    def _validate_external_layout(
        self, placeholder: fx.Node, shape: Shape
    ) -> None:
        if not self.config.require_contiguous_inputs:
            return
        tensor_meta = placeholder.meta.get("tensor_meta")
        if tensor_meta is None or not hasattr(tensor_meta, "stride"):
            return
        actual = tuple(int(x) for x in tensor_meta.stride)
        expected = self._contiguous_strides(shape)
        for extent, a, e in zip(shape, actual, expected):
            if extent != 1 and a != e:
                raise NormalizationError(
                    f"External input is not contiguous: shape={shape}, "
                    f"stride={actual}",
                    node=placeholder,
                )

    def _bind_parent_arguments(
        self, call_node: fx.Node, placeholders: Sequence[fx.Node]
    ) -> Dict[str, Tuple[str, Any]]:
        bindings: Dict[str, Tuple[str, Any]] = {}
        positional_count = len(call_node.args)
        for index, placeholder in enumerate(placeholders):
            if index < positional_count:
                bindings[placeholder.name] = ("arg", index)
                continue
            candidates = [str(placeholder.target), placeholder.name]
            key = next(
                (name for name in candidates if name in call_node.kwargs), None
            )
            if key is None:
                raise NormalizationError(
                    f"Cannot bind child placeholder {placeholder.name!r} to "
                    f"parent call arguments",
                    node=call_node,
                )
            bindings[placeholder.name] = ("kwarg", key)
        return bindings

    def _broadcast_map(
        self, value: AddressMap, output_shape: Shape, node: fx.Node
    ) -> AddressMap:
        try:
            return value.expand(output_shape)
        except ValueError as exc:
            raise NormalizationError(
                f"Address map with shape {tuple(value.shape)} cannot "
                f"broadcast to {output_shape}",
                node=node,
            ) from exc

    @staticmethod
    def _contiguous_strides(shape: Shape) -> Tuple[int, ...]:
        if not shape:
            return ()
        strides = [0] * len(shape)
        running = 1
        for index in range(len(shape) - 1, -1, -1):
            strides[index] = running
            running *= max(shape[index], 1)
        return tuple(strides)
