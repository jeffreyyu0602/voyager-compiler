from __future__ import annotations

import logging
import math
import operator
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch import fx

from .mapping_utils import (
    is_elementwise_op,
    is_nop,
    is_shape_changing_nop,
)
from ..pt2e_utils import propagate_shape

logger = logging.getLogger(__name__)


Shape = Tuple[int, ...]
NodePredicate = Callable[[fx.Node], bool]


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
    mx_node: Optional[fx.Node] = None
    selected_index: Optional[int] = None
    primary_plan_index: int = 0

    @property
    def is_multi_output(self) -> bool:
        return self.selected_index is None and len(self.external_shapes) > 1


@dataclass
class NormalizationConfig:
    """Configuration for :class:`IterationSpaceNormalizer`.

    The default target detection is intentionally conservative. Projects with
    custom operators should pass explicit predicates or annotate nodes with:

        node.meta["strong_anchor"] = True
        node.meta["pointwise"] = True
    """

    max_address_map_elements: int = 1 << 64
    require_contiguous_inputs: bool = True
    parent_shape_method: str = "view"  # "view" or "reshape"
    anchor_predicate: Optional[NodePredicate] = None
    pointwise_predicate: Optional[NodePredicate] = None

    def __post_init__(self) -> None:
        if self.max_address_map_elements <= 0:
            raise ValueError("max_address_map_elements must be positive")
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

        from .mapping import get_anchor_node

        anchor = get_anchor_node(call_node)
        # Keep only a *strong* anchor (gemm/conv/layernorm/softmax); a pointwise
        # "anchor" means the group is anchorless (iteration == output shape).
        if (
            anchor is None
            or is_elementwise_op(anchor)
            or self._is_supported_mx_op(anchor)
        ):
            anchor = None

        if anchor is None:
            iteration_shape = self._node_tensor_shape(output.iteration_node)
            anchor_inputs: set[fx.Node] = set()
            normalizable_placeholders = placeholders
        else:
            iteration_shape = self._node_tensor_shape(anchor)
            anchor_inputs = self._placeholder_ancestors(anchor)
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

        self._check_analysis_size(iteration_shape, context="iteration space")

        input_plans: Dict[str, InputPlan] = {}
        for placeholder in normalizable_placeholders:
            original_shape = self._node_tensor_shape(placeholder)
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
                propagated=propagated,
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

        output_plans = self._build_output_plans(
            output=output,
            iteration_shape=iteration_shape,
        )
        output_plan = output_plans[output.primary_plan_index]

        # No mutation occurs before this point.
        placeholder_bindings = self._bind_parent_arguments(
            call_node, placeholders
        )
        self._rewrite_parent_inputs(
            parent=parent,
            call_node=call_node,
            placeholder_bindings=placeholder_bindings,
            input_plans=input_plans,
        )
        self._rewrite_child(
            child=child,
            anchor=anchor,
            iteration_shape=iteration_shape,
            input_plans=input_plans,
            output=output,
            output_plans=output_plans,
        )
        self._rewrite_parent_output(
            parent,
            call_node,
            output,
            output_plans,
            iteration_shape,
        )

        child.graph.eliminate_dead_code()
        child.graph.lint()
        child.recompile()

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
    ) -> Optional[torch.Tensor]:
        seed_shape = seed_node.shape
        self._check_analysis_size(seed_shape, context=f"seed {seed_node.name}")
        seed = torch.arange(math.prod(seed_shape), dtype=torch.int64).reshape(
            seed_shape
        )

        env: Dict[fx.Node, Optional[torch.Tensor]] = {}
        for node in child.graph.nodes:
            if node is seed_node:
                env[node] = seed
            elif node.op in {"placeholder", "get_attr"}:
                env[node] = None
            elif anchor is not None and node is anchor:
                env[node] = None
            elif is_shape_changing_nop(node):
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
            elif is_elementwise_op(node):
                maps = [
                    env[arg]
                    for arg in self._node_args(node)
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
                        if not torch.equal(first, other):
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
        expected = torch.arange(
            math.prod(iteration_shape), dtype=torch.int64
        ).reshape(iteration_shape)
        actual = propagated.reshape(iteration_shape)
        if not torch.equal(actual, expected):
            raise NormalizationError(
                "The tail changes the order or multiplicity of the "
                "anchor-produced stream",
                node=anchor,
            )

    def _recover_and_validate_affine_map(
        self,
        propagated: torch.Tensor,
        iteration_shape: Shape,
        placeholder: fx.Node,
    ) -> Tuple[int, Shape]:
        if tuple(propagated.shape) != iteration_shape:
            raise NormalizationError(
                f"Map shape {tuple(propagated.shape)} != iteration shape "
                f"{iteration_shape}",
                node=placeholder,
            )
        origin = (0,) * len(iteration_shape)
        base = (
            int(propagated[origin].item())
            if iteration_shape
            else int(propagated.item())
        )
        strides = []
        for dim, extent in enumerate(iteration_shape):
            if extent <= 1:
                strides.append(0)
                continue
            index = [0] * len(iteration_shape)
            index[dim] = 1
            strides.append(int(propagated[tuple(index)].item()) - base)

        expected = torch.full(iteration_shape, base, dtype=torch.int64)
        for dim, (extent, stride) in enumerate(zip(iteration_shape, strides)):
            if extent <= 1 or stride == 0:
                continue
            shape = [1] * len(iteration_shape)
            shape[dim] = extent
            expected = (
                expected
                + torch.arange(extent, dtype=torch.int64).reshape(shape)
                * stride
            )

        if not torch.equal(expected, propagated):
            mismatch = self._first_mismatch(expected, propagated)
            raise NormalizationError(
                "Input map is not representable by one fixed base and one "
                f"stride per iteration dimension; first mismatch at {mismatch}",
                node=placeholder,
            )
        return base, tuple(strides)

    def _derive_boundary_shape(
        self,
        *,
        original_shape: Shape,
        iteration_shape: Shape,
        base: int,
        strides: Shape,
        propagated: torch.Tensor,
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

        candidate = torch.arange(
            math.prod(original_shape), dtype=torch.int64
        ).reshape(boundary)
        try:
            candidate = torch.broadcast_to(candidate, iteration_shape)
        except RuntimeError as exc:
            raise NormalizationError(
                f"Boundary shape {boundary} cannot broadcast to "
                f"{iteration_shape}",
                node=placeholder,
            ) from exc

        if not torch.equal(candidate, propagated):
            mismatch = self._first_mismatch(candidate, propagated)
            raise NormalizationError(
                "Affine access is not specifically expressible as an "
                "order-preserving view plus fetch-side broadcast; first "
                f"mismatch at {mismatch}",
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

        if self._is_supported_mx_op(value):
            shapes = self._tuple_output_shapes(value)
            data_node = self._mx_data_node(value)
            return _OutputDescription(
                value_node=value,
                map_target=value,
                iteration_node=data_node,
                external_shapes=shapes,
                mx_node=value,
                primary_plan_index=len(shapes) - 1,
            )

        if self._is_getitem_node(value):
            source = value.args[0]
            index = value.args[1] if len(value.args) > 1 else None
            if (
                isinstance(source, fx.Node)
                and self._is_supported_mx_op(source)
                and isinstance(index, int)
            ):
                shapes = self._tuple_output_shapes(source)
                data_node = self._mx_data_node(source)
                if index < 0:
                    index += len(shapes)
                if index < 0 or index >= len(shapes):
                    raise NormalizationError(
                        f"MX output index {index} is out of range",
                        node=value,
                    )
                return _OutputDescription(
                    value_node=value,
                    map_target=source,
                    iteration_node=data_node,
                    external_shapes=(self._node_tensor_shape(value),),
                    mx_node=source,
                    selected_index=index,
                )

        return _OutputDescription(
            value_node=value,
            map_target=value,
            iteration_node=value,
            external_shapes=(self._node_tensor_shape(value),),
        )

    def _build_output_plans(
        self,
        *,
        output: _OutputDescription,
        iteration_shape: Shape,
    ) -> Tuple[OutputPlan, ...]:
        internal_shapes = self._internal_output_shapes(output, iteration_shape)
        if len(internal_shapes) != len(output.external_shapes):
            raise NormalizationError(
                "Internal and external output arity differ",
                node=output.value_node,
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
            if output.mx_node is not None and not self._same_non_unit_dims(
                internal_shape, external_shape
            ):
                raise NormalizationError(
                    "MX output restore would regroup non-unit dimensions and "
                    "change quantization block semantics",
                    node=output.mx_node,
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

    def _internal_output_shapes(
        self,
        output: _OutputDescription,
        iteration_shape: Shape,
    ) -> Tuple[Shape, ...]:
        if output.mx_node is None:
            return (iteration_shape,)

        mx_shapes = self._mx_output_shapes(output.mx_node, iteration_shape)
        if output.selected_index is None:
            return mx_shapes
        return (mx_shapes[output.selected_index],)

    def _mx_output_shapes(
        self,
        node: fx.Node,
        data_shape: Shape,
    ) -> Tuple[Shape, ...]:
        scale_shape = self._mx_scale_shape(node, data_shape)
        if node.target == torch.ops.quantized_ops.quantize_mx.default:
            return (scale_shape, data_shape)

        if node.target == torch.ops.quantized_ops.quantize_mx_outlier.default:
            if len(data_shape) < 2:
                raise NormalizationError(
                    "quantize_mx_outlier requires at least a matrix-shaped "
                    "input",
                    node=node,
                )
            max_pct = float(self._get_arg(node, 9, "max_pct", 0.01))
            batch_shape = data_shape[:-2]
            mat_shape = data_shape[-2:]
            max_nnz = int(math.prod(mat_shape) * max_pct)
            sparse_shape = batch_shape + (max_nnz,)
            indptr_shape = batch_shape + (mat_shape[0] + 1,)
            return (
                sparse_shape,
                sparse_shape,
                indptr_shape,
                scale_shape,
                data_shape,
            )

        raise NormalizationError("Unsupported MX op", node=node)

    def _mx_scale_shape(self, node: fx.Node, data_shape: Shape) -> Shape:
        axes = self._get_arg(node, 2, "axes", None)
        block_size = self._get_arg(node, 3, "block_size", None)
        if axes is None or block_size is None:
            raise NormalizationError(
                "MX quantization requires static axes and block_size",
                node=node,
            )
        axes = (axes,) if isinstance(axes, int) else tuple(axes)
        rank = len(data_shape)
        scale_shape = list(data_shape)
        for axis in axes:
            axis = int(axis)
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                raise NormalizationError(
                    f"MX axis {axis} is out of range for shape {data_shape}",
                    node=node,
                )
            scale_shape[axis] = math.ceil(scale_shape[axis] / int(block_size))
        return tuple(scale_shape)

    def _get_arg(
        self,
        node: fx.Node,
        index: int,
        name: str,
        default: Any = None,
    ) -> Any:
        if name in node.kwargs:
            return self._resolve_static(node.kwargs[name])
        if len(node.args) > index:
            return self._resolve_static(node.args[index])
        return default

    def _same_non_unit_dims(self, left: Shape, right: Shape) -> bool:
        return tuple(dim for dim in left if dim != 1) == tuple(
            dim for dim in right if dim != 1
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
                formatted = parent.graph.call_method(
                    plan.parent_operation,
                    args=(value, *plan.boundary_shape),
                )
                # Shallow copy: a deepcopy would touch a FakeTensor ``val``.
                formatted.meta = dict(value.meta)
                self._set_shape_meta(formatted, plan.boundary_shape)
                propagate_shape(formatted, parent)
                formatted.meta["iteration_space_boundary"] = {
                    "iteration_shape": plan.iteration_shape,
                    "broadcast_dims": plan.broadcast_dims,
                    "base": plan.base,
                    "strides": plan.strides,
                }
                # Retarget the placeholder's ``source_node`` to the new ``view``
                plan.placeholder.meta["source_node"] = formatted
                if binding_kind == "arg":
                    args[binding_key] = formatted
                else:
                    kwargs[binding_key] = formatted

        call_node.args = tuple(args)
        call_node.kwargs = kwargs
        call_node.meta["iteration_shape"] = (
            next(iter(input_plans.values())).iteration_shape
            if input_plans
            else call_node.meta.get("iteration_shape")
        )
        call_node.meta["input_fetch_plans"] = {
            name: {
                "boundary_shape": plan.boundary_shape,
                "broadcast_dims": plan.broadcast_dims,
                "base": plan.base,
                "strides": plan.strides,
            }
            for name, plan in input_plans.items()
        }

    def _rewrite_child(
        self,
        *,
        child: fx.GraphModule,
        anchor: Optional[fx.Node],
        iteration_shape: Shape,
        input_plans: Mapping[str, InputPlan],
        output: _OutputDescription,
        output_plans: Tuple[OutputPlan, ...],
    ) -> None:
        for plan in input_plans.values():
            placeholder = plan.placeholder
            self._set_shape_meta(placeholder, plan.boundary_shape)
            placeholder.meta["iteration_space_boundary"] = {
                "iteration_shape": plan.iteration_shape,
                "broadcast_dims": plan.broadcast_dims,
                "base": plan.base,
                "strides": plan.strides,
            }

        anchor_ancestors = (
            self._ancestors(anchor) if anchor is not None else set()
        )
        removable = []
        for node in child.graph.nodes:
            if not is_shape_changing_nop(node):
                continue
            if anchor is not None and node in anchor_ancestors:
                # Shape operations used to construct anchor operands remain
                # part of the anchor prelude.
                continue
            removable.append(node)

        for node in removable:
            source = node.all_input_nodes[0] if node.all_input_nodes else None
            if source is None:
                raise NormalizationError(
                    "Shape no-op has no tensor input", node=node
                )
            node.replace_all_uses_with(source)

        # Erase in reverse topological order after uses have been replaced.
        for node in reversed(removable):
            if len(node.users) == 0:
                child.graph.erase_node(node)

        descendants = (
            self._descendants(anchor)
            if anchor is not None
            else set(child.graph.nodes)
        )
        for node in child.graph.nodes:
            if is_elementwise_op(node) and (
                anchor is None or node in descendants
            ):
                node.meta["iteration_shape"] = iteration_shape
                # This is the hardware logical shape. Direct PyTorch
                # execution may retain a smaller broadcast-compatible
                # intermediate shape.
                node.meta["hardware_tensor_shape"] = iteration_shape

        if output.mx_node is None:
            self._set_shape_meta(
                output.value_node, output_plans[0].internal_shape
            )
            output.value_node.meta["iteration_shape"] = iteration_shape
            return

        mx_shapes = self._mx_output_shapes(output.mx_node, iteration_shape)
        self._set_tuple_shape_meta(output.mx_node, mx_shapes)
        output.mx_node.meta["iteration_shape"] = iteration_shape
        output.mx_node.meta["hardware_tensor_shape"] = iteration_shape
        if output.selected_index is not None:
            self._set_shape_meta(
                output.value_node,
                output_plans[0].internal_shape,
            )
            output.value_node.meta["iteration_shape"] = iteration_shape

    def _rewrite_parent_output(
        self,
        parent: fx.GraphModule,
        call_node: fx.Node,
        output: _OutputDescription,
        output_plans: Tuple[OutputPlan, ...],
        iteration_shape: Shape,
    ) -> None:
        call_node.meta["iteration_shape"] = iteration_shape
        call_node.meta["output_plans"] = [
            {
                "internal_shape": plan.internal_shape,
                "external_shape": plan.external_shape,
                "parent_operation": plan.parent_operation,
            }
            for plan in output_plans
        ]

        if output.is_multi_output:
            self._set_tuple_shape_meta(
                call_node,
                tuple(plan.internal_shape for plan in output_plans),
            )
            self._rewrite_parent_tuple_output(call_node, output_plans)
            return

        plan = output_plans[0]
        self._set_shape_meta(call_node, plan.internal_shape)
        if plan.parent_operation == "identity":
            return

        old_users = list(call_node.users)
        with parent.graph.inserting_after(call_node):
            restored = parent.graph.call_method(
                plan.parent_operation,
                args=(call_node, *plan.external_shape),
            )
        # Shallow copy: a deepcopy would touch a FakeTensor ``val``.
        restored.meta = dict(call_node.meta)
        self._set_shape_meta(restored, plan.external_shape)
        self._copy_reshaped_value(restored, call_node, plan.external_shape)
        restored.meta["iteration_space_output_restore"] = {
            "internal_shape": plan.internal_shape,
            "external_shape": plan.external_shape,
        }
        for user in old_users:
            user.replace_input_with(call_node, restored)

    def _rewrite_parent_tuple_output(
        self,
        call_node: fx.Node,
        output_plans: Tuple[OutputPlan, ...],
    ) -> None:
        if all(plan.parent_operation == "identity" for plan in output_plans):
            return

        graph = call_node.graph
        for user in list(call_node.users):
            if not self._is_getitem_node(user):
                if user.op == "output":
                    user.args = (
                        self._materialize_tuple_outputs(
                            call_node, output_plans
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
            self._set_shape_meta(user, plan.internal_shape)
            if plan.parent_operation == "identity":
                continue

            old_users = list(user.users)
            with graph.inserting_after(user):
                restored = graph.call_method(
                    plan.parent_operation,
                    args=(user, *plan.external_shape),
                )
            restored.meta = dict(user.meta)
            self._set_shape_meta(restored, plan.external_shape)
            self._copy_reshaped_value(restored, user, plan.external_shape)
            restored.meta["iteration_space_output_restore"] = {
                "internal_shape": plan.internal_shape,
                "external_shape": plan.external_shape,
            }
            for old_user in old_users:
                old_user.replace_input_with(user, restored)

    def _materialize_tuple_outputs(
        self,
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
            self._set_shape_meta(item, plan.internal_shape)
            insert_after = item
            if plan.parent_operation == "identity":
                outputs.append(item)
                continue
            with graph.inserting_after(insert_after):
                restored = graph.call_method(
                    plan.parent_operation,
                    args=(item, *plan.external_shape),
                )
            restored.meta = dict(item.meta)
            self._set_shape_meta(restored, plan.external_shape)
            self._copy_reshaped_value(restored, item, plan.external_shape)
            restored.meta["iteration_space_output_restore"] = {
                "internal_shape": plan.internal_shape,
                "external_shape": plan.external_shape,
            }
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

    def _apply_shape_noop(
        self, node: fx.Node, value: torch.Tensor
    ) -> torch.Tensor:
        """Replay a shape-only op on the integer address map by rerunning the
        op with the map substituted for its tensor input.  This reindexes the
        map exactly as the op would the data, and handles this repo's aten ops
        (``view`` / ``reshape`` / ``squeeze`` / ``unsqueeze`` / ``select`` /
        identity ``slice``) generically; a non-order-preserving op yields a map
        that later fails the fixed-stride check."""
        new_args = tuple(
            value if isinstance(a, fx.Node) else a for a in node.args
        )
        new_kwargs = {
            k: (value if isinstance(v, fx.Node) else v)
            for k, v in node.kwargs.items()
        }
        try:
            return node.target(*new_args, **new_kwargs)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - surface as a normalization error
            raise NormalizationError(
                "Unsupported shape no-op", node=node
            ) from exc

    def _extract_method_shape(self, node: fx.Node) -> Shape:
        raw = node.args[1:]
        if len(raw) == 1 and isinstance(
            self._resolve_static(raw[0]), (tuple, list, torch.Size)
        ):
            raw_shape = self._resolve_static(raw[0])
        else:
            raw_shape = tuple(self._resolve_static(x) for x in raw)
        return self._normalize_shape_arg(raw_shape)

    def _normalize_shape_arg(self, value: Any) -> Shape:
        if isinstance(value, int):
            return (int(value),)
        if isinstance(value, (tuple, list, torch.Size)):
            return tuple(int(v) for v in value)
        raise NormalizationError(f"Expected a static shape, got {value!r}")

    def _is_full_range_getitem(self, index: Any, input_shape: Shape) -> bool:
        if not isinstance(index, tuple):
            index = (index,)

        expanded: list[Any] = []
        ellipsis_seen = False
        explicit_consumed = sum(1 for item in index if item is not Ellipsis)
        for item in index:
            if item is Ellipsis:
                if ellipsis_seen:
                    return False
                ellipsis_seen = True
                missing = len(input_shape) - explicit_consumed
                expanded.extend([slice(None)] * missing)
            else:
                expanded.append(item)
        expanded.extend([slice(None)] * (len(input_shape) - len(expanded)))
        if len(expanded) != len(input_shape):
            return False

        for extent, item in zip(input_shape, expanded):
            if not isinstance(item, slice):
                return False
            start = (
                0 if item.start is None else self._maybe_static_int(item.start)
            )
            stop = (
                extent
                if item.stop is None
                else self._maybe_static_int(item.stop)
            )
            step = 1 if item.step is None else self._maybe_static_int(item.step)
            if start is None or stop is None or step is None:
                return False
            if start != 0 or stop != extent or step != 1:
                return False
        return True

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
            }
        )

    def _is_getitem_node(self, node: fx.Node) -> bool:
        return node.op == "call_function" and node.target is operator.getitem

    def _mx_data_node(self, node: fx.Node) -> fx.Node:
        if not node.args or not isinstance(node.args[0], fx.Node):
            raise NormalizationError(
                "MX quantization input must be a tensor node", node=node
            )
        return node.args[0]

    def _node_tensor_shape(self, node: fx.Node) -> Shape:
        shape = getattr(node, "shape", None)
        if shape is not None:
            return self._normalize_shape_arg(shape)
        value = getattr(node, "value", None)
        if isinstance(value, torch.Tensor):
            return tuple(int(dim) for dim in value.shape)
        raise NormalizationError("Node does not have a tensor shape", node=node)

    def _tuple_output_shapes(self, node: fx.Node) -> Tuple[Shape, ...]:
        shape = getattr(node, "shape", None)
        if isinstance(shape, (tuple, list)) and shape:
            first = shape[0]
            if isinstance(first, (tuple, list, torch.Size)):
                return tuple(self._normalize_shape_arg(s) for s in shape)

        value = getattr(node, "value", None)
        if isinstance(value, (tuple, list)) and all(
            isinstance(v, torch.Tensor) for v in value
        ):
            return tuple(tuple(int(dim) for dim in v.shape) for v in value)

        raise NormalizationError(
            "MX output node does not have tuple tensor shapes", node=node
        )

    def _set_shape_meta(self, node: fx.Node, shape: Shape) -> None:
        shape = tuple(shape)
        node.shape = torch.Size(shape)
        node.meta["normalized_shape"] = tuple(shape)
        self._reshape_node_value(node, shape)
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is not None and hasattr(tensor_meta, "_replace"):
            updates: Dict[str, Any] = {"shape": torch.Size(shape)}
            if hasattr(tensor_meta, "stride"):
                updates["stride"] = self._contiguous_strides(shape)
            try:
                node.meta["tensor_meta"] = tensor_meta._replace(**updates)
            except (TypeError, ValueError):
                pass

    def _set_tuple_shape_meta(
        self,
        node: fx.Node,
        shapes: Tuple[Shape, ...],
    ) -> None:
        shapes = tuple(tuple(shape) for shape in shapes)
        node.shape = tuple(torch.Size(shape) for shape in shapes)
        node.meta["normalized_shape"] = shapes
        value = getattr(node, "value", None)
        if not isinstance(value, (tuple, list)) or len(value) != len(shapes):
            return
        reshaped = []
        changed = False
        for item, shape in zip(value, shapes):
            if isinstance(item, torch.Tensor) and item.numel() == math.prod(
                shape
            ):
                reshaped.append(item.reshape(shape).cpu().clone())
                changed = True
            else:
                reshaped.append(item)
        if changed:
            node.value = tuple(reshaped)

    def _reshape_node_value(self, node: fx.Node, shape: Shape) -> None:
        value = getattr(node, "value", None)
        if isinstance(value, torch.Tensor) and value.numel() == math.prod(
            shape
        ):
            node.value = value.reshape(shape).cpu().clone()

    def _copy_reshaped_value(
        self,
        target: fx.Node,
        source: fx.Node,
        shape: Shape,
    ) -> None:
        value = getattr(source, "value", None)
        if isinstance(value, torch.Tensor) and value.numel() == math.prod(
            shape
        ):
            target.value = value.reshape(shape).cpu().clone()

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

    def _placeholder_ancestors(self, node: fx.Node) -> set[fx.Node]:
        return {n for n in self._ancestors(node) if n.op == "placeholder"}

    def _ancestors(self, node: Optional[fx.Node]) -> set[fx.Node]:
        if node is None:
            return set()
        result: set[fx.Node] = set()
        stack = list(node.all_input_nodes)
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.add(current)
            stack.extend(current.all_input_nodes)
        return result

    def _descendants(self, node: Optional[fx.Node]) -> set[fx.Node]:
        if node is None:
            return set()
        result: set[fx.Node] = set()
        stack = list(node.users)
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.add(current)
            stack.extend(current.users)
        return result

    def _node_args(self, node: fx.Node) -> list[fx.Node]:
        found: list[fx.Node] = []

        def collect(arg: Any) -> Any:
            if isinstance(arg, fx.Node):
                found.append(arg)
            return arg

        fx.map_arg((node.args, node.kwargs), collect)
        return found

    def _broadcast_map(
        self, value: torch.Tensor, output_shape: Shape, node: fx.Node
    ) -> torch.Tensor:
        self._check_analysis_size(output_shape, context=f"node {node.name}")
        try:
            return torch.broadcast_to(value, output_shape)
        except RuntimeError as exc:
            raise NormalizationError(
                f"Address map with shape {tuple(value.shape)} cannot "
                f"broadcast to {output_shape}",
                node=node,
            ) from exc

    def _check_analysis_size(self, shape: Shape, *, context: str) -> None:
        count = math.prod(shape)
        if count > self.config.max_address_map_elements:
            raise NormalizationError(
                f"Concrete address-map analysis for {context} requires "
                f"{count} elements, exceeding limit "
                f"{self.config.max_address_map_elements}"
            )

    def _target_name(self, node: fx.Node) -> str:
        target = node.target
        if isinstance(target, str):
            return target
        module = getattr(target, "__module__", "")
        qualname = getattr(
            target, "__qualname__", getattr(target, "__name__", repr(target))
        )
        return f"{module}.{qualname}" if module else str(qualname)

    def _resolve_static(self, value: Any) -> Any:
        if isinstance(value, fx.Node):
            if "val" in value.meta and not isinstance(
                value.meta["val"], torch.Tensor
            ):
                return value.meta["val"]
            raise NormalizationError(
                "Dynamic shape arguments are not supported by this "
                "implementation",
                node=value,
            )
        return value

    @staticmethod
    def _maybe_static_int(value: Any) -> Optional[int]:
        return int(value) if isinstance(value, int) else None

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

    @staticmethod
    def _first_mismatch(
        expected: torch.Tensor, actual: torch.Tensor
    ) -> Tuple[int, ...]:
        mismatch = torch.nonzero(expected != actual, as_tuple=False)
        if mismatch.numel() == 0:
            return ()
        return tuple(int(x) for x in mismatch[0].tolist())


def normalize(
    model: fx.GraphModule, node: fx.Node, submodule: fx.GraphModule
) -> Optional[NormalizationResult]:
    """Plug-in entry called by ``fuse_operator`` after the fused submodule is
    created: normalize that submodule to the anchor's iteration space, in place.

    Populates the child node shapes (the analysis reads ``node.shape``) and
    skips the group with a warning if it cannot be expressed in one iteration
    space, leaving it fused as-is.
    """
    from .shape_prop import ShapeProp

    inputs = []
    for n in node.all_input_nodes:
        value = getattr(n, "value", None)
        inputs.append(
            value.clone() if isinstance(value, torch.Tensor) else value
        )
    ShapeProp(submodule).propagate(*inputs)

    try:
        return IterationSpaceNormalizer().normalize(model, node)
    except NormalizationError as exc:
        logger.warning("normalize: skipped %s: %s", node.name, exc)
        return None
