"""
Loop-aware output generation for bufferized FX graphs.

Three consumers of the bufferized FX dialect (``while_loop`` + ``voyager.*``
nodes), grouped here because they share the same traversal concerns:

  * ``gen_code_bufferized``    FX graph -> ``voyager`` protobuf Model
  * ``gen_compute_graph``      FX graph -> graphviz SVG
  * ``print_bufferized_graph`` FX graph -> indented text

The protobuf is the ``voyager`` schema (``voyager_ir.proto``), which models
this dialect directly and so *replaces* — rather than reuses — the legacy
``param.proto`` emitter (``codegen/mapping.py``, still used by the
non-bufferized path).  Three ideas carry the whole translation:

  * **Every instruction stands alone.**  A ``TensorBox`` — a model input, a
    weight, a ``voyager.alloc`` / ``voyager.zeros`` — declares an allocation.
    Every *use* of it is a ``TensorBoxRef`` repeating that declaration in full
    (name, dtype, memory level and address, banking) under the *operand's*
    shape, plus the window it reads — a ``voyager.subview``, which for a
    software-pipeline bank is the slot a step picks.  Aliasing ops and the
    ``subview`` collapse into the reference, the slot stays a *runtime* offset,
    and nothing has to be resolved to execute an instruction.
  * **Compute is destination-passing.**  ``voyager.insert(src, dst)`` is how the
    FX dialect emulates a write-to-destination; it is not an instruction.  It is
    dropped, and ``dst`` becomes the *producing* op's ``Output.destination`` —
    for a ``torch.cond``, on the producer inside *each* branch.  An ``insert``
    whose source is another buffer is a genuine move, and is emitted as a
    ``clone`` into that destination.
  * **Control logic is emitted**, tagged ``op: "cpu"``: index arithmetic,
    predicates and ``delinearize_index`` run on the control processor rather
    than the accelerator datapath, but they are real work the backend schedules.
    The only thing dropped as structure is the ``while_loop``'s trip test —
    ``while_loop`` is just how the builders spell a ``for``, and
    ``ForLoop.start/end/step`` already says it.
"""

import logging
import operator
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import graphviz
import numpy as np
import torch
from tabulate import tabulate
from torch.fx import GraphModule, Node
from torch.fx.operator_schemas import normalize_function

import interstellar
from voyager_compiler.codegen.node_info import (
    QMAP_PARAMS,
    is_nop,
    quant_param_arg_nodes,
    require_allocation,
)
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _is_compute,
    _produces_tensor,
)
from voyager_compiler.codegen.transform.bufferize.ops import oracle_disabled
from voyager_compiler.codegen.transform.bufferize.utils import (
    _collect_codebook_nodes,
    _passed_whole,
)
from voyager_compiler.codegen.voyager_ir_pb2 import (
    MEMORY_LEVEL_DRAM,
    MEMORY_LEVEL_IMMEDIATE,
    MEMORY_LEVEL_REGISTER,
    MEMORY_LEVEL_SCRATCHPAD,
    SCALAR_BOOL,
    SCALAR_F32,
    SCALAR_INDEX,
    Argument,
    LevelAccessCount,
    LevelTiling,
    LoopBound,
    Model,
    Operation,
    PrimOp,
    ScalarValue,
    TensorBox,
    TensorBoxRef,
    Tiling,
)
from voyager_compiler.shape_prop import ShapeProp

logger = logging.getLogger(__name__)

WHILE_LOOP = torch.ops.higher_order.while_loop
COND = torch.ops.higher_order.cond
COMMIT = torch.ops.higher_order.commit
_INSERT = torch.ops.voyager.insert.default
_ALLOC = torch.ops.voyager.alloc.default
_ZEROS = torch.ops.voyager.zeros.default
_FILL = torch.ops.voyager.fill.default
_SUBVIEW = torch.ops.voyager.subview.default
_ASYNC_COPY = torch.ops.voyager.async_copy.default

# The ops that *declare* storage; every other tensor node refers to storage one
# of these owns.
_ALLOCATORS = (_ALLOC, _ZEROS, _FILL)

# Operand names for the Python builtins the loop control uses (``operator.eq`` /
# ``and_`` / ``add`` ...), which carry no ATen schema to normalize against.
_BUILTIN_ARG_NAMES = ("input", "other")


def _target_name(target, short: bool = False) -> str:
    if isinstance(target, torch._ops.OpOverload):
        name = target._schema.name
        return name.split("::")[-1] if short else name
    return getattr(target, "__name__", str(target))


# ===========================================================================
# 0. Asynchronous tensor-file dumping
# ===========================================================================
# Every model input / parameter is written beside ``model.txt`` so the
# hardware can be driven with real data.  The writes go to a thread pool;
# ``flush_tensor_files`` blocks until they are all on disk.


# Global variable to store the custom save function
_custom_save_function = None


def register_save_function(custom_function):
    """
    Decorator to register a custom function for saving tensors.
    The custom function should accept two arguments: tensor and filename.
    """
    global _custom_save_function
    if not callable(custom_function):
        raise ValueError("The custom function must be callable.")
    _custom_save_function = custom_function
    print(f"Custom save function '{custom_function.__name__}' registered.")
    return custom_function


_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


def _write_numpy(np_array: np.ndarray, filename: str, shape: tuple):
    """Worker function: write the tensor and log when done."""
    try:
        np_array.tofile(filename)
        logger.info(f"✅ Saved tensor {shape} -> {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save tensor {shape} -> {filename}: {e}")


_pending_writes = []


def _save_tensor(tensor: torch.Tensor, filename: str):
    """Asynchronously save tensor to a binary file."""
    t = tensor.detach().cpu().contiguous().to(torch.float32)
    np_array = t.numpy()
    _pending_writes.append(
        _executor.submit(_write_numpy, np_array, filename, tuple(t.shape))
    )


def flush_tensor_files():
    """Block until every queued tensor file is on disk — the writes are handed
    to a thread pool, so a caller that reads ``tensor_files/`` as soon as
    ``compile`` returns would otherwise race them."""
    for future in _pending_writes:
        future.result()
    _pending_writes.clear()


def save_tensor(tensor, filename):
    """
    Save the tensor to a file using the custom save function if defined,
    otherwise use the default _save_tensor function.
    """
    if _custom_save_function is not None:
        _custom_save_function(tensor, filename)
    else:
        _save_tensor(tensor, filename)


# ===========================================================================
# 1. Protobuf code generation
# ===========================================================================


def _norm_extent(e):
    """Normalise a ``loop_extents`` entry to ``(start, end, step)``.

    An entry is either a bare ``end`` count (start 0, step 1) or an explicit
    ``(start, end[, step])`` tuple.
    """
    if isinstance(e, (tuple, list)):
        start, end = int(e[0]), int(e[1])
        step = int(e[2]) if len(e) > 2 else 1
        return start, end, step
    return 0, int(e), 1


def _loop_extents(node) -> List[tuple]:
    """Per-dimension ``(start, end, step)`` for a flattened-grid while_loop.

    Builders tag each ``while_loop`` with ``node.meta['loop_extents']`` — the
    extents of the tile grid it iterates.
    """
    ext = node.meta.get("loop_extents")
    if ext:
        return [_norm_extent(e) for e in ext]
    # Legacy / untagged: a single loop with an unknown bound.
    return [(0, 0, 1)]


def _loop_trip(node) -> int:
    """How many times a flattened-grid while_loop runs, at least once — the
    count the emitted ``ForLoop`` bounds describe.  An FA3 sweep whose grid
    holds a single step tags ``(1, 1, 1)`` and a legacy loop is untagged; both
    would otherwise read as zero iterations."""
    count = 1
    for start, end, step in _loop_extents(node):
        count *= max(1, -(-(end - start) // step))
    return count


def _trip_str(node) -> str:
    """Per-dimension bounds annotation for the text / graphviz views."""
    ext = node.meta.get("loop_extents")
    if not ext:
        return ""
    dims = []
    for start, end, step in (_norm_extent(e) for e in ext):
        dims.append(
            str(end) if (start, step) == (0, 1) else f"{start}:{end}:{step}"
        )
    return f" trip=[{', '.join(dims)}]"


# The interstellar architecture here is 4-level (PE / L1 / L2 / DRAM), but the
# DRAM blocking is already explicit as the emitted loop nest, so only L1 and L2
# are serialized.
_TILING_NUM_LEVELS = 3


def _build_tiling(node) -> Tiling:
    """The interstellar mapping of a tiled matrix op (stamped on the node by its
    builder), as a ``Tiling``.  ``LoopIndex`` is numbered like the interstellar
    loop ids, so they map across directly."""
    mapping, access_list = node.meta["interstellar_tiling"]
    tiling = Tiling(name=node.name)

    for level in range(1, _TILING_NUM_LEVELS):  # skip level 0 (the PE)
        lt = LevelTiling()
        loop_index = 0
        while loop_index < interstellar.le.NUM:
            matched = False
            for loop in range(interstellar.le.NUM):
                if mapping.loop_orders[loop][level] == loop_index:
                    lt.loop_bounds.append(
                        LoopBound(
                            loop=loop,
                            bound=mapping.loop_blockings[loop][level],
                        )
                    )
                    loop_index += 1
                    matched = True
                    break
            if not matched:
                break
        tiling.level_tilings.append(lt)
        tiling.level_access_counts.append(
            LevelAccessCount(
                input_access_count=int(access_list[level][0]),
                output_access_count=int(access_list[level][1]),
                weight_access_count=int(access_list[level][2]),
            )
        )
    return tiling


# --- small readers over the shape-propagated graph ---------------------------


def _value(node):
    """The value ShapeProp stamped on a node: a tensor, an index vector (a list
    of ints), or a scalar."""
    return getattr(node, "value", None)


def _is_tensor(node) -> bool:
    return isinstance(node, Node) and isinstance(_value(node), torch.Tensor)


def _is_index_vector(node) -> bool:
    """A node producing a *tile index* — the list of per-dimension coordinates
    ``delinearize_index`` returns, which the component ``getitem``s read."""
    val = _value(node)
    return isinstance(val, (tuple, list)) and all(
        isinstance(v, (int, bool)) for v in val
    )


def _dtype_str(node) -> str:
    """The logical (quantized) dtype the bufferizer derived (``nf4_6``,
    ``fp8_e5m3``) when there is one, else the physical torch dtype."""
    dtype = node.meta.get("dtype")
    if isinstance(dtype, str):
        return dtype
    return str(_value(node).dtype).split(".")[1]


def _scalar_type(value):
    if isinstance(value, bool):
        return SCALAR_BOOL
    if isinstance(value, float):
        return SCALAR_F32
    return SCALAR_INDEX


def _placeholders(gm: GraphModule) -> List[Node]:
    return [n for n in gm.graph.nodes if n.op == "placeholder"]


def _outputs_of(gm: GraphModule) -> List:
    outs = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    return list(outs) if isinstance(outs, (tuple, list)) else [outs]


def _owns_storage(node) -> bool:
    """``node`` *is* a buffer: a model input, a weight, an explicit
    ``voyager.alloc`` / ``zeros``, or a tensor the host materializes outside the
    accelerator (a ``pad`` to the hardware unrolling, a slice, a cast) — which
    the planner gives an address like any other DRAM buffer.  Everything else
    names storage one of these owns."""
    if node.op in ("placeholder", "get_attr"):
        return True
    if node.op != "call_function":
        return False
    return node.target in _ALLOCATORS or "memory" in node.meta


def _component_names(node) -> List[str]:
    """SSA name of each component of an index vector.  A component is named
    after the ``getitem`` that reads it (so the ``getitem`` itself needs no
    instruction — it *is* the name), or positionally if nothing reads it."""
    names = [f"{node.name}_{i}" for i in range(len(_value(node)))]
    for u in node.users:
        if u.target is operator.getitem and isinstance(u.args[1], int):
            names[u.args[1]] = u.name
    return names


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------


class _Emitter:
    """Walks a bufferized FX graph twice.

    **Bind** resolves identity: which storage every tensor node names (threading
    each region's placeholders to the operands bound to them), and which op
    every ``voyager.insert`` is really the store of.

    **Emit** then serializes, reading those two maps.  Splitting them is what
    lets a destination reach *backwards* into a ``cond`` branch that was already
    walked, and a bank ``select`` inside a loop body resolve to an ``alloc``
    declared outside it.
    """

    def __init__(self, model: GraphModule, dump_dir: Optional[str]):
        self.model = model
        self.dump_dir = dump_dir
        # Per-region lexical scope: a placeholder -> the ref/scalar it is bound
        # to in the enclosing region.
        self.envs: Dict[GraphModule, Dict[Node, object]] = {}
        # Declared storage.
        self.boxes: Dict[str, Node] = {}
        # Destination-passing: producer -> {output index: destination}.
        self.dest: Dict[Node, Dict[Optional[int], TensorBoxRef]] = {}

    # --- references ------------------------------------------------------

    def _ref(self, node, env, internal=frozenset()) -> TensorBoxRef:
        """``node`` as an operand: the allocation it names, described in full.

        The reference repeats the allocation — name, dtype, memory level and
        address, banking — so an instruction is executable without resolving
        anything else.  Its shape is ``node``'s own whenever an aliasing op
        (a reshape, a rank-reducing ``subview``) stands between the two.
        """
        ref = self._resolve(node, env, internal)
        value = _value(node)
        # ``value.shape`` still leads with the bank dim that ``_tensor_box``
        # factors into ``bank_count``, so an operand naming the allocation
        # itself keeps the declared shape rather than restoring that dim.  Every
        # operand of a banked buffer reaches it through ``select_bank``, whose
        # ``squeeze_dim`` has already dropped the dim.
        if not _owns_storage(node) and isinstance(value, torch.Tensor):
            del ref.box.shape[:]
            ref.box.shape.extend(int(d) for d in value.shape)
            ref.box.dtype = _dtype_str(node)
        return ref

    def _resolve(self, node, env, internal) -> TensorBoxRef:
        """The storage ``node`` names.  Views (bank slots, reshapes, casts) and
        region boundaries are transparent: they resolve to the box that owns the
        bytes, plus the window a ``voyager.subview`` reads of them.

        ``internal`` is the set of values a fused group computes.  One of those
        names the ``PrimOp`` that produced it — it has no storage at all
        (it never leaves the datapath); the reference only says *which* op of
        the fusion the operand comes from.
        """
        if node in internal:
            return TensorBoxRef(box=TensorBox(node=node.name))

        if node in env:
            ref = TensorBoxRef()
            ref.CopyFrom(env[node])
            return ref

        if _owns_storage(node):
            return TensorBoxRef(box=self._tensor_box(node))

        if node.op == "call_function":
            if node.target is _SUBVIEW:
                return self._window(node, env, internal)
            if is_nop(node):
                return self._ref(node.args[0], env, internal)
            if node.target is operator.getitem:
                ref = self.dest.get(node.args[0], {}).get(node.args[1])
                if ref is not None:
                    out = TensorBoxRef()
                    out.CopyFrom(ref)
                    return out

        if (dest := self.dest.get(node)) is not None and None in dest:
            ref = TensorBoxRef()
            ref.CopyFrom(dest[None])
            return ref

        raise ValueError(
            f"'{node.name}' ({node.target}) is used as a tensor operand but "
            f"names no storage: it is neither a buffer nor a value written to "
            f"one by voyager.insert"
        )

    def _window(self, node, env, internal) -> TensorBoxRef:
        """A ``voyager.subview`` is not an instruction: it *is* the reference
        the operand makes to the buffer it windows.  Its arguments pass straight
        through — an offset may be a runtime scalar (the bank a step writes),
        while sizes and strides are static.

        The referenced dims of a banked buffer are ``[bank_count, *shape]``, so
        dim 0 offsets the bank and the backend pitches it by
        ``bank_stride_bytes``.  A window over the *whole* referenced buffer is
        the buffer, and is left off.  ``squeeze_dim`` is not addressing — it is
        the rank the *tensor* takes, and the bytes it names are the same.
        """
        source, offsets, sizes, strides = node.args[:4]
        ref = self._ref(source, env, internal)
        if ref.offsets:
            raise ValueError(
                f"'{node.name}' windows '{source.name}', which is already a "
                f"window of '{ref.box.node}': a TensorBoxRef carries one "
                f"window, so the two would have to be composed"
            )
        shape = list(_value(source).shape)
        if (
            all(o == 0 for o in offsets)
            and list(sizes) == shape
            and all(s == 1 for s in strides)
        ):
            return ref  # the whole buffer

        ref.offsets.extend(self._scalar(o, env) for o in offsets)
        ref.sizes.extend(int(s) for s in sizes)
        ref.strides.extend(int(s) for s in strides)
        return ref

    def _scalar(self, value, env) -> ScalarValue:
        """``value`` as a scalar operand — a literal, or the SSA name of one
        computed elsewhere (a loop index, a predicate, an address term).  A
        reference states its type, so it need not be resolved against its
        defining op to know what it is; a literal's type is the arm it sets."""
        if isinstance(value, Node):
            out = ScalarValue()
            if value in env:
                out.CopyFrom(env[value])
            else:
                out.node = value.name
            known = _value(value)
            if known is not None and not isinstance(
                known, (torch.Tensor, tuple, list)
            ):
                out.type = _scalar_type(known)
            return out
        if isinstance(value, bool):
            return ScalarValue(bool_value=value)
        if isinstance(value, int):
            return ScalarValue(int_value=int(value))
        if isinstance(value, float):
            return ScalarValue(float_value=float(value))
        raise TypeError(f"not a scalar operand: {value!r}")

    def _bind_value(self, value, env):
        """The ref a region placeholder takes from the operand bound to it."""
        if _is_tensor(value):
            return self._ref(value, env)
        return self._scalar(value, env)

    def _argument(self, value, env, internal=frozenset()) -> Argument:
        arg = Argument()
        if isinstance(value, Node):
            if _is_tensor(value):
                arg.tensor_box.CopyFrom(self._ref(value, env, internal))
            elif _is_index_vector(value):
                # The whole index vector as one operand: its components by name.
                arg.scalar_list.values.extend(
                    ScalarValue(node=n, type=_scalar_type(v))
                    for n, v in zip(_component_names(value), _value(value))
                )
            else:
                arg.scalar.CopyFrom(self._scalar(value, env))
        elif isinstance(value, (list, tuple)):
            if any(_is_tensor(v) for v in value):
                arg.tensor_box_list.values.extend(
                    self._ref(v, env, internal) for v in value
                )
            else:
                arg.scalar_list.values.extend(
                    self._scalar(v, env) for v in value
                )
        elif isinstance(value, str):
            arg.str_value = value
        elif isinstance(
            value,
            (torch.dtype, torch.layout, torch.device, torch.memory_format),
        ):
            arg.str_value = str(value).split(".")[-1]
        else:
            arg.scalar.CopyFrom(self._scalar(value, env))
        return arg

    # --- bind ------------------------------------------------------------

    def bind(self, gm: GraphModule, env: Dict[Node, object]) -> None:
        self.envs[gm] = env
        named = dict(gm.named_modules(remove_duplicate=False))

        for node in gm.graph.nodes:
            if gm is self.model and _is_tensor(node) and _owns_storage(node):
                self.boxes[node.name] = node

            if node.op == "call_module":
                sub = named[str(node.target)]
                self.bind(sub, self._child_env(sub, node.args, env))
            elif node.op == "call_function":
                if node.target is WHILE_LOOP:
                    self._bind_loop(node, named, env)
                elif node.target is COND:
                    operands = list(node.args[3]) if len(node.args) > 3 else []
                    for graph_arg in (node.args[1], node.args[2]):
                        branch = named[str(graph_arg.target)]
                        self.bind(
                            branch, self._child_env(branch, operands, env)
                        )
                elif node.target is COMMIT:
                    sub = named[str(node.args[0].target)]
                    self.bind(sub, self._child_env(sub, node.args[1:], env))
                elif node.target is _INSERT:
                    self._bind_insert(node, gm, env)

    def _child_env(self, sub: GraphModule, operands, env) -> Dict[Node, object]:
        """Bind a region's placeholders to the operands passed into it, so a
        reference inside the region resolves to the caller's storage."""
        return {
            ph: self._bind_value(operand, env)
            for ph, operand in zip(_placeholders(sub), operands)
            if isinstance(operand, Node) or isinstance(operand, (int, float))
        }

    def _bind_loop(self, node, named, env) -> None:
        """A loop body's placeholders split in two: the *carried* scalars are
        the loop's own induction variable and iteration arguments (they keep
        their body-local SSA names), while the *additional* inputs are buffers
        the body borrows from the enclosing region."""
        body = named[str(node.args[1].target)]
        carried = list(node.args[2])
        extra = list(node.args[3]) if len(node.args) > 3 else []
        phs = _placeholders(body)

        child: Dict[Node, object] = {
            ph: ScalarValue(node=ph.name, type=_scalar_type(_value(ph)))
            for ph in phs[: len(carried)]
        }
        child.update(self._child_env_from(phs[len(carried) :], extra, env))
        self.bind(body, child)
        # The trip test (``cond_fn``) is loop structure, not a region: the
        # emitted ForLoop's start/end/step already says it.

    def _child_env_from(self, phs, operands, env) -> Dict[Node, object]:
        return {
            ph: self._bind_value(operand, env)
            for ph, operand in zip(phs, operands)
            if isinstance(operand, (Node, int, float))
        }

    def _bind_insert(self, node, gm, env) -> None:
        """Resolve one destination-passing store: which op(s) actually produce
        the value, and where it lands."""
        src, dst = node.args[0], node.args[1]
        destination = self._ref(dst, env)

        producers = self._producers(src, gm, env)
        if producers is None:
            # An ``insert`` writes a *computed* value to its destination; with
            # no op producing it there is nothing to carry the destination.  A
            # builder moving one buffer into another must say so — the copy is
            # real data movement, and an ``aten.clone`` gives it a producer.
            raise ValueError(
                f"'{node.name}' stores '{getattr(src, 'name', src)}' into "
                f"'{getattr(dst, 'name', dst)}', but no op produces it: a "
                f"buffer-to-buffer move must be written as "
                f"voyager.insert(src.clone(), dst)"
            )

        for producer, index in producers:
            self.dest.setdefault(producer, {})[index] = destination

    def _producers(self, node, gm, env, index=None):
        """The op(s) whose result this store writes, as ``[(node, index)]`` — or
        ``None`` when the source is a buffer (so the store is a move, not a
        compute destination).

        A ``cond`` result has one producer *per branch*: both branches write the
        same destination, so both carry it.
        """
        if not isinstance(node, Node):
            return None
        if node.op in ("placeholder", "get_attr"):
            return None
        if node.op == "call_module":
            return [(node, index)]
        if node.op != "call_function":
            return None
        if node.target in _ALLOCATORS or node.target is _SUBVIEW:
            return None  # a buffer handle
        if is_nop(node):
            return self._producers(node.args[0], gm, env, index)
        if node.target is operator.getitem:
            source, slot = node.args[0], node.args[1]
            if isinstance(source, Node) and source.target is COND:
                named = dict(gm.named_modules(remove_duplicate=False))
                producers = []
                for graph_arg in (source.args[1], source.args[2]):
                    branch = named[str(graph_arg.target)]
                    result = _outputs_of(branch)[slot]
                    found = self._producers(
                        result, branch, self.envs[branch], index
                    )
                    if found is None:
                        return None
                    producers += found
                return producers
            return self._producers(source, gm, env, slot)
        return [(node, index)]

    # --- emit ------------------------------------------------------------

    def _call(self, node, env, internal=frozenset()) -> PrimOp:
        """One op, serialized kwargs-only: every operand under its canonical
        schema keyword, so position never carries meaning.

        ``internal`` is the set of values produced *inside* the same fused
        group.  An operand that is one names the ``PrimOp`` that computed
        it, not a ``TensorBox``: it never leaves the datapath, so it has no
        storage and no address — the reference exists only so the backend can
        see which operand comes from the previous op of the fusion.

        ``op`` is where it runs: a tensor written into storage the op does not
        own is a bufferized instruction, so it runs on the datapath.
        """
        call = PrimOp(
            name=node.name,
            op=(
                "call_function"
                if _produces_tensor(node) and not _owns_storage(node)
                else "cpu"
            ),
            target=_target_name(node.target),
        )
        qmaps = quant_param_arg_nodes(node, QMAP_PARAMS)
        for key, value in self._kwargs(node).items():
            if value is None:
                continue  # an unset optional: absent, not null
            if isinstance(value, Node) and value in qmaps:
                continue  # a qmap lookup table is not emittedan operand
            call.kwargs[key].CopyFrom(self._argument(value, env, internal))
        return call

    def _kwargs(self, node) -> Dict[str, object]:
        normalized = normalize_function(
            node.target,
            node.args,
            node.kwargs,
            normalize_to_only_use_kwargs=True,
        )
        if normalized is not None:
            return dict(normalized.kwargs)
        # A Python builtin (loop-control ``eq`` / ``and_`` / ``add`` ...) has no
        # schema to normalize against; name its operands like the ATen binaries.
        kwargs = dict(node.kwargs)
        for i, arg in enumerate(node.args):
            name = (
                _BUILTIN_ARG_NAMES[i]
                if i < len(_BUILTIN_ARG_NAMES)
                else f"arg{i}"
            )
            kwargs[name] = arg
        return kwargs

    def _set_outputs(self, op: Operation, node, env) -> None:
        """An op's results.  An op that *creates* storage declares it; every
        other tensor result is a destination the op writes (never a value it
        yields); a scalar result defines an SSA name; a DMA / wait has no result
        at all."""
        if node.op == "call_function" and _owns_storage(node):
            out = op.outputs.add()
            out.name = node.name
            out.tensor_box.CopyFrom(self._tensor_box(node))
            return

        if _is_index_vector(node):
            for i, name in enumerate(_component_names(node)):
                out = op.outputs.add()
                out.name = name
                out.scalar = _scalar_type(_value(node)[i])
            return

        value = _value(node)
        if value is None:
            return  # async_copy / async_wait: side effect only

        if not isinstance(value, torch.Tensor) and not isinstance(
            value, (tuple, list)
        ):
            out = op.outputs.add()
            out.name = node.name
            out.scalar = _scalar_type(value)
            return

        destinations = self.dest.get(node)
        if not destinations:
            raise ValueError(
                f"destination-passing violation: '{node.name}' "
                f"({node.target}) produces a tensor but no voyager.insert "
                f"stores it, so it has nowhere to write"
            )
        for index, destination in sorted(
            destinations.items(), key=lambda kv: (kv[0] is not None, kv[0])
        ):
            out = op.outputs.add()
            out.name = node.name if index is None else f"{node.name}_{index}"
            out.destination.CopyFrom(destination)

    def _tensor_box(self, node) -> TensorBox:
        """Declare ``node``'s storage.  A banked buffer records its depth and
        pitch instead of a leading bank *dimension*, so ``shape`` stays the
        payload of one bank and slot ``i`` lives at
        ``address + i * bank_stride_bytes``."""
        box = TensorBox(node=node.name, dtype=_dtype_str(node))
        shape = list(_value(node).shape)

        if banks := node.meta.get("bank_count", 0):
            box.bank_count = banks
            box.bank_stride_bytes = node.meta["bank_stride"]
            shape = shape[1:]  # the bank dim is not a tensor dim
        box.shape.extend(shape)

        box.memory.level = self._memory_level(node)
        segment = node.meta.get("scratchpad") or node.meta.get("memory")
        if segment is not None:
            box.memory.address = int(segment.start)
        return box

    def _memory_level(self, node):
        """Where ``node``'s bytes live: the space bufferization annotated.

        A semaphore (``voyager.zeros`` / a credit-seeded ``voyager.fill``) is a
        counter the accelerator maps itself, so it is declared at register level
        with no address.  Anything left unannotated is in no memory at all — a
        codebook or a folded constant, programmed into the instruction that
        reads it rather than fetched from an address.
        """
        if node.op == "call_function" and node.target in (_ZEROS, _FILL):
            return MEMORY_LEVEL_REGISTER
        if node.meta.get("space") == "Scratchpad":
            return MEMORY_LEVEL_SCRATCHPAD
        if node.meta.get("space") == "DRAM":
            return MEMORY_LEVEL_DRAM
        return MEMORY_LEVEL_IMMEDIATE

    def _dump(self, node) -> None:
        """Write the tensors a hardware run needs to replay this op on its own:
        a compute op's operand tiles and its result, and a DRAM buffer (which a
        whole loop is replayed against) — an ``alloc``, or a tensor the host
        materializes and the loop then loads tiles out of (a ``pad``, a slice).
        The machinery in between — the DMA, the semaphores, the SRAM banks, the
        index vectors — the run drives itself, so it is not dumped."""
        if self.dump_dir is None:
            return
        if _is_compute(node):
            tensors = list(node.all_input_nodes) + [node]
        elif _owns_storage(node) and node.meta.get("space") == "DRAM":
            tensors = [node]  # a buffer: its own bytes
        else:
            return
        for n in tensors:
            if _is_tensor(n):
                save_tensor(
                    n.value, os.path.join(self.dump_dir, f"{n.name}.bin")
                )

    def emit_region(self, gm: GraphModule, ops) -> None:
        env = self.envs[gm]
        named = dict(gm.named_modules(remove_duplicate=False))
        for node in gm.graph.nodes:
            self._emit(node, gm, named, env, ops)

    def _emit(self, node, gm, named, env, ops) -> None:
        if node.op == "call_module":
            ops.append(self._fused(node, named, env))
            return
        if node.op != "call_function":
            return

        if node.target is WHILE_LOOP:
            ops.append(self._loop(node, named, env))
            return
        if node.target is COND:
            ops.append(self._cond(node, named, env))
            return
        if node.target is COMMIT:
            ops.append(self._async(node, named, env))
            return
        if node.target is _INSERT:
            return  # a destination-passing store is not an instruction
        if (
            is_nop(node)
            or node.target is _SUBVIEW
            or node.target is operator.getitem
        ):
            # A name for storage someone else owns — a ``subview`` window, a
            # reshape, the unpacking of a multi-output op — not an instruction.
            return

        self._dump(node)
        op = Operation(name=node.name)
        op.prim.CopyFrom(self._call(node, env))
        self._set_outputs(op, node, env)
        if "interstellar_tiling" in node.meta:
            op.tiling.CopyFrom(_build_tiling(node))
        ops.append(op)

    def _fused(self, node, named, env) -> Operation:
        """A fused group (a GEMM / conv plus its tail) is one instruction: the
        ops it chains, and the destination(s) the chain writes.  Its
        intermediate values have no storage — they never leave the datapath."""
        sub = named[str(node.target)]
        sub_env = self.envs[sub]
        self._dump(node)

        # The ops the fusion chains.  Their results are its transients: an
        # operand naming one names that ``PrimOp``, not storage (a view
        # over one is transparent, and resolves to the same ``PrimOp``).
        chain = [
            n
            for n in sub.graph.nodes
            if not is_nop(n) and n.op == "call_function"
        ]
        internal = frozenset(chain)

        op = Operation(name=node.name)
        for inner in chain:
            op.fused.op_list.append(self._call(inner, sub_env, internal))
        self._set_outputs(op, node, env)
        if "interstellar_tiling" in node.meta:
            op.tiling.CopyFrom(_build_tiling(node))
        return op

    def _loop(self, node, named, env) -> Operation:
        """A ``while_loop`` over a flattened tile grid is a counted ``for``: the
        builders emulate one with a step counter and a trip test.  The step is
        the induction variable; the remaining carried values are scalar
        iteration arguments (the software pipeline's producer / consumer
        cursors).  Buffers are *not* carried — they are referenced by name — so
        nothing but scalars is yielded."""
        extents = _loop_extents(node)
        if len(extents) != 1:
            raise ValueError(
                f"'{node.name}' has {len(extents)} grid extents; the builders "
                f"flatten the grid into a single counted loop"
            )
        start, end, step = extents[0]

        body = named[str(node.args[1].target)]
        carried = list(node.args[2])
        phs = _placeholders(body)
        results = _outputs_of(body)

        op = Operation(name=node.name)
        loop = op.loop.for_loop
        loop.iv = phs[0].name
        loop.start.int_value = start
        loop.end.int_value = end
        loop.step.int_value = step

        for i in range(1, len(carried)):
            iter_arg = loop.iter_args.add()
            iter_arg.name = phs[i].name
            iter_arg.type = _scalar_type(_value(phs[i]))
            iter_arg.initial.CopyFrom(self._scalar(carried[i], env))

        self.emit_region(body, loop.body.ops)
        body_env = self.envs[body]
        for i in range(1, len(carried)):
            loop.body.yields.append(self._scalar(results[i], body_env))

        # The final value of each iteration argument, named by the handle the
        # enclosing region reads it through.
        for i in range(1, len(carried)):
            out = op.outputs.add()
            out.name = self._result_name(node, i)
            out.scalar = _scalar_type(_value(phs[i]))
        return op

    def _result_name(self, node, index: int) -> str:
        for user in node.users:
            if user.target is operator.getitem and user.args[1] == index:
                return user.name
        return f"{node.name}_{index}"

    def _cond(self, node, named, env) -> Operation:
        """A ``torch.cond`` is a two-way region.  Its *tensor* result is not a
        value it yields: both branches write the same destination (the store
        that consumed the cond was pushed into each branch), so a branch that
        only writes buffers yields nothing."""
        op = Operation(name=node.name)
        op.cond.predicate.CopyFrom(self._scalar(node.args[0], env))

        branches = (
            (named[str(node.args[1].target)], op.cond.true_region),
            (named[str(node.args[2].target)], op.cond.false_region),
        )
        yielded = self._yielded_scalars(node, branches[0][0])
        for branch, region in branches:
            self.emit_region(branch, region.ops)
            results = _outputs_of(branch)
            for i in yielded:
                region.yields.append(
                    self._scalar(results[i], self.envs[branch])
                )

        for i in yielded:
            out = op.outputs.add()
            out.name = self._result_name(node, i)
            out.scalar = _scalar_type(_value(_outputs_of(branches[0][0])[i]))
        return op

    def _async(self, node, named, env) -> Operation:
        """A ``commit`` region: dispatched asynchronously — it waits every
        semaphore in ``dependencies``, runs its body, then signals ``post`` on
        retirement.  Emitted as an ``AsyncOp`` wrapping the body region (whose
        ops keep destination-passing, exactly like a top-level op) plus the
        dependency / post semaphore references."""
        sub = named[str(node.args[0].target)]
        op = Operation(name=node.name)
        async_op = getattr(op, "async")  # ``async`` is a Python keyword
        self.emit_region(sub, async_op.body.ops)
        for dep in node.kwargs.get("dependencies") or ():
            async_op.dependencies.add().CopyFrom(self._ref(dep, env))
        if (post := node.kwargs.get("post")) is not None:
            async_op.post.CopyFrom(self._ref(post, env))
        return op

    def _yielded_scalars(self, node, branch) -> List[int]:
        """The branch result slots the enclosing region actually reads *as
        scalars*.  A tensor slot is a destination (already threaded into the
        branches), and the ``1`` / ``0`` a guard branch returns is dead."""
        yielded = []
        for user in node.users:
            if user.target is not operator.getitem:
                continue
            slot = user.args[1]
            if _is_tensor(user) or not user.users:
                continue
            yielded.append(slot)
        return sorted(set(yielded))

    # --- entry point -----------------------------------------------------

    def build(self) -> Model:
        self.bind(self.model, {})

        model = Model()
        quant_params = _collect_codebook_nodes(self.model)
        for node in self.model.graph.nodes:
            if node.op == "placeholder" and _is_tensor(node):
                model.inputs.append(self._tensor_box(node))
            elif node.op == "get_attr" and _is_tensor(node):
                if not _passed_whole(node, quant_params):
                    model.parameters.append(self._tensor_box(node))
            else:
                continue
            is_qmap = node in quant_params and "qmap" in node.name
            if self.dump_dir is not None and not is_qmap:
                save_tensor(
                    _value(node),
                    os.path.join(self.dump_dir, f"{node.name}.bin"),
                )

        self.emit_region(self.model, model.ops)

        env = self.envs[self.model]
        for result in _outputs_of(self.model):
            if _is_tensor(result):
                owner = self.boxes.get(self._ref(result, env).box.node)
                if owner is not None:
                    model.outputs.append(self._tensor_box(owner))
        return model


def print_layer_table(
    model: GraphModule, params: Model, to_string: bool = False
):
    """Print (or return) the bufferized ``layers.txt``: one
    ``name start end group trip`` row per layer, in execution order.

    A layer is one bufferized nest.  Its nodes all carry the same
    ``meta['scope']`` — the op bufferization replaced — and a splice lands them
    as one contiguous run, so the scope changing *is* the layer boundary, and
    naming the run's first and last op delimits the whole layer: the buffers it
    allocates, the DMA and semaphores around them, and the tile loop holding
    the compute.  Everything a layer needs is therefore inside its own range,
    and a nest lowered to several kernels is still one layer.  Ops between two
    ranges belong to none — they are host-side setup.

    ``group`` is an equivalence class: two layers share one exactly when
    bufferization built them from the same cached nest
    (``meta['build_group']``), so a consumer can exercise one representative
    per class.  Classes are numbered from zero in the order they first appear
    — deterministic within a file, meaningless across files.  ``trip`` is how
    many tiles the layer's loop sweeps (the longest, if it has more than one),
    1 for an untiled layer.

    Args:
        model: The bufferized graph, named and memory-planned.
        params: The emitted Model, whose top-level ``ops`` are the instructions
            a row may name — an FX node that produced none (a store folded into
            its producer, a view) is not one, and cannot bound a range.
        to_string: Return the table instead of printing it.

    Returns:
        The table text.

    Raises:
        ValueError: If a layer's ops are not contiguous, which would leave its
            range covering another layer's ops.
    """
    emitted = {op.name for op in params.ops}
    spans: Dict[str, List[Node]] = {}
    current = None

    for n in model.graph.nodes:
        if n.name not in emitted:
            continue
        scope = n.meta.get("scope")
        if scope is None and not _is_compute(n):
            continue  # setup that belongs to no layer
        # A compute op bufferization never built has no scope, and is a layer
        # of its own under its own name.
        name = scope[0] if scope else n.name
        if name != current:
            if name in spans:
                raise ValueError(
                    f"the ops of layer '{name}' are split by '{n.name}': a "
                    f"layer is spliced as one contiguous run, so its start / "
                    f"end cannot name a range"
                )
            spans[name] = []
            current = name
        spans[name].append(n)

    rows, first = [], {}
    for name, nodes in spans.items():
        group = nodes[0].meta.get("build_group")
        trips = [
            _loop_trip(n)
            for n in nodes
            if n.op == "call_function" and n.target is WHILE_LOOP
        ]
        rows.append(
            [
                name,
                nodes[0].name,
                nodes[-1].name,
                first.setdefault(name if group is None else group, len(first)),
                max(trips, default=1),
            ]
        )

    # ``plain`` aligns the columns without rules or separators, so a row is
    # still five whitespace-separated fields that ``line.split()`` reads.
    text = tabulate(rows, tablefmt="plain")
    if to_string:
        return text
    print(text)
    return text


def gen_code_bufferized(model: GraphModule, args, output_dir=None) -> Model:
    """Generate a ``voyager`` protobuf Model from a bufferized FX graph."""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # One recursive pass stamps every node — top level and inside every
    # while_loop / cond body + fused call_module — with a ``.value``, so the
    # emitter below just reads it (no per-body ShapeProp).  oracle_disabled:
    # bodies are walked a single iteration, where a wait may precede its copy.
    with oracle_disabled():
        ShapeProp(model, recurse=True).propagate(*args)

    # One iteration is all a tile needs, but it leaves the DRAM buffers holding
    # a single tile over ``alloc``'s random fill.  A non-recursive pass runs the
    # loop for real and re-stamps only the top level, so the buffers end up with
    # the true tensors and the tiles keep theirs.
    if output_dir is not None:
        ShapeProp(model).propagate(*args)

    return _Emitter(model, output_dir).build()


# ===========================================================================
# 2. Graphviz rendering
# ===========================================================================

# A line break inside a record label.
_LABEL_BREAK = "&#92;n"


def _wrap_long_name(name: str, max_len: int = 20) -> str:
    """Break a long node name over several label lines, at the last
    non-alphabetic character before each limit, so a box stays narrow."""
    if len(name) <= max_len:
        return name
    lines = []
    start = 0
    while start + max_len < len(name):
        end = start + max_len
        break_at = end
        for i in range(end, start, -1):
            if not name[i].isalpha():
                break_at = i + 1
                break
        lines.append(name[start:break_at])
        start = break_at
    lines.append(name[start:])
    return _LABEL_BREAK.join(lines)


def _node_label(node, named) -> Optional[str]:
    """``node``'s record label: its name (wrapped), the shape(s) it produces,
    and the dtype when that is worth saying — the logical (quantized) one the
    quantizer stamped, else a physical one that is not the default float.  A
    fused submodule adds the ops it chains as a second field.

    ``None`` for a node that produces no tensor, which gets no box at all.
    """
    val = getattr(node, "value", None)
    dtype = node.meta.get("dtype")
    parts = [_wrap_long_name(node.name)]

    if isinstance(val, torch.Tensor):
        parts.append(str(tuple(val.shape)))
        if dtype is not None:
            parts.append(str(dtype))
        elif val.dtype not in (torch.float, torch.bfloat16):
            parts.append(str(val.dtype))
    elif isinstance(val, (tuple, list)):
        parts.append(", ".join(str(tuple(t.shape)) for t in val))
        dtypes = [t.dtype for t in val]
        if isinstance(dtype, (list, tuple)):
            dtypes = [d or physical for d, physical in zip(dtype, dtypes)]
        if any(d not in (torch.float, torch.bfloat16) for d in dtypes):
            parts.append(", ".join(str(d) for d in dtypes))
    else:
        return None

    label = _LABEL_BREAK.join(parts)
    sub = named.get(str(node.target)) if node.op == "call_module" else None
    if isinstance(sub, GraphModule):
        label += "|" + _LABEL_BREAK.join(
            n.name for n in sub.graph.nodes if n.op == "call_function"
        )
    # ``<`` and ``>`` are record-label port syntax, so a dtype or shape holding
    # one has to be escaped.
    return ("{" + label + "}").replace("<", r"\<").replace(">", r"\>")


def _render(model: GraphModule, output_file: str) -> None:
    """Draw one box per node and one edge per use, then write the SVG."""
    g = graphviz.Digraph()
    named = dict(model.named_modules(remove_duplicate=False))
    edges = []

    for node in model.graph.nodes:
        if node.op == "get_attr" and not require_allocation(node):
            continue
        label = _node_label(node, named)
        if label is None:
            continue
        g.node(node.name, label=label, shape="Mrecord")
        edges += [(node.name, u.name) for u in node.users]

    g.edges(edges)
    g.render(output_file, format="svg", cleanup=True)


def gen_compute_graph(
    model: GraphModule,
    output_file: str = "compute_graph",
    timeout: Optional[float] = 300,
) -> None:
    """Render an FX graph to ``<output_file>.svg``.

    Args:
        model: The graph to draw.  Shapes must already be propagated — a node
            with no value produces no box.
        output_file: Path without extension; ``.svg`` is appended.
        timeout: Seconds to allow the build + render, or ``None`` for no limit.
            Graphviz can take minutes on a large graph, and the picture is a
            debug aid, so exceeding it prints a warning and skips the render
            rather than raising.  Implemented with ``SIGALRM``, so it only
            works on the main thread of a POSIX process.
    """
    if timeout is None:
        _render(model, output_file)
        return

    class _RenderTimeout(Exception):
        pass

    def _on_timeout(signum, frame):
        raise _RenderTimeout

    old_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        _render(model, output_file)
    except _RenderTimeout:
        print(
            f"WARNING: gen_compute_graph exceeded {timeout:g}s; "
            "skipping compute graph rendering."
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# ===========================================================================
# 3. Indented text printer
# ===========================================================================

# Short names for the builtin torch dtypes used in the ``<shape x dtype>``
# annotation (custom quantized dtypes carry their own string, e.g. ``nf4_6``,
# used verbatim).
_DTYPE_SHORT = {
    torch.float32: "f32",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "bool",
}


def _type_str(node) -> str:
    """MLIR-like ``<DxDx...xdtype>`` (plus ``, space`` when known) for a node
    that produces a single tensor; ``""`` otherwise (multi-output / scalar /
    unknown).

    The dtype is the custom (quantized) ``meta['dtype']`` string (e.g.
    ``nf4_6``) used verbatim when set, else a short name for the builtin torch
    dtype (``f32`` / ``bf16`` / ...).  ``space`` is ``meta['space']`` (``DRAM``
    / ``Scratchpad``) from the bufferize pass.  Shapes come from the node's
    ShapeProp ``value`` or, inside loop bodies, the exported ``meta['val']``.
    """
    val = getattr(node, "value", None)
    if not isinstance(val, torch.Tensor):
        val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return ""
    custom = node.meta.get("dtype")
    dtype = (
        custom
        if isinstance(custom, str)
        else _DTYPE_SHORT.get(val.dtype, str(val.dtype).replace("torch.", ""))
    )
    dims = "x".join(str(int(d)) for d in val.shape)
    ty = f"{dims}x{dtype}" if dims else dtype  # 0-D scalar -> just the dtype
    space = node.meta.get("space")
    return f"<{ty}, {space}>" if space else f"<{ty}>"


def _fmt_arg(a):
    if isinstance(a, torch.fx.Node):
        # Annotate every tensor argument with its ``<shape×dtype, space>`` so a
        # DMA's direction is legible inline (e.g. ``async_copy(src<…,DRAM>,
        # dst<…,Scratchpad>)`` = a load); scalar / index args (``_type_str``
        # empty) print as the bare name.
        return f"{a.name}{_type_str(a)}"
    if isinstance(a, (list, tuple)):
        return "[" + ", ".join(_fmt_arg(x) for x in a) + "]"
    return repr(a)


def _print_loop(
    node, gm, named, lines: List[str], indent: int, pad: str
) -> None:
    """
    Print a while_loop in scf.for-like form, binding each loop-body argument to
    the value it receives so the dataflow across the loop boundary is explicit:

        while_loop = loop trip=N carried(arg0_1 = 0, arg1_1 = empty) \
                     extra(arg2_1 = x) {
          <body>
        }
    """
    body_t = str(node.args[1].target)
    body_mod = named.get(body_t) or getattr(gm, body_t, None)
    carried = list(node.args[2]) if len(node.args) > 2 else []
    extra = list(node.args[3]) if len(node.args) > 3 else []
    phs = (
        [n for n in body_mod.graph.nodes if n.op == "placeholder"]
        if isinstance(body_mod, GraphModule)
        else []
    )

    def _bindings(ph_list, src_list):
        # Annotate only the bound placeholder (LHS); the source (RHS) is a node
        # already defined and annotated above, so printing it bare (just its
        # name) avoids the noise.
        return ", ".join(
            f"{ph.name}{_type_str(ph)} = {src}"
            for ph, src in zip(ph_list, src_list)
        )

    header = (
        f"{pad}{node.name} = loop{_trip_str(node)} "
        f"carried({_bindings(phs[:len(carried)], carried)})"
    )
    if extra:
        header += f" extra({_bindings(phs[len(carried):], extra)})"
    lines.append(header + " {")
    if isinstance(body_mod, GraphModule):
        _print_graph(body_mod, lines, indent + 1, skip_placeholders=True)
    lines.append(f"{pad}}}")


def _print_cond(
    node, gm, named, lines: List[str], indent: int, pad: str
) -> None:
    """Print a ``torch.cond`` in MLIR ``scf.if`` form — both branch regions
    descended into, operands bound to each branch's placeholders, the branch
    output rendered as ``yield``:

        cond = if pred (b_arg0 = src0, ...) {
          <true region>
          yield <true results>
        } else (b_arg0 = src0, ...) {
          <false region>
          yield <false results>
        }

    Only the result slots the enclosing graph reads back are yielded.  A branch
    that just writes buffers still has to return *something* to ``torch.cond``,
    and that placeholder value (the ``0`` a guard branch returns) is dead — so
    such a branch prints no ``yield`` at all.  A guard's false branch is then
    left with nothing to print, and its ``else`` region is dropped entirely.
    """
    pred = _fmt_arg(node.args[0])
    operands = list(node.args[3]) if len(node.args) > 3 else []

    def _branch(graph_arg):
        mod = named.get(str(graph_arg.target)) or getattr(
            gm, str(graph_arg.target), None
        )
        phs = (
            [n for n in mod.graph.nodes if n.op == "placeholder"]
            if isinstance(mod, GraphModule)
            else []
        )
        binds = ", ".join(
            f"{ph.name}{_type_str(ph)} = {src}"
            for ph, src in zip(phs, operands)
        )
        return mod, binds

    true_mod, true_binds = _branch(node.args[1])
    false_mod, false_binds = _branch(node.args[2])

    getitems = [u for u in node.users if u.target is operator.getitem]
    if len(getitems) < len(node.users):
        # A user reading the result tuple whole keeps every slot live.
        live = list(range(len(_outputs_of(true_mod))))
    else:
        live = sorted({g.args[1] for g in getitems if g.users})

    def _render_branch(mod) -> List[str]:
        body: List[str] = []
        if not isinstance(mod, GraphModule):
            return body
        _print_graph(
            mod, body, indent + 1, skip_placeholders=True, terminator=None
        )
        if live:
            results = _outputs_of(mod)
            body.append(f"{pad}  yield {_fmt_arg([results[i] for i in live])}")
        return body

    lines.append(
        f"{pad}{node.name}{_type_str(node)} = if {pred} ({true_binds}) {{"
    )
    lines.extend(_render_branch(true_mod))
    if false_body := _render_branch(false_mod):
        lines.append(f"{pad}}} else ({false_binds}) {{")
        lines.extend(false_body)
    lines.append(f"{pad}}}")


def _print_commit(
    node, gm, named, lines: List[str], indent: int, pad: str
) -> None:
    """Print a ``voyager.commit`` in region form — the committed body descended
    into like a ``cond`` branch, its operands bound to the body placeholders and
    the ``dependencies`` / ``post`` semaphores shown on the header:

        commit = commit deps=[..] post=.. (b_arg0 = src0, ...) {
          <body>
          return <results>
        }

    A region that only writes buffers still has to return something (an empty
    list), which nothing downstream reads — so the ``return`` is printed only
    when the enclosing graph reads the commit's result.
    """
    subgraph, operands = node.args[0], list(node.args[1:])
    mod = named.get(str(subgraph.target)) or getattr(
        gm, str(subgraph.target), None
    )
    phs = (
        [n for n in mod.graph.nodes if n.op == "placeholder"]
        if isinstance(mod, GraphModule)
        else []
    )
    binds = ", ".join(
        f"{ph.name}{_type_str(ph)} = {src}" for ph, src in zip(phs, operands)
    )
    tags = f"deps={_fmt_arg(node.kwargs.get('dependencies', ()))}"
    post = node.kwargs.get("post", None)
    if post is not None:
        tags += f" post={_fmt_arg(post)}"
    read = any(
        user.target is not operator.getitem or user.users for user in node.users
    )
    lines.append(
        f"{pad}{node.name}{_type_str(node)} = commit {tags} ({binds}) {{"
    )
    if isinstance(mod, GraphModule):
        _print_graph(
            mod,
            lines,
            indent + 1,
            skip_placeholders=True,
            terminator="return" if read else None,
        )
    lines.append(f"{pad}}}")


def _print_graph(
    gm: GraphModule,
    lines: List[str],
    indent: int,
    skip_placeholders: bool = False,
    terminator: Optional[str] = "return",
) -> None:
    """Render ``gm``'s nodes into ``lines``.  ``terminator`` is the keyword the
    output node prints with, or ``None`` to drop it — a ``cond`` branch prints
    its own ``yield``, over the live result slots only."""
    pad = "  " * indent
    named = dict(gm.named_modules())

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # while_loop body placeholders are bound in the loop header
            # (scf.for-style iter_args), so they are skipped here.
            if not skip_placeholders:
                lines.append(f"{pad}arg {node.name}{_type_str(node)}")
            continue
        if node.op == "output":
            if terminator is not None:
                lines.append(f"{pad}{terminator} {_fmt_arg(node.args[0])}")
            continue
        if node.op == "get_attr":
            sub = getattr(gm, str(node.target), None)
            if isinstance(sub, GraphModule):
                continue  # printed at the while_loop use-site
            lines.append(
                f"{pad}{node.name}{_type_str(node)} = get_attr {node.target}"
            )
            continue

        if node.op == "call_function" and node.target is WHILE_LOOP:
            _print_loop(node, gm, named, lines, indent, pad)
            continue

        if node.op == "call_function" and node.target is COND:
            _print_cond(node, gm, named, lines, indent, pad)
            continue

        if node.op == "call_function" and node.target is COMMIT:
            _print_commit(node, gm, named, lines, indent, pad)
            continue

        if node.op == "call_module":
            # A fused submodule: just a braced block labelled by the node name
            # (its placeholders reuse their source-node names, so the body ops
            # read directly against the outer nodes — no ``arg`` lines, and the
            # ``= fused <target>`` tag only repeats the node name).
            sub = named.get(str(node.target))
            lines.append(f"{pad}{node.name}{_type_str(node)} = fused {{")
            if isinstance(sub, GraphModule):
                _print_graph(sub, lines, indent + 1, skip_placeholders=True)
            lines.append(f"{pad}}}")
            continue

        parts = [_fmt_arg(a) for a in node.args]
        parts += [f"{k}={_fmt_arg(v)}" for k, v in node.kwargs.items()]
        lines.append(
            f"{pad}{node.name}{_type_str(node)} = "
            f"{_target_name(node.target)}({', '.join(parts)})"
        )


def print_bufferized_graph(model: GraphModule, to_string: bool = False):
    """
    Print (or return) an indented textual rendering of a bufferized FX graph,
    descending into ``while_loop`` bodies and fused ``call_module`` submodules.
    """
    lines: List[str] = ["graph {"]
    _print_graph(model, lines, 1)
    lines.append("}")
    text = "\n".join(lines)
    if to_string:
        return text
    print(text)
    return text
