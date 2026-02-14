import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from ..pt2e_utils import dtype_byte_size

logger = logging.getLogger(__name__)


class MemorySpace(IntEnum):
    """Logical memory spaces available to the system/hardware."""
    DRAM        = 0  # main system memory (e.g., DDR)
    SCRATCHPAD  = 1  # local SW-managed fast memory


@dataclass
class Segment:
    start: Union[float, int]
    end: Union[float, int]
    memory_space: Optional[int] = None
    node: Optional[torch.fx.Node] = None

    def __post_init__(self) -> None:
        s_raw = self.start
        e_raw = self.end

        s = int(s_raw)          # truncate toward zero (matches your original)
        e = math.ceil(e_raw)    # round end up

        if s != s_raw:
            logger.warning("Segment start %r is not an integer. Rounding to %d.", s_raw, s)
        if e != e_raw:
            logger.warning("Segment end %r is not an integer. Rounding up to %d.", e_raw, e)

        if e < s:
            raise ValueError(f"Segment end ({e}) is less than start ({s}).")

        self.start = s
        self.end = e


def _find_user_with_target(node: torch.fx.Node, targets):
    if not isinstance(targets, set):
        if isinstance(targets, (list, tuple)):
            targets = set(targets)
        else:
            targets = {targets}

    for user in node.users:
        if user.target in targets and user.args[0] == node:
            return user

        # Check for users of fused dequantization nodes
        if (
            user.target == torch.ops.quantized_ops.dequantize.default
            and user.meta.get("fused") is True
        ):
            found = _find_user_with_target(user, targets)
            if found is not None:
                return found

        if user.op == 'call_module':
            gm = user.meta['submodule']
            placeholder = next(n for n in gm.graph.nodes if n.name == node.name)

            found = _find_user_with_target(placeholder, targets)
            if found is not None:
                return found

    return None


def _align_size(size, width):
    if width is None:
        return size
    return (size + width - 1) // width * width


def compute_tensor_size(
    node,
    shape=None,
    is_scratchpad_output=False,
    bank_width=None,
    unroll_dim=None,
):
    val = node.value
    if isinstance(val, torch.Tensor):
        dtype = node.meta.get('dtype') or val.dtype
        numel = math.prod(shape) if shape is not None else val.numel()
        tensor_size = numel * dtype_byte_size(dtype)

        conv_targets = (
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_ops.conv2d.default
        )
        conv_user = _find_user_with_target(node, conv_targets)

        if conv_user is not None:
            dim = 1 if conv_user.target == torch.ops.aten.conv2d.default else -1
            if val.shape[dim] == 3:
                logger.debug(f"Increase memory for conv2d input {node} by 3x")
                tensor_size *= 3

        # If this is an output during scratchpad allocation, we don't care
        # downstream layers
        if is_scratchpad_output:
            return tensor_size

        # TODO: Should only do this when allocating scratchpad memory for the
        # specific operation. E.g. if a node is consumed by both a softmax and
        # an add node, we shouldn't increase the size for the add path.
        if _find_user_with_target(node, torch.ops.aten.softmax.int):
            logger.debug(f"Increase memory for softmax input {node} by 2x")
            return tensor_size * 2

        if _find_user_with_target(node, torch.ops.aten.layer_norm.default):
            logger.debug(f"Increase memory for layer_norm input {node} by 2x")
            return (tensor_size + numel) * 2

        return tensor_size

    if isinstance(val, (tuple, list)):
        if shape is not None:
            key = "tiled_output_sizes"
            numel = [math.prod(s) for s in shape]
        else:
            key = "output_sizes"
            numel = [t.numel() for t in val]

        # Sparse outputs need to be aligned with hardware unroll dimension
        if unroll_dim is not None:
            numel = [_align_size(s, unroll_dim) for s in numel]

        dtypes = node.meta.get('dtype') or [None for _ in val]

        sizes = [
            _align_size(n * dtype_byte_size(dt or t.dtype), bank_width)
            for t, n, dt in zip(val, numel, dtypes)
        ]

        node.meta[key] = tuple(sizes)
        return _align_size(sum(sizes), bank_width)

    logger.warning(f"Node {node} has a non-tensor output")
    return None


class MemoryAllocator:
    """
    This class implements a simple first-fit memory manager for allocating
    memory partitions to tensors.

    Attributes:
        total_memory (int): The total amount of memory available for allocation.
        bank_width (int): The alignment requirement for memory allocations.
        memory_space (MemorySpace): The type of memory space to use for allocations.
    """

    def __init__(self, total_memory=None, bank_width=None, memory_space=None):
        self.total_memory = total_memory or (1 << 63) - 1
        self.bank_width = bank_width
        self.memory_space = memory_space or MemorySpace.DRAM  # default to DRAM

        self.segments = [Segment(0, self.total_memory, self.memory_space)]
        self.memory_map = {}
        self.snapshots = []

    def allocate_memory(self, node):
        if not hasattr(node, 'value'):
            logger.warning(f"Node {node} does not have a value attribute")
            return None

        # Skip allocation for quantization scaling factors
        if (
            isinstance(node.value, torch.Tensor)
            and node.value.numel() == 1
            and node.op == "get_attr"
        ):
            logger.info(f"Skipping allocation for scalar scale tensor: {node.name}")
            return None

        tensor_size = compute_tensor_size(node, bank_width=self.bank_width)

        if tensor_size is None:
            return None

        tensor_size = _align_size(tensor_size, self.bank_width)

        for index, segment in enumerate(self.segments):
            segment_size = segment.end - segment.start
            if segment.node is None and segment_size >= tensor_size:
                if segment_size > tensor_size:
                    new_segment = Segment(
                        start=segment.start + tensor_size,
                        end=segment.end,
                        memory_space=self.memory_space,
                    )
                    segment.end = segment.start + tensor_size
                    self.segments.insert(index + 1, new_segment)

                self.memory_map[node] = segment
                segment.node = node

                return Segment(segment.start, segment.end, self.memory_space)

        raise RuntimeError(f"Memory allocation failed for tensor {node.name}")

    def free_memory(self, node):
        if node in self.memory_map:
            segment = self.memory_map[node]
            segment.node = None
            self.merge_segments()
            del self.memory_map[node]

    def merge_segments(self):
        i = 0
        while i < len(self.segments) - 1:
            current_partition = self.segments[i]
            next_partition = self.segments[i + 1]
            if current_partition.node is None and next_partition.node is None:
                current_partition.end = next_partition.end
                self.segments.pop(i + 1)
            else:
                i += 1

    def print_layout(self):
        for segment in self.segments:
            status = 'free' if segment.node is None else segment.node.name
            print(f"Segment from {segment.start} to {segment.end}: {status}")

    def snapshot(self):
        partitions = [
            Segment(
                start=segment.start,
                end=segment.end,
                node=segment.node.name if segment.node else None,
            )
            for segment in self.segments[:-1]
        ]
        self.snapshots.append(partitions)

    def dump_snapshots(self, filename="dump_snapshot.png", colormap_name='tab20'):
        """
        Plots memory usage over time from a list of memory snapshots.
        Tensors (partitions with nodes) cycle through colors from the colormap.
        Free partitions are shown in gray.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        cmap = cm.get_cmap(colormap_name)
        color_cycle = [cmap(i) for i in range(cmap.N)]
        free_color = (0.85, 0.85, 0.85)

        id_to_color = {}
        color_index = 0

        for t, snapshot in enumerate(self.snapshots):
            for segment in snapshot:
                if segment.node is None:
                    color = free_color
                else:
                    if segment.node not in id_to_color:
                        id_to_color[segment.node] = color_cycle[color_index % len(color_cycle)]
                        color_index += 1
                    color = id_to_color[segment.node]

                ax.bar(
                    x=t,
                    height=segment.end - segment.start,
                    width=1.0,
                    bottom=segment.start,
                    color=color,
                    linewidth=0,
                )

        def format_bytes(x, _):
            if x >= 1 << 30:
                return f"{x / (1 << 30):.0f}GB"
            elif x >= 1 << 20:
                return f"{x / (1 << 20):.0f}MB"
            elif x >= 1 << 10:
                return f"{x / (1 << 10):.0f}KB"
            else:
                return f"{x:.0f}B"

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_bytes))

        max_bytes = max(p.end for snapshot in self.snapshots for p in snapshot)
        max_mb = max_bytes / (1 << 20)

        # Auto interval using base-10 logic
        locator = mticker.MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True)
        tick_vals_mb = locator.tick_values(0, max_mb)
        tick_vals_bytes = [int(mb * (1 << 20)) for mb in tick_vals_mb]

        ax.set_yticks(tick_vals_bytes)

        ax.set_title("Active Memory Timeline")
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
