import math
from collections import defaultdict
from typing import Dict, Set, Union

from voyager_compiler.codegen.lowering.ir import (
    Value,
    TensorBox,
    Loops,
    IRNode,
    Module,
)
from voyager_compiler.pt2e_utils import dtype_byte_size


class IntervalManager:
    """
    Manages free and occupied memory segments for a single memory space.
    Uses a 'First-Fit' strategy to find gaps for new allocations.
    """
    def __init__(self, alignment: int = 32):
        self.alignment = alignment
        # List of (start, size) tuples, sorted by start
        self.allocated_blocks = []
        self.peak_usage = 0

    def allocate(self, size: int) -> int:
        """Finds the first free gap that fits 'size' bytes."""
        # Align to the specified boundary (e.g., 32 bytes for vector loads/DMA)
        size = math.ceil(size / self.alignment) * self.alignment

        prev_end = 0
        insert_idx = 0
        found_offset = -1

        # Check gaps between allocated blocks
        for i, (start, length) in enumerate(self.allocated_blocks):
            gap = start - prev_end
            if gap >= size:
                found_offset = prev_end
                insert_idx = i
                break
            prev_end = start + length

        if found_offset == -1:
            # No gap found, append to the end
            found_offset = prev_end
            insert_idx = len(self.allocated_blocks)

        # Record allocation
        self.allocated_blocks.insert(insert_idx, (found_offset, size))

        # Track peak usage
        self.peak_usage = max(self.peak_usage, found_offset + size)
        return found_offset

    def free(self, offset: int):
        """Removes the block at 'offset' from the allocated list."""
        for i, (start, _) in enumerate(self.allocated_blocks):
            if start == offset:
                self.allocated_blocks.pop(i)
                return
        raise ValueError(f"Attempted to free unallocated offset: {offset}")


class MemoryAllocatorPass:
    """
    Analyzes liveness and assigns memory offsets using designated space allocators.
    """
    def __init__(self, alignment: int = 32):
        self.mapping: Dict[Value, int] = {}  # Result: Value -> Offset
        self.node_to_last_use: Dict[Value, IRNode] = {}

        # Independent allocators for distinct memory spaces
        self.allocators = {
            "Scratchpad": IntervalManager(alignment=alignment),
            "DRAM": IntervalManager(alignment=alignment)
        }

    @staticmethod
    def get_tensor_bytes(val: Value) -> int:
        """Calculates the byte size of a TensorBox."""
        if not isinstance(val, TensorBox) or not hasattr(val, 'shape') or not hasattr(val, 'dtype'):
            return 0
        numel = math.prod(val.shape) if val.shape else 1
        return int(numel * dtype_byte_size(val.dtype))

    def _analyze_liveness(self, module: Union[Module, Loops]) -> Set[Value]:
        """Recursively traverses nodes to map each value to the node of its final use."""
        inputs = set()
        outputs = set()

        if isinstance(module, Module):
            outputs.update(module.args)

        for node in module.body:
            if isinstance(node, Loops):
                loop_inputs = self._analyze_liveness(node)
                for inp in loop_inputs:
                    if inp in outputs:
                        self.node_to_last_use[inp] = node
                    else:
                        inputs.add(inp)
            else:
                for inp in node.inputs:
                    if inp in outputs:
                        self.node_to_last_use[inp] = node
                    else:
                        inputs.add(inp)
                outputs.update(node.outputs)

        return inputs

    def _allocate_tensor(self, tensor):
        """Helper to allocate a single tensor in its correct memory space."""
        if not isinstance(tensor, TensorBox) or tensor in self.mapping:
            return

        space = getattr(tensor, 'space', None)
        if space not in self.allocators:
            return  # Ignore spaces we aren't managing

        size = self.get_tensor_bytes(tensor)
        if size == 0 or not tensor.users:
            return

        offset = self.allocators[space].allocate(size)
        tensor.address = offset
        self.mapping[tensor] = offset
        print(f"Allocated [{space}] {tensor.name}: {size} bytes @ offset {offset}")

    def _free_tensor(self, tensor):
        """Helper to free a single tensor from its correct memory space."""
        if not isinstance(tensor, TensorBox) or tensor not in self.mapping:
            return

        space = getattr(tensor, 'space', None)
        if space not in self.allocators:
            return

        offset = self.mapping[tensor]
        self.allocators[space].free(offset)
        print(f"Freed [{space}] {tensor.name} from offset {offset}")

    def _process_scope(self, block: Union[Module, Loops], user_to_last_uses: Dict[IRNode, Set[Value]]):
        """Walks the IR, allocating node outputs and freeing inputs that have reached their last use."""
        # Allocate module inputs at the start of the module scope
        if isinstance(block, Module):
            for arg in block.args + block.params:
                self._allocate_tensor(arg)

        for node in block.body:
            if isinstance(node, Loops):
                print(f"Entering Loop {node.index}:")
                self._process_scope(node, user_to_last_uses)
                print(f"Exiting Loop {node.index}")
            else:
                print(f"Processing node: {node.format()}")
                for out in node.outputs:
                    self._allocate_tensor(out)

            # Free inputs whose lifetime expires at this node
            for dead_val in user_to_last_uses.get(node, []):
                self._free_tensor(dead_val)

    def run(self, module: Module) -> Dict[Value, int]:
        self.mapping.clear()
        self.node_to_last_use.clear()
        for alloc in self.allocators.values():
            alloc.allocated_blocks.clear()
            alloc.peak_usage = 0

        print("--- Running Liveness Analysis ---")
        self._analyze_liveness(module)

        user_to_last_uses: Dict[IRNode, Set[Value]] = defaultdict(set)
        for val, user_node in self.node_to_last_use.items():
            user_to_last_uses[user_node].add(val)

        print("\n--- Running Memory Allocation ---")
        self._process_scope(module, user_to_last_uses)

        return self.mapping


def run_memory_pass(module) -> Dict[Value, int]:
    allocator = MemoryAllocatorPass()
    mapping = allocator.run(module)

    print("\n--- Memory Pass Summary ---")
    for space, alloc_manager in allocator.allocators.items():
        print(f"Peak {space} Memory Required: {alloc_manager.peak_usage} bytes")

    for val, offset in mapping.items():
        val.address = offset
        print(f"{val.name}: {offset}")

    return mapping
