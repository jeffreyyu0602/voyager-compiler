import math
from collections import defaultdict
from typing import Dict, Set, Union

from voyager_compiler.codegen.lowering.ir import (
    Value,
    TensorBox,
    Loops,
    IRNode,
    Module,
    Operation,
    FusedOp,
)
from voyager_compiler.pt2e_utils import dtype_byte_size


class IntervalManager:
    """
    Manages free and occupied memory segments.
    Uses a 'First-Fit' strategy to find gaps for new allocations.
    """
    def __init__(self, base_offset=0):
        # List of (start, size) tuples, sorted by start
        self.allocated_blocks = []
        self.peak_usage = 0

    def allocate(self, size: int) -> int:
        """Finds the first free gap that fits 'size' bytes."""
        # Align to 32 bytes (common for vector loads/DMA)
        size = math.ceil(size / 32) * 32

        # Check gaps between allocated blocks
        prev_end = 0

        insert_idx = 0
        found_offset = -1

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
        end_addr = found_offset + size
        self.peak_usage = max(self.peak_usage, end_addr)

        return found_offset

    def free(self, offset: int):
        """Removes the block at 'offset' from the allocated list."""
        for i, (start, length) in enumerate(self.allocated_blocks):
            if start == offset:
                self.allocated_blocks.pop(i)
                return
        raise ValueError(f"Attempted to free unallocated offset: {offset}")


class ScratchpadAllocator:
    def __init__(self, alignment: int = 32):
        self.alignment = alignment
        self.mapping: Dict[Value, int] = {} # Result: Value -> Offset
        self.mem_manager = IntervalManager()

        # Liveness tracking
        self.node_to_last_use: Dict[Value, IRNode] = {}

    def get_tensor_bytes(self, val: TensorBox) -> int:
        if not hasattr(val, 'shape') or not hasattr(val, 'dtype'):
            return 0
        numel = 1
        for s in val.shape:
            numel *= s
        return int(numel * dtype_byte_size(val.dtype))

    def _analyze_liveness(self, module: Module):
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
                continue

            for inp in node.inputs:
                if inp in outputs:
                    self.node_to_last_use[inp] = node
                else:
                    inputs.add(inp)

            outputs.update(node.outputs)

        print(f"module inputs: {[v.name for v in inputs]}")
        print(f"module outputs: {[v.name for v in outputs]}")
        return inputs

    def _process_scope(
        self,
        block: Union[Module, Loops],
        user_to_last_uses: Dict[Value, Set[IRNode]],
        memory_space="Scratchpad"
    ):
        if isinstance(block, Module):
            for arg in block.args:
                if isinstance(arg, TensorBox) and arg.space == memory_space:
                    size = self.get_tensor_bytes(arg)
                    offset = self.mem_manager.allocate(size)
                    arg.address = offset
                    self.mapping[arg] = offset
                    print(f"Alloc module arg {arg.name} size {size} @ {offset}")

        for node in block.body:
            if isinstance(node, Loops):
                print(f"Entering loop {node.index} with body:")
                self._process_scope(node, user_to_last_uses, memory_space=memory_space)
                print(f"Exiting loop {node.index}")
            else:
                print(f"Processing node: {node.format()}")
                for out in node.outputs:
                    if (
                        isinstance(out, TensorBox)
                        and out.space == memory_space
                        and out not in self.mapping
                    ):
                        size = self.get_tensor_bytes(out)
                        offset = self.mem_manager.allocate(size)
                        out.address = offset
                        self.mapping[out] = offset
                        print(f"Alloc {out.name} size {size} @ {offset}")

            for inp in user_to_last_uses.get(node, []):
                if inp in self.mapping:
                    offset = self.mapping[inp]
                    self.mem_manager.free(offset)
                    print(f"Free {inp.name} from offset {offset}")

    def run(self, module):
        self.mapping = {}
        self.node_to_last_use = {}

        # Pass 1: Liveness Analysis
        self._analyze_liveness(module)

        print("Liveness Analysis Complete. Last uses:")
        for val, op in self.node_to_last_use.items():
            op_name = op.format()
            if isinstance(op, Loops):
                op_name = f"Loop(index={op.index})"
            print(f"{val.name}: last used at op {op_name}")

        scratchpad_last_uses: Dict[Value, Set[IRNode]] = defaultdict(set)
        dram_last_uses: Dict[Value, Set[IRNode]] = defaultdict(set)
        for inp, user in self.node_to_last_use.items():
            if isinstance(inp, TensorBox):
                if inp.space == "Scratchpad":
                    scratchpad_last_uses[user].add(inp)
                elif inp.space == "DRAM":
                    dram_last_uses[user].add(inp)

        for user, last_use in scratchpad_last_uses.items():
            if isinstance(user, Loops):
                print(f"Loop {user.index} has last uses:")
            else:
                print(f"User {user.format()} has last uses:")
            print([v.name for v in last_use])

        # Pass 2: Allocation Simulation
        print(f"Scratchpad Allocation:")
        self._process_scope(module, scratchpad_last_uses, memory_space="Scratchpad")

        print(f"\nDRAM Allocation:")
        self._process_scope(module, dram_last_uses, memory_space="DRAM")

        return self.mapping


def run_memory_pass(module):
    allocator = ScratchpadAllocator()
    mapping = allocator.run(module)

    print(f"Total Scratchpad Memory Required: {allocator.mem_manager.peak_usage} bytes")

    # You can attach the mapping to the IR if you wish
    for val, offset in mapping.items():
        val.address = offset
        print(f"{val.name}: {offset}")

    return mapping
