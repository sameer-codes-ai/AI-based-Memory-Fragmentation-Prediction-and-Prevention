"""
PREFRAG: Heap Allocator Simulator
Simulates memory allocation, free block coalescing, compaction invocation, and metric telemetry.
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple
from .memory_block import MemoryBlock
from .fragmentation_metrics import FragmentationCalculator, FragmentationSnapshot
from .compaction_engine import CompactionEngine, CompactionReport


class AllocationPolicy(str, Enum):
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"


class HeapAllocator:
    """
    Virtual heap memory manager simulating memory allocation strategies and continuous fragmentation state.
    """

    def __init__(
        self,
        total_capacity: int = 16 * 1024 * 1024, # 16 MB default capacity
        policy: AllocationPolicy = AllocationPolicy.FIRST_FIT,
        alignment: int = 16,
        severe_f1_threshold: float = 0.65
    ):
        self.total_capacity = total_capacity
        self.policy = policy
        self.alignment = alignment
        self.severe_f1_threshold = severe_f1_threshold
        
        # Initial state: single large free block covering the entire capacity
        self.blocks: List[MemoryBlock] = [
            MemoryBlock(start_address=0, size=total_capacity, is_free=True)
        ]
        
        self.compaction_engine = CompactionEngine()
        self.alloc_map: Dict[int, MemoryBlock] = {} # address -> block reference
        self.alloc_id_map: Dict[int, MemoryBlock] = {} # alloc_id -> block reference
        self._next_alloc_id = 1
        self.failures_count = 0
        self.total_allocations = 0
        self.total_deallocations = 0
        self.compaction_history: List[CompactionReport] = []

    def _align(self, size: int) -> int:
        if self.alignment <= 1:
            return size
        return ((size + self.alignment - 1) // self.alignment) * self.alignment

    def allocate(
        self,
        size: int,
        timestamp: float = 0.0,
        lifetime: Optional[float] = None,
        tag: str = "",
        call_stack_hash: Optional[int] = None,
        alloc_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Allocates memory of the requested size according to the configured allocation policy.
        Returns start address if successful, or None if allocation failed (fragmentation/capacity OOM).
        """
        aligned_size = self._align(size)
        candidate_idx: Optional[int] = None

        if self.policy == AllocationPolicy.FIRST_FIT:
            for idx, b in enumerate(self.blocks):
                if b.is_free and b.size >= aligned_size:
                    candidate_idx = idx
                    break

        elif self.policy == AllocationPolicy.BEST_FIT:
            best_diff = float("inf")
            for idx, b in enumerate(self.blocks):
                if b.is_free and b.size >= aligned_size:
                    diff = b.size - aligned_size
                    if diff < best_diff:
                        best_diff = diff
                        candidate_idx = idx
                        if diff == 0:
                            break

        elif self.policy == AllocationPolicy.WORST_FIT:
            worst_size = -1
            for idx, b in enumerate(self.blocks):
                if b.is_free and b.size >= aligned_size:
                    if b.size > worst_size:
                        worst_size = b.size
                        candidate_idx = idx

        if candidate_idx is None:
            self.failures_count += 1
            return None

        # Perform allocation and block splitting
        target_block = self.blocks[candidate_idx]
        current_alloc_id = alloc_id if alloc_id is not None else self._next_alloc_id
        self._next_alloc_id += 1

        remainder = target_block.split(
            alloc_size=aligned_size,
            alloc_id=current_alloc_id,
            alloc_time=timestamp,
            lifetime=lifetime,
            tag=tag,
            call_stack_hash=call_stack_hash
        )

        if remainder is not None:
            self.blocks.insert(candidate_idx + 1, remainder)

        self.alloc_map[target_block.start_address] = target_block
        self.alloc_id_map[current_alloc_id] = target_block
        self.total_allocations += 1
        return target_block.start_address

    def free(self, address: int, timestamp: float = 0.0) -> bool:
        """
        Frees memory at the specified start address and automatically coalesces adjacent free neighbors.
        """
        if address in self.alloc_id_map:
            return self.free_by_id(address, timestamp)

        if address not in self.alloc_map:
            return False

        block = self.alloc_map.pop(address)
        if block.alloc_id in self.alloc_id_map:
            del self.alloc_id_map[block.alloc_id]

        block.is_free = True
        block.alloc_id = None
        block.tag = ""
        block.call_stack_hash = None
        self.total_deallocations += 1

        # Coalesce adjacent free blocks
        self._coalesce()
        return True

    def free_by_id(self, alloc_id: int, timestamp: float = 0.0) -> bool:
        """
        Frees memory by unique allocation ID (stable across memory compactions).
        """
        if alloc_id not in self.alloc_id_map:
            return False

        block = self.alloc_id_map.pop(alloc_id)
        if block.start_address in self.alloc_map:
            del self.alloc_map[block.start_address]

        block.is_free = True
        block.alloc_id = None
        block.tag = ""
        block.call_stack_hash = None
        self.total_deallocations += 1

        self._coalesce()
        return True

    def _coalesce(self) -> None:
        """
        Iterates and coalesces contiguous free blocks in the heap list.
        """
        if len(self.blocks) <= 1:
            return

        new_blocks = [self.blocks[0]]
        for b in self.blocks[1:]:
            prev = new_blocks[-1]
            if prev.is_free and b.is_free:
                prev.size += b.size
            else:
                new_blocks.append(b)
        self.blocks = new_blocks

    def compact(self, timestamp: float = 0.0) -> CompactionReport:
        """
        Triggers memory compaction on the current heap state.
        """
        self.blocks, report = self.compaction_engine.compact(self.blocks, self.total_capacity, timestamp)
        
        # Re-index alloc_map
        self.alloc_map.clear()
        self.alloc_id_map.clear()
        for b in self.blocks:
            if not b.is_free:
                self.alloc_map[b.start_address] = b
                if b.alloc_id is not None:
                    self.alloc_id_map[b.alloc_id] = b

        self.compaction_history.append(report)
        return report

    def get_snapshot(self, timestamp: float = 0.0) -> FragmentationSnapshot:
        """
        Returns a complete analytical snapshot of heap fragmentation state.
        """
        return FragmentationCalculator.calculate(
            blocks=self.blocks,
            total_capacity=self.total_capacity,
            timestamp=timestamp,
            failures=self.failures_count,
            severe_threshold_f1=self.severe_f1_threshold
        )
