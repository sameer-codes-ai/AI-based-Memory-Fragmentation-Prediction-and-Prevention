"""
PREFRAG: Memory Compaction Engine
Simulates physical heap reorganization, sliding compaction, and calculates precise operation costs.
"""

from dataclasses import dataclass
from typing import List, Tuple
from .memory_block import MemoryBlock
from .fragmentation_metrics import FragmentationCalculator, FragmentationSnapshot


@dataclass(slots=True)
class CompactionReport:
    timestamp: float
    blocks_moved: int
    bytes_moved: int
    f1_before: float
    f1_after: float
    largest_free_before: int
    largest_free_after: int
    pause_latency_ms: float
    total_cost: float


class CompactionEngine:
    """
    Simulates physical memory compaction and models the CPU and latency costs.
    """

    def __init__(
        self,
        base_pause_ms: float = 0.5,
        per_block_cost_ms: float = 0.005,
        bandwidth_mb_per_sec: float = 10000.0, # 10 GB/s memory bandwidth
        cost_weight_alpha: float = 1.0,        # weight for block moves
        cost_weight_beta: float = 0.0001,      # weight for bytes moved
        cost_weight_gamma: float = 2.0         # weight for pause time
    ):
        self.base_pause_ms = base_pause_ms
        self.per_block_cost_ms = per_block_cost_ms
        self.bandwidth_bytes_per_ms = (bandwidth_mb_per_sec * 1024 * 1024) / 1000.0
        self.alpha = cost_weight_alpha
        self.beta = cost_weight_beta
        self.gamma = cost_weight_gamma

    def estimate_compaction_cost(self, blocks: List[MemoryBlock]) -> Tuple[int, int, float, float]:
        """
        Estimates (blocks_moved, bytes_moved, latency_ms, total_cost) without modifying the heap.
        """
        curr_addr = 0
        blocks_moved = 0
        bytes_moved = 0

        for b in blocks:
            if not b.is_free:
                if b.start_address != curr_addr:
                    blocks_moved += 1
                    bytes_moved += b.size
                curr_addr += b.size

        move_time_ms = float(bytes_moved) / max(1.0, self.bandwidth_bytes_per_ms)
        latency_ms = self.base_pause_ms + (blocks_moved * self.per_block_cost_ms) + move_time_ms
        cost = (self.alpha * blocks_moved) + (self.beta * bytes_moved) + (self.gamma * latency_ms)
        return blocks_moved, bytes_moved, latency_ms, cost

    def compact(
        self,
        blocks: List[MemoryBlock],
        total_capacity: int,
        timestamp: float = 0.0
    ) -> Tuple[List[MemoryBlock], CompactionReport]:
        """
        Executes sliding compaction: slides all allocated blocks to the base of memory
        and merges all free space into one large contiguous free block.
        """
        snap_before = FragmentationCalculator.calculate(blocks, total_capacity, timestamp)

        allocated_blocks = [b for b in blocks if not b.is_free]
        new_blocks: List[MemoryBlock] = []
        curr_addr = 0
        blocks_moved = 0
        bytes_moved = 0

        for b in allocated_blocks:
            if b.start_address != curr_addr:
                blocks_moved += 1
                bytes_moved += b.size
            
            new_b = MemoryBlock(
                start_address=curr_addr,
                size=b.size,
                is_free=False,
                alloc_id=b.alloc_id,
                alloc_time=b.alloc_time,
                lifetime=b.lifetime,
                tag=b.tag,
                call_stack_hash=b.call_stack_hash
            )
            new_blocks.append(new_b)
            curr_addr += b.size

        # Remainder is single contiguous free block
        remaining_free = total_capacity - curr_addr
        if remaining_free > 0:
            new_blocks.append(MemoryBlock(
                start_address=curr_addr,
                size=remaining_free,
                is_free=True
            ))

        snap_after = FragmentationCalculator.calculate(new_blocks, total_capacity, timestamp)

        move_time_ms = float(bytes_moved) / max(1.0, self.bandwidth_bytes_per_ms)
        latency_ms = self.base_pause_ms + (blocks_moved * self.per_block_cost_ms) + move_time_ms
        total_cost = (self.alpha * blocks_moved) + (self.beta * bytes_moved) + (self.gamma * latency_ms)

        report = CompactionReport(
            timestamp=timestamp,
            blocks_moved=blocks_moved,
            bytes_moved=bytes_moved,
            f1_before=snap_before.f1_largest_free_ratio,
            f1_after=snap_after.f1_largest_free_ratio,
            largest_free_before=snap_before.largest_free_block,
            largest_free_after=snap_after.largest_free_block,
            pause_latency_ms=latency_ms,
            total_cost=total_cost
        )
        return new_blocks, report
