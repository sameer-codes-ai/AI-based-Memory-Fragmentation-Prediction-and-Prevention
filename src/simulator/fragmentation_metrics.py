"""
PREFRAG: Mathematical Fragmentation Metrics
Computes F1 (Largest-Free-Block Ratio), F2 (Free-Block Entropy), F3 (Free Block Count), and spatial metrics.
"""

import math
from typing import List, Dict, Any
from dataclasses import dataclass
from .memory_block import MemoryBlock


@dataclass(slots=True)
class FragmentationSnapshot:
    timestamp: float
    total_capacity: int
    allocated_bytes: int
    total_free_bytes: int
    largest_free_block: int
    free_block_count: int
    allocated_block_count: int
    f1_largest_free_ratio: float       # F1 = 1 - (LFB / Total Free)
    f2_free_block_entropy: float      # F2 = -sum(p_i * log2(p_i))
    f3_free_block_count: int          # F3 = N_free
    memory_utilization: float         # Allocated / Total
    allocation_failures: int = 0
    severe_fragmentation_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_capacity": self.total_capacity,
            "allocated_bytes": self.allocated_bytes,
            "total_free_bytes": self.total_free_bytes,
            "largest_free_block": self.largest_free_block,
            "free_block_count": self.free_block_count,
            "allocated_block_count": self.allocated_block_count,
            "f1_largest_free_ratio": self.f1_largest_free_ratio,
            "f2_free_block_entropy": self.f2_free_block_entropy,
            "f3_free_block_count": self.f3_free_block_count,
            "memory_utilization": self.memory_utilization,
            "allocation_failures": self.allocation_failures,
            "severe_fragmentation_flag": self.severe_fragmentation_flag
        }


class FragmentationCalculator:
    """
    Computes mathematical fragmentation indicators across a collection of heap memory blocks.
    """

    @staticmethod
    def calculate(
        blocks: List[MemoryBlock],
        total_capacity: int,
        timestamp: float = 0.0,
        failures: int = 0,
        severe_threshold_f1: float = 0.65
    ) -> FragmentationSnapshot:
        free_blocks = [b for b in blocks if b.is_free and b.size > 0]
        allocated_blocks = [b for b in blocks if not b.is_free]

        total_free = sum(b.size for b in free_blocks)
        allocated_bytes = sum(b.size for b in allocated_blocks)
        largest_free = max((b.size for b in free_blocks), default=0)
        free_count = len(free_blocks)
        allocated_count = len(allocated_blocks)

        # F1: Largest Free Block Ratio
        # F1 = 1 - (LFB / Total Free) when Total Free > 0, else 1.0 (or 0.0 if empty heap)
        if total_free > 0:
            f1 = 1.0 - (float(largest_free) / float(total_free))
        else:
            f1 = 1.0 if allocated_bytes > 0 else 0.0

        # F2: Free Block Shannon Entropy
        # p_i = block_size / total_free
        # Entropy = - sum(p_i * log2(p_i))
        if total_free > 0 and free_count > 0:
            entropy = 0.0
            for b in free_blocks:
                p_i = b.size / total_free
                if p_i > 0:
                    entropy -= p_i * math.log2(p_i)
            f2 = entropy
        else:
            f2 = 0.0

        # F3: Count of free blocks
        f3 = free_count

        utilization = float(allocated_bytes) / float(total_capacity) if total_capacity > 0 else 0.0
        severe_flag = (f1 >= severe_threshold_f1) and (utilization < 0.85) and (total_free > 0)

        return FragmentationSnapshot(
            timestamp=timestamp,
            total_capacity=total_capacity,
            allocated_bytes=allocated_bytes,
            total_free_bytes=total_free,
            largest_free_block=largest_free,
            free_block_count=free_count,
            allocated_block_count=allocated_count,
            f1_largest_free_ratio=max(0.0, min(1.0, f1)),
            f2_free_block_entropy=max(0.0, f2),
            f3_free_block_count=f3,
            memory_utilization=max(0.0, min(1.0, utilization)),
            allocation_failures=failures,
            severe_fragmentation_flag=severe_flag
        )
