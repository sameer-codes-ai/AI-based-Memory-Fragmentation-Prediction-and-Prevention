"""
PREFRAG: Memory Block
Represents a contiguous segment in simulated heap memory.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryBlock:
    start_address: int
    size: int
    is_free: bool = True
    alloc_id: Optional[int] = None
    alloc_time: Optional[float] = None
    lifetime: Optional[float] = None
    tag: str = ""
    call_stack_hash: Optional[int] = None

    @property
    def end_address(self) -> int:
        return self.start_address + self.size

    def split(self, alloc_size: int, alloc_id: int, alloc_time: float, lifetime: Optional[float] = None, tag: str = "", call_stack_hash: Optional[int] = None) -> "MemoryBlock":
        """
        Splits this free block into an allocated block of `alloc_size` and a remainder free block.
        Returns the new remainder free block.
        """
        if not self.is_free or alloc_size > self.size:
            raise ValueError("Cannot split non-free or undersized block")

        remainder_size = self.size - alloc_size
        self.size = alloc_size
        self.is_free = False
        self.alloc_id = alloc_id
        self.alloc_time = alloc_time
        self.lifetime = lifetime
        self.tag = tag
        self.call_stack_hash = call_stack_hash

        if remainder_size > 0:
            return MemoryBlock(
                start_address=self.start_address + alloc_size,
                size=remainder_size,
                is_free=True
            )
        return None
