"""
PREFRAG: Memory Allocation Trace Schema
Defines the canonical event representation for dynamic memory allocation/deallocation traces.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Iterator
import csv


class AllocationOp(str, Enum):
    ALLOC = "ALLOC"
    FREE = "FREE"
    REALLOC = "REALLOC"


@dataclass(slots=True)
class MemoryEvent:
    """
    Represents a single memory management event in a workload trace.
    """
    timestamp: float              # Logical or physical timestamp (ms or normalized tick)
    op: AllocationOp              # ALLOC or FREE
    address: int                  # Virtual/heap address identifier
    size: int                     # Allocation size in bytes (0 for free if address is primary)
    lifetime: Optional[float] = None     # Object lifespan if known (derived or profiled)
    call_stack_hash: Optional[int] = None # Allocation site/context identifier
    thread_id: int = 0            # Originating thread ID
    tag: str = ""                 # Workload/context semantic tag

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['op'] = self.op.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEvent":
        op_val = AllocationOp(data["op"]) if isinstance(data["op"], str) else data["op"]
        return cls(
            timestamp=float(data["timestamp"]),
            op=op_val,
            address=int(data["address"]),
            size=int(data["size"]),
            lifetime=float(data["lifetime"]) if data.get("lifetime") is not None and data.get("lifetime") != "" else None,
            call_stack_hash=int(data["call_stack_hash"]) if data.get("call_stack_hash") is not None and data.get("call_stack_hash") != "" else None,
            thread_id=int(data.get("thread_id", 0) or 0),
            tag=str(data.get("tag", ""))
        )


class TraceStream:
    """
    Utility for streaming and saving memory allocation traces to CSV format.
    """

    @staticmethod
    def save_csv(events: List[MemoryEvent], filepath: str) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "op", "address", "size", "lifetime", "call_stack_hash", "thread_id", "tag"])
            for ev in events:
                writer.writerow([
                    ev.timestamp,
                    ev.op.value,
                    ev.address,
                    ev.size,
                    ev.lifetime if ev.lifetime is not None else "",
                    ev.call_stack_hash if ev.call_stack_hash is not None else "",
                    ev.thread_id,
                    ev.tag
                ])

    @staticmethod
    def load_csv(filepath: str) -> List[MemoryEvent]:
        events = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(MemoryEvent(
                    timestamp=float(row["timestamp"]),
                    op=AllocationOp(row["op"]),
                    address=int(row["address"]),
                    size=int(row["size"]),
                    lifetime=float(row["lifetime"]) if row.get("lifetime") and row["lifetime"] != "" else None,
                    call_stack_hash=int(row["call_stack_hash"]) if row.get("call_stack_hash") and row["call_stack_hash"] != "" else None,
                    thread_id=int(row.get("thread_id", 0) or 0),
                    tag=row.get("tag", "")
                ))
        return events

    @staticmethod
    def stream_csv(filepath: str) -> Iterator[MemoryEvent]:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield MemoryEvent(
                    timestamp=float(row["timestamp"]),
                    op=AllocationOp(row["op"]),
                    address=int(row["address"]),
                    size=int(row["size"]),
                    lifetime=float(row["lifetime"]) if row.get("lifetime") and row["lifetime"] != "" else None,
                    call_stack_hash=int(row["call_stack_hash"]) if row.get("call_stack_hash") and row["call_stack_hash"] != "" else None,
                    thread_id=int(row.get("thread_id", 0) or 0),
                    tag=row.get("tag", "")
                )
