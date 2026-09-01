"""
PREFRAG: Trace Parser
Supports ingestion and normalization of external profiling formats (MemProfiler, Heaptrack, Valgrind).
"""

import json
import csv
from typing import List, Optional
from .trace_schema import MemoryEvent, AllocationOp


class TraceParser:
    """
    Parses and normalizes external trace files into PREFRAG MemoryEvent streams.
    """

    @staticmethod
    def parse_csv_trace(filepath: str) -> List[MemoryEvent]:
        events = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                op = AllocationOp.ALLOC if "ALLOC" in row.get("op", "ALLOC").upper() else AllocationOp.FREE
                events.append(MemoryEvent(
                    timestamp=float(row.get("timestamp", 0.0)),
                    op=op,
                    address=int(row.get("address", 0), 0) if isinstance(row.get("address"), str) and row.get("address").startswith("0x") else int(row.get("address", 0)),
                    size=int(row.get("size", 0)),
                    lifetime=float(row["lifetime"]) if row.get("lifetime") and row["lifetime"] != "" else None,
                    call_stack_hash=int(row["call_stack_hash"], 0) if row.get("call_stack_hash") and str(row["call_stack_hash"]).startswith("0x") else (int(row["call_stack_hash"]) if row.get("call_stack_hash") and row["call_stack_hash"] != "" else None),
                    thread_id=int(row.get("thread_id", 0) or 0),
                    tag=row.get("tag", "")
                ))
        return events

    @staticmethod
    def parse_memprofiler_json(filepath: str) -> List[MemoryEvent]:
        """
        Parses MemProfiler JSON output format containing recorded allocation events.
        """
        events = []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_events = data if isinstance(data, list) else data.get("events", [])
            for item in raw_events:
                op_str = item.get("operation", item.get("op", "ALLOC")).upper()
                op = AllocationOp.ALLOC if "ALLOC" in op_str or "MALLOC" in op_str else AllocationOp.FREE
                events.append(MemoryEvent(
                    timestamp=float(item.get("timestamp", item.get("time", 0.0))),
                    op=op,
                    address=int(item.get("address", 0)),
                    size=int(item.get("size", 0)),
                    lifetime=float(item["lifetime"]) if item.get("lifetime") is not None else None,
                    call_stack_hash=hash(tuple(item.get("stack", []))) if item.get("stack") else None,
                    thread_id=int(item.get("thread_id", 0)),
                    tag=item.get("tag", "memprofiler")
                ))
        return events
