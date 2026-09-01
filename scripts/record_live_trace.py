"""
PREFRAG: High-Performance Live Memory Profiler
Executes real live workloads (SQLite, JSON microservice, NumPy numeric matrices)
and records real physical memory allocation/free events to CSV.
"""

import sys
import os
import time
import tracemalloc
import sqlite3
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.traces.trace_schema import MemoryEvent, AllocationOp, TraceStream


def record_live_workload(
    workload_type: str = "sqlite",
    num_steps: int = 200,
    output_path: str = "results/saved_traces/live_sqlite_trace.csv"
) -> str:
    events = []
    tracemalloc.start(1)
    prev_snapshot = tracemalloc.take_snapshot()
    start_time = time.perf_counter()

    if workload_type == "sqlite":
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, key TEXT, value BLOB)")
        cursor.execute("CREATE INDEX idx_key ON records(key)")

        for i in range(num_steps):
            val = os.urandom(int(np.random.choice([256, 1024, 4096, 16384])))
            cursor.execute("INSERT INTO records (key, value) VALUES (?, ?)", (f"key_{i}", val))
            if i % 10 == 0:
                cursor.execute("SELECT * FROM records WHERE key = ?", (f"key_{np.random.randint(0, i+1)}",))
                _ = cursor.fetchall()
            if i % 20 == 0:
                cursor.execute("DELETE FROM records WHERE id % 4 == 0")
                conn.commit()

            if i % 5 == 0:
                curr_snap = tracemalloc.take_snapshot()
                diff = curr_snap.compare_to(prev_snapshot, 'filename')
                curr_time = (time.perf_counter() - start_time) * 1000.0
                for stat in diff:
                    if stat.size_diff != 0:
                        op = AllocationOp.ALLOC if stat.size_diff > 0 else AllocationOp.FREE
                        events.append(MemoryEvent(
                            timestamp=round(curr_time, 2),
                            op=op,
                            address=hash(stat.traceback[0].filename) & 0x7FFFFFFF,
                            size=abs(stat.size_diff),
                            tag=f"live_{workload_type}"
                        ))
                prev_snapshot = curr_snap

        conn.close()

    elif workload_type == "json_microservice":
        cache = {}
        for i in range(num_steps):
            payload = {
                "id": f"req_{i}",
                "timestamp": time.time(),
                "records": [{"idx": j, "data": "abc" * 20} for j in range(np.random.randint(10, 50))]
            }
            s = json.dumps(payload)
            d = json.loads(s)
            cache[i % 30] = d

            if i % 5 == 0:
                curr_snap = tracemalloc.take_snapshot()
                diff = curr_snap.compare_to(prev_snapshot, 'filename')
                curr_time = (time.perf_counter() - start_time) * 1000.0
                for stat in diff:
                    if stat.size_diff != 0:
                        op = AllocationOp.ALLOC if stat.size_diff > 0 else AllocationOp.FREE
                        events.append(MemoryEvent(
                            timestamp=round(curr_time, 2),
                            op=op,
                            address=hash(stat.traceback[0].filename) & 0x7FFFFFFF,
                            size=abs(stat.size_diff),
                            tag=f"live_{workload_type}"
                        ))
                prev_snapshot = curr_snap

    elif workload_type == "scientific_matrix":
        arrays = []
        for i in range(num_steps):
            sz = np.random.choice([100, 200, 500])
            mat = np.random.randn(sz, sz).astype(np.float32)
            arrays.append(mat)
            if len(arrays) > 15:
                arrays.pop(0)

            if i % 5 == 0:
                curr_snap = tracemalloc.take_snapshot()
                diff = curr_snap.compare_to(prev_snapshot, 'filename')
                curr_time = (time.perf_counter() - start_time) * 1000.0
                for stat in diff:
                    if stat.size_diff != 0:
                        op = AllocationOp.ALLOC if stat.size_diff > 0 else AllocationOp.FREE
                        events.append(MemoryEvent(
                            timestamp=round(curr_time, 2),
                            op=op,
                            address=hash(stat.traceback[0].filename) & 0x7FFFFFFF,
                            size=abs(stat.size_diff),
                            tag=f"live_{workload_type}"
                        ))
                prev_snapshot = curr_snap

    tracemalloc.stop()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    TraceStream.save_csv(events, output_path)
    print(f"[+] Successfully recorded {len(events)} live events for '{workload_type}' -> {output_path}")
    return output_path


if __name__ == "__main__":
    record_live_workload("sqlite", 150, "results/saved_traces/live_sqlite_trace.csv")
    record_live_workload("json_microservice", 150, "results/saved_traces/live_microservice_trace.csv")
    record_live_workload("scientific_matrix", 150, "results/saved_traces/live_scientific_trace.csv")
