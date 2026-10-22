"""
PREFRAG: Real-Time Interactive Live Demo Server
Serves an interactive web application that visualizes heap allocations,
generates fragmentation, runs ML predictive models, and triggers cost-aware defragmentation.
"""

import sys
import os
import json
import time
import math
import logging
from typing import List, Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.traces.trace_schema import MemoryEvent, AllocationOp
from src.traces.workload_generator import WorkloadGenerator, WorkloadFamily
from src.simulator.heap_allocator import HeapAllocator, AllocationPolicy
from src.simulator.fragmentation_metrics import FragmentationCalculator, FragmentationSnapshot
from src.simulator.compaction_engine import CompactionReport
from src.features.feature_extractor import FeatureExtractor
from src.models.baselines import RandomForestBaseline, LinearRegressionBaseline, StaticThresholdBaseline
from src.controller.policies import (
    ReactivePolicy,
    AlwaysProactivePolicy,
    ThresholdPredictivePolicy,
    CostAwarePredictivePolicy
)
from src.controller.cost_model import CostBenefitOptimizer


class LiveDemoEngine:
    """
    Manages live state, workload streaming, ML model inference, and heap memory management.
    """

    def __init__(self, heap_capacity: int = 1024 * 1024): # 1 MB default for snappy visual demo
        self.heap_capacity = heap_capacity
        self.workload_family = WorkloadFamily.DATABASE
        self.allocator_policy = AllocationPolicy.BEST_FIT
        self.management_policy_name = "E5: PREFRAG (Predictive + Cost-Aware)"
        
        self.gen = WorkloadGenerator(seed=42)
        self.extractor = FeatureExtractor(window_size_ticks=20, severe_f1_threshold=0.60)
        
        self.allocator: HeapAllocator = HeapAllocator(
            total_capacity=self.heap_capacity,
            policy=self.allocator_policy,
            alignment=16,
            severe_f1_threshold=0.60
        )
        
        self.policy: Any = CostAwarePredictivePolicy(
            benefit_scale_lambda=1500.0,
            min_net_benefit_threshold=2.0,
            cooldown_ticks=10,
            severe_f1_threshold=0.60
        )
        
        self.model: Any = None
        self.trace_events: List[MemoryEvent] = []
        self.trace_cursor: int = 0
        self.current_tick: int = 0
        
        self.recent_events: List[MemoryEvent] = []
        self.snapshot_history: List[FragmentationSnapshot] = []
        self.feature_seq_history: List[np.ndarray] = []
        
        self.last_predictions: List[float] = [0.0, 0.0, 0.0, 0.0] # t+1, t+5, t+10, t+30
        self.history_records: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.last_compaction_report: Optional[Dict[str, Any]] = None
        self.last_policy_evaluation: Optional[Dict[str, Any]] = None
        self.custom_alloc_counter: int = 9000

        self._initialize_workload(self.workload_family)

    def _initialize_workload(self, family: WorkloadFamily):
        self.workload_family = family
        self.trace_events = self.gen.generate_trace(family, num_events=400)
        self.trace_cursor = 0
        self.current_tick = 0
        self.recent_events = []
        self.snapshot_history = []
        self.feature_seq_history = []
        self.history_records = []
        self.event_log = []
        self.last_compaction_report = None
        self.last_policy_evaluation = None
        
        # Reset heap
        self.allocator = HeapAllocator(
            total_capacity=self.heap_capacity,
            policy=self.allocator_policy,
            alignment=16,
            severe_f1_threshold=0.60
        )
        
        # Fast train initial baseline model for this workload
        self._train_model()
        self._record_snapshot(timestamp=0.0)

    def _train_model(self):
        """Train baseline model on a quick replay sample."""
        try:
            sample_alloc = HeapAllocator(total_capacity=self.heap_capacity, policy=self.allocator_policy)
            sample_snaps = []
            for ev in self.trace_events[:150]:
                if ev.op == AllocationOp.ALLOC:
                    sample_alloc.allocate(ev.size, ev.timestamp, alloc_id=ev.address)
                else:
                    sample_alloc.free_by_id(ev.address, ev.timestamp)
                sample_snaps.append(sample_alloc.get_snapshot(ev.timestamp))
            
            X, Y_reg, _ = self.extractor.extract_features_from_history(self.trace_events[:150], sample_snaps)
            if len(X) > 20:
                self.model = RandomForestBaseline(n_estimators=10, max_depth=5).fit(X, Y_reg)
            else:
                self.model = StaticThresholdBaseline()
        except Exception as e:
            logging.warning("Model training fallback: %s", e)
            self.model = StaticThresholdBaseline()

    def set_allocator_policy(self, policy_str: str):
        mapping = {
            "first_fit": AllocationPolicy.FIRST_FIT,
            "best_fit": AllocationPolicy.BEST_FIT,
            "worst_fit": AllocationPolicy.WORST_FIT,
            "next_fit": AllocationPolicy.NEXT_FIT
        }
        if policy_str in mapping:
            self.allocator_policy = mapping[policy_str]
            self.allocator.policy = self.allocator_policy

    def set_management_policy(self, policy_name: str):
        self.management_policy_name = policy_name
        if "E1" in policy_name:
            self.policy = ReactivePolicy(severe_threshold=0.60, cooldown_ticks=10)
        elif "E2" in policy_name:
            self.policy = AlwaysProactivePolicy(interval_ticks=20)
        elif "E4" in policy_name:
            self.policy = ThresholdPredictivePolicy(threshold=0.60, cooldown_ticks=10)
        else: # E5
            self.policy = CostAwarePredictivePolicy(
                benefit_scale_lambda=1500.0,
                min_net_benefit_threshold=2.0,
                cooldown_ticks=10,
                severe_f1_threshold=0.60
            )

    def _record_snapshot(self, timestamp: float):
        snap = self.allocator.get_snapshot(timestamp)
        self.snapshot_history.append(snap)
        if len(self.snapshot_history) > 100:
            self.snapshot_history.pop(0)

        # Predict future fragmentation using current feature vector
        pred_f1_t10 = snap.f1_largest_free_ratio
        if self.model and len(self.snapshot_history) >= 5 and len(self.recent_events) >= 5:
            try:
                feat = self.extractor.extract_single_step(self.recent_events, self.snapshot_history)
                vec = feat.to_array().reshape(1, -1)
                preds = self.model.predict(vec)[0]
                self.last_predictions = [float(preds[0]), float(preds[1]), float(preds[2]), float(preds[3])]
                pred_f1_t10 = float(self.last_predictions[2])
            except Exception:
                self.last_predictions = [snap.f1_largest_free_ratio] * 4
        else:
            self.last_predictions = [snap.f1_largest_free_ratio] * 4

        record = {
            "tick": self.current_tick,
            "f1": round(snap.f1_largest_free_ratio, 4),
            "predicted_f1_t10": round(pred_f1_t10, 4),
            "f2_entropy": round(snap.f2_free_block_entropy, 4),
            "free_blocks": snap.free_block_count,
            "utilization": round(snap.memory_utilization * 100, 1),
            "allocated_kb": round(snap.allocated_bytes / 1024, 1),
            "free_kb": round(snap.total_free_bytes / 1024, 1),
            "largest_free_kb": round(snap.largest_free_block / 1024, 1),
        }
        self.history_records.append(record)
        if len(self.history_records) > 60:
            self.history_records.pop(0)

    def step(self, count: int = 1) -> Dict[str, Any]:
        """Steps through workload trace events and applies the selected management policy."""
        results = []
        for _ in range(count):
            if self.trace_cursor >= len(self.trace_events):
                # Loop workload trace
                self.trace_cursor = 0

            ev = self.trace_events[self.trace_cursor]
            self.trace_cursor += 1
            self.current_tick += 1
            self.recent_events.append(ev)
            if len(self.recent_events) > 40:
                self.recent_events.pop(0)

            event_msg = ""
            if ev.op == AllocationOp.ALLOC:
                addr = self.allocator.allocate(
                    size=ev.size,
                    timestamp=float(self.current_tick),
                    lifetime=ev.lifetime,
                    tag=ev.tag,
                    call_stack_hash=ev.call_stack_hash,
                    alloc_id=ev.address
                )
                if addr is not None:
                    event_msg = f"Allocated {ev.size // 1024} KB ({ev.tag or 'block'}) @ 0x{addr:05X}"
                else:
                    event_msg = f"FAILED Allocation of {ev.size // 1024} KB ({ev.tag}) — OUT OF CONTIGUOUS MEMORY!"
            else:
                freed = self.allocator.free_by_id(ev.address, float(self.current_tick))
                event_msg = f"Freed alloc #{ev.address}" if freed else f"Free failed for #{ev.address}"

            self.event_log.append({
                "tick": self.current_tick,
                "type": ev.op.name,
                "msg": event_msg,
                "is_error": "FAILED" in event_msg
            })
            if len(self.event_log) > 30:
                self.event_log.pop(0)

            self._record_snapshot(float(self.current_tick))

            # Policy check: Should we reorganize?
            current_snap = self.snapshot_history[-1]
            predicted_f1 = self.last_predictions[2] # t+10 horizon
            
            should_compact = self.policy.decide(
                tick=self.current_tick,
                current_f1=current_snap.f1_largest_free_ratio,
                predicted_f1=predicted_f1,
                blocks=self.allocator.blocks
            )

            # Record policy evaluation metrics
            est_cost = 0.0
            net_benefit = 0.0
            if isinstance(self.policy, CostAwarePredictivePolicy):
                eval_res = self.policy.optimizer.evaluate(self.current_tick, predicted_f1, self.allocator.blocks)
                est_cost = eval_res.estimated_cost
                net_benefit = eval_res.net_benefit
                raw_benefit = eval_res.expected_fragmentation_reduction * self.policy.optimizer.benefit_scale
                self.last_policy_evaluation = {
                    "should_intervene": eval_res.should_intervene,
                    "expected_benefit": round(raw_benefit, 2),
                    "compaction_cost": round(eval_res.estimated_cost, 2),
                    "net_benefit": round(eval_res.net_benefit, 2),
                    "pause_ms": round(eval_res.estimated_latency_ms, 3),
                    "action": eval_res.action_name
                }

            if should_compact:
                report = self.allocator.compact(float(self.current_tick))
                self.last_compaction_report = {
                    "tick": self.current_tick,
                    "blocks_moved": report.blocks_moved,
                    "bytes_moved_kb": round(report.bytes_moved / 1024, 1),
                    "pause_ms": round(report.pause_latency_ms, 2),
                    "total_cost": round(report.total_cost, 2),
                    "f1_before": round(report.f1_before, 4),
                    "f1_after": round(report.f1_after, 4)
                }
                self.event_log.append({
                    "tick": self.current_tick,
                    "type": "COMPACT",
                    "msg": f"PROACTIVE COMPACTION: Moved {report.blocks_moved} blocks ({report.bytes_moved // 1024} KB), F1 {report.f1_before:.2f} -> {report.f1_after:.2f}, Cost: {report.total_cost:.1f}",
                    "is_compact": True
                })
                self._record_snapshot(float(self.current_tick))

        return self.get_state()

    def generate_fragmentation(self, pattern: str = "checkerboard") -> Dict[str, Any]:
        """
        Intentionally injects severe memory fragmentation:
        Allocates multiple contiguous blocks and selectively frees every alternate one,
        leaving disjoint holes that prevent larger allocations.
        """
        initial_snap = self.allocator.get_snapshot()
        self.current_tick += 1
        
        # Calculate how many blocks to span across available free space
        snap = self.allocator.get_snapshot()
        free_bytes = snap.total_free_bytes
        block_size = 32 * 1024 # 32 KB blocks
        num_blocks = max(4, free_bytes // block_size)

        alloc_ids = []
        for i in range(num_blocks):
            self.custom_alloc_counter += 1
            aid = self.custom_alloc_counter
            tag = "frag_seed_A" if i % 2 == 0 else "frag_seed_B"
            addr = self.allocator.allocate(
                size=block_size,
                timestamp=float(self.current_tick),
                tag=tag,
                alloc_id=aid
            )
            if addr is not None:
                alloc_ids.append(aid)

        # Now free alternating blocks to create isolated holes
        freed_count = 0
        for idx in range(0, len(alloc_ids), 2):
            aid = alloc_ids[idx]
            if self.allocator.free_by_id(aid, float(self.current_tick)):
                freed_count += 1

        self.event_log.append({
            "tick": self.current_tick,
            "type": "INJECT",
            "msg": f"INJECTED FRAGMENTATION: Created {len(alloc_ids)} blocks & freed {freed_count} alternating holes ({block_size//1024}KB each)",
            "is_inject": True
        })

        self._record_snapshot(float(self.current_tick))
        return self.get_state()

    def compact_manual(self) -> Dict[str, Any]:
        """Triggers manual compaction."""
        self.current_tick += 1
        report = self.allocator.compact(float(self.current_tick))
        self.last_compaction_report = {
            "tick": self.current_tick,
            "blocks_moved": report.blocks_moved,
            "bytes_moved_kb": round(report.bytes_moved / 1024, 1),
            "pause_ms": round(report.pause_latency_ms, 2),
            "total_cost": round(report.total_cost, 2),
            "f1_before": round(report.f1_before, 4),
            "f1_after": round(report.f1_after, 4)
        }
        self.event_log.append({
            "tick": self.current_tick,
            "type": "COMPACT",
            "msg": f"COMPACTION EXECUTED: Reorganized {report.blocks_moved} blocks ({report.bytes_moved // 1024} KB), F1 {report.f1_before:.2f} -> {report.f1_after:.2f}",
            "is_compact": True
        })
        self._record_snapshot(float(self.current_tick))
        return self.get_state()

    def test_allocation(self, size_kb: int = 64) -> Dict[str, Any]:
        """Attempts an allocation of size_kb to prove whether fragmentation causes allocation failure."""
        self.current_tick += 1
        self.custom_alloc_counter += 1
        size_bytes = size_kb * 1024
        aid = self.custom_alloc_counter
        addr = self.allocator.allocate(size_bytes, float(self.current_tick), tag="test_payload", alloc_id=aid)

        if addr is not None:
            msg = f"SUCCESS: Allocated {size_kb} KB block @ 0x{addr:05X}"
            is_err = False
        else:
            snap = self.allocator.get_snapshot()
            msg = (
                f"FAILED: Cannot allocate {size_kb} KB! (Total free = {snap.total_free_bytes//1024} KB, "
                f"but largest contiguous hole is only {snap.largest_free_block//1024} KB). Heap is FRAGMENTED!"
            )
            is_err = True

        self.event_log.append({
            "tick": self.current_tick,
            "type": "TEST",
            "msg": msg,
            "is_error": is_err
        })
        self._record_snapshot(float(self.current_tick))
        return self.get_state()

    def reset(self) -> Dict[str, Any]:
        self._initialize_workload(self.workload_family)
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        snap = self.allocator.get_snapshot(float(self.current_tick))
        blocks_data = []
        for b in self.allocator.blocks:
            blocks_data.append({
                "start": b.start_address,
                "end": b.start_address + b.size,
                "size": b.size,
                "size_kb": round(b.size / 1024, 1),
                "is_free": b.is_free,
                "tag": b.tag if not b.is_free else "FREE HOLE",
                "alloc_id": b.alloc_id
            })

        return {
            "tick": self.current_tick,
            "total_capacity": self.heap_capacity,
            "total_capacity_kb": self.heap_capacity // 1024,
            "allocator_policy": self.allocator_policy.value,
            "management_policy": self.management_policy_name,
            "workload_family": self.workload_family.value,
            "snapshot": snap.to_dict(),
            "blocks": blocks_data,
            "predictions": {
                "t1": round(self.last_predictions[0], 4),
                "t5": round(self.last_predictions[1], 4),
                "t10": round(self.last_predictions[2], 4),
                "t30": round(self.last_predictions[3], 4),
            },
            "history": self.history_records,
            "event_log": self.event_log,
            "last_compaction": self.last_compaction_report,
            "last_evaluation": self.last_policy_evaluation,
            "stats": {
                "total_compactions": len(self.allocator.compaction_history),
                "failures_count": self.allocator.failures_count,
                "total_blocks_count": len(self.allocator.blocks),
                "free_blocks_count": snap.free_block_count,
                "allocated_blocks_count": snap.allocated_block_count,
            }
        }


# Global engine instance
ENGINE = LiveDemoEngine()


class LiveDemoHTTPHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Any, code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
            if os.path.exists(html_path):
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "HTML frontend not found")
        elif self.path == "/api/state":
            self._send_json(ENGINE.get_state())
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            req = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception:
            req = {}

        if self.path == "/api/step":
            count = int(req.get("count", 1))
            self._send_json(ENGINE.step(count))
        elif self.path == "/api/generate_fragmentation":
            pattern = req.get("pattern", "checkerboard")
            self._send_json(ENGINE.generate_fragmentation(pattern))
        elif self.path == "/api/compact":
            self._send_json(ENGINE.compact_manual())
        elif self.path == "/api/test_allocation":
            size_kb = int(req.get("size_kb", 64))
            self._send_json(ENGINE.test_allocation(size_kb))
        elif self.path == "/api/reset":
            self._send_json(ENGINE.reset())
        elif self.path == "/api/configure":
            if "workload" in req:
                fam = WorkloadFamily(req["workload"])
                ENGINE._initialize_workload(fam)
            if "allocator_policy" in req:
                ENGINE.set_allocator_policy(req["allocator_policy"])
            if "management_policy" in req:
                ENGINE.set_management_policy(req["management_policy"])
            self._send_json(ENGINE.get_state())
        else:
            self.send_error(404, "Unknown API route")


def run_server(port: int = 8050):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, LiveDemoHTTPHandler)
    print(f"\n=======================================================")
    print(f"🚀 PREFRAG Live Memory Fragmentation Demo Server Running")
    print(f"👉 Local URL: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down demo server.")
        httpd.server_close()


if __name__ == "__main__":
    port = 8050
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
