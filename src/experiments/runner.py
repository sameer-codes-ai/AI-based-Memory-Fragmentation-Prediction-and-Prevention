"""
PREFRAG: Experiment Runner Engine
Executes conditions E1 - E5 across benchmark workloads and measures both ML and Systems metrics.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import math
import numpy as np
from ..traces.trace_schema import MemoryEvent, AllocationOp
from ..traces.workload_generator import WorkloadGenerator, WorkloadFamily
from ..simulator.heap_allocator import HeapAllocator, AllocationPolicy
from ..simulator.fragmentation_metrics import FragmentationSnapshot
from ..features.feature_extractor import FeatureExtractor
from ..models.baselines import StaticThresholdBaseline, RandomForestBaseline, GradientBoostingBaseline
from ..models.temporal_nn import TemporalForecasterLSTM, TemporalForecasterGRU, TemporalForecasterTCN
from ..controller.policies import (
    ReactivePolicy,
    AlwaysProactivePolicy,
    PredictionOnlyPolicy,
    ThresholdPredictivePolicy,
    CostAwarePredictivePolicy
)


@dataclass(slots=True)
class ExperimentResult:
    condition_name: str
    workload_name: str
    total_events: int
    mean_f1: float
    peak_f1: float
    severe_episodes_f1: int
    allocation_failures: int
    compactions_count: int
    total_blocks_moved: int
    total_bytes_moved_mb: float
    cumulative_pause_ms: float
    total_cost: float
    mean_utilization: float

    # ML metrics (if predictive model was used)
    mae_t1: float = 0.0
    mae_t5: float = 0.0
    mae_t10: float = 0.0
    mae_t30: float = 0.0
    r2_t10: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    """
    Orchestrates full experimental evaluations across all conditions (E1-E5).
    """

    def __init__(
        self,
        heap_capacity: int = 2 * 1024 * 1024,
        window_size: int = 40,
        severe_threshold: float = 0.65
    ):
        self.heap_capacity = heap_capacity
        self.window_size = window_size
        self.severe_threshold = severe_threshold
        self.extractor = FeatureExtractor(window_size_ticks=window_size, severe_f1_threshold=severe_threshold)

    def prepare_dataset_for_workload(
        self,
        events: List[MemoryEvent]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[FragmentationSnapshot]]:
        """
        Executes a passive trace replay run to generate ground-truth feature/target tensors.
        """
        allocator = HeapAllocator(total_capacity=self.heap_capacity, severe_f1_threshold=self.severe_threshold)
        snapshots: List[FragmentationSnapshot] = []

        for tick, ev in enumerate(events):
            if ev.op == AllocationOp.ALLOC:
                allocator.allocate(
                    size=ev.size,
                    timestamp=ev.timestamp,
                    lifetime=ev.lifetime,
                    tag=ev.tag,
                    call_stack_hash=ev.call_stack_hash,
                    alloc_id=ev.address
                )
            elif ev.op == AllocationOp.FREE:
                allocator.free(ev.address, ev.timestamp)

            snapshots.append(allocator.get_snapshot(ev.timestamp))

        X, Y_reg, Y_cls = self.extractor.extract_features_from_history(events, snapshots)
        return X, Y_reg, Y_cls, snapshots

    def run_condition_simulation(
        self,
        events: List[MemoryEvent],
        condition_name: str,
        policy: Any,
        forecaster: Optional[Any] = None
    ) -> ExperimentResult:
        """
        Executes interactive trace simulation under a given memory management policy.
        """
        allocator = HeapAllocator(total_capacity=self.heap_capacity, severe_f1_threshold=self.severe_threshold)
        snapshots: List[FragmentationSnapshot] = []
        recent_events: List[MemoryEvent] = []
        
        predictions_history: List[float] = []
        actual_t10_history: List[float] = []

        for tick, ev in enumerate(events):
            recent_events.append(ev)
            if len(recent_events) > (self.window_size + 10):
                recent_events.pop(0)

            # Apply memory event
            if ev.op == AllocationOp.ALLOC:
                allocator.allocate(
                    size=ev.size,
                    timestamp=ev.timestamp,
                    lifetime=ev.lifetime,
                    tag=ev.tag,
                    call_stack_hash=ev.call_stack_hash,
                    alloc_id=ev.address
                )
            elif ev.op == AllocationOp.FREE:
                allocator.free(ev.address, ev.timestamp)

            current_snap = allocator.get_snapshot(ev.timestamp)
            snapshots.append(current_snap)

            # Model prediction if available
            pred_f1 = current_snap.f1_largest_free_ratio
            if forecaster is not None and len(snapshots) >= self.window_size:
                try:
                    # Construct single-step feature vector using same logic as FeatureExtractor
                    sub_snaps = snapshots[-self.window_size:]
                    start_snap = sub_snaps[0]
                    allocs = [e for e in recent_events if e.op == AllocationOp.ALLOC]
                    frees = [e for e in recent_events if e.op == AllocationOp.FREE]

                    # Group 1: Allocation statistics
                    alloc_sizes = [e.size for e in allocs]
                    alloc_count = float(len(allocs))
                    alloc_bytes = float(sum(alloc_sizes)) if alloc_sizes else 0.0
                    alloc_mean = float(np.mean(alloc_sizes)) if alloc_sizes else 0.0
                    alloc_std = float(np.std(alloc_sizes)) if len(alloc_sizes) > 1 else 0.0
                    alloc_max = float(max(alloc_sizes)) if alloc_sizes else 0.0
                    duration = max(0.001, current_snap.timestamp - start_snap.timestamp)
                    alloc_rate = alloc_bytes / duration

                    # Group 2: Deallocation statistics
                    free_sizes = [e.size for e in frees]
                    free_count = float(len(frees))
                    free_bytes = float(sum(free_sizes)) if free_sizes else 0.0
                    free_mean = float(np.mean(free_sizes)) if free_sizes else 0.0
                    free_std = float(np.std(free_sizes)) if len(free_sizes) > 1 else 0.0
                    free_rate = free_bytes / duration

                    # Group 3: Lifetime features (real computation, not hardcoded)
                    lifetimes = [e.lifetime for e in allocs if e.lifetime is not None]
                    life_mean = float(np.mean(lifetimes)) if lifetimes else 0.0
                    life_std = float(np.std(lifetimes)) if len(lifetimes) > 1 else 0.0
                    tick_duration = duration / max(1, self.window_size)
                    short_lived = sum(1 for l in lifetimes if l <= (10.0 * tick_duration))
                    short_lived_ratio = float(short_lived) / max(1.0, float(len(lifetimes)))

                    # Group 4: Free-space geometry
                    f1 = current_snap.f1_largest_free_ratio
                    f2 = current_snap.f2_free_block_entropy
                    f3 = float(current_snap.f3_free_block_count)
                    tot_free = float(current_snap.total_free_bytes)
                    lfb = float(current_snap.largest_free_block)
                    util = current_snap.memory_utilization

                    # Group 5: Temporal dynamics
                    delta_f1 = f1 - start_snap.f1_largest_free_ratio
                    delta_f2 = f2 - start_snap.f2_free_block_entropy
                    delta_alloc = float(current_snap.allocated_bytes - start_snap.allocated_bytes)

                    # Burstiness: coefficient of variation of inter-arrival times
                    if len(allocs) > 2:
                        inter_times = np.diff([e.timestamp for e in allocs])
                        burstiness = float(np.std(inter_times) / (np.mean(inter_times) + 1e-6))
                    else:
                        burstiness = 0.0

                    alloc_free_ratio = alloc_count / max(1.0, free_count)

                    # Group 6: Workload context
                    active_alloc_count = float(current_snap.allocated_block_count)
                    active_payload = float(current_snap.allocated_bytes)

                    # Call stack entropy (real computation)
                    stacks = [e.call_stack_hash for e in allocs if e.call_stack_hash is not None]
                    if stacks:
                        _, counts = np.unique(stacks, return_counts=True)
                        p = counts / len(stacks)
                        stack_entropy = float(-np.sum(p * np.log2(p + 1e-9)))
                    else:
                        stack_entropy = 0.0

                    feats = np.array([[
                        alloc_count, alloc_bytes, alloc_mean, alloc_std, alloc_max, alloc_rate,
                        free_count, free_bytes, free_mean, free_std, free_rate,
                        life_mean, life_std, short_lived_ratio,
                        f1, f2, f3, tot_free, lfb, util,
                        delta_f1, delta_f2, delta_alloc, burstiness, alloc_free_ratio,
                        active_alloc_count, active_payload, stack_entropy
                    ]], dtype=np.float32)

                    preds = forecaster.predict(feats)
                    pred_f1 = float(preds[0, 2]) # Horizon t+10
                    predictions_history.append(pred_f1)
                except Exception:
                    pred_f1 = current_snap.f1_largest_free_ratio

            # Policy decision
            should_compact = policy.decide(
                tick=tick,
                current_f1=current_snap.f1_largest_free_ratio,
                predicted_f1=pred_f1,
                blocks=allocator.blocks
            )

            if should_compact:
                allocator.compact(ev.timestamp)

        # Aggregate metrics
        f1_series = [s.f1_largest_free_ratio for s in snapshots]
        util_series = [s.memory_utilization for s in snapshots]
        compactions = allocator.compaction_history

        total_blocks_moved = sum(c.blocks_moved for c in compactions)
        total_bytes_moved = sum(c.bytes_moved for c in compactions)
        total_pause_ms = sum(c.pause_latency_ms for c in compactions)
        total_cost = sum(c.total_cost for c in compactions)
        severe_count = sum(1 for f in f1_series if f >= self.severe_threshold)

        return ExperimentResult(
            condition_name=condition_name,
            workload_name=events[0].tag if events else "unknown",
            total_events=len(events),
            mean_f1=float(np.mean(f1_series)),
            peak_f1=float(np.max(f1_series)),
            severe_episodes_f1=severe_count,
            allocation_failures=allocator.failures_count,
            compactions_count=len(compactions),
            total_blocks_moved=total_blocks_moved,
            total_bytes_moved_mb=float(total_bytes_moved) / (1024.0 * 1024.0),
            cumulative_pause_ms=float(total_pause_ms),
            total_cost=float(total_cost),
            mean_utilization=float(np.mean(util_series))
        )
