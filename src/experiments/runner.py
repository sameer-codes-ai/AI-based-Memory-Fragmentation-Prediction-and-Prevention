"""
PREFRAG: Experiment Runner Engine
Executes conditions E1 - E5 across benchmark workloads and measures both ML and Systems metrics.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import math
import logging
import numpy as np
from sklearn.metrics import r2_score
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
        events: List[MemoryEvent],
        policy: AllocationPolicy = AllocationPolicy.FIRST_FIT
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[FragmentationSnapshot]]:
        """
        Executes a passive trace replay run to generate ground-truth feature/target tensors.
        """
        allocator = HeapAllocator(total_capacity=self.heap_capacity, policy=policy, severe_f1_threshold=self.severe_threshold)
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
        forecaster: Optional[Any] = None,
        allocation_policy: AllocationPolicy = AllocationPolicy.FIRST_FIT
    ) -> ExperimentResult:
        """
        Executes interactive trace simulation under a given memory management policy.
        """
        allocator = HeapAllocator(total_capacity=self.heap_capacity, policy=allocation_policy, severe_f1_threshold=self.severe_threshold)
        snapshots: List[FragmentationSnapshot] = []
        recent_events: List[MemoryEvent] = []
        feature_seq_history: List[np.ndarray] = []
        
        predictions_by_tick: Dict[int, Tuple[float, float, float, float]] = {}

        for tick, ev in enumerate(events):
            recent_events.append(ev)
            if len(recent_events) > (self.window_size + 20):
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
                    feat_vec = self.extractor.extract_single_step(recent_events, snapshots)
                    feature_seq_history.append(feat_vec.to_array())
                    if len(feature_seq_history) > 30:
                        feature_seq_history.pop(0)

                    feats_seq = np.array(feature_seq_history, dtype=np.float32)
                    preds = forecaster.predict(feats_seq)
                    pred_horizons = preds[-1]
                    pred_f1 = float(pred_horizons[2]) # Horizon t+10
                    predictions_by_tick[tick] = (
                        float(pred_horizons[0]),
                        float(pred_horizons[1]),
                        float(pred_horizons[2]),
                        float(pred_horizons[3])
                    )
                except Exception as exc:
                    logging.debug("Forecaster prediction exception at tick %d: %s", tick, exc)
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

        # Calculate online ML prediction accuracy if forecaster was used
        mae_t1 = 0.0
        mae_t5 = 0.0
        mae_t10 = 0.0
        mae_t30 = 0.0
        r2_t10 = 0.0

        if predictions_by_tick:
            n_snaps = len(snapshots)
            d1 = [abs(p[0] - snapshots[t + 1].f1_largest_free_ratio) for t, p in predictions_by_tick.items() if t + 1 < n_snaps]
            d5 = [abs(p[1] - snapshots[t + 5].f1_largest_free_ratio) for t, p in predictions_by_tick.items() if t + 5 < n_snaps]
            d10 = [abs(p[2] - snapshots[t + 10].f1_largest_free_ratio) for t, p in predictions_by_tick.items() if t + 10 < n_snaps]
            d30 = [abs(p[3] - snapshots[t + 30].f1_largest_free_ratio) for t, p in predictions_by_tick.items() if t + 30 < n_snaps]

            if d1:
                mae_t1 = float(np.mean(d1))
            if d5:
                mae_t5 = float(np.mean(d5))
            if d10:
                mae_t10 = float(np.mean(d10))
            if d30:
                mae_t30 = float(np.mean(d30))

            preds_10 = [p[2] for t, p in predictions_by_tick.items() if t + 10 < n_snaps]
            actuals_10 = [snapshots[t + 10].f1_largest_free_ratio for t, p in predictions_by_tick.items() if t + 10 < n_snaps]
            if len(actuals_10) > 1 and np.var(actuals_10) > 1e-8:
                try:
                    r2_t10 = float(r2_score(actuals_10, preds_10))
                except Exception:
                    r2_t10 = 0.0

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
            mean_utilization=float(np.mean(util_series)),
            mae_t1=round(mae_t1, 4),
            mae_t5=round(mae_t5, 4),
            mae_t10=round(mae_t10, 4),
            mae_t30=round(mae_t30, 4),
            r2_t10=round(r2_t10, 4)
        )
