"""
PREFRAG: Temporal Feature Engineering Pipeline
Extracts 6 structured feature groups across sliding temporal windows and prepares multi-horizon forecast targets.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import numpy as np
from ..traces.trace_schema import MemoryEvent, AllocationOp
from ..simulator.fragmentation_metrics import FragmentationSnapshot


@dataclass(slots=True)
class TemporalFeatureVector:
    timestamp: float
    # Group 1: Allocation statistics
    alloc_count: float
    alloc_bytes_total: float
    alloc_size_mean: float
    alloc_size_std: float
    alloc_size_max: float
    alloc_rate: float

    # Group 2: Deallocation statistics
    free_count: float
    free_bytes_total: float
    free_size_mean: float
    free_size_std: float
    free_rate: float

    # Group 3: Lifetime features
    lifetime_mean: float
    lifetime_std: float
    short_lived_ratio: float

    # Group 4: Free-space / Heap geometry
    current_f1: float
    current_f2: float
    current_f3: float
    total_free_bytes: float
    largest_free_block: float
    memory_utilization: float

    # Group 5: Temporal dynamics
    delta_f1: float
    delta_f2: float
    delta_allocated_bytes: float
    alloc_burstiness: float
    alloc_to_free_ratio: float

    # Group 6: Workload context
    active_alloc_count: float
    active_payload_bytes: float
    call_stack_entropy: float

    # Group 7: Lag and trend dynamics (6 features)
    f1_lag_1: float = 0.0
    f1_lag_5: float = 0.0
    f1_lag_10: float = 0.0
    f1_trend_5: float = 0.0
    f1_volatility_10: float = 0.0
    alloc_rate_trend: float = 0.0

    # Multi-horizon ground truth targets (None during live inference)
    target_f1_t1: Optional[float] = None
    target_f1_t5: Optional[float] = None
    target_f1_t10: Optional[float] = None
    target_f1_t30: Optional[float] = None
    target_severe_warning: Optional[int] = None # 1 if F1(t+k) >= F_critical else 0

    def to_array(self, include_targets: bool = False) -> np.ndarray:
        feats = [
            self.alloc_count, self.alloc_bytes_total, self.alloc_size_mean, self.alloc_size_std, self.alloc_size_max, self.alloc_rate,
            self.free_count, self.free_bytes_total, self.free_size_mean, self.free_size_std, self.free_rate,
            self.lifetime_mean, self.lifetime_std, self.short_lived_ratio,
            self.current_f1, self.current_f2, self.current_f3, self.total_free_bytes, self.largest_free_block, self.memory_utilization,
            self.delta_f1, self.delta_f2, self.delta_allocated_bytes, self.alloc_burstiness, self.alloc_to_free_ratio,
            self.active_alloc_count, self.active_payload_bytes, self.call_stack_entropy,
            self.f1_lag_1, self.f1_lag_5, self.f1_lag_10, self.f1_trend_5, self.f1_volatility_10, self.alloc_rate_trend
        ]
        return np.array(feats, dtype=np.float32)

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "alloc_count", "alloc_bytes_total", "alloc_size_mean", "alloc_size_std", "alloc_size_max", "alloc_rate",
            "free_count", "free_bytes_total", "free_size_mean", "free_size_std", "free_rate",
            "lifetime_mean", "lifetime_std", "short_lived_ratio",
            "current_f1", "current_f2", "current_f3", "total_free_bytes", "largest_free_block", "memory_utilization",
            "delta_f1", "delta_f2", "delta_allocated_bytes", "alloc_burstiness", "alloc_to_free_ratio",
            "active_alloc_count", "active_payload_bytes", "call_stack_entropy",
            "f1_lag_1", "f1_lag_5", "f1_lag_10", "f1_trend_5", "f1_volatility_10", "alloc_rate_trend"
        ]

    @classmethod
    def feature_groups(cls) -> Dict[str, List[int]]:
        return {
            "allocation": list(range(0, 6)),
            "deallocation": list(range(6, 11)),
            "lifetime": list(range(11, 14)),
            "freespace": list(range(14, 20)),
            "temporal": list(range(20, 25)),
            "workload": list(range(25, 28)),
            "lag_trend": list(range(28, 34))
        }


class FeatureExtractor:
    """
    Constructs feature vectors from trace windows and simulator snapshots.
    """

    def __init__(
        self,
        window_size_ticks: int = 50,
        severe_f1_threshold: float = 0.65,
        horizons: Tuple[int, int, int, int] = (1, 5, 10, 30)
    ):
        self.window_size = window_size_ticks
        self.severe_f1_threshold = severe_f1_threshold
        self.horizons = horizons

    def extract_features_from_history(
        self,
        events: List[MemoryEvent],
        snapshots: List[FragmentationSnapshot]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts temporal feature matrix X, regression targets Y_reg, and classification targets Y_cls.
        Returns:
          X: shape (N, feature_dim)
          Y_reg: shape (N, 4) -> [t+1, t+5, t+10, t+30]
          Y_cls: shape (N,) -> binary indicator for severe risk at horizon t+10
        """
        if len(snapshots) < self.window_size + max(self.horizons):
            raise ValueError(f"Need at least {self.window_size + max(self.horizons)} timesteps to build dataset")

        feature_vectors: List[TemporalFeatureVector] = []
        max_h = max(self.horizons)
        n_samples = len(snapshots) - max_h

        # Pre-index events by timestamp
        time_to_events: Dict[float, List[MemoryEvent]] = {}
        for ev in events:
            time_to_events.setdefault(ev.timestamp, []).append(ev)

        for t in range(self.window_size, n_samples):
            current_snap = snapshots[t]
            start_snap = snapshots[t - self.window_size]

            # Gather unique events in window [t - window_size, t] without duplicate counting
            seen_timestamps = set()
            window_events: List[MemoryEvent] = []
            for past_t in range(t - self.window_size, t + 1):
                past_time = snapshots[past_t].timestamp
                if past_time in time_to_events and past_time not in seen_timestamps:
                    window_events.extend(time_to_events[past_time])
                    seen_timestamps.add(past_time)

            allocs = [e for e in window_events if e.op == AllocationOp.ALLOC]
            frees = [e for e in window_events if e.op == AllocationOp.FREE]

            # Group 1: Allocations
            alloc_count = float(len(allocs))
            alloc_sizes = [e.size for e in allocs]
            alloc_bytes = float(sum(alloc_sizes))
            alloc_mean = float(np.mean(alloc_sizes)) if alloc_sizes else 0.0
            alloc_std = float(np.std(alloc_sizes)) if len(alloc_sizes) > 1 else 0.0
            alloc_max = float(max(alloc_sizes)) if alloc_sizes else 0.0
            duration = max(0.001, current_snap.timestamp - start_snap.timestamp)
            alloc_rate = alloc_bytes / duration

            # Group 2: Deallocations
            free_count = float(len(frees))
            free_sizes = [e.size for e in frees]
            free_bytes = float(sum(free_sizes))
            free_mean = float(np.mean(free_sizes)) if free_sizes else 0.0
            free_std = float(np.std(free_sizes)) if len(free_sizes) > 1 else 0.0
            free_rate = free_bytes / duration

            # Group 3: Lifetimes (bounded by observation duration to avoid oracle leakage)
            lifetimes = [min(e.lifetime, duration) for e in allocs if e.lifetime is not None]
            life_mean = float(np.mean(lifetimes)) if lifetimes else 0.0
            life_std = float(np.std(lifetimes)) if len(lifetimes) > 1 else 0.0
            short_lived = sum(1 for l in lifetimes if l <= (10.0 * (duration / max(1, self.window_size))))
            short_ratio = float(short_lived) / max(1.0, float(len(lifetimes)))

            # Group 4: Free-space
            f1 = current_snap.f1_largest_free_ratio
            f2 = current_snap.f2_free_block_entropy
            f3 = float(current_snap.f3_free_block_count)
            tot_free = float(current_snap.total_free_bytes)
            lfb = float(current_snap.largest_free_block)
            util = current_snap.memory_utilization

            # Group 5: Dynamics
            delta_f1 = f1 - start_snap.f1_largest_free_ratio
            delta_f2 = f2 - start_snap.f2_free_block_entropy
            delta_alloc = float(current_snap.allocated_bytes - start_snap.allocated_bytes)
            
            # Burstiness: std of inter-arrival timestamps
            if len(allocs) > 2:
                inter_times = np.diff([e.timestamp for e in allocs])
                burstiness = float(np.std(inter_times) / (np.mean(inter_times) + 1e-6))
            else:
                burstiness = 0.0

            alloc_free_ratio = float(alloc_count) / max(1.0, float(free_count))

            # Group 6: Workload Context
            active_alloc_count = float(current_snap.allocated_block_count)
            active_payload = float(current_snap.allocated_bytes)
            
            # Call stack entropy
            stacks = [e.call_stack_hash for e in allocs if e.call_stack_hash is not None]
            if stacks:
                _, counts = np.unique(stacks, return_counts=True)
                p = counts / len(stacks)
                stack_entropy = float(-np.sum(p * np.log2(p + 1e-9)))
            else:
                stack_entropy = 0.0

            # Group 7: Lag and trend dynamics
            f1_lag_1 = snapshots[t - 1].f1_largest_free_ratio if t >= 1 else f1
            f1_lag_5 = snapshots[t - 5].f1_largest_free_ratio if t >= 5 else f1
            f1_lag_10 = snapshots[t - 10].f1_largest_free_ratio if t >= 10 else f1
            f1_trend_5 = float(f1 - f1_lag_5) / 5.0
            
            window_10_f1 = [snapshots[t - k].f1_largest_free_ratio for k in range(min(10, t + 1))]
            f1_volatility_10 = float(np.std(window_10_f1)) if len(window_10_f1) > 1 else 0.0
            alloc_rate_trend = float(alloc_rate - (alloc_bytes / max(0.001, duration * 2.0)))

            # Multi-horizon ground truth
            t1_val = snapshots[t + self.horizons[0]].f1_largest_free_ratio
            t5_val = snapshots[t + self.horizons[1]].f1_largest_free_ratio
            t10_val = snapshots[t + self.horizons[2]].f1_largest_free_ratio
            t30_val = snapshots[t + self.horizons[3]].f1_largest_free_ratio
            severe_warn = 1 if max(t1_val, t5_val, t10_val) >= self.severe_f1_threshold else 0

            vec = TemporalFeatureVector(
                timestamp=current_snap.timestamp,
                alloc_count=alloc_count,
                alloc_bytes_total=alloc_bytes,
                alloc_size_mean=alloc_mean,
                alloc_size_std=alloc_std,
                alloc_size_max=alloc_max,
                alloc_rate=alloc_rate,
                free_count=free_count,
                free_bytes_total=free_bytes,
                free_size_mean=free_mean,
                free_size_std=free_std,
                free_rate=free_rate,
                lifetime_mean=life_mean,
                lifetime_std=life_std,
                short_lived_ratio=short_ratio,
                current_f1=f1,
                current_f2=f2,
                current_f3=f3,
                total_free_bytes=tot_free,
                largest_free_block=lfb,
                memory_utilization=util,
                delta_f1=delta_f1,
                delta_f2=delta_f2,
                delta_allocated_bytes=delta_alloc,
                alloc_burstiness=burstiness,
                alloc_to_free_ratio=alloc_free_ratio,
                active_alloc_count=active_alloc_count,
                active_payload_bytes=active_payload,
                call_stack_entropy=stack_entropy,
                f1_lag_1=f1_lag_1,
                f1_lag_5=f1_lag_5,
                f1_lag_10=f1_lag_10,
                f1_trend_5=f1_trend_5,
                f1_volatility_10=f1_volatility_10,
                alloc_rate_trend=alloc_rate_trend,
                target_f1_t1=t1_val,
                target_f1_t5=t5_val,
                target_f1_t10=t10_val,
                target_f1_t30=t30_val,
                target_severe_warning=severe_warn
            )
            feature_vectors.append(vec)

        X = np.array([v.to_array() for v in feature_vectors], dtype=np.float32)
        Y_reg = np.array([[v.target_f1_t1, v.target_f1_t5, v.target_f1_t10, v.target_f1_t30] for v in feature_vectors], dtype=np.float32)
        Y_cls = np.array([v.target_severe_warning for v in feature_vectors], dtype=np.int64)

        return X, Y_reg, Y_cls

    def extract_single_step(
        self,
        recent_events: List[MemoryEvent],
        snapshots: List[FragmentationSnapshot]
    ) -> TemporalFeatureVector:
        """
        Extracts a single 34-dimensional feature vector for online inference.
        """
        if len(snapshots) < 2:
            current_snap = snapshots[-1] if snapshots else None
            f1 = current_snap.f1_largest_free_ratio if current_snap else 0.0
            return TemporalFeatureVector(
                timestamp=current_snap.timestamp if current_snap else 0.0,
                alloc_count=0.0, alloc_bytes_total=0.0, alloc_size_mean=0.0, alloc_size_std=0.0, alloc_size_max=0.0, alloc_rate=0.0,
                free_count=0.0, free_bytes_total=0.0, free_size_mean=0.0, free_size_std=0.0, free_rate=0.0,
                lifetime_mean=0.0, lifetime_std=0.0, short_lived_ratio=0.0,
                current_f1=f1, current_f2=0.0, current_f3=1.0, total_free_bytes=0.0, largest_free_block=0.0, memory_utilization=0.0,
                delta_f1=0.0, delta_f2=0.0, delta_allocated_bytes=0.0, alloc_burstiness=0.0, alloc_to_free_ratio=1.0,
                active_alloc_count=0.0, active_payload_bytes=0.0, call_stack_entropy=0.0,
                f1_lag_1=f1, f1_lag_5=f1, f1_lag_10=f1, f1_trend_5=0.0, f1_volatility_10=0.0, alloc_rate_trend=0.0
            )

        current_snap = snapshots[-1]
        w_size = min(self.window_size, len(snapshots) - 1)
        start_snap = snapshots[-w_size - 1]

        allocs = [e for e in recent_events if e.op == AllocationOp.ALLOC]
        frees = [e for e in recent_events if e.op == AllocationOp.FREE]

        alloc_count = float(len(allocs))
        alloc_sizes = [e.size for e in allocs]
        alloc_bytes = float(sum(alloc_sizes))
        alloc_mean = float(np.mean(alloc_sizes)) if alloc_sizes else 0.0
        alloc_std = float(np.std(alloc_sizes)) if len(alloc_sizes) > 1 else 0.0
        alloc_max = float(max(alloc_sizes)) if alloc_sizes else 0.0
        duration = max(0.001, current_snap.timestamp - start_snap.timestamp)
        alloc_rate = alloc_bytes / duration

        free_count = float(len(frees))
        free_sizes = [e.size for e in frees]
        free_bytes = float(sum(free_sizes))
        free_mean = float(np.mean(free_sizes)) if free_sizes else 0.0
        free_std = float(np.std(free_sizes)) if len(free_sizes) > 1 else 0.0
        free_rate = free_bytes / duration

        lifetimes = [min(e.lifetime, duration) for e in allocs if e.lifetime is not None]
        life_mean = float(np.mean(lifetimes)) if lifetimes else 0.0
        life_std = float(np.std(lifetimes)) if len(lifetimes) > 1 else 0.0
        short_lived = sum(1 for l in lifetimes if l <= (10.0 * (duration / max(1, w_size))))
        short_ratio = float(short_lived) / max(1.0, float(len(lifetimes)))

        f1 = current_snap.f1_largest_free_ratio
        f2 = current_snap.f2_free_block_entropy
        f3 = float(current_snap.f3_free_block_count)
        tot_free = float(current_snap.total_free_bytes)
        lfb = float(current_snap.largest_free_block)
        util = current_snap.memory_utilization

        delta_f1 = f1 - start_snap.f1_largest_free_ratio
        delta_f2 = f2 - start_snap.f2_free_block_entropy
        delta_alloc = float(current_snap.allocated_bytes - start_snap.allocated_bytes)

        if len(allocs) > 2:
            inter_times = np.diff([e.timestamp for e in allocs])
            burstiness = float(np.std(inter_times) / (np.mean(inter_times) + 1e-6))
        else:
            burstiness = 0.0

        alloc_free_ratio = float(alloc_count) / max(1.0, float(free_count))
        active_alloc_count = float(current_snap.allocated_block_count)
        active_payload = float(current_snap.allocated_bytes)

        stacks = [e.call_stack_hash for e in allocs if e.call_stack_hash is not None]
        if stacks:
            _, counts = np.unique(stacks, return_counts=True)
            p = counts / len(stacks)
            stack_entropy = float(-np.sum(p * np.log2(p + 1e-9)))
        else:
            stack_entropy = 0.0

        f1_lag_1 = snapshots[-2].f1_largest_free_ratio if len(snapshots) >= 2 else f1
        f1_lag_5 = snapshots[-6].f1_largest_free_ratio if len(snapshots) >= 6 else f1
        f1_lag_10 = snapshots[-11].f1_largest_free_ratio if len(snapshots) >= 11 else f1
        f1_trend_5 = float(f1 - f1_lag_5) / 5.0

        window_10_f1 = [s.f1_largest_free_ratio for s in snapshots[-10:]]
        f1_volatility_10 = float(np.std(window_10_f1)) if len(window_10_f1) > 1 else 0.0
        alloc_rate_trend = float(alloc_rate - (alloc_bytes / max(0.001, duration * 2.0)))

        return TemporalFeatureVector(
            timestamp=current_snap.timestamp,
            alloc_count=alloc_count,
            alloc_bytes_total=alloc_bytes,
            alloc_size_mean=alloc_mean,
            alloc_size_std=alloc_std,
            alloc_size_max=alloc_max,
            alloc_rate=alloc_rate,
            free_count=free_count,
            free_bytes_total=free_bytes,
            free_size_mean=free_mean,
            free_size_std=free_std,
            free_rate=free_rate,
            lifetime_mean=life_mean,
            lifetime_std=life_std,
            short_lived_ratio=short_ratio,
            current_f1=f1,
            current_f2=f2,
            current_f3=f3,
            total_free_bytes=tot_free,
            largest_free_block=lfb,
            memory_utilization=util,
            delta_f1=delta_f1,
            delta_f2=delta_f2,
            delta_allocated_bytes=delta_alloc,
            alloc_burstiness=burstiness,
            alloc_to_free_ratio=alloc_free_ratio,
            active_alloc_count=active_alloc_count,
            active_payload_bytes=active_payload,
            call_stack_entropy=stack_entropy,
            f1_lag_1=f1_lag_1,
            f1_lag_5=f1_lag_5,
            f1_lag_10=f1_lag_10,
            f1_trend_5=f1_trend_5,
            f1_volatility_10=f1_volatility_10,
            alloc_rate_trend=alloc_rate_trend
        )
