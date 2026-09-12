"""
PREFRAG: Cost-Benefit Model
Computes expected fragmentation reduction vs physical compaction cost for rational proactive decisions.
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
from ..simulator.memory_block import MemoryBlock
from ..simulator.compaction_engine import CompactionEngine


@dataclass(slots=True)
class DecisionEvaluation:
    action_name: str
    predicted_future_f1_without_action: float
    estimated_future_f1_with_action: float
    expected_fragmentation_reduction: float
    estimated_blocks_moved: int
    estimated_bytes_moved: int
    estimated_latency_ms: float
    estimated_cost: float
    net_benefit: float
    should_intervene: bool


class CostBenefitOptimizer:
    """
    Evaluates whether preventive memory compaction is cost-justified.
    """

    def __init__(
        self,
        benefit_scale_lambda: float = 100.0, # Value assigned per unit of F1 fragmentation reduction
        min_net_benefit_threshold: float = 5.0, # Minimum net gain required to justify action
        cooldown_period_ticks: int = 15,     # Hysteresis to prevent thrashing
        severe_f1_threshold: float = 0.65
    ):
        self.benefit_scale = benefit_scale_lambda
        self.min_net_benefit = min_net_benefit_threshold
        self.cooldown_ticks = cooldown_period_ticks
        self.severe_threshold = severe_f1_threshold
        self.compaction_engine = CompactionEngine()
        self.last_intervention_tick: Optional[int] = None

    def evaluate(
        self,
        current_tick: int,
        predicted_future_f1: float,
        blocks: List[MemoryBlock]
    ) -> DecisionEvaluation:
        # Check cooldown
        if self.last_intervention_tick is not None:
            if (current_tick - self.last_intervention_tick) < self.cooldown_ticks:
                return DecisionEvaluation(
                    action_name="cooldown_wait",
                    predicted_future_f1_without_action=predicted_future_f1,
                    estimated_future_f1_with_action=predicted_future_f1,
                    expected_fragmentation_reduction=0.0,
                    estimated_blocks_moved=0,
                    estimated_bytes_moved=0,
                    estimated_latency_ms=0.0,
                    estimated_cost=0.0,
                    net_benefit=-1.0,
                    should_intervene=False
                )

        # Estimate compaction cost on current block state
        blocks_moved, bytes_moved, latency_ms, cost = self.compaction_engine.estimate_compaction_cost(blocks)

        # Realistic post-compaction F1 estimate:
        # After sliding compaction, all free space is merged into one contiguous tail block.
        # However, in practice, alignment constraints and pinned objects prevent perfect compaction.
        # We estimate residual fragmentation based on the number of allocated blocks.
        allocated_blocks = [b for b in blocks if not b.is_free]
        num_allocated = len(allocated_blocks)
        if num_allocated > 0:
            # Even after perfect compaction, there's minimal residual fragmentation
            # from block alignment overhead (typically 0.01 - 0.05 for well-packed heaps)
            f1_after_compaction = min(0.05, 0.005 * num_allocated)
        else:
            f1_after_compaction = 0.0
        reduction = max(0.0, predicted_future_f1 - f1_after_compaction)
        raw_benefit = reduction * self.benefit_scale

        # Net Benefit = Benefit - Cost
        net_benefit = raw_benefit - cost

        # Decision rule: intervene if net benefit is strictly positive AND predicted F1 crosses critical threshold
        should_act = (net_benefit >= self.min_net_benefit) and (predicted_future_f1 >= self.severe_threshold)

        return DecisionEvaluation(
            action_name="compact" if should_act else "no_action",
            predicted_future_f1_without_action=predicted_future_f1,
            estimated_future_f1_with_action=f1_after_compaction,
            expected_fragmentation_reduction=reduction,
            estimated_blocks_moved=blocks_moved,
            estimated_bytes_moved=bytes_moved,
            estimated_latency_ms=latency_ms,
            estimated_cost=cost,
            net_benefit=net_benefit,
            should_intervene=should_act
        )

    def record_intervention(self, tick: int) -> None:
        self.last_intervention_tick = tick
