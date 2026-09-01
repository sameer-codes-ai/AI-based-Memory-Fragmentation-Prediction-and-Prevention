"""
PREFRAG: Memory Management Decision Policies (E1 - E5)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from .cost_model import CostBenefitOptimizer, DecisionEvaluation
from ..simulator.memory_block import MemoryBlock


class BaseMemoryPolicy(ABC):
    @abstractmethod
    def decide(
        self,
        tick: int,
        current_f1: float,
        predicted_f1: float,
        blocks: List[MemoryBlock]
    ) -> bool:
        pass


class ReactivePolicy(BaseMemoryPolicy):
    """
    E1: Reactive Baseline — Triggers compaction only after fragmentation is already severe.
    """

    def __init__(self, severe_threshold: float = 0.65, cooldown_ticks: int = 15):
        self.severe_threshold = severe_threshold
        self.cooldown = cooldown_ticks
        self.last_tick: Optional[int] = None

    def decide(self, tick: int, current_f1: float, predicted_f1: float, blocks: List[MemoryBlock]) -> bool:
        if self.last_tick is not None and (tick - self.last_tick) < self.cooldown:
            return False
        if current_f1 >= self.severe_threshold:
            self.last_tick = tick
            return True
        return False


class AlwaysProactivePolicy(BaseMemoryPolicy):
    """
    E2: Always-Proactive Baseline — Unconditionally triggers compaction at fixed intervals.
    """

    def __init__(self, interval_ticks: int = 25):
        self.interval = interval_ticks

    def decide(self, tick: int, current_f1: float, predicted_f1: float, blocks: List[MemoryBlock]) -> bool:
        return (tick > 0) and (tick % self.interval == 0)


class PredictionOnlyPolicy(BaseMemoryPolicy):
    """
    E3: Prediction Only — Tracks forecasts without triggering intervention.
    """

    def decide(self, tick: int, current_f1: float, predicted_f1: float, blocks: List[MemoryBlock]) -> bool:
        return False


class ThresholdPredictivePolicy(BaseMemoryPolicy):
    """
    E4: ML + Threshold Predictive — Triggers compaction whenever predicted future fragmentation crosses threshold.
    """

    def __init__(self, threshold: float = 0.65, cooldown_ticks: int = 15):
        self.threshold = threshold
        self.cooldown = cooldown_ticks
        self.last_tick: Optional[int] = None

    def decide(self, tick: int, current_f1: float, predicted_f1: float, blocks: List[MemoryBlock]) -> bool:
        if self.last_tick is not None and (tick - self.last_tick) < self.cooldown:
            return False
        if predicted_f1 >= self.threshold:
            self.last_tick = tick
            return True
        return False


class CostAwarePredictivePolicy(BaseMemoryPolicy):
    """
    E5: PREFRAG Core — Cost-Aware Preventive Memory Reorganization.
    Intervenes if and only if the expected benefit strictly outweighs the operational cost.
    """

    def __init__(
        self,
        benefit_scale_lambda: float = 100.0,
        min_net_benefit_threshold: float = 5.0,
        cooldown_ticks: int = 15,
        severe_f1_threshold: float = 0.65
    ):
        self.optimizer = CostBenefitOptimizer(
            benefit_scale_lambda=benefit_scale_lambda,
            min_net_benefit_threshold=min_net_benefit_threshold,
            cooldown_period_ticks=cooldown_ticks,
            severe_f1_threshold=severe_f1_threshold
        )
        self.last_decision: Optional[DecisionEvaluation] = None

    def decide(self, tick: int, current_f1: float, predicted_f1: float, blocks: List[MemoryBlock]) -> bool:
        decision = self.optimizer.evaluate(tick, predicted_f1, blocks)
        self.last_decision = decision
        if decision.should_intervene:
            self.optimizer.record_intervention(tick)
            return True
        return False
