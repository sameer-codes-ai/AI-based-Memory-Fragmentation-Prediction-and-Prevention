"""
PREFRAG: Unit and Property Tests
Tests trace generation, heap allocator mechanics, feature engineering, models, and cost controllers.
"""

import unittest
import numpy as np
from src.traces.trace_schema import MemoryEvent, AllocationOp, TraceStream
from src.traces.workload_generator import WorkloadGenerator, WorkloadFamily
from src.simulator.memory_block import MemoryBlock
from src.simulator.fragmentation_metrics import FragmentationCalculator, FragmentationSnapshot
from src.simulator.heap_allocator import HeapAllocator, AllocationPolicy
from src.simulator.compaction_engine import CompactionEngine
from src.features.feature_extractor import FeatureExtractor, TemporalFeatureVector
from src.models.baselines import StaticThresholdBaseline, MovingAverageBaseline, LinearRegressionBaseline, RandomForestBaseline
from src.models.temporal_nn import SimpleMLPForecaster, TemporalForecasterLSTM
from src.controller.cost_model import CostBenefitOptimizer
from src.controller.policies import ReactivePolicy, AlwaysProactivePolicy, CostAwarePredictivePolicy


class TestPREFRAGTraces(unittest.TestCase):
    def test_workload_generation(self):
        gen = WorkloadGenerator(seed=42)
        trace = gen.generate_trace(WorkloadFamily.WEB_SERVER, num_events=100)
        self.assertEqual(len(trace), 100)
        allocs = [e for e in trace if e.op == AllocationOp.ALLOC]
        frees = [e for e in trace if e.op == AllocationOp.FREE]
        self.assertGreater(len(allocs), 0)
        self.assertGreater(len(frees), 0)

    def test_trace_csv_roundtrip(self):
        gen = WorkloadGenerator(seed=123)
        trace = gen.generate_trace(WorkloadFamily.DATABASE, num_events=50)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            TraceStream.save_csv(trace, tmp_path)
            loaded = TraceStream.load_csv(tmp_path)
            self.assertEqual(len(trace), len(loaded))
            self.assertEqual(trace[0].address, loaded[0].address)
            self.assertEqual(trace[0].size, loaded[0].size)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestPREFRAGSimulator(unittest.TestCase):
    def test_allocation_and_free(self):
        allocator = HeapAllocator(total_capacity=1024 * 1024, policy=AllocationPolicy.FIRST_FIT)
        addr1 = allocator.allocate(1024, timestamp=1.0)
        addr2 = allocator.allocate(2048, timestamp=2.0)
        self.assertIsNotNone(addr1)
        self.assertIsNotNone(addr2)
        self.assertNotEqual(addr1, addr2)

        snap = allocator.get_snapshot(timestamp=2.0)
        self.assertEqual(snap.allocated_block_count, 2)

        # Free first block
        freed = allocator.free(addr1, timestamp=3.0)
        self.assertTrue(freed)
        snap2 = allocator.get_snapshot(timestamp=3.0)
        self.assertEqual(snap2.allocated_block_count, 1)

    def test_adjacent_coalescing(self):
        allocator = HeapAllocator(total_capacity=10000)
        a1 = allocator.allocate(1000)
        a2 = allocator.allocate(1000)
        a3 = allocator.allocate(1000)

        # Free middle, then left and right
        allocator.free(a1)
        allocator.free(a3)
        self.assertEqual(len([b for b in allocator.blocks if b.is_free]), 2)

        allocator.free(a2) # Now all 3 contiguous free spaces should coalesce into 1 block
        free_blocks = [b for b in allocator.blocks if b.is_free]
        self.assertEqual(len(free_blocks), 1)
        self.assertEqual(free_blocks[0].size, 10000)

    def test_compaction(self):
        allocator = HeapAllocator(total_capacity=10000, alignment=1)
        a1 = allocator.allocate(1000) # [0, 1000)
        a2 = allocator.allocate(1000) # [1000, 2000)
        a3 = allocator.allocate(1000) # [2000, 3000)
        
        allocator.free(a2) # creates a hole in the middle
        snap_before = allocator.get_snapshot()
        self.assertGreater(snap_before.f1_largest_free_ratio, 0.0)

        report = allocator.compact(timestamp=10.0)
        self.assertGreater(report.blocks_moved, 0)
        self.assertEqual(report.f1_after, 0.0)
        self.assertEqual(report.largest_free_after, 8000)


class TestPREFRAGFeatures(unittest.TestCase):
    def test_feature_extraction(self):
        gen = WorkloadGenerator(seed=42)
        trace = gen.generate_trace(WorkloadFamily.MICROSERVICE, num_events=300)
        from src.experiments.runner import ExperimentRunner
        runner = ExperimentRunner(heap_capacity=4 * 1024 * 1024, window_size=30)
        X, Y_reg, Y_cls, snapshots = runner.prepare_dataset_for_workload(trace)

        self.assertEqual(X.shape[1], 28)
        self.assertEqual(Y_reg.shape[1], 4)
        self.assertEqual(len(X), len(Y_reg))
        self.assertEqual(len(X), len(Y_cls))


class TestPREFRAGModels(unittest.TestCase):
    def test_baselines_and_neural_models(self):
        np.random.seed(42)
        X = np.random.randn(100, 28).astype(np.float32)
        Y = np.random.uniform(0.1, 0.9, (100, 4)).astype(np.float32)

        # Static Threshold Baseline
        b1 = StaticThresholdBaseline()
        p1 = b1.predict(X)
        self.assertEqual(p1.shape, (100, 4))

        # Ridge Baseline
        b3 = LinearRegressionBaseline().fit(X[:80], Y[:80])
        p3 = b3.predict(X[80:])
        self.assertEqual(p3.shape, (20, 4))

        # Random Forest Baseline
        b4 = RandomForestBaseline(n_estimators=10, max_depth=4).fit(X[:80], Y[:80])
        p4 = b4.predict(X[80:])
        self.assertEqual(p4.shape, (20, 4))

        # LSTM Forecaster
        m1 = TemporalForecasterLSTM(input_dim=28, hidden_dim=16, output_dim=4)
        m1.fit(X[:80], Y[:80])
        pm1 = m1.predict(X[80:])
        self.assertEqual(pm1.shape, (20, 4))


class TestPREFRAGController(unittest.TestCase):
    def test_cost_benefit_optimizer(self):
        opt = CostBenefitOptimizer(benefit_scale_lambda=100.0, min_net_benefit_threshold=5.0)
        blocks = [
            MemoryBlock(start_address=0, size=1000, is_free=False),
            MemoryBlock(start_address=1000, size=2000, is_free=True),
            MemoryBlock(start_address=3000, size=1000, is_free=False),
            MemoryBlock(start_address=4000, size=6000, is_free=True)
        ]

        # Case 1: Low predicted fragmentation -> No intervention
        dec1 = opt.evaluate(current_tick=1, predicted_future_f1=0.20, blocks=blocks)
        self.assertFalse(dec1.should_intervene)

        # Case 2: Severe predicted fragmentation (F1=0.85) -> Intervene
        dec2 = opt.evaluate(current_tick=2, predicted_future_f1=0.85, blocks=blocks)
        self.assertTrue(dec2.should_intervene)
        self.assertGreater(dec2.net_benefit, 0.0)


if __name__ == "__main__":
    unittest.main()
