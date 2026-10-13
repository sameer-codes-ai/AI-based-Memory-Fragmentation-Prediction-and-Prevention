"""
PREFRAG: Unit and Property Tests
Tests trace generation across 10 families, heap allocator mechanics for 4 strategies,
feature engineering (34-D), baselines B1-B5, deep models M1-M3, and cost controllers.
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
from src.models.baselines import (
    StaticThresholdBaseline,
    MovingAverageBaseline,
    LinearRegressionBaseline,
    RandomForestBaseline,
    GradientBoostingBaseline
)
from src.models.temporal_nn import (
    TemporalForecasterLSTM,
    TemporalForecasterGRU,
    TemporalForecasterTCN
)
from src.controller.cost_model import CostBenefitOptimizer
from src.controller.policies import (
    ReactivePolicy,
    AlwaysProactivePolicy,
    PredictionOnlyPolicy,
    ThresholdPredictivePolicy,
    CostAwarePredictivePolicy
)


class TestPREFRAGTraces(unittest.TestCase):
    def test_workload_generation_all_10_families(self):
        gen = WorkloadGenerator(seed=42)
        for fam in WorkloadFamily:
            trace = gen.generate_trace(fam, num_events=60)
            self.assertEqual(len(trace), 60, f"Failed for family {fam}")
            allocs = [e for e in trace if e.op == AllocationOp.ALLOC]
            self.assertGreater(len(allocs), 0, f"No allocs for {fam}")

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
    def test_allocation_and_free_all_policies(self):
        policies = [
            AllocationPolicy.FIRST_FIT,
            AllocationPolicy.BEST_FIT,
            AllocationPolicy.WORST_FIT,
            AllocationPolicy.NEXT_FIT
        ]
        for pol in policies:
            allocator = HeapAllocator(total_capacity=1024 * 1024, policy=pol)
            addr1 = allocator.allocate(1024, timestamp=1.0)
            addr2 = allocator.allocate(2048, timestamp=2.0)
            self.assertIsNotNone(addr1, f"Failed addr1 with policy {pol}")
            self.assertIsNotNone(addr2, f"Failed addr2 with policy {pol}")
            self.assertNotEqual(addr1, addr2)

            snap = allocator.get_snapshot(timestamp=2.0)
            self.assertEqual(snap.allocated_block_count, 2)

            freed = allocator.free(addr1, timestamp=3.0)
            self.assertTrue(freed, f"Failed free with policy {pol}")
            snap2 = allocator.get_snapshot(timestamp=3.0)
            self.assertEqual(snap2.allocated_block_count, 1)

    def test_next_fit_roving_pointer(self):
        allocator = HeapAllocator(total_capacity=10000, policy=AllocationPolicy.NEXT_FIT)
        a1 = allocator.allocate(1000)
        a2 = allocator.allocate(1000)
        a3 = allocator.allocate(1000)
        self.assertGreater(a2, a1)
        self.assertGreater(a3, a2)
        # Verify next allocation starts searching from after a3
        a4 = allocator.allocate(1000)
        self.assertGreater(a4, a3)

    def test_adjacent_coalescing(self):
        allocator = HeapAllocator(total_capacity=10000)
        a1 = allocator.allocate(1000)
        a2 = allocator.allocate(1000)
        a3 = allocator.allocate(1000)

        allocator.free(a1)
        allocator.free(a3)
        self.assertEqual(len([b for b in allocator.blocks if b.is_free]), 2)

        allocator.free(a2)
        free_blocks = [b for b in allocator.blocks if b.is_free]
        self.assertEqual(len(free_blocks), 1)
        self.assertEqual(free_blocks[0].size, 10000)

    def test_compaction(self):
        allocator = HeapAllocator(total_capacity=10000, alignment=1)
        a1 = allocator.allocate(1000)
        a2 = allocator.allocate(1000)
        a3 = allocator.allocate(1000)
        
        allocator.free(a2)
        snap_before = allocator.get_snapshot()
        self.assertGreater(snap_before.f1_largest_free_ratio, 0.0)

        report = allocator.compact(timestamp=10.0)
        self.assertGreater(report.blocks_moved, 0)
        self.assertEqual(report.f1_after, 0.0)
        self.assertEqual(report.largest_free_after, 8000)

    def test_allocator_edge_cases(self):
        allocator = HeapAllocator(total_capacity=1024, alignment=1)
        addr = allocator.allocate(1024)
        self.assertIsNotNone(addr)
        # Attempt allocation when heap is full
        failed_addr = allocator.allocate(64)
        self.assertIsNone(failed_addr)
        self.assertEqual(allocator.failures_count, 1)

        # Free valid allocation
        freed = allocator.free(addr)
        self.assertTrue(freed)

        # Double free should return False
        freed_again = allocator.free(addr)
        self.assertFalse(freed_again)

        # Free non-existent address
        self.assertFalse(allocator.free(0x999999))


class TestPREFRAGFeatures(unittest.TestCase):
    def test_feature_extraction_34d(self):
        gen = WorkloadGenerator(seed=42)
        trace = gen.generate_trace(WorkloadFamily.MICROSERVICE, num_events=300)
        from src.experiments.runner import ExperimentRunner
        runner = ExperimentRunner(heap_capacity=4 * 1024 * 1024, window_size=30)
        X, Y_reg, Y_cls, snapshots = runner.prepare_dataset_for_workload(trace)

        self.assertEqual(X.shape[1], 34, "Feature matrix must have 34 dimensions")
        self.assertEqual(Y_reg.shape[1], 4, "Regression target must have 4 horizons")
        self.assertEqual(len(X), len(Y_reg))
        self.assertEqual(len(X), len(Y_cls))

        # Test extract_single_step consistency
        extractor = FeatureExtractor(window_size_ticks=30)
        single_vec = extractor.extract_single_step(trace[-40:], snapshots)
        self.assertEqual(len(single_vec.to_array()), 34)


class TestPREFRAGModels(unittest.TestCase):
    def test_baselines_and_neural_models(self):
        np.random.seed(42)
        X = np.random.randn(100, 34).astype(np.float32)
        Y = np.random.uniform(0.1, 0.9, (100, 4)).astype(np.float32)

        # B1: Static Threshold
        b1 = StaticThresholdBaseline()
        p1 = b1.predict(X)
        self.assertEqual(p1.shape, (100, 4))

        # B2: Moving Average
        b2 = MovingAverageBaseline()
        p2 = b2.predict(X)
        self.assertEqual(p2.shape, (100, 4))

        # B3: Ridge Regression
        b3 = LinearRegressionBaseline().fit(X[:80], Y[:80])
        p3 = b3.predict(X[80:])
        self.assertEqual(p3.shape, (20, 4))

        # B4: Random Forest
        b4 = RandomForestBaseline(n_estimators=10, max_depth=4).fit(X[:80], Y[:80])
        p4 = b4.predict(X[80:])
        self.assertEqual(p4.shape, (20, 4))

        # B5: Gradient Boosted Trees
        b5 = GradientBoostingBaseline(n_estimators=10, max_depth=3).fit(X[:80], Y[:80])
        p5 = b5.predict(X[80:])
        self.assertEqual(p5.shape, (20, 4))

        # M1: LSTM
        m1 = TemporalForecasterLSTM(input_dim=34, hidden_dim=16, output_dim=4, seq_len=5, epochs=5)
        m1.fit(X[:80], Y[:80])
        pm1 = m1.predict(X[80:])
        self.assertEqual(pm1.shape, (20, 4))

        # M2: GRU
        m2 = TemporalForecasterGRU(input_dim=34, hidden_dim=16, output_dim=4, seq_len=5, epochs=5)
        m2.fit(X[:80], Y[:80])
        pm2 = m2.predict(X[80:])
        self.assertEqual(pm2.shape, (20, 4))

        # M3: TCN
        m3 = TemporalForecasterTCN(input_dim=34, hidden_dim=32, output_dim=4, seq_len=5, epochs=5)
        m3.fit(X[:80], Y[:80])
        pm3 = m3.predict(X[80:])
        self.assertEqual(pm3.shape, (20, 4))


class TestPREFRAGController(unittest.TestCase):
    def test_cost_benefit_optimizer(self):
        opt = CostBenefitOptimizer(benefit_scale_lambda=100.0, min_net_benefit_threshold=5.0)
        blocks = [
            MemoryBlock(start_address=0, size=1000, is_free=False),
            MemoryBlock(start_address=1000, size=2000, is_free=True),
            MemoryBlock(start_address=3000, size=1000, is_free=False),
            MemoryBlock(start_address=4000, size=6000, is_free=True)
        ]

        dec1 = opt.evaluate(current_tick=1, predicted_future_f1=0.20, blocks=blocks)
        self.assertFalse(dec1.should_intervene)

        dec2 = opt.evaluate(current_tick=2, predicted_future_f1=0.85, blocks=blocks)
        self.assertTrue(dec2.should_intervene)
        self.assertGreater(dec2.net_benefit, 0.0)

    def test_policies(self):
        blocks = [
            MemoryBlock(start_address=0, size=1000, is_free=False),
            MemoryBlock(start_address=1000, size=2000, is_free=True)
        ]
        
        # ReactivePolicy
        r_pol = ReactivePolicy(severe_threshold=0.50, cooldown_ticks=5)
        self.assertFalse(r_pol.decide(1, current_f1=0.30, predicted_f1=0.80, blocks=blocks))
        self.assertTrue(r_pol.decide(2, current_f1=0.60, predicted_f1=0.80, blocks=blocks))
        # Cooldown prevents immediate re-trigger
        self.assertFalse(r_pol.decide(3, current_f1=0.60, predicted_f1=0.80, blocks=blocks))

        # AlwaysProactivePolicy
        p_pol = AlwaysProactivePolicy(interval_ticks=10)
        self.assertFalse(p_pol.decide(5, current_f1=0.1, predicted_f1=0.1, blocks=blocks))
        self.assertTrue(p_pol.decide(10, current_f1=0.1, predicted_f1=0.1, blocks=blocks))

        # CostAwarePredictivePolicy
        ca_pol = CostAwarePredictivePolicy(benefit_scale_lambda=1500.0, min_net_benefit_threshold=2.0, cooldown_ticks=5, severe_f1_threshold=0.40)
        self.assertFalse(ca_pol.decide(1, current_f1=0.10, predicted_f1=0.20, blocks=blocks))
        self.assertTrue(ca_pol.decide(2, current_f1=0.20, predicted_f1=0.80, blocks=blocks))


class TestPREFRAGSimulation(unittest.TestCase):
    def test_simulation_with_ml_forecaster(self):
        from src.experiments.runner import ExperimentRunner
        gen = WorkloadGenerator(seed=42)
        trace = gen.generate_trace(WorkloadFamily.WEB_SERVER, num_events=120)
        runner = ExperimentRunner(heap_capacity=1024 * 1024, window_size=20, severe_threshold=0.40)
        policy = ThresholdPredictivePolicy(threshold=0.40, cooldown_ticks=5)
        forecaster = StaticThresholdBaseline()
        res = runner.run_condition_simulation(
            events=trace,
            condition_name="E4: ML + Threshold",
            policy=policy,
            forecaster=forecaster
        )
        self.assertEqual(res.total_events, 120)
        self.assertGreaterEqual(res.mean_f1, 0.0)
        self.assertGreaterEqual(res.mae_t1, 0.0)
        self.assertGreaterEqual(res.mae_t10, 0.0)
        self.assertIn("mae_t10", res.to_dict())


if __name__ == "__main__":
    unittest.main()

