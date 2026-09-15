"""
PREFRAG Master Benchmark & Experiment Orchestrator
Runs end-to-end experiments (E1-E5), model comparisons, generalization, ablation, and exports publication reports.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.traces.workload_generator import WorkloadGenerator, WorkloadFamily
from src.traces.trace_schema import TraceStream, AllocationOp
from src.simulator.heap_allocator import HeapAllocator
from src.features.feature_extractor import FeatureExtractor
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
from src.controller.policies import (
    ReactivePolicy,
    AlwaysProactivePolicy,
    PredictionOnlyPolicy,
    ThresholdPredictivePolicy,
    CostAwarePredictivePolicy
)
from src.experiments.runner import ExperimentRunner, ExperimentResult
from src.experiments.generalization import GeneralizationSuite, FeatureAblationSuite, ExplainabilitySuite


def run_full_benchmark_suite():
    print("=" * 80)
    print("PREFRAG: Predictive and Cost-Aware Memory Fragmentation Management")
    print("Executing End-to-End Research Benchmark Pipeline...")
    print("=" * 80)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    traces_dir = os.path.join(results_dir, "saved_traces")
    os.makedirs(traces_dir, exist_ok=True)

    # Step 1: Workload Trace Generation
    print("\n[Step 1/5] Generating Workload Traces for 5 Benchmark Families...")
    generator = WorkloadGenerator(seed=42)
    workload_traces = {}
    num_events = 25000

    for fam in WorkloadFamily:
        trace = generator.generate_trace(fam, num_events=num_events)
        workload_traces[fam.value] = trace
        trace_path = os.path.join(traces_dir, f"{fam.value}_trace.csv")
        TraceStream.save_csv(trace, trace_path)
        print(f"  -> Generated {fam.value}: {len(trace)} events (saved to {trace_path})")

    # Step 2: Multi-Horizon Model Comparison
    print("\n[Step 2/5] Training and Benchmarking Multi-Horizon Predictors...")
    X_list, Y_list = [], []
    for fam in WorkloadFamily:
        tr = workload_traces[fam.value]
        peak_bytes = sum(e.size for e in tr[:300] if e.op == AllocationOp.ALLOC)
        cap = max(512 * 1024, int(peak_bytes * 1.3))
        r = ExperimentRunner(heap_capacity=cap, window_size=40, severe_threshold=0.40)
        x_f, y_f, _, _ = r.prepare_dataset_for_workload(tr)
        X_list.append(x_f)
        Y_list.append(y_f)

    X_all = np.vstack(X_list)
    Y_all = np.vstack(Y_list)

    np.random.seed(42)
    perm = np.random.permutation(len(X_all))
    split = int(0.75 * len(X_all))
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, Y_train = X_all[train_idx], Y_all[train_idx]
    X_test, Y_test = X_all[test_idx], Y_all[test_idx]

    models = {}
    print("  -> Initializing B1: Static Threshold...")
    models["B1: Static Threshold"] = StaticThresholdBaseline()

    print("  -> Initializing B2: Moving Average...")
    models["B2: Moving Average"] = MovingAverageBaseline()

    print("  -> Training B3: Linear Regression (Ridge)...")
    models["B3: Linear Regression"] = LinearRegressionBaseline().fit(X_train, Y_train)

    print("  -> Training B4: Random Forest (60 estimators)...")
    models["B4: Random Forest"] = RandomForestBaseline(n_estimators=60, max_depth=10).fit(X_train, Y_train)

    print("  -> Training B5: Gradient Boosted Trees (60 estimators)...")
    models["B5: Gradient Boosted Trees"] = GradientBoostingBaseline(n_estimators=60, max_depth=4).fit(X_train, Y_train)

    print("  -> Training M1: Temporal LSTM (Backpropagation Through Time)...")
    models["M1: Temporal LSTM"] = TemporalForecasterLSTM(input_dim=X_train.shape[1], hidden_dim=48, output_dim=4).fit(X_train, Y_train, verbose=True)

    print("  -> Training M2: Temporal GRU (Gated Recurrent Unit)...")
    models["M2: Temporal GRU"] = TemporalForecasterGRU(input_dim=X_train.shape[1], hidden_dim=40, output_dim=4).fit(X_train, Y_train, verbose=True)

    print("  -> Training M3: Temporal TCN (Dilated Causal Convolutions)...")
    models["M3: Temporal TCN"] = TemporalForecasterTCN(input_dim=X_train.shape[1], hidden_dim=64, output_dim=4).fit(X_train, Y_train, verbose=True)

    model_metrics = {}
    print(f"\n{'Model':<30} | {'MAE (t+1)':<10} | {'MAE (t+5)':<10} | {'MAE (t+10)':<10} | {'MAE (t+30)':<10} | {'R2 (t+10)':<10}")
    print("-" * 88)

    for name, model in models.items():
        preds = model.predict(X_test)
        mae1 = float(mean_absolute_error(Y_test[:, 0], preds[:, 0]))
        mae5 = float(mean_absolute_error(Y_test[:, 1], preds[:, 1]))
        mae10 = float(mean_absolute_error(Y_test[:, 2], preds[:, 2]))
        mae30 = float(mean_absolute_error(Y_test[:, 3], preds[:, 3]))
        r2_10 = float(r2_score(Y_test[:, 2], preds[:, 2]))

        model_metrics[name] = {
            "mae_t1": round(mae1, 4),
            "mae_t5": round(mae5, 4),
            "mae_t10": round(mae10, 4),
            "mae_t30": round(mae30, 4),
            "r2_t10": round(r2_10, 4)
        }
        print(f"{name:<30} | {mae1:<10.4f} | {mae5:<10.4f} | {mae10:<10.4f} | {mae30:<10.4f} | {r2_10:<10.4f}")

    # Step 3: 5-Way System Strategy Simulation (E1 - E5)
    print("\n[Step 3/5] Simulating 5 Memory Management Regimes (E1 - E5)...")
    primary_forecaster = models["M1: Temporal LSTM"]

    test_workloads = [
        WorkloadFamily.MICROSERVICE,
        WorkloadFamily.SCIENTIFIC,
        WorkloadFamily.DATABASE,
        WorkloadFamily.ADVERSARIAL_CHURN
    ]
    
    policies = {
        "E1: Reactive Baseline": ReactivePolicy(severe_threshold=0.40, cooldown_ticks=15),
        "E2: Always-Proactive": AlwaysProactivePolicy(interval_ticks=30),
        "E3: Prediction Only": PredictionOnlyPolicy(),
        "E4: ML + Threshold": ThresholdPredictivePolicy(threshold=0.40, cooldown_ticks=15),
        "E5: ML + Cost-Aware (PREFRAG)": CostAwarePredictivePolicy(benefit_scale_lambda=1500.0, min_net_benefit_threshold=2.0, cooldown_ticks=15, severe_f1_threshold=0.40)
    }

    system_results = []
    print(f"\n{'Condition':<30} | {'Workload':<15} | {'Mean F1':<8} | {'Peak F1':<8} | {'Severe Ep':<10} | {'Compactions':<12} | {'Moved (MB)':<12} | {'Pause (ms)':<10} | {'Cost':<8}")
    print("-" * 130)

    for w_fam in test_workloads:
        w_trace = workload_traces[w_fam.value]
        # Size heap capacity to create realistic memory pressure (~50-80% peak utilization)
        peak_bytes = sum(e.size for e in w_trace[:300] if e.op == AllocationOp.ALLOC)
        dynamic_capacity = max(512 * 1024, int(peak_bytes * 1.3))
        sim_runner = ExperimentRunner(heap_capacity=dynamic_capacity, window_size=40, severe_threshold=0.40)

        for cond_name, policy in policies.items():
            fc = primary_forecaster if ("ML" in cond_name or "Prediction" in cond_name) else None
            res = sim_runner.run_condition_simulation(
                events=w_trace,
                condition_name=cond_name,
                policy=policy,
                forecaster=fc
            )
            system_results.append(res)
            print(f"{cond_name:<30} | {w_fam.value:<15} | {res.mean_f1:<8.4f} | {res.peak_f1:<8.4f} | {res.severe_episodes_f1:<10} | {res.compactions_count:<12} | {res.total_bytes_moved_mb:<12.2f} | {res.cumulative_pause_ms:<10.2f} | {res.total_cost:<8.1f}")

    # Step 4: Cross-Workload Generalization
    print("\n[Step 4/5] Evaluating Cross-Workload Generalization...")
    gen_extractor = FeatureExtractor(window_size_ticks=40, severe_f1_threshold=0.40)
    gen_suite = GeneralizationSuite(gen_extractor)
    gen_matrix = gen_suite.run_cross_workload_matrix(num_events=2000)

    print(f"\n{'Target Workload Family':<25} | {'MAE (t+10)':<12} | {'RMSE (t+10)':<12} | {'R2 (t+10)':<12} | {'Domain':<12}")
    print("-" * 80)
    for wname, m in gen_matrix.items():
        dom = "In-Domain" if m["in_domain"] else "Cross-Domain"
        print(f"{wname:<25} | {m['mae_t10']:<12.4f} | {m['rmse_t10']:<12.4f} | {m['r2_t10']:<12.4f} | {dom:<12}")

    # Step 5: Feature Ablation and Explainability
    print("\n[Step 5/5] Conducting Feature Ablation and Importance Attribution...")
    ablation_suite = FeatureAblationSuite()
    ablation_results = ablation_suite.run_ablation(X_train, Y_train, X_test, Y_test)

    print(f"\n{'Ablation Configuration':<35} | {'MAE (t+10)':<12} | {'R2 (t+10)':<12} | {'Delta Error %':<15}")
    print("-" * 80)
    for conf, m in ablation_results.items():
        print(f"{conf:<35} | {m['mae_t10']:<12.4f} | {m['r2_t10']:<12.4f} | {m['delta_mae_pct']:+14.2f}%")

    top_features = ExplainabilitySuite.get_feature_importances(X_train, Y_train[:, 2])
    print("\nTop 8 Most Predictive Features for Fragmentation:")
    for rank, (fname, imp) in enumerate(top_features[:8], 1):
        print(f"  {rank}. {fname:<25}: Importance Score = {imp:.4f}")

    # Save summary artifacts
    summary_report_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_comparison": model_metrics,
            "system_simulation": [r.to_dict() for r in system_results],
            "cross_workload_generalization": gen_matrix,
            "feature_ablation": ablation_results,
            "top_features": top_features
        }, f, indent=2)

    # Generate Markdown Report
    md_report_path = os.path.join(results_dir, "benchmark_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# PREFRAG: Experimental Benchmark Report\n\n")
        f.write("## 1. Multi-Horizon Model Performance Comparison\n\n")
        f.write("| Model | MAE (t+1) | MAE (t+5) | MAE (t+10) | MAE (t+30) | R² (t+10) |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|\n")
        for mname, m in model_metrics.items():
            f.write(f"| **{mname}** | {m['mae_t1']:.4f} | {m['mae_t5']:.4f} | {m['mae_t10']:.4f} | {m['mae_t30']:.4f} | {m['r2_t10']:.4f} |\n")

        f.write("\n## 2. Five-Way Memory Management Regime Comparison (E1 - E5)\n\n")
        f.write("| Condition | Workload | Mean F₁ | Peak F₁ | Severe Episodes | Compactions | Moved (MB) | Pause (ms) | Total Cost |\n")
        f.write("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in system_results:
            f.write(f"| **{r.condition_name}** | {r.workload_name} | {r.mean_f1:.4f} | {r.peak_f1:.4f} | {r.severe_episodes_f1} | {r.compactions_count} | {r.total_bytes_moved_mb:.2f} MB | {r.cumulative_pause_ms:.2f} ms | {r.total_cost:.1f} |\n")

        f.write("\n## 3. Cross-Workload Generalization Matrix\n\n")
        f.write("| Target Workload Family | MAE (t+10) | RMSE (t+10) | R² (t+10) | Domain Status |\n")
        f.write("|---|:---:|:---:|:---:|:---:|\n")
        for wname, m in gen_matrix.items():
            dom = "In-Domain" if m["in_domain"] else "Cross-Domain"
            f.write(f"| **{wname}** | {m['mae_t10']:.4f} | {m['rmse_t10']:.4f} | {m['r2_t10']:.4f} | {dom} |\n")

        f.write("\n## 4. Feature Group Ablation Study\n\n")
        f.write("| Ablation Variant | MAE (t+10) | R² (t+10) | Error Increase (%) |\n")
        f.write("|---|:---:|:---:|:---:|:---:|\n")
        for cname, m in ablation_results.items():
            f.write(f"| **{cname}** | {m['mae_t10']:.4f} | {m['r2_t10']:.4f} | {m['delta_mae_pct']:+.2f}% |\n")

        f.write("\n## 5. Feature Attribution Ranking\n\n")
        f.write("| Rank | Feature Name | Importance Score |\n")
        f.write("|---|---|:---:|\n")
        for rank, (fname, imp) in enumerate(top_features[:10], 1):
            f.write(f"| {rank} | `{fname}` | {imp:.4f} |\n")

    print("\n" + "=" * 80)
    print(f"Benchmark run complete! Results saved to:")
    print(f"  - {summary_report_path}")
    print(f"  - {md_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_benchmark_suite()
