# PREFRAG: Experimental Benchmark Report

## 1. Allocation Strategy Comparison (First-Fit vs Best-Fit vs Worst-Fit vs Next-Fit)

| Workload | Policy | Mean F₁ | Peak F₁ | Severe Episodes (F₁ ≥ 0.40) | Failures |
|---|:---:|:---:|:---:|:---:|:---:|
| database | **first_fit** | 0.8402 | 0.9637 | 9178 | 349 |
| database | **best_fit** | 0.8077 | 0.9588 | 9070 | 271 |
| database | **worst_fit** | 0.9522 | 0.9870 | 9739 | 905 |
| database | **next_fit** | 0.9304 | 0.9755 | 9739 | 578 |
| scientific | **first_fit** | 0.1295 | 0.3813 | 0 | 0 |
| scientific | **best_fit** | 0.1473 | 0.4043 | 4 | 0 |
| scientific | **worst_fit** | 0.4875 | 0.7701 | 7910 | 0 |
| scientific | **next_fit** | 0.2039 | 0.5872 | 1223 | 0 |
| microservice | **first_fit** | 0.0028 | 0.0177 | 0 | 0 |
| microservice | **best_fit** | 0.0028 | 0.0135 | 0 | 0 |
| microservice | **worst_fit** | 0.3160 | 0.5812 | 3626 | 0 |
| microservice | **next_fit** | 0.2310 | 0.5106 | 2211 | 0 |
| adversarial_churn | **first_fit** | 0.0594 | 0.0917 | 0 | 0 |
| adversarial_churn | **best_fit** | 0.0488 | 0.0805 | 0 | 0 |
| adversarial_churn | **worst_fit** | 0.9264 | 0.9867 | 9514 | 0 |
| adversarial_churn | **next_fit** | 0.9093 | 0.9800 | 9514 | 0 |
| game_engine | **first_fit** | 0.6837 | 0.9115 | 9080 | 127 |
| game_engine | **best_fit** | 0.6313 | 0.8718 | 8815 | 165 |
| game_engine | **worst_fit** | 0.8613 | 0.9622 | 9533 | 135 |
| game_engine | **next_fit** | 0.7669 | 0.9192 | 9533 | 120 |

**Selected Allocator Policy:** `BEST_FIT` (empirically evaluated across all benchmark families).

## 2. Multi-Horizon Model Performance Comparison (Chronological Splitting, 34 Features)

| Model | MAE (t+1) | MAE (t+5) | MAE (t+10) | MAE (t+30) | R² (t+10) |
|---|:---:|:---:|:---:|:---:|:---:|
| **B1: Static Threshold** | 0.0032 | 0.0100 | 0.0164 | 0.0340 | 0.9888 |
| **B2: Moving Average** | 0.0090 | 0.0217 | 0.0391 | 0.0612 | 0.9597 |
| **B3: Linear Regression** | 0.0036 | 0.0107 | 0.0167 | 0.0309 | 0.9906 |
| **B4: Random Forest** | 0.0056 | 0.0125 | 0.0180 | 0.0300 | 0.9897 |
| **B5: Gradient Boosted Trees** | 0.0168 | 0.0214 | 0.0260 | 0.0370 | 0.9887 |
| **M1: Temporal LSTM** | 0.0073 | 0.0153 | 0.0218 | 0.0347 | 0.9852 |
| **M2: Temporal GRU** | 0.0097 | 0.0176 | 0.0231 | 0.0371 | 0.9850 |
| **M3: Temporal TCN** | 0.0087 | 0.0149 | 0.0208 | 0.0343 | 0.9880 |

## 3. Five-Way Memory Management Regime Comparison (E1 - E5)

| Condition | Workload | Mean F₁ | Peak F₁ | Severe Episodes | Compactions | Moved (MB) | Pause (ms) | Total Cost |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1: Reactive Baseline** | database | 0.2340 | 0.8476 | 1216 | 312 | 745.74 MB | 628.78 ms | 159096.0 |
| **E2: Always-Proactive** | database | 0.2372 | 0.8627 | 2271 | 333 | 765.52 MB | 660.24 ms | 165029.1 |
| **E3: Prediction Only** | database | 0.8077 | 0.9588 | 9070 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | database | 0.1906 | 0.8170 | 1282 | 474 | 1101.38 MB | 942.43 ms | 236431.3 |
| **E5: ML + Cost-Aware (PREFRAG)** | database | 0.1902 | 0.8128 | 1325 | 463 | 1081.03 MB | 921.72 ms | 231621.8 |
| **E1: Reactive Baseline** | scientific | 0.1473 | 0.4043 | 4 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E2: Always-Proactive** | scientific | 0.0273 | 0.1389 | 0 | 333 | 3247.23 MB | 571.44 ms | 357683.7 |
| **E3: Prediction Only** | scientific | 0.1473 | 0.4043 | 4 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | scientific | 0.1473 | 0.4043 | 4 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E5: ML + Cost-Aware (PREFRAG)** | scientific | 0.1473 | 0.4043 | 4 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E1: Reactive Baseline** | microservice | 0.0028 | 0.0135 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E2: Always-Proactive** | microservice | 0.0014 | 0.0624 | 0 | 333 | 1.09 MB | 194.48 ms | 6077.1 |
| **E3: Prediction Only** | microservice | 0.0028 | 0.0135 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | microservice | 0.0028 | 0.0135 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E5: ML + Cost-Aware (PREFRAG)** | microservice | 0.0028 | 0.0135 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E1: Reactive Baseline** | adversarial_churn | 0.0488 | 0.0805 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E2: Always-Proactive** | adversarial_churn | 0.0064 | 0.0229 | 0 | 333 | 41.19 MB | 559.76 ms | 83267.2 |
| **E3: Prediction Only** | adversarial_churn | 0.0488 | 0.0805 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | adversarial_churn | 0.0488 | 0.0805 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E5: ML + Cost-Aware (PREFRAG)** | adversarial_churn | 0.0488 | 0.0805 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E1: Reactive Baseline** | game_engine | 0.6282 | 0.8718 | 8753 | 1 | 3.41 MB | 1.15 ms | 420.5 |
| **E2: Always-Proactive** | game_engine | 0.1469 | 0.7379 | 1271 | 333 | 543.98 MB | 285.80 ms | 70592.0 |
| **E3: Prediction Only** | game_engine | 0.6313 | 0.8718 | 8815 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | game_engine | 0.6313 | 0.8718 | 8815 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E5: ML + Cost-Aware (PREFRAG)** | game_engine | 0.6313 | 0.8718 | 8815 | 0 | 0.00 MB | 0.00 ms | 0.0 |

## 4. Cross-Workload Generalization Matrix

| Target Workload Family | MAE (t+10) | RMSE (t+10) | R² (t+10) | Domain Status |
|---|:---:|:---:|:---:|:---:|
| **web_server** | 0.1079 | 0.1277 | -28.6198 | In-Domain |
| **database** | 0.6707 | 0.7335 | -4.2344 | Cross-Domain |
| **scientific** | 0.3150 | 0.4128 | -6.6260 | Cross-Domain |
| **microservice** | 0.1587 | 0.1989 | -373555.5000 | Cross-Domain |
| **adversarial_churn** | 0.2994 | 0.3671 | -8771.2969 | Cross-Domain |

## 5. Feature Group Ablation Study (7 Semantic Groups)

| Ablation Variant | MAE (t+10) | R² (t+10) | Error Increase (%) |
|---|:---:|:---:|:---:|:---:|
| **Full Model (All Features)** | 0.0217 | 0.9860 | +0.00% |
| **Ablate: Allocation Features** | 0.0211 | 0.9856 | -2.91% |
| **Ablate: Deallocation Features** | 0.0211 | 0.9860 | -2.82% |
| **Ablate: Lifetime Features** | 0.0205 | 0.9857 | -5.74% |
| **Ablate: Freespace Features** | 0.0206 | 0.9879 | -5.07% |
| **Ablate: Temporal Features** | 0.0206 | 0.9858 | -5.26% |
| **Ablate: Workload Features** | 0.0199 | 0.9879 | -8.62% |
| **Ablate: Lag_trend Features** | 0.0207 | 0.9875 | -4.52% |

## 6. Feature Attribution Ranking

| Rank | Feature Name | Importance Score |
|---|---|:---:|
| 1 | `current_f1` | 0.9900 |
| 2 | `current_f2` | 0.0017 |
| 3 | `largest_free_block` | 0.0013 |
| 4 | `active_payload_bytes` | 0.0006 |
| 5 | `call_stack_entropy` | 0.0005 |
| 6 | `memory_utilization` | 0.0005 |
| 7 | `total_free_bytes` | 0.0004 |
| 8 | `delta_f1` | 0.0004 |
| 9 | `delta_allocated_bytes` | 0.0004 |
| 10 | `alloc_size_max` | 0.0003 |
