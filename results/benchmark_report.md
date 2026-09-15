# PREFRAG: Experimental Benchmark Report

## 1. Multi-Horizon Model Performance Comparison

| Model | MAE (t+1) | MAE (t+5) | MAE (t+10) | MAE (t+30) | R² (t+10) |
|---|:---:|:---:|:---:|:---:|:---:|
| **B1: Static Threshold** | 0.0032 | 0.0099 | 0.0166 | 0.0349 | 0.9898 |
| **B2: Moving Average** | 0.0086 | 0.0204 | 0.0368 | 0.0570 | 0.9660 |
| **B3: Linear Regression** | 0.0035 | 0.0104 | 0.0157 | 0.0238 | 0.9926 |
| **B4: Random Forest** | 0.0043 | 0.0088 | 0.0116 | 0.0155 | 0.9958 |
| **B5: Gradient Boosted Trees** | 0.0168 | 0.0210 | 0.0242 | 0.0299 | 0.9916 |
| **M1: Temporal LSTM** | 0.0056 | 0.0103 | 0.0140 | 0.0204 | 0.9937 |
| **M2: Temporal GRU** | 0.0079 | 0.0139 | 0.0159 | 0.0219 | 0.9930 |
| **M3: Temporal TCN** | 0.0063 | 0.0113 | 0.0150 | 0.0215 | 0.9936 |

## 2. Five-Way Memory Management Regime Comparison (E1 - E5)

| Condition | Workload | Mean F₁ | Peak F₁ | Severe Episodes | Compactions | Moved (MB) | Pause (ms) | Total Cost |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1: Reactive Baseline** | microservice | 0.0029 | 0.0201 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E2: Always-Proactive** | microservice | 0.0014 | 0.0624 | 0 | 833 | 2.75 MB | 485.64 ms | 15032.2 |
| **E3: Prediction Only** | microservice | 0.0029 | 0.0201 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | microservice | 0.0029 | 0.0201 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E5: ML + Cost-Aware (PREFRAG)** | microservice | 0.0029 | 0.0201 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E1: Reactive Baseline** | scientific | 0.1335 | 0.4011 | 1 | 1 | 6.60 MB | 1.32 ms | 727.4 |
| **E2: Always-Proactive** | scientific | 0.0280 | 0.1542 | 0 | 833 | 8179.39 MB | 1432.68 ms | 900185.1 |
| **E3: Prediction Only** | scientific | 0.1336 | 0.4011 | 1 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | scientific | 0.1336 | 0.4011 | 1 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E5: ML + Cost-Aware (PREFRAG)** | scientific | 0.1336 | 0.4011 | 1 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E1: Reactive Baseline** | database | 0.8608 | 0.9646 | 23652 | 5 | 12.32 MB | 10.91 ms | 2749.2 |
| **E2: Always-Proactive** | database | 0.2354 | 0.8726 | 5130 | 833 | 2039.12 MB | 1746.39 ms | 442506.2 |
| **E3: Prediction Only** | database | 0.8754 | 0.9646 | 24178 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | database | 0.2211 | 0.8699 | 2760 | 869 | 2167.75 MB | 1819.14 ms | 464515.5 |
| **E5: ML + Cost-Aware (PREFRAG)** | database | 0.2247 | 0.8699 | 2738 | 830 | 2068.54 MB | 1734.65 ms | 442931.7 |
| **E1: Reactive Baseline** | adversarial_churn | 0.0015 | 0.0052 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E2: Always-Proactive** | adversarial_churn | 0.0015 | 0.0052 | 0 | 833 | 12.78 MB | 544.40 ms | 27753.3 |
| **E3: Prediction Only** | adversarial_churn | 0.0015 | 0.0052 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | adversarial_churn | 0.0015 | 0.0052 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E5: ML + Cost-Aware (PREFRAG)** | adversarial_churn | 0.0015 | 0.0052 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |

## 3. Cross-Workload Generalization Matrix

| Target Workload Family | MAE (t+10) | RMSE (t+10) | R² (t+10) | Domain Status |
|---|:---:|:---:|:---:|:---:|
| **web_server** | 0.0857 | 0.1042 | -18.7256 | In-Domain |
| **database** | 0.7758 | 0.8251 | -5.6239 | Cross-Domain |
| **scientific** | 0.1940 | 0.2553 | -1.9166 | Cross-Domain |
| **microservice** | 0.1802 | 0.2214 | -463101.5625 | Cross-Domain |
| **adversarial_churn** | 0.0003 | 0.0004 | -1.5394 | Cross-Domain |

## 4. Feature Group Ablation Study

| Ablation Variant | MAE (t+10) | R² (t+10) | Error Increase (%) |
|---|:---:|:---:|:---:|:---:|
| **Full Model (All Features)** | 0.0140 | 0.9937 | +0.00% |
| **Ablate: Allocation Features** | 0.0135 | 0.9938 | -3.84% |
| **Ablate: Deallocation Features** | 0.0139 | 0.9937 | -0.47% |
| **Ablate: Lifetime Features** | 0.0136 | 0.9938 | -2.68% |
| **Ablate: Freespace Features** | 0.0343 | 0.9660 | +145.12% |
| **Ablate: Temporal Features** | 0.0139 | 0.9937 | -0.66% |
| **Ablate: Workload Features** | 0.0137 | 0.9937 | -2.35% |

## 5. Feature Attribution Ranking

| Rank | Feature Name | Importance Score |
|---|---|:---:|
| 1 | `current_f1` | 0.9368 |
| 2 | `current_f2` | 0.0561 |
| 3 | `delta_f1` | 0.0020 |
| 4 | `free_count` | 0.0010 |
| 5 | `alloc_to_free_ratio` | 0.0008 |
| 6 | `largest_free_block` | 0.0007 |
| 7 | `active_alloc_count` | 0.0002 |
| 8 | `active_payload_bytes` | 0.0002 |
| 9 | `lifetime_mean` | 0.0002 |
| 10 | `memory_utilization` | 0.0002 |
