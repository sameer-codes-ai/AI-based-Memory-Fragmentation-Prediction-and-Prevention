# PREFRAG: Experimental Benchmark Report

## 1. Multi-Horizon Model Performance Comparison

| Model | MAE (t+1) | MAE (t+5) | MAE (t+10) | MAE (t+30) | R² (t+10) |
|---|:---:|:---:|:---:|:---:|:---:|
| **B1: Static Threshold** | 0.0032 | 0.0099 | 0.0166 | 0.0349 | 0.9898 |
| **B2: Moving Average** | 0.0086 | 0.0204 | 0.0368 | 0.0570 | 0.9660 |
| **B3: Linear Regression** | 0.0035 | 0.0104 | 0.0157 | 0.0238 | 0.9926 |
| **B4: Random Forest** | 0.0043 | 0.0088 | 0.0116 | 0.0155 | 0.9958 |
| **B5: Gradient Boosted Trees** | 0.0168 | 0.0210 | 0.0242 | 0.0299 | 0.9916 |
| **M1: Temporal LSTM** | 0.0067 | 0.0105 | 0.0136 | 0.0208 | 0.9939 |
| **M2: Temporal GRU** | 0.0076 | 0.0104 | 0.0144 | 0.0218 | 0.9935 |
| **M3: Temporal TCN** | 0.0078 | 0.0124 | 0.0146 | 0.0199 | 0.9940 |

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
| **E4: ML + Threshold** | scientific | 0.0545 | 0.2806 | 0 | 192 | 2138.64 MB | 367.41 ms | 236496.2 |
| **E5: ML + Cost-Aware (PREFRAG)** | scientific | 0.1318 | 0.3972 | 0 | 6 | 34.96 MB | 7.46 ms | 3872.3 |
| **E1: Reactive Baseline** | database | 0.8608 | 0.9646 | 23652 | 5 | 12.32 MB | 10.91 ms | 2749.2 |
| **E2: Always-Proactive** | database | 0.2354 | 0.8726 | 5130 | 833 | 2039.12 MB | 1746.39 ms | 442506.2 |
| **E3: Prediction Only** | database | 0.8754 | 0.9646 | 24178 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | database | 0.8734 | 0.9646 | 24111 | 1 | 2.42 MB | 2.13 ms | 535.1 |
| **E5: ML + Cost-Aware (PREFRAG)** | database | 0.8609 | 0.9646 | 23658 | 5 | 12.34 MB | 10.86 ms | 2740.2 |
| **E1: Reactive Baseline** | adversarial_churn | 0.0015 | 0.0052 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E2: Always-Proactive** | adversarial_churn | 0.0015 | 0.0052 | 0 | 833 | 12.78 MB | 544.40 ms | 27753.3 |
| **E3: Prediction Only** | adversarial_churn | 0.0015 | 0.0052 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E4: ML + Threshold** | adversarial_churn | 0.0015 | 0.0052 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |
| **E5: ML + Cost-Aware (PREFRAG)** | adversarial_churn | 0.0015 | 0.0052 | 0 | 0 | 0.00 MB | 0.00 ms | 0.0 |

## 3. Cross-Workload Generalization Matrix

| Target Workload Family | MAE (t+10) | RMSE (t+10) | R² (t+10) | Domain Status |
|---|:---:|:---:|:---:|:---:|
| **web_server** | 0.0760 | 0.0847 | -12.0243 | In-Domain |
| **database** | 0.4992 | 0.5292 | -1.7244 | Cross-Domain |
| **scientific** | 0.7132 | 0.7674 | -25.3580 | Cross-Domain |
| **microservice** | 0.0673 | 0.0798 | -60209.2123 | Cross-Domain |
| **adversarial_churn** | 0.0055 | 0.0212 | -6191.6184 | Cross-Domain |

## 4. Feature Group Ablation Study

| Ablation Variant | MAE (t+10) | R² (t+10) | Error Increase (%) |
|---|:---:|:---:|:---:|:---:|
| **Full Model (All Features)** | 0.0136 | 0.9939 | +0.00% |
| **Ablate: Allocation Features** | 0.0140 | 0.9938 | +2.97% |
| **Ablate: Deallocation Features** | 0.0143 | 0.9937 | +4.74% |
| **Ablate: Lifetime Features** | 0.0150 | 0.9932 | +9.97% |
| **Ablate: Freespace Features** | 0.0298 | 0.9742 | +119.16% |
| **Ablate: Temporal Features** | 0.0141 | 0.9939 | +3.47% |
| **Ablate: Workload Features** | 0.0144 | 0.9938 | +5.68% |

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
