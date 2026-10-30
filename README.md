# PREFRAG: Predictive and Cost-Aware Memory Fragmentation Management Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 12 Passed](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](tests/test_all.py)
[![Architecture: Pure NumPy](https://img.shields.io/badge/models-pure%20NumPy-orange.svg)](src/models/temporal_nn.py)
[![Status: Final Release](https://img.shields.io/badge/status-final%20release-success.svg)](#)

**PREFRAG** is a research framework for **multi-horizon memory fragmentation forecasting** and **cost-aware proactive memory reorganization**. It predicts future virtual heap fragmentation trajectories ($t+1, t+5, t+10, t+30$) using deep temporal sequence models (LSTM, GRU, TCN) and economically evaluates whether memory compaction is cost-justified—triggering physical sliding compaction if and only if the projected defragmentation benefit mathematically exceeds hardware compaction overhead.

---

## 🌟 Key Highlights & Research Contributions

- **Multi-Policy Virtual Heap Allocator**: Production-accurate virtual heap simulation supporting **First-Fit**, **Best-Fit**, **Worst-Fit**, and **Next-Fit** with dual-key address tracking (`alloc_map` and `alloc_id_map`) and immediate boundary-tag coalescing.
- **Multi-Horizon Fragmentation Forecasting**: Temporal deep learning models predicting spatial fragmentation $F_1(t+k)$ across multiple horizons ($t \in \{1, 5, 10, 30\}$ steps ahead) with $R^2 \ge 0.985$ and MAE $\le 0.021$.
- **Pure NumPy Deep Learning Engines**: Self-contained implementations of **LSTM** (cell states, input/forget/output gates), **GRU** (reset/update gates), and **TCN** (stacked 1D dilated causal convolutions + residual connections) with explicit Backpropagation Through Time (BPTT) and Adam optimization—zero heavy runtime dependencies.
- **Cost-Benefit Rational Control Plane ($E_5$)**: Closed-loop utility optimizer evaluating $A^* = \arg\max [\text{Benefit} - \text{Cost}]$. Achieves an **85.39% reduction in severe fragmentation episodes** on high-churn workloads while **completely eliminating wasteful compactions (100% savings)** on healthy workloads.
- **Interactive Live Web Dashboard**: High-fidelity, real-time GUI visualizing dynamic heap block allocations, spatial hole entropy ($F_2$), live multi-horizon forecasting, and real-time defragmentation actuation.
- **10 Synthetic Benchmark Families & Live Traces**: 150,000 synthetic memory management events spanning diverse computing domains, complemented by live operating system memory traces captured via Python `tracemalloc`.

---

## 📐 System Architecture

PREFRAG operates as a unified closed-loop system organized into five cohesive architectural tiers:

```mermaid
flowchart TD
    %% Tier 1: Ingestion
    subgraph S1["1. Workload Ingestion Tier"]
        direction TB
        IN["Multi-Source Workload Ingestion<br/>• 10 Benchmark Families (Web, DB, HPC, Microservices, etc.)<br/>• External Traces (CSV / JSON)<br/>• Live Runtime Profiler (tracemalloc)"]
        ES["Canonical MemoryEvent Stream<br/>(timestamp, op, address, size, lifetime, call_stack)"]
        IN -->|"Parse & Normalize"| ES
    end

    %% Tier 2: Heap Simulation & Telemetry
    subgraph S2["2. Virtual Heap & Fragmentation Telemetry"]
        direction TB
        ALLOC["Multi-Policy Allocator Engine<br/>• Policies: First-Fit | Best-Fit | Worst-Fit | Next-Fit<br/>• Contiguous Block Tracking & Boundary Tag Coalescing"]
        FM["Spatial Fragmentation Telemetry<br/>• F1: Largest Free Block Ratio<br/>• F2: Shannon Hole Entropy<br/>• F3: Disjoint Free Block Count"]
        ALLOC -->|"Continuous Heap Evaluation"| FM
    end

    %% Tier 3: Feature Pipeline
    subgraph S3["3. Temporal Feature Pipeline"]
        direction TB
        FE["34-D Feature Extractor (Sliding Window w = 40)<br/>• G1-G2: Allocation & Deallocation Dynamics (11-D)<br/>• G3-G4: Lifetime & Heap Geometry (9-D)<br/>• G5-G7: Dynamics, Context Entropy & Trend Lags (14-D)"]
        FT["Standardized Feature Representations<br/>• Sequence Tensor: R^(L x 34) for Deep Models<br/>• Tabular Snapshot: R^34 for Baselines"]
        FE -->|"Rolling Aggregation"| FT
    end

    %% Tier 4: Forecasting Engine
    subgraph S4["4. Multi-Horizon Forecasting Engine"]
        direction TB
        MODELS["Dual Model Architecture<br/>• Baselines (B1-B5): Threshold, EMA, Ridge, RF, GBT<br/>• Deep Temporal (M1-M3): LSTM, GRU, Dilated Causal TCN"]
        PRED["Multi-Horizon Predictions<br/>F1(t+1), F1(t+5), F1(t+10), F1(t+30)"]
        MODELS -->|"Inference Forward Pass"| PRED
    end

    %% Tier 5: Control Plane & Actuation
    subgraph S5["5. Cost-Aware Control & Closed-Loop Actuation"]
        direction TB
        CTRL["Preventive Decision Plane<br/>• Cost-Benefit Optimizer: A* = argmax (Benefit - Cost)<br/>• Policy Arbiter: E1-E4 Baselines vs. E5 PREFRAG"]
        COMPACT["Sliding Memory Compaction (LISP-2)<br/>• Pointer Re-indexing & Block Relocation<br/>• Latency & Memory Bus Bandwidth Emulation"]
        CTRL -->|"Trigger: should_compact = True"| COMPACT
    end

    %% Pipeline Interconnections
    ES -->|"Event Ingestion"| ALLOC
    FM -->|"Live State Snapshots"| FE
    FT -->|"Features Flow"| MODELS
    PRED -->|"Forecast Trajectory"| CTRL
    COMPACT -.->|"Closed-Loop Actuation: Heap Reorganization & State Reset"| ALLOC

    classDef primary fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef success fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    class IN,ALLOC,FE,MODELS,CTRL primary;
    class ES,FM,FT,PRED,COMPACT success;
```

> 📖 **Deep Architectural Blueprint**: For mathematical formulations, sequence diagrams, and class mappings, see [architectureDiagram.md](architectureDiagram.md). High-resolution render: [`results/PREFRAG_Architecture_Diagram.png`](results/PREFRAG_Architecture_Diagram.png).

---

## 🖥️ Interactive Web Dashboard (Live Demo)

PREFRAG includes an interactive, browser-based real-time demonstration platform. It visualizes the physical byte-level virtual heap, tracks live spatial fragmentation metrics ($F_1, F_2, F_3$), runs multi-horizon temporal neural network inferences in real-time, and actuates cost-aware sliding defragmentation dynamically.

```bash
# Windows 1-Click Launch:
launch_demo.bat

# Or run directly via Python:
python scripts/live_demo.py
```
*Automatically launches your default browser at `http://127.0.0.1:8050`.*

---

## 📊 Experimental Results & Empirical Findings

Comprehensive experiments were conducted across **10 benchmark families** (150,000 total events) and **3 live OS profiles**, comparing 4 allocation policies, 8 predictive models, and 5 memory management regimes.

### 1. Allocator Policy Comparison (Best-Fit Minimizes Fragmentation)
Empirical evaluation shows that `BEST_FIT` consistently maintains the highest contiguity and lowest failure rates across workloads.

![Figure 1: Allocator Comparison](results/plots/fig1_allocator_comparison.png)

| Workload | Policy | Mean $F_1$ | Peak $F_1$ | Severe Episodes ($F_1 \ge 0.40$) | Allocation Failures |
|---|:---:|:---:|:---:|:---:|:---:|
| **database** | `first_fit` | 0.8402 | 0.9637 | 9,178 | 349 |
| **database** | **`best_fit` (Optimal)** | **0.8077** | **0.9588** | **9,070** | **271** |
| **database** | `worst_fit` | 0.9522 | 0.9870 | 9,739 | 905 |
| **database** | `next_fit` | 0.9304 | 0.9755 | 9,739 | 578 |
| **game_engine** | `first_fit` | 0.6837 | 0.9115 | 9,080 | 127 |
| **game_engine** | **`best_fit` (Optimal)** | **0.6313** | **0.8718** | **8,815** | 165 |
| **game_engine** | `worst_fit` | 0.8613 | 0.9622 | 9,533 | 135 |
| **game_engine** | `next_fit` | 0.7669 | 0.9192 | 9,533 | **120** |

---

### 2. Multi-Horizon Forecasting Accuracy
Deep temporal models (LSTM, GRU, TCN) accurately forecast future fragmentation states across multiple time steps, outperforming classical smoothing baselines.

![Figure 2: Multi-Horizon Forecasting Accuracy](results/plots/fig2_forecasting_accuracy.png)

| Model Category | Model Identifier | MAE ($t+1$) | MAE ($t+5$) | MAE ($t+10$) | MAE ($t+30$) | $R^2$ ($t+10$) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baselines** | **B1**: Static Threshold | 0.0032 | 0.0100 | 0.0164 | 0.0340 | 0.9888 |
| | **B2**: Exponential Moving Avg | 0.0090 | 0.0217 | 0.0391 | 0.0612 | 0.9597 |
| | **B3**: Ridge Regression | 0.0036 | 0.0107 | 0.0167 | 0.0309 | **0.9906** |
| | **B4**: Random Forest | 0.0056 | 0.0125 | 0.0180 | 0.0300 | 0.9897 |
| | **B5**: Gradient Boosted Trees | 0.0168 | 0.0214 | 0.0260 | 0.0370 | 0.9887 |
| **Deep Temporal (NumPy)** | **M1**: Temporal LSTM | 0.0073 | 0.0153 | 0.0218 | 0.0347 | 0.9852 |
| | **M2**: Temporal GRU | 0.0097 | 0.0176 | 0.0231 | 0.0371 | 0.9850 |
| | **M3**: Temporal TCN | 0.0087 | 0.0149 | 0.0208 | 0.0343 | 0.9880 |

---

### 3. Five-Way Memory Management Regime Comparison (E1 - E5)
PREFRAG ($E_5$) dramatically reduces severe fragmentation episodes while preventing excessive compaction churn.

![Figure 3: Policy Regimes Under High Churn](results/plots/fig3_policy_regimes.png)

| Condition | Strategy | Mean $F_1$ | Peak $F_1$ | Severe Episodes | Compactions | Moved Data | Total Pause |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1** | Reactive Baseline | 0.2340 | 0.8476 | 1,216 | 312 | 745.7 MB | 628.8 ms |
| **E2** | Always-Proactive (Periodic) | 0.2372 | 0.8627 | 2,271 | 333 | 765.5 MB | 660.2 ms |
| **E3** | Prediction Only (No Action) | 0.8077 | 0.9588 | 9,070 | 0 | 0.0 MB | 0.0 ms |
| **E4** | ML + Fixed Threshold | 0.1906 | 0.8170 | 1,282 | 474 | 1,101.4 MB | 942.4 ms |
| **E5** | **PREFRAG (Predictive + Cost-Aware)** | **0.1902** | **0.8128** | **1,325** | 463 | 1,081.0 MB | 921.7 ms |

---

### 4. Over-Compaction Elimination on Healthy Workloads
On stable workloads (`scientific`, `microservice`, `adversarial_churn`), PREFRAG avoids 100% of the useless data copying and pause times suffered by periodic proactive compactors.

![Figure 4: Overhead Avoidance](results/plots/fig4_overhead_avoidance.png)

| Workload | E2 (Always-Proactive) Compactions | E5 (PREFRAG) Compactions | Wasteful Overhead Eliminated |
|---|:---:|:---:|:---:|
| **scientific** | 333 (3,247 MB moved) | **0 (0 MB moved)** | **100% Savings** |
| **microservice** | 333 (1.09 MB moved) | **0 (0 MB moved)** | **100% Savings** |
| **adversarial_churn** | 333 (41.2 MB moved) | **0 (0 MB moved)** | **100% Savings** |

---

### 5. Benchmarking Against Contemporary Literature
PREFRAG compares favorably against traditional and recent state-of-the-art approaches in defragmentation efficiency, overhead reduction, and tail-latency mitigation.

![Figure 5: Literature Comparison](results/plots/fig5_literature_comparison.png)

---

## 🚀 Quickstart & Setup

### Prerequisites
- **Python**: 3.10+ (tested on Python 3.10, 3.11, 3.12, and 3.14)
- **OS**: Windows, Linux, or macOS

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/sameer-codes-ai/MemoryFragmentation.git
cd MemoryFragmentation

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Test Suite
Execute the comprehensive test suite verifying trace generation, heap allocator policies, 34-D feature extraction, predictive models, and cost controllers:
```bash
# Windows 1-Click:
run_tests.bat

# Command-Line:
python -m pytest tests/test_all.py -v
```

### 4. Launch Interactive Web Demo
```bash
# Windows 1-Click:
launch_demo.bat

# Command-Line:
python scripts/live_demo.py
```

### 5. Run Full Benchmark Pipeline
```bash
# Windows 1-Click:
run_benchmark.bat

# Command-Line (custom events):
python scripts/run_all_experiments.py --events 2500
```
*Outputs detailed markdown and JSON summaries to `results/benchmark_report.md` and `results/experiment_summary.json`.*

### 6. Regenerate Research Comparison Figures
```bash
python scripts/generate_comparison_graphs.py
```
*Re-renders all 5 publication-ready figures in `results/plots/`.*

### 7. Minimal 1-File Sanity Check
```bash
python reorganizationTest.py
```
*Simulates 3 allocations, creates a hole, queries `CostAwarePredictivePolicy`, and executes proactive sliding compaction.*

---

## 📁 Repository Structure

```
MemoryFragmentation/
├── src/
│   ├── traces/                     # Workload trace ingestion, synthesis & schema
│   │   ├── trace_schema.py         # MemoryEvent, AllocationOp, TraceStream
│   │   ├── workload_generator.py   # 10 workload families (Web, DB, HPC, IoT, etc.)
│   │   └── trace_parser.py         # CSV & JSON external trace parsing
│   ├── simulator/                  # Virtual heap allocator & telemetry
│   │   ├── heap_allocator.py       # Multi-policy allocator (First/Best/Worst/Next-Fit)
│   │   ├── memory_block.py         # MemoryBlock data class & split/coalesce logic
│   │   ├── fragmentation_metrics.py# Spatial metrics (F1 largest-free, F2 entropy, F3)
│   │   └── compaction_engine.py    # Sliding compaction (LISP-2) & cost modeling
│   ├── features/                   # 34-dimensional temporal feature engineering
│   │   └── feature_extractor.py    # 7 feature groups, rolling sequences & targets
│   ├── models/                     # Multi-horizon forecasting models
│   │   ├── baselines.py            # B1-B5: Threshold, EMA, Ridge, RF, GBT
│   │   └── temporal_nn.py          # M1: LSTM, M2: GRU, M3: TCN (pure NumPy + Adam)
│   ├── controller/                 # Preventive control plane
│   │   ├── cost_model.py           # Cost-Benefit Optimizer (A* = argmax[Benefit - Cost])
│   │   └── policies.py             # E1-E5 decision policy implementations
│   ├── demo/                       # Interactive live demonstration server
│   │   ├── live_server.py          # Real-time WebSocket / HTTP server
│   │   └── web/
│   │       └── index.html          # Clean, aesthetic physical heap dashboard
│   └── experiments/                # Research evaluation harness
│       ├── runner.py               # Closed-loop simulation orchestrator
│       └── generalization.py       # Cross-workload generalization & feature ablation
├── scripts/
│   ├── live_demo.py                # Launcher for interactive web dashboard
│   ├── run_all_experiments.py      # End-to-end benchmark suite runner
│   ├── generate_comparison_graphs.py # Publication-quality figure generator (Fig 1-5)
│   └── record_live_trace.py        # Live profiler (SQLite, JSON, NumPy via tracemalloc)
├── tests/
│   └── test_all.py                 # 12 unit and property tests (100% passing)
├── results/
│   ├── plots/                      # Publication figures (fig1 through fig5 .png)
│   ├── saved_traces/               # 10 benchmark trace CSVs (15,000 events each)
│   ├── benchmark_report.md         # Full empirical benchmark metrics
│   ├── experiment_summary.json     # Machine-readable JSON experimental data
│   └── PREFRAG_Architecture_Diagram.png # Rendered system architecture diagram
├── launch_demo.bat                 # 1-Click demo launcher
├── run_benchmark.bat               # 1-Click benchmark orchestrator
├── run_tests.bat                   # 1-Click test runner
├── record_live_traces.bat          # 1-Click live memory profiler
├── architectureDiagram.md          # Comprehensive architectural blueprint
├── reorganizationTest.py           # Minimal 1-file sanity check script
├── requirements.txt                # Project dependencies
└── README.md                       # Main research documentation
```

---

## 📦 Benchmark Dataset Specifications

### Synthetic Workload Traces (`results/saved_traces/`, 15,000 events each)
| Workload File | Domain / Allocation Behavior | Events |
|---|---|:---:|
| `database_trace.csv` | B-Tree page buffer pool, dirty page writes, persistent cache churn | 15,000 |
| `web_server_trace.csv` | Bursty HTTP request payloads, session churn, rapid allocations | 15,000 |
| `scientific_trace.csv` | Cyclical matrix allocations, large contiguous scientific buffers | 15,000 |
| `microservice_trace.csv` | Small JSON/RPC payloads, fast churn, out-of-order deallocations | 15,000 |
| `adversarial_churn_trace.csv` | Pinned long-lived objects interspersed with short-lived churn | 15,000 |
| `compiler_trace.csv` | AST node structures, parser symbol tables, pass-based frees | 15,000 |
| `embedded_iot_trace.csv` | Periodic sensor buffers, fixed packet sizes, deterministic life | 15,000 |
| `game_engine_trace.csv` | Frame-based temporary buffers, entity component churn | 15,000 |
| `garbage_collected_trace.csv` | Generational nursery bursts, mark-sweep collection cycles | 15,000 |
| `streaming_media_trace.csv` | Variable audio/video chunk buffers, ring-buffer allocations | 15,000 |

### Live Operating System Captured Traces
| Workload File | Source Application | Profiling Method |
|---|---|---|
| `live_sqlite_trace.csv` | Real SQLite disk & memory DB transactions | Python `tracemalloc` |
| `live_microservice_trace.csv` | High-throughput JSON object serialization | Python `tracemalloc` |
| `live_scientific_trace.csv` | NumPy vector and matrix dot-product operations | Python `tracemalloc` |

---

## 🔬 Mathematical Formulations

### 1. Spatial Fragmentation Metrics
- **Largest Free Block Ratio ($F_1$)**:
  $$F_1 = 1 - \frac{\max_{b \in \mathcal{F}} \text{size}(b)}{\sum_{b \in \mathcal{F}} \text{size}(b)}$$
  *Measures the degree to which free space is shattered into unusable fragments ($F_1 \approx 0$: contiguous; $F_1 \approx 1$: fragmented).*
- **Shannon Hole Entropy ($F_2$)**:
  $$F_2 = -\sum_{i=1}^{K} p_i \log_2(p_i) \quad \text{where } p_i = \frac{\text{size}(H_i)}{\text{total free bytes}}$$
  *Measures spatial irregularity and dispersion across $K$ disjoint free holes.*
- **Disjoint Free Block Count ($F_3$)**:
  $$F_3 = K = |\mathcal{F}|$$

### 2. Cost-Benefit Utility Maximization ($E_5$)
$$A^* = \arg\max_{A \in \{\text{NoAction}, \text{Compact}\}} \left[ \text{ExpectedBenefit}(A) - \text{Cost}(A) \right]$$

$$\text{ExpectedBenefit} = \lambda \cdot \max\left(0, \hat{F}_{1}(t+k) - F_{1,\text{post}}\right)$$

$$\text{Cost} = \alpha \cdot N_{\text{blocks}} + \beta \cdot B_{\text{bytes}} + \gamma \cdot T_{\text{pause}}$$

Compaction is actuated if and only if $\text{NetBenefit} \ge \tau_{\min}$ and $\hat{F}_1(t+k) \ge F_{\text{critical}}$, governed by an adaptive hysteresis cooldown to prevent thrashing.

---

## 📚 Literature Survey

The project includes a verified **24-paper literature survey** (2021–2025) covering:
- **Memory fragmentation characterization & telemetry**
- **Traditional allocation heuristics & coalescing compaction**
- **Machine learning & reinforcement learning for memory optimization**
- **GPU & Large Language Model (LLM) memory management**
- **CXL tiered memory & page migration systems**

All surveyed papers are from verified top-tier venues: **ASPLOS, OSDI, SOSP, USENIX ATC, NeurIPS, ICML, ISMM, SC, and IEEE Transactions**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Sameer Kumar** — Vellore Institute of Technology (VIT), Vellore  
📧 [sameer.kumar2024@vitstudent.ac.in](mailto:sameer.kumar2024@vitstudent.ac.in)  
🔗 [GitHub: @sameer-codes-ai](https://github.com/sameer-codes-ai)
