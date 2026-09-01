# PREFRAG: Predictive and Cost-Aware Memory Fragmentation Management Using Machine Learning

PREFRAG is a research framework for **multi-horizon memory fragmentation forecasting** and **cost-aware preventive memory reorganization**.

---

## 🚀 Quickstart & Setup

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Test Suite
```bash
python -m unittest discover tests
```

### 4. Run End-to-End Benchmark Experiments
```bash
python scripts/run_all_experiments.py
```

---

## 📁 Repository Structure

```
MemoryFragmentation/
├── src/
│   ├── traces/          # Trace schema, workload generators (Web, DB, Scientific, Microservice, Adversarial)
│   ├── simulator/       # Virtual heap allocator, block coalescing, compaction engine, F1/F2/F3 metrics
│   ├── features/        # 28-dimensional temporal feature extraction pipeline across 6 feature groups
│   ├── models/          # Baselines (B1-B5) and Deep Temporal Forecasters (M1-M3)
│   ├── controller/      # Cost-benefit optimizer and decision policies (E1-E5)
│   └── experiments/     # Simulation harness, cross-workload generalization, and ablation suites
├── scripts/
│   └── run_all_experiments.py  # End-to-end benchmark orchestrator
├── tests/
│   └── test_all.py      # Unit & property test suite
├── results/             # Benchmark outputs, CSV traces, and markdown reports
├── requirements.txt     # Dependency list
├── literature_review.md # Verified 22-paper survey
├── novelty_analysis.md  # Adversarial kill-test and research gap identification
├── methodology.md       # Full dataset audit and experimental methodology
└── research_design.md   # Mathematical formulation and systems architecture
```

---

## 🔬 Core Novelty & Methodology

- **Multi-Horizon Fragmentation Forecasting**: Temporal sequence models ($M_1$ LSTM, $M_2$ GRU, $M_3$ TCN) predict future fragmentation states ($t+1, t+5, t+10, t+30$) from historical allocation dynamics.
- **Cost-Aware Preventive Intervention**: Optimizes $A^* = \arg\max_{A \in \mathcal{A}} [\text{ExpectedBenefit}(A) - \text{Cost}(A)]$ to eliminate unnecessary compactions and avoid pause-time overhead.
- **Cross-Workload Generalization**: Verified across multiple workload distributions without memorization.
