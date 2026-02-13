# Causal Evaluation of Alert-Based Information Quality Governance in Biomedical Entity Recognition

[![License: ODbL](https://img.shields.io/badge/License-ODbL-brightgreen.svg)](https://opendatacommons.org/licenses/odbl/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.13026%2Fp5pn--ty93-blue)](https://doi.org/10.13026/p5pn-ty93)

This repository contains the complete code and analysis pipeline for the research paper:

**"Causal Evaluation of Alert-Based Information Quality Governance in Biomedical Entity Recognition under Long-Tail Distributional Shift"**

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This study develops and validates a **causal framework** for evaluating alert-based Information Quality (IQ) governance in biomedical named entity recognition (NER) and normalization (NEN) systems operating under long-tail distributional shift.

### Research Questions
1. **RQ1 (Causal Effect)**: Can quasi-experimental designs identify the causal effect of alert-based monitoring on performance degradation?
2. **RQ2 (Mechanism)**: How can mediation analysis decompose monitoring effects into pathway-specific components?
3. **RQ3 (Optimization)**: How can cost–benefit frameworks determine optimal alert thresholds?
4. **RQ4 (Heterogeneity)**: Where does monitoring deliver the greatest value?

### Key Findings
- **DiD Effect**: Alert monitoring reduces performance degradation by **5.82 percentage points** (p < 0.001)
- **Mechanism**: **106.9%** of the effect operates through drift detection (complete mediation)
- **Cost Savings**: Optimal threshold reduces operational costs by **71%** vs. no monitoring
- **Heterogeneity**: Effects vary **31-fold** across deployment contexts

## ✨ Key Features

### Methodological Innovations

- **Quasi-Experimental Design**: Difference-in-Differences (DiD) with parallel trends validation
- **Instrumental Variables**: Two-stage least squares with weak instrument diagnostics
- **Mediation Analysis**: Causal pathway decomposition with bootstrap confidence intervals
- **Cost-Benefit Optimization**: ROC-based threshold selection minimizing total operational costs
- **Heterogeneous Treatment Effects**: Context-specific impact estimation
- **Markov Temporal Modeling**: State transition probabilities and steady-state analysis

### Technical Capabilities

- External benchmark validation (BC5CDR, NCBI-disease style)
- SOTA comparison with statistical significance testing
- Error analysis by entity complexity and rarity
- Sensitivity analysis (ROPE, subsampling)
- 12 publication-quality figures (300 DPI)
- 13 comprehensive analysis tables

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/alert-iq-governance-biomedical-ner.git
cd alert-iq-governance-biomedical-ner

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core packages:
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Statistical functions
- `scikit-learn>=1.0.0` - Machine learning utilities
- `matplotlib>=3.4.0` - Visualization
- `seaborn>=0.11.0` - Statistical plotting

See `requirements.txt` for complete list.

## 📊 Data

### Source

This analysis uses the **PhysioNet Synthetic Mention Corpora for Disease Entity Recognition and Normalization (v1.0.0)**:

**Citation:**
```
Sasse, K., & Osborne, J. D. (2025). Synthetic Mention Corpora for Disease 
Entity Recognition and Normalization (version 1.0.0). PhysioNet. 
RRID:SCR_007345. https://doi.org/10.13026/p5pn-ty93
```

**PhysioNet Citation:**
```
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., 
Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and 
PhysioNet: Components of a new research resource for complex physiologic 
signals. Circulation [Online]. 101(23), pp. e215–e220.
```

### Dataset Characteristics

- **Size**: 128,945 disease mention observations
- **Entities**: 47,654 unique UMLS CUIs (rare diseases, ≤5 mentions each)
- **Time Periods**: 1-7 periods per CUI (median = 3)
- **Generation**: Fine-tuned LLaMA-2-13B-Chat on SemEval 2015 Task 14 data
- **License**: Open Database License (ODbL v1.0)

## 🚀 Usage

### Option 1: Run Complete Pipeline

Execute the entire analysis with a single command:

```bash
python scripts/10_run_full_pipeline.py
```

This will:
1. Download raw data
2. Engineer features
3. Run all causal analyses
4. Generate all figures and tables
5. Create comprehensive report

**Expected runtime**: ~15-20 minutes

### Option 2: Run Individual Analyses

Execute analyses step-by-step:

```bash
# Step 1: Download and preprocess data
python scripts/01_download_data.py
python scripts/02_preprocess_data.py

# Step 2: Run causal inference
python scripts/03_run_causal_analysis.py

# Step 3: Cost-benefit optimization
python scripts/04_run_cost_benefit.py

# Step 4: Heterogeneous effects
python scripts/05_run_heterogeneity.py

# Step 5: External validation
python scripts/06_run_validation.py

# Step 6: Markov modeling
python scripts/07_run_markov.py

# Step 7: Generate outputs
python scripts/08_generate_figures.py
python scripts/09_generate_tables.py
```

### Option 3: Interactive Exploration

Use Jupyter notebooks for interactive analysis:

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## 📁 Repository Structure

```
alert-iq-governance-biomedical-ner/
│
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                            # Open Database License (ODbL v1.0)
├── .gitignore                        # Git ignore file
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration parameters
│   ├── data_loader.py                # Data loading and preprocessing
│   ├── feature_engineering.py        # Feature creation
│   ├── alert_signals.py              # Alert signal generation
│   ├── performance_simulation.py     # Performance degradation simulation
│   ├── causal_inference.py           # DiD, IV, Mediation analyses
│   ├── cost_benefit.py               # ROC and cost optimization
│   ├── heterogeneity.py              # HTE analysis
│   ├── markov_model.py               # Markov chain modeling
│   ├── visualization.py              # All plotting functions
│   └── utils.py                      # Helper functions
│
├── scripts/                          # Analysis scripts
│   ├── 01_download_data.py           # Download raw data
│   ├── 02_preprocess_data.py         # Preprocess and engineer features
│   ├── 03_run_causal_analysis.py     # Run DiD, IV, Mediation
│   ├── 04_run_cost_benefit.py        # Cost-benefit optimization
│   ├── 05_run_heterogeneity.py       # Context-specific effects
│   ├── 06_run_validation.py          # External validation & SOTA
│   ├── 07_run_markov.py              # Markov modeling
│   ├── 08_generate_figures.py        # Generate all figures
│   ├── 09_generate_tables.py         # Generate all tables
│   └── 10_run_full_pipeline.py       # Master script
│
├── outputs/                          # All outputs
│   ├── figures/                      # 12 figures
│   │   ├── fig1_did.png
│   │   ├── fig2_parallel_trends.png
│   │   ├── fig3_event_study.png
│   │   ├── fig4_performance_drift.png
│   │   ├── fig5_mediation.png
│   │   ├── fig6_hte.png
│   │   ├── fig7_iv_first_stage.png
│   │   ├── fig8_roc_curve.png
│   │   ├── fig9_sota_comparison.png
│   │   ├── fig10_external_benchmarks.png
│   │   ├── fig11_markov_transition.png
│   │   └── fig12_error_analysis.png
│   │
│   ├── tables/                       # 13 CSV tables
│       ├── table1_did_results.csv
│       ├── table2_parallel_trends.csv
│       ├── table3_iv_analysis.csv
│       ├── table4_mediation.csv
│       ├── table5_hte_context.csv
│       ├── table6_sota_comparison.csv
│       ├── table7_external_benchmarks.csv
│       ├── table8_cost_benefit.csv
│       ├── table9_sensitivity.csv
│       ├── table10_error_analysis.csv
│       ├── table11_case_studies.csv
│       ├── table12_markov_transition.csv
│       └── table13_ablation.csv      
│
└── docs/                             # Documentation
    ├── DATA_DICTIONARY.md            # Variable descriptions
    └── REPRODUCTION.md               # Reproduction guide
```

See [Repository Structure](#repository-structure) section for details.

## 📈 Results

### Main Deliverables

#### Figures (12 publication-quality PNGs)

1. **fig1_did.png** - Difference-in-Differences visualization
2. **fig2_parallel_trends.png** - Pre-treatment trends validation
3. **fig3_event_study.png** - Dynamic treatment effects
4. **fig4_performance_drift.png** - Performance degradation curve
5. **fig5_mediation.png** - Causal mediation path diagram
6. **fig6_hte.png** - Heterogeneous treatment effects
7. **fig7_iv_first_stage.png** - Instrumental variable diagnostics
8. **fig8_roc_curve.png** - ROC curve and optimal threshold
9. **fig9_sota_comparison.png** - SOTA baseline comparison
10. **fig10_external_benchmarks.png** - External validation
11. **fig11_markov_transition.png** - State transition matrix
12. **fig12_error_analysis.png** - Error breakdown by complexity

#### Tables (13 comprehensive CSVs)

1. **table1_did_results.csv** - DiD coefficients with robust SE
2. **table2_parallel_trends.csv** - Pre-treatment test results
3. **table3_iv_analysis.csv** - IV diagnostics (F-stat, Cragg-Donald)
4. **table4_mediation.csv** - Mediation decomposition
5. **table5_hte_context.csv** - Context-specific treatment effects
6. **table6_sota_comparison.csv** - SOTA method comparison
7. **table7_external_benchmarks.csv** - External validation results
8. **table8_cost_benefit.csv** - Operational cost analysis
9. **table9_sensitivity.csv** - Robustness checks
10. **table10_error_analysis.csv** - Performance by complexity/rarity
11. **table11_case_studies.csv** - Top/bottom performer analysis
12. **table12_markov_transition.csv** - Transition probabilities
13. **table13_ablation.csv** - Component ablation study

### Key Metrics Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **DiD Estimate** | -0.0582 | Alert monitoring prevents 5.82pp degradation |
| **IV Estimate** | -0.3351 | Endogeneity-corrected effect (6× larger) |
| **Mediation %** | 106.9% | Complete mediation via drift detection |
| **Cost Savings** | 71% | vs. no monitoring baseline |
| **Optimal Threshold** | 0.105 | Jaccard distance for alert trigger |
| **ROC AUC** | 0.854 | Alert discriminative capacity |
| **HTE Range** | 31-fold | Context 4: -0.0376 to Context 1: -0.0012 |

## 📝 Dataset Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{sasse2025synthetic,
  author = {Sasse, Kuleen and Osborne, John David},
  title = {Synthetic Mention Corpora for Disease Entity Recognition and Normalization},
  year = {2025},
  publisher = {PhysioNet},
  version = {1.0.0},
  doi = {10.13026/p5pn-ty93},
  url = {https://doi.org/10.13026/p5pn-ty93}
}
```

## 📄 License

This project is licensed under the **Open Database License (ODbL v1.0)** - see the [LICENSE](LICENSE) file for details.

### Key Points

- **Data**: Open Database License (ODbL v1.0)
- **Code**: MIT License
- **Attribution Required**: Please cite both the paper and the dataset

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Reporting Issues

Found a bug or have a feature request? Please open an issue on GitHub.

## 📧 Contact

**Yong-Jae Lee**  
AI Lab, Cloud Group  
Future Technology Research Institute, TOBESOFT  
Email: yj11021@tobesoft.com

## 🙏 Acknowledgments

- **Dataset**: Kuleen Sasse & John David Osborne (PhysioNet)
- **Compute**: NVIDIA GPU Grant Program
- **Funding**: NIH grants P30AR072583 and R01AG057684
- **Infrastructure**: UAB Research Computing

## 📚 Additional Resources

- [Full Paper](docs/paper.pdf) (when published)
- [Supplementary Materials](docs/supplementary.pdf)
- [Methodology Details](docs/METHODOLOGY.md)
- [Reproduction Guide](docs/REPRODUCTION.md)
- [Data Dictionary](docs/DATA_DICTIONARY.md)
