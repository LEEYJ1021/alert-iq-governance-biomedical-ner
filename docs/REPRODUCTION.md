# Reproduction Guide

This guide provides step-by-step instructions for reproducing all results in the paper.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 500MB for data and outputs

### Software Dependencies

See `requirements.txt` for complete list. Core packages:

```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.13.0
```

## Quick Start (5 minutes)

Run the entire pipeline with a single command:

```bash
# Clone repository
git clone https://github.com/yourusername/alert-iq-governance-biomedical-ner.git
cd alert-iq-governance-biomedical-ner

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python scripts/10_run_full_pipeline.py
```

**Expected Output:**
- Processed data: `data/processed/analysis_ready.parquet`
- 12 figures: `outputs/figures/fig*.png`
- 13 tables: `outputs/tables/table*.csv`
- Report: `outputs/reports/COMPREHENSIVE_FINAL_REPORT.txt`

**Runtime**: ~15-20 minutes on standard laptop

## Step-by-Step Reproduction

### Step 1: Data Acquisition

Download the PhysioNet dataset:

```bash
python scripts/01_download_data.py
```

**Output**: `data/raw/SYNTHETIC_MENTIONS.csv` (128,945 rows)

**Verification**:
```python
import pandas as pd
df = pd.read_csv('data/raw/SYNTHETIC_MENTIONS.csv')
print(f"Rows: {len(df):,}")  # Should be: 128,945
print(f"CUIs: {df['cui'].nunique():,}")  # Should be: 47,654
```

### Step 2: Feature Engineering

Process raw data and engineer features:

```bash
python scripts/02_preprocess_data.py
```

**Outputs**:
- `data/processed/analysis_ready.parquet`
- Features: mention tokens, complexity, TTR, clinical indicators
- Panel structure: time_period, cui_id
- Deployment contexts: 5 K-means clusters

**Verification**:
```python
df = pd.read_parquet('data/processed/analysis_ready.parquet')
assert 'NER_F1' in df.columns
assert 'Alert_S' in df.columns
assert df['context_id'].nunique() == 5
print("✓ Preprocessing successful")
```

### Step 3: Causal Analysis

Run quasi-experimental analyses (DiD, IV, Mediation):

```bash
python scripts/03_run_causal_analysis.py
```

**Outputs**:
- `outputs/tables/table1_did_results.csv` - DiD coefficients
- `outputs/tables/table2_parallel_trends.csv` - Pre-treatment test
- `outputs/tables/table3_iv_analysis.csv` - IV diagnostics
- `outputs/tables/table4_mediation.csv` - Mediation decomposition

**Key Results to Verify**:
```python
did = pd.read_csv('outputs/tables/table1_did_results.csv')
did_effect = did.loc[did['Variable'] == 'Treated×Post (DiD)', 'Coefficient'].values[0]
print(f"DiD Effect: {did_effect:.4f}")  # Should be: ~-0.0582

med = pd.read_csv('outputs/tables/table4_mediation.csv')
pct_med = med.loc[med['Path'] == 'Indirect (a×b)', 'Percent_Mediated'].values[0]
print(f"% Mediated: {pct_med:.1f}%")  # Should be: ~106.9%
```

### Step 4: Cost-Benefit Optimization

Determine optimal alert threshold:

```bash
python scripts/04_run_cost_benefit.py
```

**Outputs**:
- `outputs/tables/table8_cost_benefit.csv` - Cost analysis
- `outputs/figures/fig8_roc_curve.png` - ROC curve

**Key Results**:
```python
cost = pd.read_csv('outputs/tables/table8_cost_benefit.csv')
opt_thresh = cost.loc[cost['Metric'] == 'Optimal Threshold', 'Value'].values[0]
print(f"Optimal Threshold: {opt_thresh:.4f}")  # Should be: ~0.105

savings = cost.loc[cost['Metric'] == 'Saving %', 'Value'].values[0]
print(f"Cost Savings: {savings:.1f}%")  # Should be: ~71%
```

### Step 5: Heterogeneous Effects

Estimate context-specific treatment effects:

```bash
python scripts/05_run_heterogeneity.py
```

**Outputs**:
- `outputs/tables/table5_hte_context.csv`
- `outputs/figures/fig6_hte.png`

**Verification**:
```python
hte = pd.read_csv('outputs/tables/table5_hte_context.csv')
print("Context-Specific ATEs:")
print(hte[['Context', 'ATE_Adjusted']].to_string(index=False))
# Context 4 should have largest effect (~-0.0376)
```

### Step 6: External Validation

Simulate external benchmark performance:

```bash
python scripts/06_run_validation.py
```

**Outputs**:
- `outputs/tables/table6_sota_comparison.csv`
- `outputs/tables/table7_external_benchmarks.csv`
- `outputs/figures/fig9_sota_comparison.png`
- `outputs/figures/fig10_external_benchmarks.png`

### Step 7: Markov Modeling

Estimate state transition probabilities:

```bash
python scripts/07_run_markov.py
```

**Outputs**:
- `outputs/tables/table12_markov_transition.csv`
- `outputs/figures/fig11_markov_transition.png`

### Step 8: Generate All Figures

Create publication-quality figures (300 DPI):

```bash
python scripts/08_generate_figures.py
```

**Outputs**: 12 PNG files in `outputs/figures/`

1. `fig1_did.png` - DiD visualization
2. `fig2_parallel_trends.png` - Pre-treatment trends
3. `fig3_event_study.png` - Dynamic effects
4. `fig4_performance_drift.png` - Degradation curve
5. `fig5_mediation.png` - Mediation diagram
6. `fig6_hte.png` - Heterogeneous effects
7. `fig7_iv_first_stage.png` - IV diagnostics
8. `fig8_roc_curve.png` - ROC curve
9. `fig9_sota_comparison.png` - SOTA comparison
10. `fig10_external_benchmarks.png` - External validation
11. `fig11_markov_transition.png` - State transitions
12. `fig12_error_analysis.png` - Error breakdown

### Step 9: Generate All Tables

Create CSV tables for manuscript:

```bash
python scripts/09_generate_tables.py
```

**Outputs**: 13 CSV files in `outputs/tables/`

## Troubleshooting

### Issue: Download fails

```bash
# Manual download
curl -o data/raw/SYNTHETIC_MENTIONS.csv \
  https://raw.githubusercontent.com/KuleenS/synth-der-den/master/data/SYNTHETIC_MENTIONS.csv
```

### Issue: Import errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Memory errors

Reduce sample size in `src/config.py`:

```python
SAMPLE_SIZE = 50_000  # Use subset for testing
```

### Issue: Figures don't generate

Check matplotlib backend:

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

## Verification Checklist

After running the pipeline, verify these key results:

- [ ] DiD effect: ~-0.0582 (p < 0.001)
- [ ] IV effect: ~-0.3351 (F-stat > 3000)
- [ ] Mediation: ~106.9% mediated
- [ ] Cost savings: ~71% vs. no monitoring
- [ ] Optimal threshold: ~0.105
- [ ] HTE range: 31-fold (Context 4 to Context 1)
- [ ] 12 figures generated (300 DPI PNG)
- [ ] 13 tables generated (CSV)

## Expected Runtime

| Step | Time | Notes |
|------|------|-------|
| Data download | 1-2 min | Depends on internet speed |
| Preprocessing | 3-5 min | Feature engineering |
| Causal analysis | 5-7 min | DiD, IV, mediation |
| Cost-benefit | 2-3 min | ROC optimization |
| HTE | 2-3 min | Context-specific effects |
| Validation | 1-2 min | SOTA, external |
| Markov | 1-2 min | State transitions |
| Figures | 3-5 min | 12 high-res plots |
| Tables | 1 min | 13 CSV exports |
| **Total** | **15-20 min** | On standard laptop |

## Hardware Tested

Pipeline successfully tested on:

- **MacBook Pro** (M1, 16GB RAM) - 12 minutes
- **Ubuntu 20.04** (Intel i7, 16GB RAM) - 18 minutes
- **Windows 11** (AMD Ryzen 7, 32GB RAM) - 15 minutes

## Random Seed

All stochastic processes use `RANDOM_SEED = 42` (set in `src/config.py`).

Results may vary slightly due to:
- K-means initialization (mitigated with `n_init=10`)
- Bootstrap sampling (mitigated with large B=10,000)
- Performance simulation noise (controlled with fixed seed)

For exact reproduction, ensure:
```python
import numpy as np
np.random.seed(42)
```

## Docker (Optional)

For complete environment isolation:

```bash
docker build -t alert-iq-analysis .
docker run -v $(pwd)/outputs:/app/outputs alert-iq-analysis
```

See `Dockerfile` for details.

## Contact

Issues with reproduction? Please open a GitHub issue or contact:

**Yong-Jae Lee**  
Email: yj11021@tobesoft.com

## Last Updated

February 2025
