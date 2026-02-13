# Data Dictionary

Complete description of all variables in the analysis dataset.

## Table of Contents

1. [Raw Data](#raw-data)
2. [Mention-Level Features](#mention-level-features)
3. [CUI-Level Features](#cui-level-features)
4. [Panel Structure](#panel-structure)
5. [Alert Signals](#alert-signals)
6. [Performance Metrics](#performance-metrics)
7. [Causal Analysis Variables](#causal-analysis-variables)
8. [Deployment Contexts](#deployment-contexts)

## Raw Data

### Source Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `cui` | string | UMLS Concept Unique Identifier | "C0011849" |
| `matched_output` | string | Synthetic clinical text with tagged disease mention | "Patient presents with <1CUI>diabetes mellitus</1CUI> and hypertension." |
| `mention` | string | Extracted disease mention (between tags) | "diabetes mellitus" |

**Source**: PhysioNet Synthetic Mention Corpora v1.0.0  
**Citation**: Sasse & Osborne (2025). DOI: 10.13026/p5pn-ty93

## Mention-Level Features

### Lexical Features

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `mention_tokens` | int | 1-50+ | Number of tokens (words) in mention |
| `mention_length_chars` | int | 1-200+ | Character length of mention text |
| `unique_tokens` | int | 1-50+ | Number of unique tokens in mention |
| `type_token_ratio` | float | 0-1 | Lexical diversity: unique tokens / total tokens |
| `complexity_score` | int | 1-50+ | Base complexity metric (= mention_tokens) |
| `normalized_complexity` | float | 0-20 | Complexity normalized by log(char_length) |

**Example**:
```python
mention = "acute anterior myocardial infarction"
# mention_tokens = 4
# unique_tokens = 4
# type_token_ratio = 1.0 (all tokens unique)
```

### Clinical Features (Binary)

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `has_conjunction` | int | 0, 1 | Contains: and, or, with, but, versus |
| `has_modifier` | int | 0, 1 | Contains: acute, chronic, severe, mild, moderate, etc. |
| `has_anatomical` | int | 0, 1 | Contains: left, right, upper, lower, bilateral, etc. |

**Patterns**:
- **Conjunction**: `r'\b(and|or|with|but|versus)\b'`
- **Modifier**: `r'\b(acute|chronic|severe|mild|moderate|secondary|primary|recurrent)\b'`
- **Anatomical**: `r'\b(left|right|upper|lower|bilateral|anterior|posterior)\b'`

## CUI-Level Features

Aggregated statistics computed per CUI.

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `total_mentions` | int | 1-7 | Total number of mention observations for this CUI |
| `unique_mentions` | int | 1-7 | Number of distinct mention texts for this CUI |
| `avg_complexity` | float | 1-48 | Mean complexity across all mentions |
| `std_complexity` | float | 0-10 | Standard deviation of complexity |
| `avg_ttr` | float | 0-1 | Mean type-token ratio |
| `is_rare` | int | 0, 1 | 1 if total_mentions ≤ 5, else 0 |

**Example**:
```python
# CUI "C0011849" (diabetes mellitus)
# total_mentions = 5
# unique_mentions = 3  # "diabetes", "diabetes mellitus", "DM"
# avg_complexity = 2.2
# is_rare = 0 (has exactly 5 mentions)
```

## Panel Structure

Variables defining longitudinal data structure.

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `time_period` | int | 1-7 | Sequential time index within CUI (1 = first mention) |
| `cui_id` | int | 0-47653 | Numeric CUI identifier (for efficient indexing) |

**Note**: Data is sorted by `(cui, mention)` before assigning `time_period`.

## Alert Signals

### Structural Alert (Vocabulary Expansion)

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `Alert_S` | int | 0, 1 | Structural alert triggered (new surface form) |
| `cum_Alert_S` | int | 0-7 | Cumulative structural alerts for CUI |

**Trigger Logic**:
```python
# Alert fires when mention text is seen for first time (per CUI)
if mention.lower() not in previously_seen_mentions:
    Alert_S = 1
```

**Distribution**:
- Alert rate: ~41.3% of mentions
- CUIs with ≥1 alert: ~99%

### Relational Alert (Semantic Drift)

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `jaccard_distance` | float | 0-1 | Jaccard distance between consecutive mentions |
| `Alert_R` | int | 0, 1 | Relational alert triggered (distance > threshold) |
| `cum_Alert_R` | int | 0-7 | Cumulative relational alerts for CUI |

**Jaccard Distance**:
```
d = 1 - |A ∩ B| / |A ∪ B|

where A, B are token sets of consecutive mentions
```

**Default Threshold**: 0.7

**Distribution**:
- Alert rate: ~2.9% of mentions
- Mean Jaccard distance: ~0.45

### Drift Accumulation

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `undetected_drift` | float | 0-2 | Time-decayed drift from alerts |
| `cum_undetected_drift` | float | 0-10+ | Cumulative undetected drift |

**Formula**:
```
undetected_drift = (Alert_S + Alert_R) × exp(-0.1 × time_period)
cum_undetected_drift = Σ undetected_drift
```

## Performance Metrics

Simulated downstream performance scores.

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `NER_F1` | float | 0.3-1.0 | Named Entity Recognition F1 score |
| `NEN_Accuracy` | float | 0.2-1.0 | Named Entity Normalization accuracy |
| `alert_acted_upon` | int | 0, 1 | Whether structural alert triggered action (70% prob) |

**Simulation Formula**:
```python
NER_F1 = 0.85 - 0.02 × cum_undetected_drift - context_penalty + ε
NEN_Accuracy = 0.78 - 0.024 × cum_undetected_drift - 0.8 × context_penalty + ε
```

where `ε ~ N(0, 0.02)` or `N(0, 0.025)`

**Benchmark Alignment**:
- Baseline NER F1: 0.85 (calibrated to BC5CDR, NCBI-disease)
- Baseline NEN Acc: 0.78 (Leaman et al., 2013)

## Causal Analysis Variables

### Treatment Assignment

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `treated_cui` | int | 0, 1 | CUI assigned to high-alert monitoring group |
| `post_treatment` | int | 0, 1 | Observation occurs after period 4 |
| `did_term` | int | 0, 1 | Interaction: treated_cui × post_treatment |

**Treatment Definition**:
```python
alert_rate_by_cui = df.groupby('cui')['Alert_S'].mean()
treated_cui = (alert_rate_by_cui > median(alert_rate_by_cui))
```

### Instrumental Variable

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `complexity_lag2` | float | 1-48 | Complexity score lagged by 2 periods |

**Usage**: Instrument for endogenous alert exposure in 2SLS regression.

## Deployment Contexts

5 distinct deployment environments from K-means clustering.

| Context ID | Interpretation | CUIs | Mentions | Avg Complexity | Use Case |
|------------|----------------|------|----------|----------------|----------|
| 0 | Medium complexity, moderate volume | 11,743 | 26,561 | 7.13 | General clinical |
| 1 | High complexity, low volume | 1,470 | 3,693 | 12.41 | Specialty care |
| 2 | Low complexity, high volume | 22,073 | 44,611 | 3.03 | Primary care |
| 3 | Outlier (excluded) | 34 | 25 | 43.08 | - |
| 4 | Low complexity, very high volume | 12,364 | 54,055 | 3.06 | Billing/coding |

**Context 3** excluded from analysis (N=34, 0.07% of CUIs, extreme outliers).

**Clustering Features**:
```python
['avg_complexity', 'avg_ttr', 'total_mentions', 'is_rare']
# Standardized before K-means (k=5, random_state=42)
```

## Variable Name Conventions

### Prefixes

- `avg_` - Mean value across observations
- `std_` - Standard deviation
- `cum_` - Cumulative sum over time
- `is_` - Binary indicator
- `has_` - Binary presence/absence

### Suffixes

- `_score` - Computed metric
- `_ratio` - Normalized metric
- `_id` - Identifier variable
- `_rate` - Frequency or proportion

## Data Types

| Type | Description | Examples |
|------|-------------|----------|
| `string` | Text variable | cui, mention |
| `int` | Integer | time_period, Alert_S |
| `float` | Continuous | NER_F1, jaccard_distance |
| `binary` | 0/1 indicator | is_rare, treated_cui |

## Missing Values

**No missing values** in analysis dataset after preprocessing.

Raw data may contain:
- Missing `mention` if extraction failed → **dropped**
- Missing `std_complexity` if CUI has 1 mention → **filled with 0**

## Data Validation

### Assertions

```python
# Panel structure
assert df.groupby('cui')['time_period'].min().min() == 1
assert df.groupby('cui')['time_period'].is_monotonic_increasing.all()

# Alert logic
assert (df['Alert_S'].isin([0, 1])).all()
assert (df['cum_Alert_S'] >= df['Alert_S']).all()

# Performance bounds
assert df['NER_F1'].between(0, 1).all()
assert df['NEN_Accuracy'].between(0, 1).all()
```

## File Locations

- **Raw data**: `data/raw/SYNTHETIC_MENTIONS.csv`
- **Processed data**: `data/processed/analysis_ready.parquet`
- **Analysis outputs**: `outputs/tables/*.csv`

## References

- **UMLS**: Bodenreider, O. (2004). The Unified Medical Language System (UMLS). Nucleic Acids Research, 32(D1), D267-D270.
- **BC5CDR**: Li et al. (2016). BioCreative V CDR task corpus. Database, 2016.
- **NCBI-disease**: Doğan et al. (2014). NCBI disease corpus. Journal of Biomedical Informatics, 47, 1-10.

## Version History

- **v1.0.0** (Feb 2025) - Initial release
  - 128,945 observations
  - 47,654 unique CUIs
  - 5 deployment contexts

---

**Last Updated**: February 2025  
**Maintainer**: Yong-Jae Lee (yj11021@tobesoft.com)
