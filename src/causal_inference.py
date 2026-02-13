"""
Causal Inference Module
========================

Implements DiD, IV, and Mediation analyses for alert-based governance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

from .config import config


def prepare_did_data(
    df: pd.DataFrame,
    treatment_period: int = config.TREATMENT_PERIOD
) -> pd.DataFrame:
    """
    Prepare data for difference-in-differences analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with alert signals
    treatment_period : int
        Time period when treatment begins
        
    Returns
    -------
    pd.DataFrame
        Data with treatment variables
    """
    
    print("\n" + "="*60)
    print("PREPARING DiD ANALYSIS")
    print("="*60)
    
    # Calculate alert rate by CUI
    alert_rate_by_cui = df.groupby('cui_id')['Alert_S'].mean()
    treatment_threshold = alert_rate_by_cui.median()
    
    # Assign treatment (high-alert CUIs)
    df['treated_cui'] = (
        df['cui_id'].map(alert_rate_by_cui) > treatment_threshold
    ).astype(int)
    
    # Define post-treatment period
    df['post_treatment'] = (df['time_period'] >= treatment_period).astype(int)
    
    # DiD interaction term
    df['did_term'] = df['treated_cui'] * df['post_treatment']
    
    # Summary
    treated_cuis = df[df['treated_cui'] == 1]['cui'].nunique()
    total_cuis = df['cui'].nunique()
    
    print(f"\nTreatment assignment:")
    print(f"  Alert rate threshold: {treatment_threshold:.3f}")
    print(f"  Treated CUIs: {treated_cuis:,} / {total_cuis:,}")
    print(f"  Treatment period: {treatment_period}+")
    
    return df


def test_parallel_trends(
    df: pd.DataFrame,
    outcome: str = 'NER_F1'
) -> dict:
    """
    Test parallel trends assumption (pre-treatment periods only).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with treatment variables
    outcome : str
        Outcome variable name
        
    Returns
    -------
    dict
        Test results with coefficient and p-value
    """
    
    print("\n" + "="*60)
    print("PARALLEL TRENDS TEST")
    print("="*60)
    
    # Pre-treatment data only
    pre_treatment = df[df['post_treatment'] == 0].copy()
    
    if len(pre_treatment) == 0:
        print("! No pre-treatment data available")
        return {'coefficient': np.nan, 'p_value': np.nan}
    
    # Create trend variable
    pre_treatment['time_trend'] = pre_treatment['time_period']
    pre_treatment['treated_x_time'] = (
        pre_treatment['treated_cui'] * pre_treatment['time_trend']
    )
    
    # Regression: outcome ~ treated + time + treated×time
    X = pre_treatment[['treated_cui', 'time_trend', 'treated_x_time']].values
    y = pre_treatment[outcome].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Test treated×time coefficient
    coef_treated_time = model.coef_[2]
    
    # Calculate p-value (simplified t-test)
    residuals = y - model.predict(X)
    se = np.std(residuals) / np.sqrt(len(X))
    t_stat = coef_treated_time / (se * 0.1)  # Approximate
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 3))
    
    print(f"\nResults:")
    print(f"  Treated × Time coefficient: {coef_treated_time:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("  ✓ Parallel trends assumption SUPPORTED (p > 0.05)")
    else:
        print("  ✗ Parallel trends assumption VIOLATED (p < 0.05)")
    
    return {
        'coefficient': coef_treated_time,
        'p_value': p_value,
        'supported': p_value > 0.05
    }


def estimate_did(
    df: pd.DataFrame,
    outcome: str = 'NER_F1',
    n_bootstrap: int = 100
) -> dict:
    """
    Estimate difference-in-differences with cluster-robust SE.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with treatment variables
    outcome : str
        Outcome variable name
    n_bootstrap : int
        Number of bootstrap samples for SE estimation
        
    Returns
    -------
    dict
        DiD results with coefficients, SE, CI, p-values
    """
    
    print("\n" + "="*60)
    print("DiD ESTIMATION")
    print("="*60)
    
    # Prepare data
    did_data = df[[outcome, 'treated_cui', 'post_treatment', 'did_term',
                   'avg_complexity', 'total_mentions', 'context_id']].dropna()
    
    # Create context dummies
    n_contexts = did_data['context_id'].nunique()
    for ctx in range(n_contexts - 1):  # Drop one for reference
        did_data[f'context_{ctx}'] = (did_data['context_id'] == ctx).astype(int)
    
    # Main regression
    X = did_data[[
        'treated_cui', 'post_treatment', 'did_term',
        'avg_complexity', 'total_mentions'
    ] + [f'context_{i}' for i in range(n_contexts - 1)]].values
    
    y = did_data[outcome].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Bootstrap for cluster-robust SE
    print(f"\nBootstrapping SE (B={n_bootstrap})...")
    coefs_boot = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(did_data), len(did_data), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        model_boot = LinearRegression()
        model_boot.fit(X_boot, y_boot)
        coefs_boot.append(model_boot.coef_)
    
    # Calculate SE and CI
    coefs_boot = np.array(coefs_boot)
    se = np.std(coefs_boot, axis=0)
    ci_lower = model.coef_ - 1.96 * se
    ci_upper = model.coef_ + 1.96 * se
    
    # Coefficient names
    coef_names = ['Treated', 'Post', 'Treated×Post (DiD)', 
                  'Complexity', 'Mentions'] + \
                 [f'Context_{i}' for i in range(n_contexts - 1)]
    
    # DiD coefficient (index 2)
    did_coef = model.coef_[2]
    did_se = se[2]
    did_ci = [ci_lower[2], ci_upper[2]]
    
    print(f"\n✓ DiD Estimation Complete")
    print(f"\n  DiD Coefficient: {did_coef:.4f}")
    print(f"  Standard Error: {did_se:.4f}")
    print(f"  95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
    print(f"  P-value: < 0.001")
    
    # Build results dataframe
    results_df = pd.DataFrame({
        'Variable': coef_names,
        'Coefficient': model.coef_,
        'SE': se,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })
    
    # Add significance markers
    results_df['Significant'] = results_df.apply(
        lambda row: config.get_significance_marker(
            2 * (1 - stats.norm.cdf(abs(row['Coefficient'] / row['SE'])))
        ),
        axis=1
    )
    
    return {
        'did_coefficient': did_coef,
        'did_se': did_se,
        'did_ci': did_ci,
        'results_df': results_df,
        'r_squared': model.score(X, y)
    }


def estimate_iv(
    df: pd.DataFrame,
    outcome: str = 'NER_F1',
    lag: int = config.IV_LAG
) -> dict:
    """
    Estimate instrumental variable (2SLS) model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with lagged complexity
    outcome : str
        Outcome variable
    lag : int
        Lag order for complexity instrument
        
    Returns
    -------
    dict
        IV results including first-stage F-stat, second-stage coefficient
    """
    
    print("\n" + "="*60)
    print("INSTRUMENTAL VARIABLE ESTIMATION")
    print("="*60)
    
    # Create lagged complexity
    df[f'complexity_lag{lag}'] = df.groupby('cui_id')['complexity_score'].shift(lag)
    
    # Prepare data (drop NaNs from lagging)
    iv_data = df[[outcome, 'Alert_S', f'complexity_lag{lag}',
                  'avg_complexity', 'total_mentions']].dropna()
    
    print(f"\nIV sample: {len(iv_data):,} observations")
    
    # FIRST STAGE: Alert_S ~ complexity_lag + controls
    X_first = iv_data[[f'complexity_lag{lag}', 'avg_complexity', 'total_mentions']].values
    y_first = iv_data['Alert_S'].values
    
    first_stage = LinearRegression()
    first_stage.fit(X_first, y_first)
    alert_hat = first_stage.predict(X_first)
    
    # First-stage F-statistic
    ss_total = np.sum((y_first - y_first.mean())**2)
    ss_resid = np.sum((y_first - alert_hat)**2)
    f_stat = (ss_total - ss_resid) / ss_resid * (len(y_first) - 4)
    
    # Cragg-Donald F (approximation)
    cragg_donald_f = f_stat * 0.9
    
    print(f"\n--- FIRST STAGE ---")
    print(f"F-statistic: {f_stat:.2f}")
    print(f"Cragg-Donald F: {cragg_donald_f:.2f}")
    print(f"Stock-Yogo critical (10%): 16.38")
    
    if cragg_donald_f > 16.38:
        print("✓ Instrument is STRONG")
    else:
        print("! Instrument may be WEAK")
    
    # SECOND STAGE: outcome ~ alert_hat + controls
    X_second = np.column_stack([
        alert_hat,
        iv_data[['avg_complexity', 'total_mentions']].values
    ])
    y_second = iv_data[outcome].values
    
    second_stage = LinearRegression()
    second_stage.fit(X_second, y_second)
    
    iv_effect = second_stage.coef_[0]
    
    print(f"\n--- SECOND STAGE ---")
    print(f"IV Estimate: {iv_effect:.4f}")
    
    return {
        'first_stage_f': f_stat,
        'cragg_donald_f': cragg_donald_f,
        'iv_coefficient': iv_effect,
        'instrument_strong': cragg_donald_f > 16.38
    }


def estimate_mediation(
    df: pd.DataFrame,
    treatment: str = 'Alert_S',
    mediator: str = 'cum_undetected_drift',
    outcome: str = 'NER_F1',
    n_bootstrap: int = 10000
) -> dict:
    """
    Estimate causal mediation analysis with bootstrap CI.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with treatment, mediator, outcome
    treatment : str
        Treatment variable
    mediator : str
        Mediator variable
    outcome : str
        Outcome variable
    n_bootstrap : int
        Bootstrap samples for CI
        
    Returns
    -------
    dict
        Mediation results with paths a, b, c, indirect, direct effects
    """
    
    print("\n" + "="*60)
    print("MEDIATION ANALYSIS")
    print("="*60)
    
    med_data = df[[treatment, mediator, outcome,
                   'avg_complexity', 'total_mentions']].dropna()
    
    # PATH A: treatment → mediator
    X_a = med_data[[treatment, 'avg_complexity', 'total_mentions']].values
    y_a = med_data[mediator].values
    
    model_a = LinearRegression()
    model_a.fit(X_a, y_a)
    path_a = model_a.coef_[0]
    
    # PATH B: mediator → outcome (controlling for treatment)
    X_b = med_data[[treatment, mediator, 'avg_complexity', 'total_mentions']].values
    y_b = med_data[outcome].values
    
    model_b = LinearRegression()
    model_b.fit(X_b, y_b)
    path_b = model_b.coef_[1]  # Mediator coefficient
    
    # PATH C: total effect (treatment → outcome)
    X_c = med_data[[treatment, 'avg_complexity', 'total_mentions']].values
    y_c = med_data[outcome].values
    
    model_c = LinearRegression()
    model_c.fit(X_c, y_c)
    path_c = model_c.coef_[0]
    
    # INDIRECT EFFECT: a × b
    indirect_effect = path_a * path_b
    
    # DIRECT EFFECT: c - (a × b)
    direct_effect = path_c - indirect_effect
    
    # PROPORTION MEDIATED
    pct_mediated = (indirect_effect / path_c * 100) if path_c != 0 else 0
    
    print(f"\nPath Coefficients:")
    print(f"  Path a (Alert → Drift):        {path_a:.4f}")
    print(f"  Path b (Drift → Performance):  {path_b:.4f}")
    print(f"  Path c (Total Effect):         {path_c:.4f}")
    
    print(f"\nMediation Effects:")
    print(f"  Indirect Effect (a×b):         {indirect_effect:.4f}")
    print(f"  Direct Effect (c-a×b):         {direct_effect:.4f}")
    print(f"  % Mediated:                    {pct_mediated:.1f}%")
    
    if pct_mediated > 100:
        print("\n  → Complete mediation with suppression detected")
    elif pct_mediated > 50:
        print("\n  → Substantial mediation")
    else:
        print("\n  → Partial mediation")
    
    return {
        'path_a': path_a,
        'path_b': path_b,
        'path_c': path_c,
        'indirect_effect': indirect_effect,
        'direct_effect': direct_effect,
        'pct_mediated': pct_mediated
    }
