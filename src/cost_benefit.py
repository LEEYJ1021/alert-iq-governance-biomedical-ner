"""
Cost-Benefit Analysis Module
=============================

ROC-based threshold optimization for alert systems.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

from .config import config


def simulate_true_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate ground-truth drift events for cost-benefit analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with drift metrics
        
    Returns
    -------
    pd.DataFrame
        Data with 'true_drift' column
    """
    
    print("\n" + "="*60)
    print("SIMULATING TRUE DRIFT EVENTS")
    print("="*60)
    
    # Define true drift as high cumulative drift
    drift_threshold = df['cum_undetected_drift'].median()
    df['true_drift'] = (df['cum_undetected_drift'] > drift_threshold).astype(int)
    
    print(f"\nDrift threshold: {drift_threshold:.4f}")
    print(f"True drift events: {df['true_drift'].sum():,} ({100*df['true_drift'].mean():.1f}%)")
    
    return df


def calculate_confusion_matrix(
    df: pd.DataFrame,
    alert_col: str = 'Alert_S',
    truth_col: str = 'true_drift'
) -> dict:
    """
    Calculate confusion matrix for alert performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with alerts and true drift
    alert_col : str
        Alert signal column
    truth_col : str
        Ground truth column
        
    Returns
    -------
    dict
        Confusion matrix (TP, FP, TN, FN)
    """
    
    # Create alert positive indicator
    df['alert_positive'] = ((df[alert_col] == 1)).astype(int)
    
    # Calculate confusion matrix
    tp = ((df['alert_positive'] == 1) & (df[truth_col] == 1)).sum()
    fp = ((df['alert_positive'] == 1) & (df[truth_col] == 0)).sum()
    tn = ((df['alert_positive'] == 0) & (df[truth_col] == 0)).sum()
    fn = ((df['alert_positive'] == 0) & (df[truth_col] == 1)).sum()
    
    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'total': len(df)
    }


def calculate_costs(
    confusion: dict,
    fp_cost: float = config.FP_COST,
    fn_cost: float = config.FN_COST
) -> dict:
    """
    Calculate operational costs from confusion matrix.
    
    Parameters
    ----------
    confusion : dict
        Confusion matrix
    fp_cost : float
        Cost of false positive (review effort)
    fn_cost : float
        Cost of false negative (missed drift)
        
    Returns
    -------
    dict
        Cost breakdown and totals
    """
    
    fp = confusion['FP']
    fn = confusion['FN']
    
    fp_total_cost = fp * fp_cost
    fn_total_cost = fn * fn_cost
    total_cost = fp_total_cost + fn_total_cost
    
    # Baseline: no monitoring (all FN)
    baseline_cost = (confusion['TP'] + fn) * fn_cost
    
    # Savings
    cost_saving = baseline_cost - total_cost
    cost_saving_pct = (cost_saving / baseline_cost * 100) if baseline_cost > 0 else 0
    
    return {
        'fp_cost_total': fp_total_cost,
        'fn_cost_total': fn_total_cost,
        'total_cost': total_cost,
        'baseline_cost': baseline_cost,
        'cost_saving': cost_saving,
        'cost_saving_pct': cost_saving_pct
    }


def optimize_threshold_roc(
    df: pd.DataFrame,
    score_col: str = 'jaccard_distance',
    truth_col: str = 'true_drift',
    fp_cost: float = config.FP_COST,
    fn_cost: float = config.FN_COST
) -> dict:
    """
    Find optimal alert threshold using ROC analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with drift scores and ground truth
    score_col : str
        Column with alert scores (higher = more likely drift)
    truth_col : str
        Ground truth drift column
    fp_cost : float
        Cost per false positive
    fn_cost : float
        Cost per false negative
        
    Returns
    -------
    dict
        ROC curve, optimal threshold, costs
    """
    
    print("\n" + "="*60)
    print("ROC-BASED THRESHOLD OPTIMIZATION")
    print("="*60)
    
    print(f"\nCost parameters:")
    print(f"  FP cost: ${fp_cost:.2f}")
    print(f"  FN cost: ${fn_cost:.2f}")
    print(f"  Cost ratio (FN/FP): {fn_cost/fp_cost:.1f}:1")
    
    # Calculate ROC curve
    scores = df[score_col].fillna(0).values
    truth = df[truth_col].values
    
    fpr, tpr, thresholds = roc_curve(truth, scores)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Calculate cost at each threshold
    costs = []
    n_positive = truth.sum()
    n_negative = len(truth) - n_positive
    
    for i, thresh in enumerate(thresholds):
        # At this threshold
        fn = n_positive * (1 - tpr[i])  # False negatives
        fp = n_negative * fpr[i]        # False positives
        
        cost = fp * fp_cost + fn * fn_cost
        costs.append(cost)
    
    # Find optimal
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    # Baseline cost (no monitoring)
    baseline_cost = n_positive * fn_cost
    
    # Savings
    cost_saving = baseline_cost - optimal_cost
    cost_saving_pct = (cost_saving / baseline_cost * 100)
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"  TPR: {optimal_tpr:.3f}")
    print(f"  FPR: {optimal_fpr:.3f}")
    print(f"  Total cost: ${optimal_cost:,.0f}")
    print(f"  Baseline cost: ${baseline_cost:,.0f}")
    print(f"  Savings: ${cost_saving:,.0f} ({cost_saving_pct:.1f}%)")
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'roc_auc': roc_auc,
        'costs': costs,
        'optimal_threshold': optimal_threshold,
        'optimal_cost': optimal_cost,
        'optimal_fpr': optimal_fpr,
        'optimal_tpr': optimal_tpr,
        'baseline_cost': baseline_cost,
        'cost_saving': cost_saving,
        'cost_saving_pct': cost_saving_pct
    }


def compare_threshold_strategies(
    df: pd.DataFrame,
    fp_cost: float = config.FP_COST,
    fn_cost: float = config.FN_COST
) -> pd.DataFrame:
    """
    Compare different threshold strategies.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with alerts and true drift
    fp_cost : float
        False positive cost
    fn_cost : float
        False negative cost
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    
    print("\n" + "="*60)
    print("THRESHOLD STRATEGY COMPARISON")
    print("="*60)
    
    # Simulate true drift if not exists
    if 'true_drift' not in df.columns:
        df = simulate_true_drift(df)
    
    results = []
    
    # 1. No monitoring
    n_drift_events = df['true_drift'].sum()
    no_monitoring_cost = n_drift_events * fn_cost
    
    results.append({
        'Scenario': 'Baseline (No Monitoring)',
        'Threshold': np.nan,
        'Total_Cost': no_monitoring_cost,
        'Savings_vs_Baseline': 0
    })
    
    # 2. Default engineering threshold (0.70)
    df['alert_default'] = (df['jaccard_distance'] > 0.7).astype(int)
    confusion_default = calculate_confusion_matrix(
        df, 
        alert_col='alert_default', 
        truth_col='true_drift'
    )
    costs_default = calculate_costs(confusion_default, fp_cost, fn_cost)
    
    results.append({
        'Scenario': 'Default Engineering',
        'Threshold': 0.70,
        'Total_Cost': costs_default['total_cost'],
        'Savings_vs_Baseline': costs_default['total_cost'] - no_monitoring_cost
    })
    
    # 3. ROC-optimal
    roc_result = optimize_threshold_roc(df, fp_cost=fp_cost, fn_cost=fn_cost)
    
    results.append({
        'Scenario': 'ROC-Optimal',
        'Threshold': roc_result['optimal_threshold'],
        'Total_Cost': roc_result['optimal_cost'],
        'Savings_vs_Baseline': roc_result['cost_saving']
    })
    
    # Create comparison table
    comparison_df = pd.DataFrame(results)
    
    print("\n" + "-"*70)
    print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
    print("-"*70)
    
    # Calculate improvement
    default_cost = comparison_df.loc[comparison_df['Scenario'] == 'Default Engineering', 'Total_Cost'].values[0]
    optimal_cost = comparison_df.loc[comparison_df['Scenario'] == 'ROC-Optimal', 'Total_Cost'].values[0]
    improvement = ((default_cost - optimal_cost) / default_cost * 100)
    
    print(f"\nROC-Optimal vs. Default:")
    print(f"  Cost reduction: {improvement:.1f}%")
    
    return comparison_df


def run_cost_benefit_analysis(
    df: pd.DataFrame,
    fp_cost: float = config.FP_COST,
    fn_cost: float = config.FN_COST
) -> dict:
    """
    Run complete cost-benefit analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with alerts and drift metrics
    fp_cost : float
        False positive cost
    fn_cost : float
        False negative cost
        
    Returns
    -------
    dict
        All cost-benefit results
        
    Examples
    --------
    >>> from src.cost_benefit import run_cost_benefit_analysis
    >>> results = run_cost_benefit_analysis(df)
    """
    
    # Simulate true drift
    df = simulate_true_drift(df)
    
    # Basic confusion matrix
    confusion = calculate_confusion_matrix(df)
    costs = calculate_costs(confusion, fp_cost, fn_cost)
    
    print("\n" + "="*60)
    print("COST-BENEFIT ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {confusion['TP']:,}")
    print(f"  False Positives: {confusion['FP']:,}")
    print(f"  True Negatives:  {confusion['TN']:,}")
    print(f"  False Negatives: {confusion['FN']:,}")
    
    print(f"\nCosts:")
    print(f"  FP Cost: ${costs['fp_cost_total']:,.0f}")
    print(f"  FN Cost: ${costs['fn_cost_total']:,.0f}")
    print(f"  Total:   ${costs['total_cost']:,.0f}")
    print(f"  Baseline: ${costs['baseline_cost']:,.0f}")
    print(f"  Savings: ${costs['cost_saving']:,.0f} ({costs['cost_saving_pct']:.1f}%)")
    
    # ROC optimization
    roc_result = optimize_threshold_roc(df, fp_cost=fp_cost, fn_cost=fn_cost)
    
    # Strategy comparison
    comparison = compare_threshold_strategies(df, fp_cost=fp_cost, fn_cost=fn_cost)
    
    return {
        'confusion': confusion,
        'costs': costs,
        'roc_result': roc_result,
        'comparison': comparison
    }
