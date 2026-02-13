"""
Performance Simulation Module
==============================

Simulates NER F1 and NEN accuracy scores with performance degradation.
"""

import pandas as pd
import numpy as np

from .config import config


def calculate_context_penalties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate context-specific baseline penalties.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'context_id' and 'avg_complexity' columns
        
    Returns
    -------
    pd.DataFrame
        Data with 'context_baseline_penalty' column
    """
    
    print("\n" + "="*60)
    print("CALCULATING CONTEXT PENALTIES")
    print("="*60)
    
    # Calculate mean complexity per context
    context_baselines = df.groupby('context_id')['avg_complexity'].mean()
    
    # Normalize to 0-0.1 range (penalty)
    context_baselines = (
        (context_baselines - context_baselines.min()) / 
        (context_baselines.max() - context_baselines.min()) * 0.1
    )
    
    # Map to dataframe
    df['context_baseline_penalty'] = df['context_id'].map(context_baselines)
    
    print("\nContext Penalties:")
    for ctx in sorted(context_baselines.index):
        print(f"  Context {ctx}: {context_baselines[ctx]:.4f}")
    
    return df


def simulate_ner_performance(
    df: pd.DataFrame,
    baseline: float = config.BASELINE_NER_F1,
    degradation_rate: float = config.DEGRADATION_RATE,
    noise_std: float = 0.02,
    random_state: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """
    Simulate NER F1 scores with drift-driven degradation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'cum_undetected_drift' and 'context_baseline_penalty'
    baseline : float
        Baseline NER F1 score (no drift, no penalty)
    degradation_rate : float
        Performance loss per unit of undetected drift
    noise_std : float
        Standard deviation of random noise
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Data with 'NER_F1' column
        
    Notes
    -----
    Formula:
        NER_F1 = baseline - degradation_rate × cum_drift - context_penalty + ε
        where ε ~ N(0, noise_std)
    """
    
    print("\n" + "="*60)
    print("SIMULATING NER F1 SCORES")
    print("="*60)
    
    print(f"Baseline: {baseline:.4f}")
    print(f"Degradation rate: {degradation_rate:.4f}")
    print(f"Noise std: {noise_std:.4f}")
    
    # Set random seed
    np.random.seed(random_state)
    
    # Generate performance scores
    df['NER_F1'] = (
        baseline - 
        degradation_rate * df['cum_undetected_drift'] -
        df['context_baseline_penalty'] +
        np.random.normal(0, noise_std, len(df))
    ).clip(0.3, 1.0)  # Realistic bounds
    
    print(f"\n✓ NER F1 simulated")
    print(f"  Mean: {df['NER_F1'].mean():.4f}")
    print(f"  Std: {df['NER_F1'].std():.4f}")
    print(f"  Range: [{df['NER_F1'].min():.4f}, {df['NER_F1'].max():.4f}]")
    
    return df


def simulate_nen_performance(
    df: pd.DataFrame,
    baseline: float = config.BASELINE_NEN_ACC,
    degradation_rate: float = config.DEGRADATION_RATE * 1.2,
    noise_std: float = 0.025,
    random_state: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """
    Simulate NEN accuracy scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'cum_undetected_drift' and 'context_baseline_penalty'
    baseline : float
        Baseline NEN accuracy (typically lower than NER)
    degradation_rate : float
        Performance loss per unit drift (higher than NER)
    noise_std : float
        Standard deviation of random noise
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Data with 'NEN_Accuracy' column
        
    Notes
    -----
    NEN typically degrades faster than NER because it requires
    both recognition AND correct normalization to standard vocabulary.
    """
    
    print("\n" + "="*60)
    print("SIMULATING NEN ACCURACY")
    print("="*60)
    
    print(f"Baseline: {baseline:.4f}")
    print(f"Degradation rate: {degradation_rate:.4f}")
    
    # Set random seed
    np.random.seed(random_state + 1)  # Different seed for independence
    
    # Generate performance scores (more degradation, more noise than NER)
    df['NEN_Accuracy'] = (
        baseline - 
        degradation_rate * df['cum_undetected_drift'] -
        df['context_baseline_penalty'] * 0.8 +  # Slightly less context penalty
        np.random.normal(0, noise_std, len(df))
    ).clip(0.2, 1.0)
    
    print(f"\n✓ NEN Accuracy simulated")
    print(f"  Mean: {df['NEN_Accuracy'].mean():.4f}")
    print(f"  Std: {df['NEN_Accuracy'].std():.4f}")
    print(f"  Gap from NER: {(df['NER_F1'] - df['NEN_Accuracy']).mean():.4f}")
    
    return df


def simulate_alert_actions(
    df: pd.DataFrame,
    action_rate: float = config.ALERT_ACTION_RATE,
    random_state: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """
    Simulate organizational response to alerts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'Alert_S' column
    action_rate : float
        Probability that a structural alert triggers action (0-1)
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Data with 'alert_acted_upon' column (1 = action taken)
        
    Notes
    -----
    Reflects realistic organizational constraints:
    - Alert fatigue
    - Resource limitations
    - Triage priorities
    
    Not all alerts can be acted upon immediately.
    """
    
    print("\n" + "="*60)
    print("SIMULATING ALERT ACTIONS")
    print("="*60)
    
    print(f"Action probability: {action_rate:.1%}")
    
    # Set random seed
    np.random.seed(random_state + 2)
    
    # Simulate action: alert must fire AND random draw succeeds
    df['alert_acted_upon'] = (
        (df['Alert_S'] == 1) & 
        (np.random.random(len(df)) < action_rate)
    ).astype(int)
    
    total_alerts = df['Alert_S'].sum()
    actions_taken = df['alert_acted_upon'].sum()
    action_rate_actual = actions_taken / total_alerts if total_alerts > 0 else 0
    
    print(f"\n✓ Alert actions simulated")
    print(f"  Total structural alerts: {total_alerts:,}")
    print(f"  Actions taken: {actions_taken:,}")
    print(f"  Actual action rate: {action_rate_actual:.1%}")
    
    return df


def create_counterfactual_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create counterfactual performance (what would have happened without alerts).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'NER_F1' and 'alert_acted_upon' columns
        
    Returns
    -------
    pd.DataFrame
        Data with 'NER_F1_counterfactual' column
        
    Notes
    -----
    For observations where alerts were acted upon, we boost performance
    by a small amount to represent the benefit of the intervention.
    This is used for visualization and effect estimation.
    """
    
    print("\n" + "="*60)
    print("CREATING COUNTERFACTUAL TRAJECTORIES")
    print("="*60)
    
    # Copy actual performance
    df['NER_F1_counterfactual'] = df['NER_F1'].copy()
    
    # For acted-upon alerts, boost by expected intervention benefit
    intervention_benefit = 0.03
    df.loc[df['alert_acted_upon'] == 1, 'NER_F1_counterfactual'] += intervention_benefit
    
    # Calculate average treatment effect on the treated (ATT)
    treated = df[df['alert_acted_upon'] == 1]
    if len(treated) > 0:
        att = treated['NER_F1_counterfactual'].mean() - treated['NER_F1'].mean()
        print(f"\n✓ Counterfactual created")
        print(f"  ATT (intervention benefit): {att:.4f}")
    
    return df


def simulate_all_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run complete performance simulation pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with required features
        
    Returns
    -------
    pd.DataFrame
        Data with all performance metrics
        
    Examples
    --------
    >>> from src.performance_simulation import simulate_all_performance
    >>> df = simulate_all_performance(df)
    """
    
    df = calculate_context_penalties(df)
    df = simulate_ner_performance(df)
    df = simulate_nen_performance(df)
    df = simulate_alert_actions(df)
    df = create_counterfactual_performance(df)
    
    print("\n" + "="*60)
    print("PERFORMANCE SIMULATION COMPLETE")
    print("="*60)
    
    # Summary statistics
    print("\nPerformance Summary:")
    print(f"  NER F1:         {df['NER_F1'].mean():.4f} ± {df['NER_F1'].std():.4f}")
    print(f"  NEN Accuracy:   {df['NEN_Accuracy'].mean():.4f} ± {df['NEN_Accuracy'].std():.4f}")
    print(f"  Alerts acted:   {df['alert_acted_upon'].sum():,} / {df['Alert_S'].sum():,}")
    
    # Correlation
    corr = df[['cum_undetected_drift', 'NER_F1']].corr().iloc[0, 1]
    print(f"\nDrift-Performance Correlation: {corr:.3f}")
    
    return df
