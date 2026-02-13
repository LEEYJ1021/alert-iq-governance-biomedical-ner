"""
Alert Signal Generation Module
===============================

Generates structural and relational alert signals for drift detection.
"""

import pandas as pd
import numpy as np

from .config import config


def calculate_structural_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate structural alerts (new unique mentions within CUI).
    
    Structural alerts flag when a previously unseen surface form appears
    for a given disease concept, indicating vocabulary expansion.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'cui' and 'mention' columns, sorted by time_period
        
    Returns
    -------
    pd.DataFrame
        Data with 'Alert_S' column (1 = structural alert triggered)
        
    Notes
    -----
    Alert triggers when:
    - A mention text (lowercased) appears for the first time for that CUI
    """
    
    print("\n" + "="*60)
    print("CALCULATING STRUCTURAL ALERTS")
    print("="*60)
    
    def calc_structural_change(group):
        seen = set()
        changes = []
        for mention in group['mention']:
            prev_size = len(seen)
            seen.add(mention.lower().strip())
            changes.append(1 if len(seen) > prev_size else 0)
        return pd.Series(changes, index=group.index)
    
    print("- Computing novelty for each mention...")
    df['Alert_S'] = df.groupby('cui', group_keys=False).apply(
        calc_structural_change
    ).values
    
    # Cumulative count
    df['cum_Alert_S'] = df.groupby('cui')['Alert_S'].cumsum()
    
    alert_rate = df['Alert_S'].mean()
    total_alerts = df['Alert_S'].sum()
    
    print(f"\n✓ Structural alerts computed")
    print(f"  Total alerts: {total_alerts:,}")
    print(f"  Alert rate: {100*alert_rate:.2f}%")
    print(f"  CUIs with ≥1 alert: {df[df['cum_Alert_S']>0]['cui'].nunique():,}")
    
    return df


def calculate_relational_alerts(
    df: pd.DataFrame,
    threshold: float = 0.7
) -> pd.DataFrame:
    """
    Calculate relational alerts (semantic drift via Jaccard distance).
    
    Relational alerts flag when consecutive mentions of the same CUI
    have significantly different token sets, indicating semantic shift.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'cui' and 'mention' columns, sorted by time_period
    threshold : float
        Jaccard distance threshold for triggering alert
        
    Returns
    -------
    pd.DataFrame
        Data with 'Alert_R' and 'jaccard_distance' columns
        
    Notes
    -----
    Jaccard distance = 1 - |intersection| / |union|
    Alert triggers when distance > threshold
    """
    
    print("\n" + "="*60)
    print("CALCULATING RELATIONAL ALERTS")
    print("="*60)
    print(f"Jaccard threshold: {threshold}")
    
    def calc_jaccard_distance(group):
        distances = [0.0]  # First mention has no predecessor
        for i in range(1, len(group)):
            current = set(group.iloc[i]['mention'].lower().split())
            previous = set(group.iloc[i-1]['mention'].lower().split())
            union = current | previous
            
            if len(union) == 0:
                distances.append(0.0)
            else:
                intersection = current & previous
                distances.append(1 - len(intersection) / len(union))
        
        return pd.Series(distances, index=group.index)
    
    print("- Computing Jaccard distances...")
    df['jaccard_distance'] = df.groupby('cui', group_keys=False).apply(
        calc_jaccard_distance
    ).values
    
    # Alert when distance exceeds threshold
    df['Alert_R'] = (df['jaccard_distance'] > threshold).astype(int)
    
    # Cumulative count
    df['cum_Alert_R'] = df.groupby('cui')['Alert_R'].cumsum()
    
    alert_rate = df['Alert_R'].mean()
    total_alerts = df['Alert_R'].sum()
    
    print(f"\n✓ Relational alerts computed")
    print(f"  Total alerts: {total_alerts:,}")
    print(f"  Alert rate: {100*alert_rate:.2f}%")
    print(f"  Mean Jaccard distance: {df['jaccard_distance'].mean():.3f}")
    print(f"  CUIs with ≥1 alert: {df[df['cum_Alert_R']>0]['cui'].nunique():,}")
    
    return df


def calculate_drift_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate undetected drift accumulation proxy.
    
    This metric represents the accumulation of quality degradation
    from alert events that were not acted upon, with exponential decay.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'Alert_S', 'Alert_R', 'time_period' columns
        
    Returns
    -------
    pd.DataFrame
        Data with 'undetected_drift' and 'cum_undetected_drift' columns
    """
    
    print("\n" + "="*60)
    print("CALCULATING DRIFT ACCUMULATION")
    print("="*60)
    
    # Drift accumulates from alerts with time decay
    df['undetected_drift'] = (
        (df['Alert_S'] + df['Alert_R']) * 
        np.exp(-0.1 * df['time_period'])
    )
    
    # Cumulative drift
    df['cum_undetected_drift'] = df.groupby('cui')['undetected_drift'].cumsum()
    
    print(f"\n✓ Drift metrics computed")
    print(f"  Mean undetected drift: {df['undetected_drift'].mean():.4f}")
    print(f"  Mean cumulative drift: {df['cum_undetected_drift'].mean():.4f}")
    print(f"  Max cumulative drift: {df['cum_undetected_drift'].max():.4f}")
    
    return df


def generate_all_alerts(
    df: pd.DataFrame,
    jaccard_threshold: float = config.DEFAULT_JACCARD_THRESHOLD
) -> pd.DataFrame:
    """
    Generate all alert signals and drift metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with required columns
    jaccard_threshold : float
        Threshold for relational alerts
        
    Returns
    -------
    pd.DataFrame
        Data with all alert signals
        
    Examples
    --------
    >>> from src.alert_signals import generate_all_alerts
    >>> df = generate_all_alerts(df)
    """
    
    df = calculate_structural_alerts(df)
    df = calculate_relational_alerts(df, threshold=jaccard_threshold)
    df = calculate_drift_proxy(df)
    
    print("\n" + "="*60)
    print("ALERT GENERATION COMPLETE")
    print("="*60)
    
    return df
