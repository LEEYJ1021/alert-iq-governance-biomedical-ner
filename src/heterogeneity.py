"""
Heterogeneous Treatment Effects Module
=======================================

Estimates context-specific treatment effects.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

from .config import config


def estimate_context_ate(
    df: pd.DataFrame,
    outcome: str = 'NER_F1'
) -> pd.DataFrame:
    """
    Estimate average treatment effects by deployment context.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with treatment, context, and outcome
    outcome : str
        Outcome variable
        
    Returns
    -------
    pd.DataFrame
        Context-specific ATEs with SE and significance
    """
    
    print("\n" + "="*60)
    print("HETEROGENEOUS TREATMENT EFFECTS ANALYSIS")
    print("="*60)
    
    results = []
    
    for ctx in sorted(df['context_id'].unique()):
        ctx_data = df[df['context_id'] == ctx].copy()
        
        if len(ctx_data) < 50:
            print(f"\n⚠ Context {ctx}: Too few observations ({len(ctx_data)}), skipping")
            continue
        
        print(f"\n--- Context {ctx} ---")
        
        # Simple ATE (difference in means)
        treated = ctx_data[ctx_data['treated_cui'] == 1][outcome].mean()
        control = ctx_data[ctx_data['treated_cui'] == 0][outcome].mean()
        ate_simple = treated - control
        
        # Adjusted ATE (regression with controls)
        X = ctx_data[['treated_cui', 'avg_complexity', 'total_mentions']].values
        y = ctx_data[outcome].values
        
        model = LinearRegression()
        model.fit(X, y)
        ate_adjusted = model.coef_[0]
        
        # Bootstrap SE
        n_boot = 50
        ate_boot = []
        for _ in range(n_boot):
            indices = np.random.choice(len(ctx_data), len(ctx_data), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            model_boot = LinearRegression()
            model_boot.fit(X_boot, y_boot)
            ate_boot.append(model_boot.coef_[0])
        
        ate_se = np.std(ate_boot)
        
        # Significance test
        t_stat = ate_adjusted / ate_se if ate_se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        sig = config.get_significance_marker(p_value)
        
        print(f"  N: {len(ctx_data):,} observations")
        print(f"  N_CUIs: {ctx_data['cui'].nunique():,}")
        print(f"  ATE (Simple): {ate_simple:.4f}")
        print(f"  ATE (Adjusted): {ate_adjusted:.4f} {sig}")
        print(f"  SE: {ate_se:.4f}")
        
        results.append({
            'Context': ctx,
            'N_CUIs': ctx_data['cui'].nunique(),
            'N_Obs': len(ctx_data),
            'Rare_Rate': ctx_data['is_rare'].mean(),
            'Avg_Complexity': ctx_data['avg_complexity'].mean(),
            'ATE_Simple': ate_simple,
            'ATE_Adjusted': ate_adjusted,
            'SE': ate_se,
            'Significant': sig
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("CONTEXT-SPECIFIC TREATMENT EFFECTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Calculate range
    if len(results_df) > 0:
        ate_range = results_df['ATE_Adjusted'].max() - results_df['ATE_Adjusted'].min()
        ate_ratio = abs(results_df['ATE_Adjusted'].max() / results_df['ATE_Adjusted'].min())
        
        print(f"\nHeterogeneity:")
        print(f"  ATE Range: {ate_range:.4f}")
        print(f"  ATE Ratio: {ate_ratio:.1f}-fold")
    
    return results_df


def identify_effect_modifiers(
    df: pd.DataFrame,
    outcome: str = 'NER_F1'
) -> dict:
    """
    Identify characteristics that modify treatment effects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with treatment and covariates
    outcome : str
        Outcome variable
        
    Returns
    -------
    dict
        Effect modification analysis
    """
    
    print("\n" + "="*60)
    print("EFFECT MODIFICATION ANALYSIS")
    print("="*60)
    
    modifiers = {}
    
    # By volume (total mentions)
    print("\n--- By Volume ---")
    df['volume_group'] = pd.qcut(
        df['total_mentions'], 
        q=3, 
        labels=['Low', 'Medium', 'High']
    )
    
    for vol in ['Low', 'Medium', 'High']:
        vol_data = df[df['volume_group'] == vol]
        treated = vol_data[vol_data['treated_cui'] == 1][outcome].mean()
        control = vol_data[vol_data['treated_cui'] == 0][outcome].mean()
        ate = treated - control
        print(f"  {vol}: {ate:.4f}")
        modifiers[f'volume_{vol}'] = ate
    
    # By complexity
    print("\n--- By Complexity ---")
    df['complexity_group'] = pd.qcut(
        df['avg_complexity'], 
        q=3, 
        labels=['Low', 'Medium', 'High'],
        duplicates='drop'
    )
    
    for comp in df['complexity_group'].unique():
        comp_data = df[df['complexity_group'] == comp]
        treated = comp_data[comp_data['treated_cui'] == 1][outcome].mean()
        control = comp_data[comp_data['treated_cui'] == 0][outcome].mean()
        ate = treated - control
        print(f"  {comp}: {ate:.4f}")
        modifiers[f'complexity_{comp}'] = ate
    
    # By rarity
    print("\n--- By Rarity ---")
    for rare in [0, 1]:
        rare_label = 'Rare' if rare == 1 else 'Common'
        rare_data = df[df['is_rare'] == rare]
        
        if len(rare_data) == 0:
            continue
            
        treated = rare_data[rare_data['treated_cui'] == 1][outcome].mean()
        control = rare_data[rare_data['treated_cui'] == 0][outcome].mean()
        ate = treated - control
        print(f"  {rare_label}: {ate:.4f}")
        modifiers[f'rarity_{rare_label}'] = ate
    
    return modifiers


def plot_hte_summary(hte_df: pd.DataFrame) -> dict:
    """
    Create summary statistics for HTE visualization.
    
    Parameters
    ----------
    hte_df : pd.DataFrame
        Context-specific ATEs
        
    Returns
    -------
    dict
        Summary for plotting
    """
    
    summary = {
        'contexts': hte_df['Context'].tolist(),
        'ate_values': hte_df['ATE_Adjusted'].tolist(),
        'se_values': hte_df['SE'].tolist(),
        'n_obs': hte_df['N_Obs'].tolist(),
        'interpretations': []
    }
    
    # Add interpretations
    for _, row in hte_df.iterrows():
        if row['Avg_Complexity'] > 10:
            interp = f"Context {row['Context']}\n(High complexity, Low volume)"
        elif row['N_Obs'] > 40000:
            interp = f"Context {row['Context']}\n(Low complexity, Very high volume)"
        elif row['N_Obs'] > 25000:
            interp = f"Context {row['Context']}\n(Low complexity, High volume)"
        else:
            interp = f"Context {row['Context']}\n(Medium complexity, Moderate volume)"
        
        summary['interpretations'].append(interp)
    
    return summary


def run_hte_analysis(df: pd.DataFrame, outcome: str = 'NER_F1') -> dict:
    """
    Run complete heterogeneous treatment effects analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with all required variables
    outcome : str
        Outcome variable
        
    Returns
    -------
    dict
        All HTE results
        
    Examples
    --------
    >>> from src.heterogeneity import run_hte_analysis
    >>> results = run_hte_analysis(df)
    """
    
    # Estimate context-specific ATEs
    hte_df = estimate_context_ate(df, outcome=outcome)
    
    # Identify effect modifiers
    modifiers = identify_effect_modifiers(df, outcome=outcome)
    
    # Summary for plotting
    plot_summary = plot_hte_summary(hte_df)
    
    print("\n" + "="*60)
    print("HTE ANALYSIS COMPLETE")
    print("="*60)
    
    return {
        'hte_df': hte_df,
        'modifiers': modifiers,
        'plot_summary': plot_summary
    }
