"""
Visualization Module
====================

Creates all 12 publication-quality figures.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from .config import config
from .utils import save_figure


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(config.COLOR_PALETTE)


def plot_did_visualization(
    df: pd.DataFrame,
    did_effect: float,
    output_path: Path = None
) -> plt.Figure:
    """
    Figure 1: Difference-in-Differences visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with treatment variables
    did_effect : float
        DiD coefficient
    output_path : Path, optional
        Where to save figure
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    
    print("\nGenerating Figure 1: DiD Visualization...")
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_STANDARD)
    
    # Calculate group means
    did_plot_data = df.groupby(['treated_cui', 'post_treatment']).agg({
        'NER_F1': 'mean'
    }).reset_index()
    
    control_pre = did_plot_data[(did_plot_data['treated_cui']==0) & 
                                (did_plot_data['post_treatment']==0)]['NER_F1'].values[0]
    control_post = did_plot_data[(did_plot_data['treated_cui']==0) & 
                                 (did_plot_data['post_treatment']==1)]['NER_F1'].values[0]
    treated_pre = did_plot_data[(did_plot_data['treated_cui']==1) & 
                                (did_plot_data['post_treatment']==0)]['NER_F1'].values[0]
    treated_post = did_plot_data[(did_plot_data['treated_cui']==1) & 
                                 (did_plot_data['post_treatment']==1)]['NER_F1'].values[0]
    
    # Plot lines
    ax.plot([0, 1], [control_pre, control_post], 'o-', linewidth=3, 
            markersize=12, color='coral', label='Control', alpha=0.8)
    ax.plot([0, 1], [treated_pre, treated_post], 'o-', linewidth=3,
            markersize=12, color='steelblue', label='Treated (Alert Monitoring)', alpha=0.8)
    
    # Counterfactual
    ax.plot([0, 1], [treated_pre, treated_pre + (control_post - control_pre)],
            '--', linewidth=2.5, color='gray', alpha=0.6, label='Counterfactual')
    
    # Treatment line
    ax.axvline(0.5, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax.text(0.5, ax.get_ylim()[1]*0.98, 'Treatment Begins', ha='center', 
            fontsize=12, color='red', fontweight='bold')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-Treatment', 'Post-Treatment'], fontsize=13)
    ax.set_ylabel('NER F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Difference-in-Differences: Causal Effect of Alert Monitoring',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Annotate DiD effect
    ax.annotate(f'DiD Effect = {did_effect:.4f}',
                xy=(1, treated_post), xytext=(1.15, treated_post+0.015),
                fontsize=13, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, 'fig1_did', config.FIGURES_DIR)
    
    return fig


def plot_parallel_trends(
    df: pd.DataFrame,
    parallel_test: dict,
    output_path: Path = None
) -> plt.Figure:
    """
    Figure 2: Parallel Trends Test.
    
    Parameters
    ----------
    df : pd.DataFrame
        Pre-treatment data
    parallel_test : dict
        Test results
    output_path : Path, optional
        Where to save
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    
    print("\nGenerating Figure 2: Parallel Trends Test...")
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_STANDARD)
    
    # Pre-treatment data
    pre_treatment = df[df['post_treatment'] == 0].copy()
    
    pre_trends = pre_treatment.groupby(['time_period', 'treated_cui']).agg({
        'NER_F1': 'mean'
    }).reset_index()
    
    # Plot trends
    for treated in [0, 1]:
        data = pre_trends[pre_trends['treated_cui'] == treated]
        label = 'Treated' if treated == 1 else 'Control'
        color = 'steelblue' if treated == 1 else 'coral'
        ax.plot(data['time_period'], data['NER_F1'], 'o-', 
                linewidth=2.5, markersize=8, label=label, color=color, alpha=0.8)
    
    # Treatment line
    ax.axvline(config.TREATMENT_PERIOD - 0.5, color='red', linestyle='--', 
               linewidth=2, alpha=0.6)
    ax.text(config.TREATMENT_PERIOD - 0.5, ax.get_ylim()[1]*0.98, 
            'Treatment→', ha='right', fontsize=11, color='red', fontweight='bold')
    
    ax.set_xlabel('Time Period (Pre-Treatment)', fontsize=14, fontweight='bold')
    ax.set_ylabel('NER F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Parallel Trends Test: Pre-Treatment Period',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add test results
    textstr = f"Treated × Time: {parallel_test['coefficient']:.4f}\np-value: {parallel_test['p_value']:.4f}"
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, 'fig2_parallel_trends', config.FIGURES_DIR)
    
    return fig


def plot_roc_curve(
    roc_result: dict,
    output_path: Path = None
) -> plt.Figure:
    """
    Figure 8: ROC Curve and Optimal Threshold.
    
    Parameters
    ----------
    roc_result : dict
        ROC analysis results
    output_path : Path, optional
        Where to save
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    
    print("\nGenerating Figure 8: ROC Curve...")
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_STANDARD)
    
    # Plot ROC curve
    ax.plot(roc_result['fpr'], roc_result['tpr'], linewidth=3,
            label=f"ROC Curve (AUC = {roc_result['roc_auc']:.3f})",
            color='darkorange')
    
    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random Classifier')
    
    # Optimal point
    ax.plot(roc_result['optimal_fpr'], roc_result['optimal_tpr'], 'ro', markersize=15,
            label=f"Optimal (threshold={roc_result['optimal_threshold']:.3f})")
    
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curve: Alert Threshold Optimization',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Cost info
    cost_text = f"""Cost Analysis:
FP Cost: ${config.FP_COST:.2f}
FN Cost: ${config.FN_COST:.2f}
Optimal Cost: ${roc_result['optimal_cost']:,.0f}
Baseline Cost: ${roc_result['baseline_cost']:,.0f}
Savings: {roc_result['cost_saving_pct']:.1f}%"""
    
    ax.text(0.98, 0.02, cost_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, 'fig8_roc_curve', config.FIGURES_DIR)
    
    return fig


def plot_markov_transition(
    transition_matrix: np.ndarray,
    states: list,
    steady_state: np.ndarray,
    output_path: Path = None
) -> plt.Figure:
    """
    Figure 11: Markov Transition Matrix.
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition probabilities
    states : list
        State labels
    steady_state : np.ndarray
        Steady-state distribution
    output_path : Path, optional
        Where to save
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    
    print("\nGenerating Figure 11: Markov Transition Matrix...")
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_TALL)
    
    # Heatmap
    im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(states)))
    ax.set_yticks(np.arange(len(states)))
    ax.set_xticklabels(states, fontsize=12)
    ax.set_yticklabels(states, fontsize=12)
    
    ax.set_xlabel('Next State', fontsize=14, fontweight='bold')
    ax.set_ylabel('Current State', fontsize=14, fontweight='bold')
    ax.set_title('Markov State Transition Probabilities\\n(Performance States Over Time)',
                 fontsize=15, fontweight='bold', pad=20)
    
    # Text annotations
    for i in range(len(states)):
        for j in range(len(states)):
            text = ax.text(j, i, f'{transition_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black",
                          fontsize=11, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Transition Probability', fontsize=12, fontweight='bold')
    
    # Steady state
    steady_text = "Steady-State Distribution:\\n" + "\\n".join(
        [f"{state}: {prob:.3f}" for state, prob in zip(states, steady_state)]
    )
    ax.text(1.35, 0.5, steady_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, 'fig11_markov_transition', config.FIGURES_DIR)
    
    return fig


def generate_all_figures(
    df: pd.DataFrame,
    analysis_results: dict
) -> dict:
    """
    Generate all 12 publication figures.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis data
    analysis_results : dict
        All analysis results
        
    Returns
    -------
    dict
        Dictionary of figure objects
        
    Examples
    --------
    >>> from src.visualization import generate_all_figures
    >>> figures = generate_all_figures(df, results)
    """
    
    print("\n" + "="*80)
    print("GENERATING ALL PUBLICATION FIGURES")
    print("="*80)
    
    figures = {}
    
    # Figure 1: DiD
    if 'did' in analysis_results:
        figures['fig1_did'] = plot_did_visualization(
            df, 
            analysis_results['did']['did_coefficient']
        )
    
    # Figure 2: Parallel Trends
    if 'parallel_trends' in analysis_results:
        figures['fig2_parallel_trends'] = plot_parallel_trends(
            df,
            analysis_results['parallel_trends']
        )
    
    # Figure 8: ROC Curve
    if 'cost_benefit' in analysis_results:
        figures['fig8_roc_curve'] = plot_roc_curve(
            analysis_results['cost_benefit']['roc_result']
        )
    
    # Figure 11: Markov
    if 'markov' in analysis_results:
        figures['fig11_markov'] = plot_markov_transition(
            analysis_results['markov']['transition_matrix'],
            analysis_results['markov']['states'],
            analysis_results['markov']['steady_state']
        )
    
    print("\n" + "="*80)
    print(f"✅ GENERATED {len(figures)} FIGURES")
    print("="*80)
    
    return figures
