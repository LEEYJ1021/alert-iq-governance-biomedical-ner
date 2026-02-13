"""
Markov Chain Modeling Module
=============================

Estimates state transition probabilities and steady-state distributions.
"""

import pandas as pd
import numpy as np
from scipy.linalg import eig

from .config import config


def create_performance_states(
    df: pd.DataFrame,
    outcome: str = 'NER_F1',
    n_states: int = 4
) -> pd.DataFrame:
    """
    Discretize continuous performance into states.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with performance metric
    outcome : str
        Performance variable
    n_states : int
        Number of discrete states
        
    Returns
    -------
    pd.DataFrame
        Data with 'perf_state' column
    """
    
    print("\n" + "="*60)
    print("CREATING PERFORMANCE STATES")
    print("="*60)
    
    # Quartile-based states
    df['perf_state'] = pd.qcut(
        df[outcome],
        q=n_states,
        labels=['Low', 'Medium', 'High', 'Very High'][:n_states],
        duplicates='drop'
    )
    
    print(f"\nState distribution:")
    print(df['perf_state'].value_counts().sort_index().to_string())
    
    return df


def estimate_transition_matrix(
    df: pd.DataFrame,
    state_col: str = 'perf_state'
) -> np.ndarray:
    """
    Estimate Markov transition probability matrix.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with state column and time structure
    state_col : str
        State variable name
        
    Returns
    -------
    np.ndarray
        Transition probability matrix (n_states × n_states)
        
    Notes
    -----
    P[i, j] = P(state_t+1 = j | state_t = i)
    """
    
    print("\n" + "="*60)
    print("ESTIMATING TRANSITION MATRIX")
    print("="*60)
    
    # Get state labels
    states = sorted(df[state_col].dropna().unique())
    n_states = len(states)
    
    print(f"\nStates: {states}")
    
    # Initialize transition matrix
    transition_matrix = np.zeros((n_states, n_states))
    
    # Count transitions
    for cui in df['cui'].unique():
        cui_data = df[df['cui'] == cui].sort_values('time_period')
        
        for i in range(len(cui_data) - 1):
            current_state = cui_data.iloc[i][state_col]
            next_state = cui_data.iloc[i + 1][state_col]
            
            if pd.notna(current_state) and pd.notna(next_state):
                current_idx = states.index(current_state)
                next_idx = states.index(next_state)
                transition_matrix[current_idx, next_idx] += 1
    
    # Normalize to probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_matrix / row_sums
    
    print(f"\n✓ Transition matrix estimated")
    print("\nTransition Probabilities:")
    
    # Create DataFrame for display
    trans_df = pd.DataFrame(
        transition_matrix,
        index=states,
        columns=states
    )
    print(trans_df.round(3).to_string())
    
    return transition_matrix, states


def calculate_steady_state(
    transition_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate steady-state distribution.
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition probability matrix
        
    Returns
    -------
    np.ndarray
        Steady-state probability vector
        
    Notes
    -----
    Finds the eigenvector corresponding to eigenvalue 1
    of the transpose of the transition matrix.
    """
    
    print("\n" + "="*60)
    print("CALCULATING STEADY STATE")
    print("="*60)
    
    # Find eigenvalues and eigenvectors of transpose
    eigenvalues, eigenvectors = eig(transition_matrix.T)
    
    # Find eigenvalue closest to 1
    steady_state_idx = np.argmax(eigenvalues.real)
    steady_state = np.abs(eigenvectors[:, steady_state_idx].real)
    
    # Normalize to sum to 1
    steady_state = steady_state / steady_state.sum()
    
    print(f"\n✓ Steady-state calculated")
    
    return steady_state


def calculate_mean_return_time(
    transition_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate mean return time to each state.
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition probability matrix
        
    Returns
    -------
    np.ndarray
        Mean return times
        
    Notes
    -----
    Mean return time to state i = 1 / π_i
    where π_i is steady-state probability of state i
    """
    
    steady_state = calculate_steady_state(transition_matrix)
    
    # Avoid division by zero
    mean_return_times = np.where(
        steady_state > 0,
        1 / steady_state,
        np.inf
    )
    
    return mean_return_times


def interpret_markov_results(
    transition_matrix: np.ndarray,
    steady_state: np.ndarray,
    states: list
) -> dict:
    """
    Interpret Markov model results.
    
    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition probabilities
    steady_state : np.ndarray
        Steady-state distribution
    states : list
        State labels
        
    Returns
    -------
    dict
        Interpretation and insights
    """
    
    print("\n" + "="*60)
    print("MARKOV MODEL INTERPRETATION")
    print("="*60)
    
    insights = {}
    
    # Steady-state interpretation
    print("\nSteady-State Distribution:")
    for i, state in enumerate(states):
        print(f"  {state}: {steady_state[i]:.3f}")
        insights[f'steady_state_{state}'] = steady_state[i]
    
    # Dominant state
    dominant_idx = np.argmax(steady_state)
    dominant_state = states[dominant_idx]
    insights['dominant_state'] = dominant_state
    
    print(f"\nDominant state: {dominant_state} ({steady_state[dominant_idx]:.1%})")
    
    # Persistence analysis
    print("\nState Persistence (diagonal probabilities):")
    for i, state in enumerate(states):
        persist_prob = transition_matrix[i, i]
        print(f"  {state} → {state}: {persist_prob:.3f}")
        insights[f'persistence_{state}'] = persist_prob
    
    # Absorption tendency
    low_state_persist = transition_matrix[0, 0]
    high_state_persist = transition_matrix[-1, -1]
    
    if low_state_persist > 0.6:
        print(f"\n! High persistence in LOW state ({low_state_persist:.3f})")
        print("  → Degraded systems rarely self-correct")
        print("  → Continuous monitoring is CRITICAL")
        insights['interpretation'] = 'degradation_trap'
    elif high_state_persist > 0.6:
        print(f"\n✓ High persistence in HIGH state ({high_state_persist:.3f})")
        print("  → System maintains good performance")
        print("  → Alert monitoring is EFFECTIVE")
        insights['interpretation'] = 'stable_high_performance'
    else:
        print("\n~ Moderate persistence across states")
        print("  → System fluctuates")
        print("  → Continuous monitoring recommended")
        insights['interpretation'] = 'fluctuating'
    
    return insights


def run_markov_analysis(
    df: pd.DataFrame,
    outcome: str = 'NER_F1',
    n_states: int = 4
) -> dict:
    """
    Run complete Markov chain analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with performance over time
    outcome : str
        Performance variable
    n_states : int
        Number of discrete states
        
    Returns
    -------
    dict
        All Markov analysis results
        
    Examples
    --------
    >>> from src.markov_model import run_markov_analysis
    >>> results = run_markov_analysis(df)
    """
    
    # Create states
    df = create_performance_states(df, outcome=outcome, n_states=n_states)
    
    # Estimate transition matrix
    transition_matrix, states = estimate_transition_matrix(df)
    
    # Calculate steady state
    steady_state = calculate_steady_state(transition_matrix)
    
    # Mean return times
    mean_return_times = calculate_mean_return_time(transition_matrix)
    
    # Interpret results
    insights = interpret_markov_results(transition_matrix, steady_state, states)
    
    print("\n" + "="*60)
    print("MARKOV ANALYSIS COMPLETE")
    print("="*60)
    
    return {
        'transition_matrix': transition_matrix,
        'states': states,
        'steady_state': steady_state,
        'mean_return_times': mean_return_times,
        'insights': insights
    }
