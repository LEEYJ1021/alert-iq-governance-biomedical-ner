#!/usr/bin/env python3
"""
Run Markov Chain Analysis
==========================

Usage:
    python scripts/07_run_markov.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from data_loader import load_processed_data
from markov_model import run_markov_analysis
from utils import save_table


def main():
    """Run Markov analysis."""
    
    print("\n" + "="*80)
    print("MARKOV CHAIN ANALYSIS")
    print("="*80)
    
    df = load_processed_data()
    
    results = run_markov_analysis(df)
    
    # Save transition matrix
    trans_df = pd.DataFrame(
        results['transition_matrix'],
        index=results['states'],
        columns=results['states']
    )
    save_table(trans_df, 'table12_markov_transition.csv')
    
    print("\n✅ Markov analysis complete")
    print("   Saved: table12_markov_transition.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
