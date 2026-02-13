#!/usr/bin/env python3
"""
Run Heterogeneous Treatment Effects Analysis
=============================================

Usage:
    python scripts/05_run_heterogeneity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_processed_data
from heterogeneity import run_hte_analysis
from utils import save_table


def main():
    """Run HTE analysis."""
    
    print("\n" + "="*80)
    print("HETEROGENEOUS TREATMENT EFFECTS")
    print("="*80)
    
    df = load_processed_data()
    
    results = run_hte_analysis(df)
    
    save_table(results['hte_df'], 'table5_hte_context.csv')
    
    print("\n✅ HTE analysis complete")
    print("   Saved: table5_hte_context.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
