#!/usr/bin/env python3
"""
Run Cost-Benefit Analysis
==========================

ROC-based threshold optimization.

Usage:
    python scripts/04_run_cost_benefit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from data_loader import load_processed_data
from cost_benefit import run_cost_benefit_analysis
from utils import save_table


def main():
    """Run cost-benefit analysis."""
    
    print("\n" + "="*80)
    print("COST-BENEFIT ANALYSIS")
    print("="*80)
    
    df = load_processed_data()
    
    results = run_cost_benefit_analysis(df)
    
    # Save results
    cost_df = pd.DataFrame([
        {'Metric': 'True Positives', 'Value': results['confusion']['TP']},
        {'Metric': 'False Positives', 'Value': results['confusion']['FP']},
        {'Metric': 'True Negatives', 'Value': results['confusion']['TN']},
        {'Metric': 'False Negatives', 'Value': results['confusion']['FN']},
        {'Metric': 'Total Cost', 'Value': results['costs']['total_cost']},
        {'Metric': 'Baseline Cost (No Monitoring)', 'Value': results['costs']['baseline_cost']},
        {'Metric': 'Cost Saving', 'Value': results['costs']['cost_saving']},
        {'Metric': 'Saving %', 'Value': results['costs']['cost_saving_pct']},
        {'Metric': 'Optimal Threshold', 'Value': results['roc_result']['optimal_threshold']},
        {'Metric': 'ROC AUC', 'Value': results['roc_result']['roc_auc']}
    ])
    
    save_table(cost_df, 'table8_cost_benefit.csv')
    
    print("\n✅ Cost-benefit analysis complete")
    print("   Saved: table8_cost_benefit.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
