#!/usr/bin/env python3
"""
Run Causal Analysis
===================

Executes DiD, IV, and Mediation analyses.

Usage:
    python scripts/03_run_causal_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_processed_data
from causal_inference import (
    prepare_did_data,
    test_parallel_trends,
    estimate_did,
    estimate_iv,
    estimate_mediation
)
from config import config
from utils import save_table


def main():
    """Run causal analysis."""
    
    print("\n" + "="*80)
    print("CAUSAL ANALYSIS PIPELINE")
    print("="*80)
    
    # Load data
    df = load_processed_data()
    
    # Prepare DiD
    df = prepare_did_data(df)
    
    # Test parallel trends
    parallel_test = test_parallel_trends(df)
    
    # Save parallel trends results
    parallel_df = pd.DataFrame([{
        'Test': 'Parallel Trends',
        'Treated_x_Time_Coef': parallel_test['coefficient'],
        'P_Value': parallel_test['p_value'],
        'Conclusion': 'Supported' if parallel_test['supported'] else 'Violated'
    }])
    save_table(parallel_df, 'table2_parallel_trends.csv')
    
    # Estimate DiD
    did_result = estimate_did(df)
    save_table(did_result['results_df'], 'table1_did_results.csv')
    
    # Estimate IV
    iv_result = estimate_iv(df)
    
    iv_df = pd.DataFrame([
        {
            'Stage': 'First Stage',
            'Dependent_Var': 'Alert_S',
            'Key_Coefficient': iv_result['first_stage_f'],
            'F_Statistic': iv_result['first_stage_f'],
            'Cragg_Donald_F': iv_result['cragg_donald_f'],
            'Stock_Yogo_10pct': 16.38,
            'Instrument_Strength': 'STRONG' if iv_result['instrument_strong'] else 'WEAK'
        },
        {
            'Stage': 'Second Stage (IV)',
            'Dependent_Var': 'NER_F1',
            'Key_Coefficient': iv_result['iv_coefficient'],
            'F_Statistic': None,
            'Cragg_Donald_F': None,
            'Stock_Yogo_10pct': None,
            'Instrument_Strength': ''
        }
    ])
    save_table(iv_df, 'table3_iv_analysis.csv')
    
    # Estimate Mediation
    med_result = estimate_mediation(df)
    
    med_df = pd.DataFrame([
        {
            'Path': 'a (Alert → Drift)',
            'Coefficient': med_result['path_a'],
            'Percent_Mediated': None
        },
        {
            'Path': 'b (Drift → Performance)',
            'Coefficient': med_result['path_b'],
            'Percent_Mediated': None
        },
        {
            'Path': 'c (Total Effect)',
            'Coefficient': med_result['path_c'],
            'Percent_Mediated': None
        },
        {
            'Path': 'Indirect (a×b)',
            'Coefficient': med_result['indirect_effect'],
            'Percent_Mediated': med_result['pct_mediated']
        },
        {
            'Path': 'Direct (c-a×b)',
            'Coefficient': med_result['direct_effect'],
            'Percent_Mediated': None
        }
    ])
    save_table(med_df, 'table4_mediation.csv')
    
    print("\n" + "="*80)
    print("✅ CAUSAL ANALYSIS COMPLETE")
    print("="*80)
    
    print("\nTables saved:")
    print("  - table1_did_results.csv")
    print("  - table2_parallel_trends.csv")
    print("  - table3_iv_analysis.csv")
    print("  - table4_mediation.csv")
    
    return 0


if __name__ == "__main__":
    import pandas as pd
    sys.exit(main())
