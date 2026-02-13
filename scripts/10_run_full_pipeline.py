#!/usr/bin/env python3
"""
Master Pipeline Script
======================

Runs the complete analysis pipeline from data download to final outputs.

Usage:
    python scripts/10_run_full_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime

# Import our modules
from config import config
from data_loader import download_data, load_raw_data, save_processed_data
from feature_engineering import engineer_all_features
from alert_signals import generate_all_alerts


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def main():
    """Run complete analysis pipeline."""
    
    start_time = datetime.now()
    
    print_section("ALERT-BASED IQ GOVERNANCE ANALYSIS PIPELINE")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ========================================================================
    # STEP 1: DATA ACQUISITION
    # ========================================================================
    
    print_section("STEP 1: DATA ACQUISITION")
    
    # Download if needed
    if not config.RAW_DATA_FILE.exists():
        download_data()
    else:
        print(f"✓ Raw data exists: {config.RAW_DATA_FILE}")
    
    # Load raw data
    df = load_raw_data()
    
    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    
    print_section("STEP 2: FEATURE ENGINEERING")
    
    df = engineer_all_features(df)
    
    # ========================================================================
    # STEP 3: ALERT SIGNAL GENERATION
    # ========================================================================
    
    print_section("STEP 3: ALERT SIGNAL GENERATION")
    
    df = generate_all_alerts(df)
    
    # ========================================================================
    # STEP 4: PERFORMANCE SIMULATION
    # ========================================================================
    
    print_section("STEP 4: PERFORMANCE SIMULATION")
    
    print("Simulating NER F1 and NEN Accuracy scores...")
    
    # Context-specific baselines
    context_baselines = df.groupby('context_id')['avg_complexity'].mean()
    context_baselines = (
        (context_baselines - context_baselines.min()) / 
        (context_baselines.max() - context_baselines.min()) * 0.1
    )
    df['context_baseline_penalty'] = df['context_id'].map(context_baselines)
    
    # NER F1 score
    np.random.seed(config.RANDOM_SEED)
    df['NER_F1'] = (
        config.BASELINE_NER_F1 - 
        config.DEGRADATION_RATE * df['cum_undetected_drift'] -
        df['context_baseline_penalty'] +
        np.random.normal(0, 0.02, len(df))
    ).clip(0.3, 1.0)
    
    # NEN Accuracy
    df['NEN_Accuracy'] = (
        config.BASELINE_NEN_ACC - 
        config.DEGRADATION_RATE * 1.2 * df['cum_undetected_drift'] -
        df['context_baseline_penalty'] * 0.8 +
        np.random.normal(0, 0.025, len(df))
    ).clip(0.2, 1.0)
    
    # Alert action simulation
    df['alert_acted_upon'] = (
        (df['Alert_S'] == 1) & 
        (np.random.random(len(df)) < config.ALERT_ACTION_RATE)
    ).astype(int)
    
    print(f"✓ Performance simulation complete")
    print(f"  Mean NER F1: {df['NER_F1'].mean():.4f}")
    print(f"  Mean NEN Accuracy: {df['NEN_Accuracy'].mean():.4f}")
    print(f"  Alerts acted upon: {df['alert_acted_upon'].sum():,}")
    
    # ========================================================================
    # STEP 5: SAVE PROCESSED DATA
    # ========================================================================
    
    print_section("STEP 5: SAVING PROCESSED DATA")
    
    save_processed_data(df)
    
    # ========================================================================
    # STEP 6: SUMMARY STATISTICS
    # ========================================================================
    
    print_section("STEP 6: SUMMARY STATISTICS")
    
    print("\nDataset Overview:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Unique CUIs: {df['cui'].nunique():,}")
    print(f"  Time periods: {df['time_period'].min()}-{df['time_period'].max()}")
    print(f"  Deployment contexts: {df['context_id'].nunique()}")
    
    print("\nAlert Statistics:")
    print(f"  Structural alerts: {df['Alert_S'].sum():,} ({100*df['Alert_S'].mean():.1f}%)")
    print(f"  Relational alerts: {df['Alert_R'].sum():,} ({100*df['Alert_R'].mean():.1f}%)")
    print(f"  Mean drift: {df['cum_undetected_drift'].mean():.4f}")
    
    print("\nPerformance Statistics:")
    print(f"  NER F1: {df['NER_F1'].mean():.4f} ± {df['NER_F1'].std():.4f}")
    print(f"  NEN Acc: {df['NEN_Accuracy'].mean():.4f} ± {df['NEN_Accuracy'].std():.4f}")
    print(f"  Correlation (drift, NER): {df[['cum_undetected_drift', 'NER_F1']].corr().iloc[0,1]:.3f}")
    
    print("\nContext Distribution:")
    context_summary = df.groupby('context_id').agg({
        'cui': 'nunique',
        'mention': 'count',
        'avg_complexity': 'mean',
        'NER_F1': 'mean'
    })
    context_summary.columns = ['CUIs', 'Mentions', 'Avg_Complexity', 'Mean_F1']
    print(context_summary.round(3).to_string())
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_section("✅ PIPELINE COMPLETE")
    
    print(f"\nExecution time: {duration}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run causal analysis:")
    print("   python scripts/03_run_causal_analysis.py")
    print("\n2. Run cost-benefit optimization:")
    print("   python scripts/04_run_cost_benefit.py")
    print("\n3. Generate all figures:")
    print("   python scripts/08_generate_figures.py")
    print("\n4. Generate all tables:")
    print("   python scripts/09_generate_tables.py")
    
    print("\nProcessed data saved to:")
    print(f"  {config.PROCESSED_DATA_FILE}")
    
    print("\n" + "="*80)
    
    return df


if __name__ == "__main__":
    df = main()
