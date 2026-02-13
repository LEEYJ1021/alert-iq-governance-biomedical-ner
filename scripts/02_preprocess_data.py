#!/usr/bin/env python3
"""
Preprocess Data
===============

Feature engineering and alert signal generation.

Usage:
    python scripts/02_preprocess_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_raw_data, save_processed_data
from feature_engineering import engineer_all_features
from alert_signals import generate_all_alerts
from performance_simulation import simulate_all_performance


def main():
    """Run preprocessing pipeline."""
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load raw data
    df = load_raw_data()
    
    # Engineer features
    df = engineer_all_features(df)
    
    # Generate alert signals
    df = generate_all_alerts(df)
    
    # Simulate performance
    df = simulate_all_performance(df)
    
    # Save
    save_processed_data(df)
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
