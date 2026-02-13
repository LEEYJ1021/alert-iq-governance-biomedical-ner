#!/usr/bin/env python3
"""
Generate All Figures
====================

Creates all 12 publication figures.

Usage:
    python scripts/08_generate_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_processed_data
from visualization import generate_all_figures


def main():
    """Generate all figures."""
    
    print("\n" + "="*80)
    print("GENERATING PUBLICATION FIGURES")
    print("="*80)
    
    df = load_processed_data()
    
    # Collect all analysis results
    # (In production, these would be loaded from saved results)
    analysis_results = {}
    
    # Run simplified analyses for figure generation
    from causal_inference import prepare_did_data, test_parallel_trends, estimate_did
    from cost_benefit import run_cost_benefit_analysis
    from markov_model import run_markov_analysis
    
    df = prepare_did_data(df)
    
    analysis_results['parallel_trends'] = test_parallel_trends(df)
    analysis_results['did'] = estimate_did(df)
    analysis_results['cost_benefit'] = run_cost_benefit_analysis(df)
    analysis_results['markov'] = run_markov_analysis(df)
    
    # Generate figures
    figures = generate_all_figures(df, analysis_results)
    
    print("\n✅ Figure generation complete")
    print(f"   Generated {len(figures)} figures in outputs/figures/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
