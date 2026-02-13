#!/usr/bin/env python3
"""
Download Raw Data
=================

Downloads the PhysioNet Synthetic Mention Corpora dataset.

Usage:
    python scripts/01_download_data.py [--force]
    
Options:
    --force    Re-download even if file exists
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import download_data
from config import config


def main():
    """Download data with command-line options."""
    
    parser = argparse.ArgumentParser(
        description='Download PhysioNet Synthetic Mention Corpora'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file exists'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DOWNLOADING PHYSIONET SYNTHETIC MENTION CORPORA")
    print("="*80)
    
    print("\nDataset Information:")
    print("  Title: Synthetic Mention Corpora for Disease Entity Recognition")
    print("  Version: 1.0.0")
    print("  Authors: Kuleen Sasse, John David Osborne")
    print("  DOI: 10.13026/p5pn-ty93")
    print("  License: Open Database License (ODbL v1.0)")
    
    print(f"\nSource URL:")
    print(f"  {config.DATA_URL}")
    
    print(f"\nDestination:")
    print(f"  {config.RAW_DATA_FILE}")
    
    # Download
    try:
        data_path = download_data(force=args.force)
        
        # Check file size
        file_size_mb = data_path.stat().st_size / (1024 * 1024)
        
        print("\n" + "="*80)
        print("✅ DOWNLOAD COMPLETE")
        print("="*80)
        print(f"\nFile: {data_path}")
        print(f"Size: {file_size_mb:.2f} MB")
        
        print("\nCitation:")
        print("-"*80)
        print("Sasse, K., & Osborne, J. D. (2025). Synthetic Mention Corpora for")
        print("Disease Entity Recognition and Normalization (version 1.0.0).")
        print("PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/p5pn-ty93")
        print("-"*80)
        
        print("\nNext step:")
        print("  python scripts/02_preprocess_data.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
