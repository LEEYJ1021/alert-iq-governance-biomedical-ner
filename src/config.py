"""
Configuration module for Alert-Based IQ Governance Analysis
===========================================================

This module contains all configuration parameters for the analysis.
"""

from pathlib import Path
from typing import Dict, List


class Config:
    """Central configuration for all analysis parameters."""
    
    # ========================================================================
    # PATHS
    # ========================================================================
    
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    OUTPUT_DIR = BASE_DIR / "outputs"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    TABLES_DIR = OUTPUT_DIR / "tables"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    
    # Create directories
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                      FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # DATA SOURCE
    # ========================================================================
    
    DATA_URL = "https://raw.githubusercontent.com/KuleenS/synth-der-den/master/data/SYNTHETIC_MENTIONS.csv"
    RAW_DATA_FILE = RAW_DATA_DIR / "SYNTHETIC_MENTIONS.csv"
    PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "analysis_ready.parquet"
    
    # ========================================================================
    # ANALYSIS PARAMETERS
    # ========================================================================
    
    # Sampling
    SAMPLE_SIZE = None  # None = use full dataset
    RANDOM_SEED = 42
    
    # Entity classification
    MIN_CUI_OBS = 5
    RARE_THRESHOLD = 5  # CUIs with ≤5 mentions are "rare"
    
    # Multi-context simulation
    N_CONTEXTS = 5
    
    # Quasi-experimental design
    TREATMENT_PERIOD = 4  # Period when treatment begins
    
    # ========================================================================
    # PERFORMANCE SIMULATION
    # ========================================================================
    
    BASELINE_NER_F1 = 0.85
    BASELINE_NEN_ACC = 0.78
    DEGRADATION_RATE = 0.02
    
    # Alert response probability
    ALERT_ACTION_RATE = 0.7  # 70% of alerts are acted upon
    
    # ========================================================================
    # CAUSAL INFERENCE
    # ========================================================================
    
    # Instrumental Variable
    IV_LAG = 2  # Use lag-2 complexity as instrument
    
    # Mediation Analysis
    MEDIATION_BOOTSTRAP_SAMPLES = 10_000
    MEDIATION_CI_METHOD = "bca"  # Bias-corrected and accelerated
    
    # ========================================================================
    # COST-BENEFIT PARAMETERS
    # ========================================================================
    
    FP_COST = 1.0   # False positive alert cost (USD)
    FN_COST = 5.0   # False negative (missed drift) cost (USD)
    
    # Alert threshold
    DEFAULT_JACCARD_THRESHOLD = 0.7  # Default engineering threshold
    
    # ========================================================================
    # EXTERNAL BENCHMARKS
    # ========================================================================
    
    EXTERNAL_BENCHMARKS: List[str] = ['BC5CDR', 'NCBI-disease', 'Internal']
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    DPI = 300
    FIGURE_FORMAT = 'png'
    
    FIGURE_SIZE_STANDARD = (10, 7)
    FIGURE_SIZE_WIDE = (12, 7)
    FIGURE_SIZE_TALL = (10, 8)
    FIGURE_SIZE_PANEL = (16, 7)
    
    COLOR_PALETTE = "colorblind"  # seaborn palette
    
    # ========================================================================
    # TABLES
    # ========================================================================
    
    TABLE_FLOAT_FORMAT = "{:.4f}"
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    SIGNIFICANCE_LEVELS = {
        0.001: '***',
        0.01: '**',
        0.05: '*',
        1.0: 'ns'
    }
    
    @classmethod
    def get_significance_marker(cls, p_value: float) -> str:
        """Get significance marker for p-value."""
        for threshold, marker in cls.SIGNIFICANCE_LEVELS.items():
            if p_value < threshold:
                return marker
        return 'ns'


# Create singleton instance
config = Config()
