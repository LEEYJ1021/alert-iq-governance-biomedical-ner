"""
Alert-Based IQ Governance Analysis Package
===========================================

This package provides tools for causal evaluation of alert-based 
information quality governance in biomedical entity recognition systems.

Modules
-------
config
    Configuration parameters for all analyses
data_loader
    Data downloading and loading utilities
feature_engineering
    Feature creation from raw mentions
alert_signals
    Alert signal generation (structural & relational)
performance_simulation
    NER/NEN performance simulation
causal_inference
    DiD, IV, and Mediation analyses
cost_benefit
    ROC-based threshold optimization
heterogeneity
    Heterogeneous treatment effects
markov_model
    Markov chain temporal modeling
visualization
    Publication figure generation
utils
    Helper functions

Quick Start
-----------
>>> from src.data_loader import get_data
>>> from src.feature_engineering import engineer_all_features
>>> from src.alert_signals import generate_all_alerts
>>> 
>>> # Load and process data
>>> df = get_data()
>>> df = engineer_all_features(df)
>>> df = generate_all_alerts(df)

Author
------
Yong-Jae Lee
AI Lab, TOBESOFT
Email: yj11021@tobesoft.com

Version
-------
1.0.0

License
-------
MIT License (code)
ODbL v1.0 (data)
"""

__version__ = '1.0.0'
__author__ = 'Yong-Jae Lee'
__email__ = 'yj11021@tobesoft.com'
__institution__ = 'TOBESOFT AI Lab'

# Import configuration first
from .config import config, Config

# Import data handling
from .data_loader import (
    download_data,
    load_raw_data,
    load_processed_data,
    save_processed_data,
    get_data
)

# Import feature engineering
from .feature_engineering import (
    create_mention_features,
    create_cui_features,
    create_panel_structure,
    create_deployment_contexts,
    engineer_all_features
)

# Import alert signals
from .alert_signals import (
    calculate_structural_alerts,
    calculate_relational_alerts,
    calculate_drift_proxy,
    generate_all_alerts
)

# Import performance simulation
from .performance_simulation import (
    calculate_context_penalties,
    simulate_ner_performance,
    simulate_nen_performance,
    simulate_alert_actions,
    simulate_all_performance
)

# Import causal inference
from .causal_inference import (
    prepare_did_data,
    test_parallel_trends,
    estimate_did,
    estimate_iv,
    estimate_mediation
)

# Import cost-benefit analysis
from .cost_benefit import (
    simulate_true_drift,
    calculate_confusion_matrix,
    calculate_costs,
    optimize_threshold_roc,
    compare_threshold_strategies,
    run_cost_benefit_analysis
)

# Import heterogeneity analysis
from .heterogeneity import (
    estimate_context_ate,
    identify_effect_modifiers,
    run_hte_analysis
)

# Import Markov modeling
from .markov_model import (
    create_performance_states,
    estimate_transition_matrix,
    calculate_steady_state,
    run_markov_analysis
)

# Import visualization
from .visualization import (
    plot_did_visualization,
    plot_parallel_trends,
    plot_roc_curve,
    plot_markov_transition,
    generate_all_figures
)

# Import utilities
from .utils import (
    print_section,
    print_subsection,
    format_number,
    format_pvalue,
    save_table,
    save_figure,
    calculate_effect_size,
    bootstrap_ci,
    check_data_quality,
    validate_panel_structure
)

# Define public API
__all__ = [
    # Configuration
    'config',
    'Config',
    
    # Data loading
    'download_data',
    'load_raw_data',
    'load_processed_data',
    'save_processed_data',
    'get_data',
    
    # Feature engineering
    'create_mention_features',
    'create_cui_features',
    'create_panel_structure',
    'create_deployment_contexts',
    'engineer_all_features',
    
    # Alert signals
    'calculate_structural_alerts',
    'calculate_relational_alerts',
    'calculate_drift_proxy',
    'generate_all_alerts',
    
    # Performance simulation
    'calculate_context_penalties',
    'simulate_ner_performance',
    'simulate_nen_performance',
    'simulate_alert_actions',
    'simulate_all_performance',
    
    # Causal inference
    'prepare_did_data',
    'test_parallel_trends',
    'estimate_did',
    'estimate_iv',
    'estimate_mediation',
    
    # Cost-benefit
    'simulate_true_drift',
    'calculate_confusion_matrix',
    'calculate_costs',
    'optimize_threshold_roc',
    'compare_threshold_strategies',
    'run_cost_benefit_analysis',
    
    # Heterogeneity
    'estimate_context_ate',
    'identify_effect_modifiers',
    'run_hte_analysis',
    
    # Markov modeling
    'create_performance_states',
    'estimate_transition_matrix',
    'calculate_steady_state',
    'run_markov_analysis',
    
    # Visualization
    'plot_did_visualization',
    'plot_parallel_trends',
    'plot_roc_curve',
    'plot_markov_transition',
    'generate_all_figures',
    
    # Utilities
    'print_section',
    'print_subsection',
    'format_number',
    'format_pvalue',
    'save_table',
    'save_figure',
    'calculate_effect_size',
    'bootstrap_ci',
    'check_data_quality',
    'validate_panel_structure',
]


def print_package_info():
    """Print package information."""
    print(f"""
    {'='*70}
    Alert-Based IQ Governance Analysis Package
    {'='*70}
    
    Version:     {__version__}
    Author:      {__author__}
    Email:       {__email__}
    Institution: {__institution__}
    
    Quick Start:
    ------------
    from src import get_data, engineer_all_features, generate_all_alerts
    
    df = get_data()
    df = engineer_all_features(df)
    df = generate_all_alerts(df)
    
    Documentation:
    --------------
    See README.md for full documentation
    See docs/REPRODUCTION.md for step-by-step guide
    See docs/DATA_DICTIONARY.md for variable reference
    
    Citation:
    ---------
    Lee, Y.-J. (2025). Causal Evaluation of Alert-Based Information 
    Quality Governance in Biomedical Entity Recognition under Long-Tail 
    Distributional Shift. TOBESOFT AI Lab.
    
    {'='*70}
    """)


# Print info when package is imported in interactive mode
import sys
if hasattr(sys, 'ps1'):  # Interactive mode
    print_package_info()
