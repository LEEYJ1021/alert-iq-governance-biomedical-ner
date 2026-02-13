"""
Utility Functions
=================

Helper functions used across analysis modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from .config import config


def print_section(title: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "="*width)
    print(title)
    print("="*width)


def print_subsection(title: str, width: int = 60):
    """Print formatted subsection header."""
    print("\n" + "-"*width)
    print(title)
    print("-"*width)


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format number with commas and decimals.
    
    Parameters
    ----------
    num : float
        Number to format
    decimals : int
        Number of decimal places
        
    Returns
    -------
    str
        Formatted string
    """
    if pd.isna(num):
        return "N/A"
    
    if abs(num) >= 1000:
        return f"{num:,.{decimals}f}"
    else:
        return f"{num:.{decimals}f}"


def format_pvalue(p: float) -> str:
    """
    Format p-value with appropriate precision.
    
    Parameters
    ----------
    p : float
        P-value
        
    Returns
    -------
    str
        Formatted p-value
    """
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def save_table(
    df: pd.DataFrame,
    filename: str,
    output_dir: Path = config.TABLES_DIR
) -> Path:
    """
    Save dataframe as CSV table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    filename : str
        Output filename (without path)
    output_dir : Path
        Output directory
        
    Returns
    -------
    Path
        Path to saved file
    """
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    return output_path


def save_figure(
    fig,
    filename: str,
    output_dir: Path = config.FIGURES_DIR,
    dpi: int = config.DPI,
    format: str = config.FIGURE_FORMAT
) -> Path:
    """
    Save matplotlib figure.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    output_dir : Path
        Output directory
    dpi : int
        Resolution
    format : str
        File format
        
    Returns
    -------
    Path
        Path to saved file
    """
    if not filename.endswith(f'.{format}'):
        filename = f"{filename}.{format}"
    
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', format=format)
    print(f"✓ Saved: {output_path}")
    return output_path


def calculate_effect_size(
    treated: np.ndarray,
    control: np.ndarray
) -> float:
    """
    Calculate Cohen's d effect size.
    
    Parameters
    ----------
    treated : np.ndarray
        Treated group outcomes
    control : np.ndarray
        Control group outcomes
        
    Returns
    -------
    float
        Cohen's d
    """
    mean_diff = treated.mean() - control.mean()
    pooled_std = np.sqrt(
        ((len(treated) - 1) * treated.std()**2 + 
         (len(control) - 1) * control.std()**2) / 
        (len(treated) + len(control) - 2)
    )
    
    return mean_diff / pooled_std if pooled_std > 0 else 0


def create_summary_stats(
    df: pd.DataFrame,
    variables: list,
    group_by: str = None
) -> pd.DataFrame:
    """
    Create summary statistics table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data
    variables : list
        Variables to summarize
    group_by : str, optional
        Group by variable
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    if group_by:
        summary = df.groupby(group_by)[variables].agg(['mean', 'std', 'min', 'median', 'max'])
    else:
        summary = df[variables].agg(['mean', 'std', 'min', 'median', 'max']).T
    
    return summary


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95
) -> tuple:
    """
    Calculate bootstrap confidence interval.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    statistic : callable
        Statistic function
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence level
        
    Returns
    -------
    tuple
        (point_estimate, ci_lower, ci_upper)
    """
    point_estimate = statistic(data)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check data quality and report issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to check
        
    Returns
    -------
    dict
        Quality report
    """
    report = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'missing_values': {},
        'duplicates': 0,
        'warnings': []
    }
    
    # Missing values
    missing = df.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    # Duplicates
    report['duplicates'] = df.duplicated().sum()
    
    # Warnings
    if len(report['missing_values']) > 0:
        report['warnings'].append(f"Missing values in {len(report['missing_values'])} columns")
    
    if report['duplicates'] > 0:
        report['warnings'].append(f"{report['duplicates']} duplicate rows")
    
    return report


def memory_usage_report(df: pd.DataFrame) -> None:
    """
    Print memory usage report for dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze
    """
    print("\n" + "="*60)
    print("MEMORY USAGE REPORT")
    print("="*60)
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nTotal memory: {memory_mb:.2f} MB")
    
    print("\nBy column:")
    col_memory = df.memory_usage(deep=True).sort_values(ascending=False)
    for col, mem in col_memory.head(10).items():
        print(f"  {col}: {mem/1024**2:.2f} MB")


def validate_panel_structure(df: pd.DataFrame) -> bool:
    """
    Validate panel data structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Panel data
        
    Returns
    -------
    bool
        True if valid
    """
    print("\n" + "="*60)
    print("VALIDATING PANEL STRUCTURE")
    print("="*60)
    
    checks = []
    
    # Check 1: time_period starts at 1
    min_period = df.groupby('cui_id')['time_period'].min().min()
    check1 = (min_period == 1)
    checks.append(check1)
    print(f"\n✓ Time periods start at 1: {check1}")
    
    # Check 2: time_period is monotonic within CUI
    monotonic = df.groupby('cui_id')['time_period'].is_monotonic_increasing.all()
    checks.append(monotonic)
    print(f"✓ Time periods monotonic: {monotonic}")
    
    # Check 3: No missing CUIs
    missing_cuis = df['cui'].isnull().sum()
    check3 = (missing_cuis == 0)
    checks.append(check3)
    print(f"✓ No missing CUIs: {check3} (missing: {missing_cuis})")
    
    all_valid = all(checks)
    
    if all_valid:
        print("\n✅ Panel structure is VALID")
    else:
        print("\n❌ Panel structure has ISSUES")
    
    return all_valid


def create_latex_table(
    df: pd.DataFrame,
    caption: str = "",
    label: str = ""
) -> str:
    """
    Convert DataFrame to LaTeX table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data
    caption : str
        Table caption
    label : str
        LaTeX label
        
    Returns
    -------
    str
        LaTeX table code
    """
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label
    )
    
    return latex
