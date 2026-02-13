"""
Data Loading Module
===================

Handles downloading and loading the PhysioNet synthetic disease mention corpus.
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .config import config


def download_data(
    url: str = config.DATA_URL,
    output_path: Path = config.RAW_DATA_FILE,
    force: bool = False
) -> Path:
    """
    Download raw data from PhysioNet GitHub repository.
    
    Parameters
    ----------
    url : str
        URL to download data from
    output_path : Path
        Where to save downloaded file
    force : bool
        If True, re-download even if file exists
        
    Returns
    -------
    Path
        Path to downloaded file
        
    Examples
    --------
    >>> data_path = download_data()
    >>> print(f"Data downloaded to: {data_path}")
    """
    
    # Check if file already exists
    if output_path.exists() and not force:
        print(f"Data already exists at: {output_path}")
        print("Use force=True to re-download")
        return output_path
    
    print(f"Downloading data from: {url}")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, 
                  desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Data downloaded successfully to: {output_path}")
    return output_path


def load_raw_data(
    data_path: Path = config.RAW_DATA_FILE,
    sample_size: Optional[int] = config.SAMPLE_SIZE,
    random_state: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """
    Load raw synthetic mention data.
    
    Parameters
    ----------
    data_path : Path
        Path to raw CSV file
    sample_size : Optional[int]
        If provided, randomly sample this many rows
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Raw data with columns: cui, matched_output
        
    Examples
    --------
    >>> df = load_raw_data()
    >>> print(f"Loaded {len(df):,} rows")
    """
    
    print(f"\nLoading data from: {data_path}")
    
    # Load CSV
    df = pd.read_csv(data_path)
    
    print(f"Initial rows: {len(df):,}")
    
    # Extract mention from tagged output
    print("Extracting disease mentions...")
    df["mention"] = df["matched_output"].str.extract(r"<1CUI>(.*?)</1CUI>")[0]
    
    # Drop rows without valid mentions
    df = df.dropna(subset=["mention"])
    print(f"Rows with valid mentions: {len(df):,}")
    
    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        print(f"Sampling {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=random_state)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"✓ Loaded {len(df):,} disease mentions")
    print(f"  Unique CUIs: {df['cui'].nunique():,}")
    
    return df


def save_processed_data(
    df: pd.DataFrame,
    output_path: Path = config.PROCESSED_DATA_FILE
) -> Path:
    """
    Save processed data to parquet format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe
    output_path : Path
        Where to save processed data
        
    Returns
    -------
    Path
        Path to saved file
    """
    
    print(f"\nSaving processed data to: {output_path}")
    
    df.to_parquet(output_path, index=False, compression='snappy')
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(df):,} rows ({file_size_mb:.2f} MB)")
    
    return output_path


def load_processed_data(
    data_path: Path = config.PROCESSED_DATA_FILE
) -> pd.DataFrame:
    """
    Load processed data from parquet file.
    
    Parameters
    ----------
    data_path : Path
        Path to processed parquet file
        
    Returns
    -------
    pd.DataFrame
        Processed data ready for analysis
    """
    
    print(f"\nLoading processed data from: {data_path}")
    
    df = pd.read_parquet(data_path)
    
    print(f"✓ Loaded {len(df):,} rows")
    print(f"  Features: {len(df.columns)} columns")
    
    return df


# Convenience function
def get_data(
    force_download: bool = False,
    force_reprocess: bool = False
) -> pd.DataFrame:
    """
    Get analysis-ready data (download and/or process if needed).
    
    Parameters
    ----------
    force_download : bool
        Force re-download of raw data
    force_reprocess : bool
        Force re-processing of features
        
    Returns
    -------
    pd.DataFrame
        Analysis-ready data
        
    Examples
    --------
    >>> df = get_data()  # Uses cached data if available
    >>> df = get_data(force_reprocess=True)  # Re-runs preprocessing
    """
    
    # Check if processed data exists
    if config.PROCESSED_DATA_FILE.exists() and not force_reprocess:
        return load_processed_data()
    
    # Download raw data if needed
    if not config.RAW_DATA_FILE.exists() or force_download:
        download_data(force=force_download)
    
    # Load and return raw data (preprocessing happens in separate module)
    return load_raw_data()
