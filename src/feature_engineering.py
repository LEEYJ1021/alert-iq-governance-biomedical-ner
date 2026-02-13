"""
Feature Engineering Module
===========================

Creates all features used in causal analysis from raw disease mentions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .config import config


def create_mention_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create mention-level features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data with 'mention' column
        
    Returns
    -------
    pd.DataFrame
        Data with added mention-level features
    """
    
    print("\n" + "="*60)
    print("CREATING MENTION-LEVEL FEATURES")
    print("="*60)
    
    # Tokenization
    print("- Tokenizing mentions...")
    df['mention_tokens'] = df['mention'].str.split().str.len()
    df['mention_length_chars'] = df['mention'].str.len()
    
    # Lexical diversity
    print("- Computing lexical diversity...")
    df['unique_tokens'] = df['mention'].apply(
        lambda x: len(set(str(x).lower().split())) if pd.notna(x) else 0
    )
    df['type_token_ratio'] = (
        df['unique_tokens'] / df['mention_tokens'].replace(0, np.nan)
    )
    
    # Complexity metrics
    print("- Computing complexity scores...")
    df['complexity_score'] = df['mention_tokens']
    df['normalized_complexity'] = (
        df['mention_tokens'] / np.log1p(df['mention_length_chars'])
    )
    
    # Clinical/domain features
    print("- Extracting clinical features...")
    df['has_conjunction'] = df['mention'].str.contains(
        r'\b(and|or|with|but|versus)\b', case=False, regex=True
    ).astype(int)
    
    df['has_modifier'] = df['mention'].str.contains(
        r'\b(acute|chronic|severe|mild|moderate|secondary|primary|recurrent)\b',
        case=False, regex=True
    ).astype(int)
    
    df['has_anatomical'] = df['mention'].str.contains(
        r'\b(left|right|upper|lower|bilateral|anterior|posterior)\b',
        case=False, regex=True
    ).astype(int)
    
    print(f"✓ Created {sum([c.startswith('mention_') or c.startswith('has_') 
                           for c in df.columns])} mention features")
    
    return df


def create_cui_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create CUI-level aggregated features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with mention-level features
        
    Returns
    -------
    pd.DataFrame
        Data with added CUI-level features
    """
    
    print("\n" + "="*60)
    print("CREATING CUI-LEVEL FEATURES")
    print("="*60)
    
    # Aggregate statistics
    print("- Computing CUI-level statistics...")
    cui_stats = df.groupby('cui').agg({
        'mention': ['count', 'nunique'],
        'complexity_score': ['mean', 'std'],
        'type_token_ratio': 'mean'
    }).reset_index()
    
    cui_stats.columns = [
        'cui', 'total_mentions', 'unique_mentions', 
        'avg_complexity', 'std_complexity', 'avg_ttr'
    ]
    
    # Fill missing standard deviations (single mention CUIs)
    cui_stats['std_complexity'] = cui_stats['std_complexity'].fillna(0)
    
    # Rarity indicator
    cui_stats['is_rare'] = (
        cui_stats['total_mentions'] <= config.RARE_THRESHOLD
    ).astype(int)
    
    # Merge back
    df = df.merge(cui_stats, on='cui', how='left')
    
    print(f"✓ Created {len(cui_stats.columns)-1} CUI features")
    print(f"  Rare CUIs: {cui_stats['is_rare'].sum():,} / {len(cui_stats):,} "
          f"({100*cui_stats['is_rare'].mean():.1f}%)")
    
    return df


def create_panel_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create panel data structure with time periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with CUI features
        
    Returns
    -------
    pd.DataFrame
        Data with panel structure (time_period, cui_id)
    """
    
    print("\n" + "="*60)
    print("CREATING PANEL STRUCTURE")
    print("="*60)
    
    # Sort by CUI and mention (ensures consistent ordering)
    print("- Sorting observations...")
    df = df.sort_values(['cui', 'mention']).reset_index(drop=True)
    
    # Create time periods (cumulative count within CUI)
    print("- Assigning time periods...")
    df['time_period'] = df.groupby('cui').cumcount() + 1
    
    # Create numeric CUI ID (for efficient processing)
    df['cui_id'] = pd.Categorical(df['cui']).codes
    
    print(f"✓ Panel structure created")
    print(f"  CUIs: {df['cui'].nunique():,}")
    print(f"  Time periods per CUI: {df['time_period'].min()}-{df['time_period'].max()}")
    print(f"  Median periods: {df.groupby('cui')['time_period'].max().median():.0f}")
    
    return df


def create_deployment_contexts(
    df: pd.DataFrame,
    n_contexts: int = config.N_CONTEXTS,
    random_state: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """
    Create deployment contexts using K-means clustering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with CUI features
    n_contexts : int
        Number of contexts to create
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Data with 'context_id' column
    """
    
    print("\n" + "="*60)
    print(f"CREATING {n_contexts} DEPLOYMENT CONTEXTS")
    print("="*60)
    
    # Features for clustering
    print("- Selecting features for clustering...")
    cui_features = df.groupby('cui').agg({
        'avg_complexity': 'first',
        'avg_ttr': 'first',
        'total_mentions': 'first',
        'is_rare': 'first'
    }).fillna(0)
    
    # Standardize features
    print("- Standardizing features...")
    scaler = StandardScaler()
    cui_features_scaled = scaler.fit_transform(cui_features)
    
    # K-means clustering
    print(f"- Running K-means (k={n_contexts})...")
    kmeans = KMeans(
        n_clusters=n_contexts,
        random_state=random_state,
        n_init=10
    )
    cui_features['context_id'] = kmeans.fit_predict(cui_features_scaled)
    
    # Merge back
    df = df.merge(
        cui_features[['context_id']].reset_index(),
        on='cui',
        how='left'
    )
    
    # Print context distribution
    print(f"\n✓ Created {n_contexts} contexts")
    print("\nContext Distribution:")
    
    context_dist = df.groupby('context_id').agg({
        'cui': 'nunique',
        'mention': 'count',
        'is_rare': 'mean',
        'avg_complexity': 'mean'
    })
    context_dist.columns = ['CUIs', 'Mentions', 'Rare_Rate', 'Avg_Complexity']
    
    print(context_dist.round(3).to_string())
    
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run complete feature engineering pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data with 'cui' and 'mention' columns
        
    Returns
    -------
    pd.DataFrame
        Data with all engineered features
        
    Examples
    --------
    >>> from src.data_loader import load_raw_data
    >>> from src.feature_engineering import engineer_all_features
    >>> 
    >>> df = load_raw_data()
    >>> df = engineer_all_features(df)
    """
    
    # Create features
    df = create_mention_features(df)
    df = create_cui_features(df)
    df = create_panel_structure(df)
    df = create_deployment_contexts(df)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    
    return df
