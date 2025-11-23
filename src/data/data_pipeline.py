"""
Data pipeline for temporal recommendation systems.

This module handles data loading, preprocessing, and temporal splitting
for recommendation system datasets.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TemporalDataLoader:
    """Load and preprocess temporal recommendation data."""
    
    def __init__(self, data_dir: str = "data") -> None:
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_interactions(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load interaction data.
        
        Args:
            file_path: Path to interactions file. If None, uses default path.
            
        Returns:
            DataFrame with interaction data
        """
        if file_path is None:
            file_path = self.data_dir / "interactions.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Interactions file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_columns = ['user_id', 'item_id', 'rating', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} interactions for {df['user_id'].nunique()} users and {df['item_id'].nunique()} items")
        
        return df
    
    def load_items(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load item metadata.
        
        Args:
            file_path: Path to items file. If None, uses default path.
            
        Returns:
            DataFrame with item metadata
        """
        if file_path is None:
            file_path = self.data_dir / "items.csv"
        
        if not Path(file_path).exists():
            logger.warning(f"Items file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        
        # Ensure item_id column exists
        if 'item_id' not in df.columns:
            raise ValueError("Items file must contain 'item_id' column")
        
        logger.info(f"Loaded metadata for {len(df)} items")
        
        return df
    
    def load_users(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load user metadata.
        
        Args:
            file_path: Path to users file. If None, uses default path.
            
        Returns:
            DataFrame with user metadata
        """
        if file_path is None:
            file_path = self.data_dir / "users.csv"
        
        if not Path(file_path).exists():
            logger.warning(f"Users file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        
        # Ensure user_id column exists
        if 'user_id' not in df.columns:
            raise ValueError("Users file must contain 'user_id' column")
        
        logger.info(f"Loaded metadata for {len(df)} users")
        
        return df


class TemporalSplitter:
    """Split data temporally for recommendation system evaluation."""
    
    def __init__(self, test_ratio: float = 0.2, validation_ratio: float = 0.1) -> None:
        """
        Initialize the temporal splitter.
        
        Args:
            test_ratio: Ratio of data to use for testing
            validation_ratio: Ratio of data to use for validation
        """
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        
    def split_by_time(
        self,
        interactions: pd.DataFrame,
        time_column: str = 'timestamp'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split interactions temporally.
        
        Args:
            interactions: DataFrame with interaction data
            time_column: Column containing timestamps
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        # Sort by timestamp
        interactions_sorted = interactions.sort_values(time_column)
        
        # Calculate split points
        n_total = len(interactions_sorted)
        n_test = int(n_total * self.test_ratio)
        n_val = int(n_total * self.validation_ratio)
        
        # Split temporally
        test_data = interactions_sorted.tail(n_test)
        remaining_data = interactions_sorted.head(n_total - n_test)
        
        val_data = remaining_data.tail(n_val)
        train_data = remaining_data.head(len(remaining_data) - n_val)
        
        logger.info(f"Temporal split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
    
    def split_by_user_time(
        self,
        interactions: pd.DataFrame,
        time_column: str = 'timestamp',
        n_test_interactions: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split interactions by user, keeping last N interactions for testing.
        
        Args:
            interactions: DataFrame with interaction data
            time_column: Column containing timestamps
            n_test_interactions: Number of interactions per user to use for testing
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        train_data = []
        val_data = []
        test_data = []
        
        for user_id in interactions['user_id'].unique():
            user_interactions = interactions[
                interactions['user_id'] == user_id
            ].sort_values(time_column)
            
            n_user_interactions = len(user_interactions)
            
            if n_user_interactions <= n_test_interactions:
                # Not enough interactions for splitting
                train_data.append(user_interactions)
                continue
            
            # Last N interactions for test
            test_interactions = user_interactions.tail(n_test_interactions)
            remaining_interactions = user_interactions.head(n_user_interactions - n_test_interactions)
            
            # Split remaining into train/val
            if len(remaining_interactions) > 1:
                train_user, val_user = train_test_split(
                    remaining_interactions,
                    test_size=0.2,
                    random_state=42
                )
                train_data.append(train_user)
                val_data.append(val_user)
            else:
                train_data.append(remaining_interactions)
            
            test_data.append(test_interactions)
        
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        logger.info(f"User-based temporal split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return train_df, val_df, test_df


class DataPreprocessor:
    """Preprocess data for recommendation systems."""
    
    def __init__(self, min_interactions: int = 5) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            min_interactions: Minimum number of interactions required for users/items
        """
        self.min_interactions = min_interactions
        
    def filter_sparse_users_items(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out users and items with too few interactions.
        
        Args:
            interactions: DataFrame with interaction data
            
        Returns:
            Filtered DataFrame
        """
        original_len = len(interactions)
        
        # Filter users
        user_counts = interactions['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        interactions_filtered = interactions[interactions['user_id'].isin(valid_users)]
        
        # Filter items
        item_counts = interactions_filtered['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.min_interactions].index
        interactions_filtered = interactions_filtered[interactions_filtered['item_id'].isin(valid_items)]
        
        logger.info(f"Filtered from {original_len} to {len(interactions_filtered)} interactions")
        logger.info(f"Users: {len(valid_users)}, Items: {len(valid_items)}")
        
        return interactions_filtered
    
    def add_temporal_features(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to interactions.
        
        Args:
            interactions: DataFrame with interaction data
            
        Returns:
            DataFrame with additional temporal features
        """
        interactions = interactions.copy()
        
        # Time-based features
        interactions['hour'] = interactions['timestamp'].dt.hour
        interactions['day_of_week'] = interactions['timestamp'].dt.dayofweek
        interactions['month'] = interactions['timestamp'].dt.month
        interactions['year'] = interactions['timestamp'].dt.year
        
        # Time since first interaction
        first_timestamp = interactions['timestamp'].min()
        interactions['days_since_start'] = (
            interactions['timestamp'] - first_timestamp
        ).dt.days
        
        # Recency score (higher for more recent interactions)
        max_timestamp = interactions['timestamp'].max()
        interactions['recency_score'] = (
            interactions['timestamp'] - first_timestamp
        ).dt.days / (max_timestamp - first_timestamp).days
        
        return interactions
    
    def create_user_item_matrix(
        self,
        interactions: pd.DataFrame,
        rating_column: str = 'rating'
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
        """
        Create user-item rating matrix.
        
        Args:
            interactions: DataFrame with interaction data
            rating_column: Column containing ratings
            
        Returns:
            Tuple of (matrix, user_mapping, item_mapping)
        """
        # Create mappings
        users = sorted(interactions['user_id'].unique())
        items = sorted(interactions['item_id'].unique())
        
        user_mapping = {user: idx for idx, user in enumerate(users)}
        item_mapping = {item: idx for idx, item in enumerate(items)}
        
        # Create matrix
        n_users = len(users)
        n_items = len(items)
        matrix = np.zeros((n_users, n_items))
        
        for _, row in interactions.iterrows():
            user_idx = user_mapping[row['user_id']]
            item_idx = item_mapping[row['item_id']]
            matrix[user_idx, item_idx] = row[rating_column]
        
        logger.info(f"Created user-item matrix: {n_users} users x {n_items} items")
        
        return matrix, user_mapping, item_mapping


def create_sample_data(data_dir: str = "data") -> None:
    """
    Create sample temporal recommendation data.
    
    Args:
        data_dir: Directory to save the data files
    """
    from src.models.temporal_recommender import TemporalDataGenerator
    
    logger.info("Creating sample temporal recommendation data...")
    
    # Generate data
    generator = TemporalDataGenerator(
        n_users=1000,
        n_items=500,
        n_interactions=10000,
        seed=42
    )
    
    interactions = generator.generate_interactions()
    items = generator.generate_items()
    
    # Save data
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    interactions.to_csv(data_path / "interactions.csv", index=False)
    items.to_csv(data_path / "items.csv", index=False)
    
    logger.info(f"Sample data saved to {data_path}")


if __name__ == "__main__":
    # Create sample data if running directly
    create_sample_data()
