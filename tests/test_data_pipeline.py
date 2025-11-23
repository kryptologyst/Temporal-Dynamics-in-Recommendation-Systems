"""
Unit tests for data pipeline functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.data.data_pipeline import (
    TemporalDataLoader,
    TemporalSplitter,
    DataPreprocessor,
    create_sample_data
)


class TestTemporalDataLoader:
    """Test temporal data loader."""
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = TemporalDataLoader("test_data")
        assert loader.data_dir == Path("test_data")
        assert loader.data_dir.exists()
    
    def test_load_interactions_missing_file(self):
        """Test loading interactions when file doesn't exist."""
        loader = TemporalDataLoader("nonexistent")
        
        with pytest.raises(FileNotFoundError):
            loader.load_interactions()
    
    def test_load_items_missing_file(self):
        """Test loading items when file doesn't exist."""
        loader = TemporalDataLoader("nonexistent")
        
        result = loader.load_items()
        assert result.empty
    
    def test_load_users_missing_file(self):
        """Test loading users when file doesn't exist."""
        loader = TemporalDataLoader("nonexistent")
        
        result = loader.load_users()
        assert result.empty


class TestTemporalSplitter:
    """Test temporal data splitting."""
    
    def test_splitter_initialization(self):
        """Test splitter initialization."""
        splitter = TemporalSplitter(test_ratio=0.2, validation_ratio=0.1)
        assert splitter.test_ratio == 0.2
        assert splitter.validation_ratio == 0.1
    
    def test_split_by_time(self):
        """Test temporal splitting."""
        splitter = TemporalSplitter(test_ratio=0.2, validation_ratio=0.1)
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1'] * 10,
            'item_id': [f'item{i}' for i in range(10)],
            'rating': [5.0] * 10,
            'timestamp': [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        })
        
        train_data, val_data, test_data = splitter.split_by_time(interactions)
        
        # Check that splits are temporal
        assert len(train_data) + len(val_data) + len(test_data) == len(interactions)
        assert len(test_data) == 2  # 20% of 10
        assert len(val_data) == 1   # 10% of 10
        
        # Check temporal ordering
        if not train_data.empty:
            assert train_data['timestamp'].max() <= val_data['timestamp'].min()
        if not val_data.empty:
            assert val_data['timestamp'].max() <= test_data['timestamp'].min()
    
    def test_split_by_user_time(self):
        """Test user-based temporal splitting."""
        splitter = TemporalSplitter()
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1'] * 10 + ['user2'] * 8,
            'item_id': [f'item{i}' for i in range(18)],
            'rating': [5.0] * 18,
            'timestamp': [datetime(2023, 1, 1) + timedelta(days=i) for i in range(18)]
        })
        
        train_data, val_data, test_data = splitter.split_by_user_time(
            interactions, n_test_interactions=2
        )
        
        # Check that each user has at most 2 test interactions
        for user_id in interactions['user_id'].unique():
            user_test = test_data[test_data['user_id'] == user_id]
            assert len(user_test) <= 2


class TestDataPreprocessor:
    """Test data preprocessing."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(min_interactions=5)
        assert preprocessor.min_interactions == 5
    
    def test_filter_sparse_users_items(self):
        """Test filtering sparse users and items."""
        preprocessor = DataPreprocessor(min_interactions=2)
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1', 'user2', 'user3'],
            'item_id': ['item1', 'item2', 'item3', 'item1', 'item1'],
            'rating': [5.0, 4.0, 3.0, 2.0, 1.0],
            'timestamp': [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
        })
        
        filtered_interactions = preprocessor.filter_sparse_users_items(interactions)
        
        # user1 has 3 interactions, user2 and user3 have 1 each
        # Only user1 should remain
        assert len(filtered_interactions) == 3
        assert filtered_interactions['user_id'].nunique() == 1
        assert filtered_interactions['user_id'].iloc[0] == 'user1'
    
    def test_add_temporal_features(self):
        """Test adding temporal features."""
        preprocessor = DataPreprocessor()
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user1'],
            'item_id': ['item1', 'item2'],
            'rating': [5.0, 4.0],
            'timestamp': [
                datetime(2023, 1, 1, 10, 30),  # Sunday, 10:30 AM
                datetime(2023, 6, 15, 14, 45)  # Thursday, 2:45 PM
            ]
        })
        
        enhanced_interactions = preprocessor.add_temporal_features(interactions)
        
        # Check that temporal features were added
        assert 'hour' in enhanced_interactions.columns
        assert 'day_of_week' in enhanced_interactions.columns
        assert 'month' in enhanced_interactions.columns
        assert 'year' in enhanced_interactions.columns
        assert 'days_since_start' in enhanced_interactions.columns
        assert 'recency_score' in enhanced_interactions.columns
        
        # Check specific values
        assert enhanced_interactions['hour'].iloc[0] == 10
        assert enhanced_interactions['day_of_week'].iloc[0] == 6  # Sunday
        assert enhanced_interactions['month'].iloc[0] == 1
        assert enhanced_interactions['year'].iloc[0] == 2023
    
    def test_create_user_item_matrix(self):
        """Test creating user-item matrix."""
        preprocessor = DataPreprocessor()
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2'],
            'item_id': ['item1', 'item2', 'item1'],
            'rating': [5.0, 4.0, 3.0],
            'timestamp': [datetime(2023, 1, 1) + timedelta(days=i) for i in range(3)]
        })
        
        matrix, user_mapping, item_mapping = preprocessor.create_user_item_matrix(interactions)
        
        # Check matrix dimensions
        assert matrix.shape == (2, 2)  # 2 users, 2 items
        
        # Check mappings
        assert len(user_mapping) == 2
        assert len(item_mapping) == 2
        assert 'user1' in user_mapping
        assert 'user2' in user_mapping
        assert 'item1' in item_mapping
        assert 'item2' in item_mapping
        
        # Check matrix values
        user1_idx = user_mapping['user1']
        user2_idx = user_mapping['user2']
        item1_idx = item_mapping['item1']
        item2_idx = item_mapping['item2']
        
        assert matrix[user1_idx, item1_idx] == 5.0
        assert matrix[user1_idx, item2_idx] == 4.0
        assert matrix[user2_idx, item1_idx] == 3.0
        assert matrix[user2_idx, item2_idx] == 0.0


class TestSampleDataCreation:
    """Test sample data creation."""
    
    def test_create_sample_data(self):
        """Test creating sample data."""
        test_dir = "test_sample_data"
        
        try:
            create_sample_data(test_dir)
            
            # Check that files were created
            interactions_file = Path(test_dir) / "interactions.csv"
            items_file = Path(test_dir) / "items.csv"
            
            assert interactions_file.exists()
            assert items_file.exists()
            
            # Check file contents
            interactions = pd.read_csv(interactions_file)
            items = pd.read_csv(items_file)
            
            assert len(interactions) > 0
            assert len(items) > 0
            assert 'user_id' in interactions.columns
            assert 'item_id' in interactions.columns
            assert 'rating' in interactions.columns
            assert 'timestamp' in interactions.columns
            
        finally:
            # Clean up
            import shutil
            if Path(test_dir).exists():
                shutil.rmtree(test_dir)


if __name__ == "__main__":
    pytest.main([__file__])
