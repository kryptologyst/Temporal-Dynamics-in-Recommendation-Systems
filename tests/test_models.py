"""
Unit tests for temporal recommendation models.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.temporal_recommender import (
    TemporalDataGenerator,
    TimeWeightedCollaborativeFiltering,
    TemporalMatrixFactorization
)


class TestTemporalDataGenerator:
    """Test temporal data generation."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = TemporalDataGenerator(n_users=100, n_items=50, n_interactions=1000)
        assert generator.n_users == 100
        assert generator.n_items == 50
        assert generator.n_interactions == 1000
        assert generator.seed == 42
    
    def test_generate_interactions(self):
        """Test interaction data generation."""
        generator = TemporalDataGenerator(n_users=10, n_items=5, n_interactions=50)
        interactions = generator.generate_interactions()
        
        assert len(interactions) == 50
        assert 'user_id' in interactions.columns
        assert 'item_id' in interactions.columns
        assert 'rating' in interactions.columns
        assert 'timestamp' in interactions.columns
        
        # Check rating range
        assert interactions['rating'].min() >= 1.0
        assert interactions['rating'].max() <= 5.0
        
        # Check timestamp range
        assert isinstance(interactions['timestamp'].iloc[0], pd.Timestamp)
    
    def test_generate_items(self):
        """Test item metadata generation."""
        generator = TemporalDataGenerator(n_items=10)
        items = generator.generate_items()
        
        assert len(items) == 10
        assert 'item_id' in items.columns
        assert 'title' in items.columns
        assert 'category' in items.columns
        assert 'price' in items.columns
        assert 'popularity' in items.columns


class TestTimeWeightedCollaborativeFiltering:
    """Test time-weighted collaborative filtering."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = TimeWeightedCollaborativeFiltering(
            decay_factor=0.1,
            min_interactions=5,
            n_factors=20
        )
        assert model.decay_factor == 0.1
        assert model.min_interactions == 5
        assert model.n_factors == 20
        assert not model.is_fitted
    
    def test_time_decay_application(self):
        """Test time decay application."""
        model = TimeWeightedCollaborativeFiltering(decay_factor=0.1)
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'item_id': ['item1', 'item2', 'item3'],
            'rating': [5.0, 4.0, 3.0],
            'timestamp': [
                datetime(2023, 1, 1),
                datetime(2023, 6, 1),
                datetime(2023, 12, 1)
            ]
        })
        
        weighted_interactions = model._apply_time_decay(interactions)
        
        assert 'time_decay' in weighted_interactions.columns
        assert 'weighted_rating' in weighted_interactions.columns
        
        # More recent interactions should have higher decay values
        assert weighted_interactions['time_decay'].iloc[2] > weighted_interactions['time_decay'].iloc[0]
    
    def test_model_fitting(self):
        """Test model fitting."""
        model = TimeWeightedCollaborativeFiltering(min_interactions=1, n_factors=5)
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'item_id': ['item1', 'item2', 'item1', 'item3'],
            'rating': [5.0, 4.0, 3.0, 2.0],
            'timestamp': [
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                datetime(2023, 3, 1),
                datetime(2023, 4, 1)
            ]
        })
        
        model.fit(interactions)
        
        assert model.is_fitted
        assert model.user_factors is not None
        assert model.item_factors is not None
        assert model.user_mapping is not None
        assert model.item_mapping is not None
    
    def test_recommendations(self):
        """Test recommendation generation."""
        model = TimeWeightedCollaborativeFiltering(min_interactions=1, n_factors=5)
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'item_id': ['item1', 'item2', 'item1', 'item3'],
            'rating': [5.0, 4.0, 3.0, 2.0],
            'timestamp': [
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                datetime(2023, 3, 1),
                datetime(2023, 4, 1)
            ]
        })
        
        model.fit(interactions)
        
        recommendations = model.recommend('user1', n_recommendations=3)
        
        assert len(recommendations) <= 3
        assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)
        assert all(isinstance(score, (int, float)) for _, score in recommendations)


class TestTemporalMatrixFactorization:
    """Test temporal matrix factorization."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = TemporalMatrixFactorization(
            n_factors=20,
            learning_rate=0.01,
            regularization=0.01,
            n_epochs=10
        )
        assert model.n_factors == 20
        assert model.learning_rate == 0.01
        assert model.regularization == 0.01
        assert model.n_epochs == 10
        assert not model.is_fitted
    
    def test_model_fitting(self):
        """Test model fitting."""
        model = TemporalMatrixFactorization(n_factors=5, n_epochs=5)
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'item_id': ['item1', 'item2', 'item1', 'item3'],
            'rating': [5.0, 4.0, 3.0, 2.0],
            'timestamp': [
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                datetime(2023, 3, 1),
                datetime(2023, 4, 1)
            ]
        })
        
        model.fit(interactions)
        
        assert model.is_fitted
        assert model.user_factors is not None
        assert model.item_factors is not None
        assert model.user_bias is not None
        assert model.item_bias is not None
        assert model.global_bias is not None
    
    def test_recommendations(self):
        """Test recommendation generation."""
        model = TemporalMatrixFactorization(n_factors=5, n_epochs=5)
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'item_id': ['item1', 'item2', 'item1', 'item3'],
            'rating': [5.0, 4.0, 3.0, 2.0],
            'timestamp': [
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
                datetime(2023, 3, 1),
                datetime(2023, 4, 1)
            ]
        })
        
        model.fit(interactions)
        
        recommendations = model.recommend('user1', n_recommendations=3)
        
        assert len(recommendations) <= 3
        assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)
        assert all(isinstance(score, (int, float)) for _, score in recommendations)


class TestModelIntegration:
    """Test model integration and comparison."""
    
    def test_model_comparison(self):
        """Test comparing different models."""
        # Create sample data
        generator = TemporalDataGenerator(n_users=20, n_items=10, n_interactions=100)
        interactions = generator.generate_interactions()
        
        # Initialize models
        models = {
            "Time-Weighted CF": TimeWeightedCollaborativeFiltering(min_interactions=1, n_factors=5),
            "Temporal MF": TemporalMatrixFactorization(n_factors=5, n_epochs=5)
        }
        
        # Train models
        for model in models.values():
            model.fit(interactions)
        
        # Test recommendations from both models
        user_id = interactions['user_id'].iloc[0]
        
        for name, model in models.items():
            recommendations = model.recommend(user_id, n_recommendations=5)
            assert len(recommendations) <= 5
            assert all(isinstance(rec, tuple) for rec in recommendations)
    
    def test_error_handling(self):
        """Test error handling in models."""
        model = TimeWeightedCollaborativeFiltering()
        
        # Test recommendation without fitting
        with pytest.raises(ValueError):
            model.recommend('user1')
        
        # Test recommendation for unknown user
        interactions = pd.DataFrame({
            'user_id': ['user1'],
            'item_id': ['item1'],
            'rating': [5.0],
            'timestamp': [datetime(2023, 1, 1)]
        })
        
        model.fit(interactions)
        recommendations = model.recommend('unknown_user')
        assert recommendations == []


if __name__ == "__main__":
    pytest.main([__file__])
