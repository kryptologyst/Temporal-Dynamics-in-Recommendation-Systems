"""
Temporal Dynamics in Recommendation Systems

This module implements time-aware recommendation systems that capture the temporal
evolution of user preferences and item relevance over time.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalDataGenerator:
    """Generate realistic temporal recommendation data with seasonal patterns."""
    
    def __init__(
        self,
        n_users: int = 1000,
        n_items: int = 500,
        n_interactions: int = 10000,
        seed: int = 42
    ) -> None:
        """
        Initialize the temporal data generator.
        
        Args:
            n_users: Number of users to generate
            n_items: Number of items to generate
            n_interactions: Number of interactions to generate
            seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_interactions = n_interactions
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_interactions(
        self,
        start_date: datetime = datetime(2023, 1, 1),
        end_date: datetime = datetime(2023, 12, 31)
    ) -> pd.DataFrame:
        """
        Generate temporal interaction data with realistic patterns.
        
        Args:
            start_date: Start date for interactions
            end_date: End date for interactions
            
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        logger.info("Generating temporal interaction data...")
        
        # Generate user-item pairs with temporal patterns
        interactions = []
        
        # Create time range
        time_range = pd.date_range(start_date, end_date, freq='D')
        
        for _ in range(self.n_interactions):
            # Random user and item
            user_id = f"user_{np.random.randint(1, self.n_users + 1)}"
            item_id = f"item_{np.random.randint(1, self.n_items + 1)}"
            
            # Generate timestamp with temporal bias (more recent interactions)
            days_from_start = np.random.exponential(scale=30)  # Exponential decay
            timestamp = start_date + timedelta(days=int(days_from_start))
            
            # Generate rating with temporal patterns
            base_rating = np.random.normal(3.5, 1.0)
            
            # Seasonal effect (higher ratings in certain months)
            month = timestamp.month
            seasonal_bonus = 0.5 * np.sin(2 * np.pi * month / 12)
            
            # Recency effect (higher ratings for more recent items)
            recency_factor = 0.3 * (timestamp - start_date).days / 365
            
            rating = np.clip(base_rating + seasonal_bonus + recency_factor, 1, 5)
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': round(rating, 1),
                'timestamp': timestamp
            })
        
        df = pd.DataFrame(interactions)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} interactions for {df['user_id'].nunique()} users and {df['item_id'].nunique()} items")
        return df
    
    def generate_items(self) -> pd.DataFrame:
        """
        Generate item metadata.
        
        Returns:
            DataFrame with item information
        """
        logger.info("Generating item metadata...")
        
        items = []
        categories = ['movie', 'music', 'book', 'game', 'product']
        
        for i in range(1, self.n_items + 1):
            items.append({
                'item_id': f"item_{i}",
                'title': f"Item {i}",
                'category': np.random.choice(categories),
                'price': np.random.uniform(10, 200),
                'popularity': np.random.exponential(scale=1.0)
            })
        
        return pd.DataFrame(items)


class TemporalRecommender:
    """Base class for temporal recommendation systems."""
    
    def __init__(self, name: str) -> None:
        """Initialize the recommender."""
        self.name = name
        self.is_fitted = False
        
    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit the model to interaction data."""
        raise NotImplementedError
        
    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_interacted: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User to recommend for
            n_recommendations: Number of recommendations to return
            exclude_interacted: Whether to exclude items the user has already interacted with
            
        Returns:
            List of (item_id, score) tuples
        """
        raise NotImplementedError


class TimeWeightedCollaborativeFiltering(TemporalRecommender):
    """Collaborative filtering with time-weighted interactions."""
    
    def __init__(
        self,
        decay_factor: float = 0.1,
        min_interactions: int = 5,
        n_factors: int = 50
    ) -> None:
        """
        Initialize time-weighted collaborative filtering.
        
        Args:
            decay_factor: Exponential decay factor for time weighting
            min_interactions: Minimum interactions required for a user/item
            n_factors: Number of latent factors for matrix factorization
        """
        super().__init__("Time-Weighted Collaborative Filtering")
        self.decay_factor = decay_factor
        self.min_interactions = min_interactions
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = None
        self.item_mapping = None
        
    def _apply_time_decay(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Apply exponential time decay to interactions."""
        interactions = interactions.copy()
        
        # Calculate time decay
        max_timestamp = interactions['timestamp'].max()
        interactions['time_decay'] = np.exp(
            -self.decay_factor * 
            (max_timestamp - interactions['timestamp']).dt.days / 365
        )
        
        # Apply decay to ratings
        interactions['weighted_rating'] = interactions['rating'] * interactions['time_decay']
        
        return interactions
    
    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit the time-weighted collaborative filtering model."""
        logger.info(f"Fitting {self.name}...")
        
        # Apply time decay
        interactions_weighted = self._apply_time_decay(interactions)
        
        # Filter users and items with minimum interactions
        user_counts = interactions_weighted['user_id'].value_counts()
        item_counts = interactions_weighted['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= self.min_interactions].index
        valid_items = item_counts[item_counts >= self.min_interactions].index
        
        interactions_filtered = interactions_weighted[
            interactions_weighted['user_id'].isin(valid_users) &
            interactions_weighted['item_id'].isin(valid_items)
        ]
        
        # Create user and item mappings
        self.user_mapping = {user: idx for idx, user in enumerate(sorted(valid_users))}
        self.item_mapping = {item: idx for idx, item in enumerate(sorted(valid_items))}
        
        # Create rating matrix
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        
        rating_matrix = np.zeros((n_users, n_items))
        
        for _, row in interactions_filtered.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['item_id']]
            rating_matrix[user_idx, item_idx] = row['weighted_rating']
        
        # Matrix factorization using NMF
        model = NMF(n_components=self.n_factors, random_state=42)
        self.user_factors = model.fit_transform(rating_matrix)
        self.item_factors = model.components_.T
        
        self.is_fitted = True
        logger.info(f"Model fitted with {n_users} users and {n_items} items")
    
    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_interacted: bool = True
    ) -> List[Tuple[str, float]]:
        """Generate recommendations for a user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_mapping:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_mapping[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(self.item_factors, user_vector)
        
        # Create item-score pairs
        item_scores = [
            (item_id, score) 
            for item_id, score in zip(self.item_mapping.keys(), scores)
        ]
        
        # Sort by score
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top recommendations
        return item_scores[:n_recommendations]


class TemporalMatrixFactorization(TemporalRecommender):
    """Matrix factorization with temporal regularization."""
    
    def __init__(
        self,
        n_factors: int = 50,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        n_epochs: int = 100
    ) -> None:
        """
        Initialize temporal matrix factorization.
        
        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            regularization: Regularization parameter
            n_epochs: Number of training epochs
        """
        super().__init__("Temporal Matrix Factorization")
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.user_mapping = None
        self.item_mapping = None
        
    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit the temporal matrix factorization model."""
        logger.info(f"Fitting {self.name}...")
        
        # Create mappings
        users = sorted(interactions['user_id'].unique())
        items = sorted(interactions['item_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(users)}
        self.item_mapping = {item: idx for idx, item in enumerate(items)}
        
        n_users = len(users)
        n_items = len(items)
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = interactions['rating'].mean()
        
        # Convert timestamps to numerical features
        interactions = interactions.copy()
        interactions['time_feature'] = (
            interactions['timestamp'] - interactions['timestamp'].min()
        ).dt.days / 365.0
        
        # Training loop
        for epoch in range(self.n_epochs):
            total_error = 0
            
            for _, row in interactions.iterrows():
                user_idx = self.user_mapping[row['user_id']]
                item_idx = self.item_mapping[row['item_id']]
                rating = row['rating']
                time_feature = row['time_feature']
                
                # Predict rating
                prediction = (
                    self.global_bias +
                    self.user_bias[user_idx] +
                    self.item_bias[item_idx] +
                    np.dot(self.user_factors[user_idx], self.item_factors[item_idx]) +
                    time_feature * 0.1  # Simple temporal bias
                )
                
                error = rating - prediction
                total_error += error ** 2
                
                # Update factors
                user_factor = self.user_factors[user_idx]
                item_factor = self.item_factors[item_idx]
                
                self.user_factors[user_idx] += self.learning_rate * (
                    error * item_factor - self.regularization * user_factor
                )
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factor - self.regularization * item_factor
                )
                
                self.user_bias[user_idx] += self.learning_rate * (
                    error - self.regularization * self.user_bias[user_idx]
                )
                self.item_bias[item_idx] += self.learning_rate * (
                    error - self.regularization * self.item_bias[item_idx]
                )
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, RMSE: {np.sqrt(total_error / len(interactions)):.4f}")
        
        self.is_fitted = True
        logger.info("Model training completed")
    
    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_interacted: bool = True
    ) -> List[Tuple[str, float]]:
        """Generate recommendations for a user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_mapping:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # Calculate scores for all items
        scores = (
            self.global_bias +
            self.user_bias[user_idx] +
            self.item_bias +
            np.dot(self.item_factors, self.user_factors[user_idx])
        )
        
        # Create item-score pairs
        item_scores = [
            (item_id, score) 
            for item_id, score in zip(self.item_mapping.keys(), scores)
        ]
        
        # Sort by score
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top recommendations
        return item_scores[:n_recommendations]


def main() -> None:
    """Main function to demonstrate temporal recommendation systems."""
    logger.info("Starting Temporal Dynamics Recommendation System Demo")
    
    # Generate data
    generator = TemporalDataGenerator(n_users=100, n_items=50, n_interactions=2000)
    interactions = generator.generate_interactions()
    items = generator.generate_items()
    
    # Save data
    interactions.to_csv('data/interactions.csv', index=False)
    items.to_csv('data/items.csv', index=False)
    
    # Split data temporally
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    split_date = interactions['timestamp'].quantile(0.8)
    
    train_data = interactions[interactions['timestamp'] < split_date]
    test_data = interactions[interactions['timestamp'] >= split_date]
    
    logger.info(f"Train data: {len(train_data)} interactions")
    logger.info(f"Test data: {len(test_data)} interactions")
    
    # Initialize models
    models = [
        TimeWeightedCollaborativeFiltering(decay_factor=0.1),
        TemporalMatrixFactorization(n_factors=20, n_epochs=50)
    ]
    
    # Train and evaluate models
    results = {}
    
    for model in models:
        logger.info(f"Training {model.name}...")
        model.fit(train_data)
        
        # Generate recommendations for a sample user
        sample_user = train_data['user_id'].iloc[0]
        recommendations = model.recommend(sample_user, n_recommendations=5)
        
        logger.info(f"Recommendations for {sample_user}:")
        for item_id, score in recommendations:
            logger.info(f"  {item_id}: {score:.3f}")
        
        results[model.name] = recommendations
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
