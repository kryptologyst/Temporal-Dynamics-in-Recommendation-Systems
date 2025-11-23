"""
Evaluation metrics and model comparison for temporal recommendation systems.

This module provides comprehensive evaluation metrics including temporal-aware
metrics for recommendation systems.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Calculate various recommendation metrics."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]) -> None:
        """
        Initialize metrics calculator.
        
        Args:
            k_values: List of k values for top-k metrics
        """
        self.k_values = k_values
        
    def precision_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        
        return relevant_in_top_k / k
    
    def recall_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        
        return relevant_in_top_k / len(relevant_items)
    
    def ndcg_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int
    ) -> float:
        """
        Calculate NDCG@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int
    ) -> float:
        """
        Calculate Hit Rate@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant items
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        return 1.0 if len(set(top_k_recs) & set(relevant_items)) > 0 else 0.0
    
    def map_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        k: int
    ) -> float:
        """
        Calculate MAP@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items)


class TemporalMetrics:
    """Calculate temporal-aware recommendation metrics."""
    
    def __init__(self, time_decay_factor: float = 0.1) -> None:
        """
        Initialize temporal metrics calculator.
        
        Args:
            time_decay_factor: Exponential decay factor for time weighting
        """
        self.time_decay_factor = time_decay_factor
    
    def temporal_precision_at_k(
        self,
        recommendations: List[str],
        relevant_items: List[str],
        timestamps: List[pd.Timestamp],
        k: int,
        current_time: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Calculate time-weighted Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            timestamps: List of timestamps for relevant items
            k: Number of top recommendations to consider
            current_time: Current time for decay calculation
            
        Returns:
            Time-weighted Precision@K score
        """
        if k == 0:
            return 0.0
        
        if current_time is None:
            current_time = pd.Timestamp.now()
        
        top_k_recs = recommendations[:k]
        
        # Calculate time-weighted precision
        weighted_precision = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                # Find timestamp for this item
                item_idx = relevant_items.index(item)
                item_timestamp = timestamps[item_idx]
                
                # Calculate time decay
                days_diff = (current_time - item_timestamp).days
                time_weight = np.exp(-self.time_decay_factor * days_diff / 365)
                
                weighted_precision += time_weight
        
        return weighted_precision / k
    
    def recency_bias_score(
        self,
        recommendations: List[str],
        item_timestamps: Dict[str, pd.Timestamp],
        k: int,
        current_time: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Calculate recency bias score (higher = more recent items).
        
        Args:
            recommendations: List of recommended item IDs
            item_timestamps: Dictionary mapping item IDs to timestamps
            k: Number of top recommendations to consider
            current_time: Current time for recency calculation
            
        Returns:
            Recency bias score
        """
        if k == 0:
            return 0.0
        
        if current_time is None:
            current_time = pd.Timestamp.now()
        
        top_k_recs = recommendations[:k]
        recency_scores = []
        
        for item in top_k_recs:
            if item in item_timestamps:
                days_diff = (current_time - item_timestamps[item]).days
                recency_score = np.exp(-self.time_decay_factor * days_diff / 365)
                recency_scores.append(recency_score)
        
        return np.mean(recency_scores) if recency_scores else 0.0


class ModelEvaluator:
    """Evaluate recommendation models comprehensively."""
    
    def __init__(
        self,
        k_values: List[int] = [5, 10, 20],
        time_decay_factor: float = 0.1
    ) -> None:
        """
        Initialize the model evaluator.
        
        Args:
            k_values: List of k values for evaluation
            time_decay_factor: Time decay factor for temporal metrics
        """
        self.k_values = k_values
        self.metrics_calculator = RecommendationMetrics(k_values)
        self.temporal_calculator = TemporalMetrics(time_decay_factor)
    
    def evaluate_model(
        self,
        model,
        test_data: pd.DataFrame,
        items_data: Optional[pd.DataFrame] = None,
        n_recommendations: int = 20
    ) -> Dict[str, float]:
        """
        Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model
            test_data: Test interaction data
            items_data: Item metadata (optional)
            n_recommendations: Number of recommendations to generate
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model: {model.name}")
        
        results = {}
        
        # Get unique users from test data
        test_users = test_data['user_id'].unique()
        
        # Calculate metrics for each k value
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            hit_rate_scores = []
            map_scores = []
            temporal_precision_scores = []
            recency_bias_scores = []
            
            for user_id in test_users:
                # Get user's test interactions
                user_test_items = test_data[
                    test_data['user_id'] == user_id
                ]['item_id'].tolist()
                
                if not user_test_items:
                    continue
                
                # Generate recommendations
                try:
                    recommendations = model.recommend(
                        user_id, 
                        n_recommendations=n_recommendations
                    )
                    rec_items = [item for item, _ in recommendations]
                except Exception as e:
                    logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                    continue
                
                # Calculate standard metrics
                precision_scores.append(
                    self.metrics_calculator.precision_at_k(rec_items, user_test_items, k)
                )
                recall_scores.append(
                    self.metrics_calculator.recall_at_k(rec_items, user_test_items, k)
                )
                ndcg_scores.append(
                    self.metrics_calculator.ndcg_at_k(rec_items, user_test_items, k)
                )
                hit_rate_scores.append(
                    self.metrics_calculator.hit_rate_at_k(rec_items, user_test_items, k)
                )
                map_scores.append(
                    self.metrics_calculator.map_at_k(rec_items, user_test_items, k)
                )
                
                # Calculate temporal metrics
                if items_data is not None:
                    user_test_timestamps = test_data[
                        test_data['user_id'] == user_id
                    ]['timestamp'].tolist()
                    
                    temporal_precision_scores.append(
                        self.temporal_calculator.temporal_precision_at_k(
                            rec_items, user_test_items, user_test_timestamps, k
                        )
                    )
                    
                    # Recency bias
                    item_timestamps = {}
                    for item in rec_items:
                        if item in items_data['item_id'].values:
                            # Use a default timestamp for items without timestamp
                            item_timestamps[item] = pd.Timestamp.now()
                    
                    recency_bias_scores.append(
                        self.temporal_calculator.recency_bias_score(
                            rec_items, item_timestamps, k
                        )
                    )
            
            # Average metrics
            results[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
            results[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
            results[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
            results[f'hit_rate@{k}'] = np.mean(hit_rate_scores) if hit_rate_scores else 0.0
            results[f'map@{k}'] = np.mean(map_scores) if map_scores else 0.0
            
            if temporal_precision_scores:
                results[f'temporal_precision@{k}'] = np.mean(temporal_precision_scores)
            if recency_bias_scores:
                results[f'recency_bias@{k}'] = np.mean(recency_bias_scores)
        
        logger.info(f"Evaluation completed for {model.name}")
        return results
    
    def compare_models(
        self,
        models: List,
        test_data: pd.DataFrame,
        items_data: Optional[pd.DataFrame] = None,
        n_recommendations: int = 20
    ) -> pd.DataFrame:
        """
        Compare multiple models and return results as DataFrame.
        
        Args:
            models: List of trained models
            test_data: Test interaction data
            items_data: Item metadata (optional)
            n_recommendations: Number of recommendations to generate
            
        Returns:
            DataFrame with model comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        all_results = []
        
        for model in models:
            results = self.evaluate_model(model, test_data, items_data, n_recommendations)
            results['model'] = model.name
            all_results.append(results)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        
        # Sort by primary metric (NDCG@10)
        if 'ndcg@10' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('ndcg@10', ascending=False)
        
        logger.info("Model comparison completed")
        return comparison_df


def calculate_coverage_metrics(
    recommendations: List[List[str]],
    all_items: List[str]
) -> Dict[str, float]:
    """
    Calculate coverage metrics for recommendations.
    
    Args:
        recommendations: List of recommendation lists for each user
        all_items: List of all available items
        
    Returns:
        Dictionary of coverage metrics
    """
    # Catalog coverage
    recommended_items = set()
    for user_recs in recommendations:
        recommended_items.update(user_recs)
    
    catalog_coverage = len(recommended_items) / len(all_items)
    
    # User coverage
    users_with_recs = len([recs for recs in recommendations if recs])
    user_coverage = users_with_recs / len(recommendations) if recommendations else 0.0
    
    return {
        'catalog_coverage': catalog_coverage,
        'user_coverage': user_coverage
    }


def calculate_diversity_metrics(
    recommendations: List[List[str]],
    item_features: Optional[Dict[str, List[str]]] = None
) -> Dict[str, float]:
    """
    Calculate diversity metrics for recommendations.
    
    Args:
        recommendations: List of recommendation lists for each user
        item_features: Dictionary mapping items to feature lists (optional)
        
    Returns:
        Dictionary of diversity metrics
    """
    # Intra-list diversity (average pairwise dissimilarity within lists)
    intra_list_diversities = []
    
    for user_recs in recommendations:
        if len(user_recs) < 2:
            continue
        
        # Simple Jaccard dissimilarity
        dissimilarities = []
        for i in range(len(user_recs)):
            for j in range(i + 1, len(user_recs)):
                # For simplicity, use item ID dissimilarity
                dissimilarity = 1.0 if user_recs[i] != user_recs[j] else 0.0
                dissimilarities.append(dissimilarity)
        
        if dissimilarities:
            intra_list_diversities.append(np.mean(dissimilarities))
    
    intra_list_diversity = np.mean(intra_list_diversities) if intra_list_diversities else 0.0
    
    return {
        'intra_list_diversity': intra_list_diversity
    }
