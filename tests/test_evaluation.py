"""
Unit tests for evaluation metrics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.evaluation.metrics import (
    RecommendationMetrics,
    TemporalMetrics,
    ModelEvaluator,
    calculate_coverage_metrics,
    calculate_diversity_metrics
)


class TestRecommendationMetrics:
    """Test recommendation metrics calculation."""
    
    def test_metrics_initialization(self):
        """Test metrics calculator initialization."""
        metrics = RecommendationMetrics(k_values=[5, 10, 20])
        assert metrics.k_values == [5, 10, 20]
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        precision_5 = metrics.precision_at_k(recommendations, relevant_items, 5)
        precision_3 = metrics.precision_at_k(recommendations, relevant_items, 3)
        
        # Precision@5: 2 relevant out of 5 = 0.4
        assert precision_5 == 0.4
        # Precision@3: 2 relevant out of 3 = 0.667
        assert abs(precision_3 - 0.667) < 0.01
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        recall_5 = metrics.recall_at_k(recommendations, relevant_items, 5)
        recall_3 = metrics.recall_at_k(recommendations, relevant_items, 3)
        
        # Recall@5: 2 relevant found out of 3 total = 0.667
        assert abs(recall_5 - 0.667) < 0.01
        # Recall@3: 2 relevant found out of 3 total = 0.667
        assert abs(recall_3 - 0.667) < 0.01
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        ndcg_5 = metrics.ndcg_at_k(recommendations, relevant_items, 5)
        
        # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.631 + 0.5 = 2.131
        # NDCG = 1.5 / 2.131 ≈ 0.704
        assert ndcg_5 > 0.7
        assert ndcg_5 < 0.8
    
    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        hit_rate_5 = metrics.hit_rate_at_k(recommendations, relevant_items, 5)
        hit_rate_2 = metrics.hit_rate_at_k(recommendations, relevant_items, 2)
        
        # Hit Rate@5: 1 (has relevant items)
        assert hit_rate_5 == 1.0
        # Hit Rate@2: 1 (has relevant items)
        assert hit_rate_2 == 1.0
    
    def test_map_at_k(self):
        """Test MAP@K calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        map_5 = metrics.map_at_k(recommendations, relevant_items, 5)
        
        # AP = (1/1 + 2/3) / 3 = (1 + 0.667) / 3 = 0.556
        assert abs(map_5 - 0.556) < 0.01
    
    def test_edge_cases(self):
        """Test edge cases for metrics."""
        metrics = RecommendationMetrics()
        
        # Empty recommendations
        precision = metrics.precision_at_k([], ['item1'], 5)
        assert precision == 0.0
        
        # Empty relevant items
        recall = metrics.recall_at_k(['item1'], [], 5)
        assert recall == 0.0
        
        # k = 0
        precision = metrics.precision_at_k(['item1'], ['item1'], 0)
        assert precision == 0.0


class TestTemporalMetrics:
    """Test temporal metrics calculation."""
    
    def test_temporal_metrics_initialization(self):
        """Test temporal metrics initialization."""
        temporal_metrics = TemporalMetrics(time_decay_factor=0.1)
        assert temporal_metrics.time_decay_factor == 0.1
    
    def test_temporal_precision_at_k(self):
        """Test temporal precision calculation."""
        temporal_metrics = TemporalMetrics(time_decay_factor=0.1)
        
        recommendations = ['item1', 'item2', 'item3']
        relevant_items = ['item1', 'item2']
        timestamps = [
            datetime(2023, 1, 1),  # Older
            datetime(2023, 6, 1)   # More recent
        ]
        current_time = datetime(2023, 12, 1)
        
        temporal_precision = temporal_metrics.temporal_precision_at_k(
            recommendations, relevant_items, timestamps, 3, current_time
        )
        
        # Should be positive and consider recency
        assert temporal_precision > 0
        assert temporal_precision <= 1.0
    
    def test_recency_bias_score(self):
        """Test recency bias score calculation."""
        temporal_metrics = TemporalMetrics(time_decay_factor=0.1)
        
        recommendations = ['item1', 'item2', 'item3']
        item_timestamps = {
            'item1': datetime(2023, 1, 1),  # Older
            'item2': datetime(2023, 6, 1),  # More recent
            'item3': datetime(2023, 11, 1)  # Most recent
        }
        current_time = datetime(2023, 12, 1)
        
        recency_bias = temporal_metrics.recency_bias_score(
            recommendations, item_timestamps, 3, current_time
        )
        
        # Should be positive and reflect recency
        assert recency_bias > 0
        assert recency_bias <= 1.0


class TestModelEvaluator:
    """Test model evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(k_values=[5, 10], time_decay_factor=0.1)
        assert evaluator.k_values == [5, 10]
        assert evaluator.temporal_calculator.time_decay_factor == 0.1
    
    def test_evaluate_model_mock(self):
        """Test model evaluation with mock model."""
        evaluator = ModelEvaluator(k_values=[5])
        
        # Create mock model
        class MockModel:
            def __init__(self):
                self.name = "Mock Model"
                self.is_fitted = True
            
            def recommend(self, user_id, n_recommendations=10):
                return [('item1', 0.9), ('item2', 0.8), ('item3', 0.7)]
        
        # Create test data
        test_data = pd.DataFrame({
            'user_id': ['user1', 'user1'],
            'item_id': ['item1', 'item2'],
            'rating': [5.0, 4.0],
            'timestamp': [datetime(2023, 1, 1), datetime(2023, 1, 2)]
        })
        
        model = MockModel()
        results = evaluator.evaluate_model(model, test_data)
        
        # Check that results contain expected metrics
        assert 'precision@5' in results
        assert 'recall@5' in results
        assert 'ndcg@5' in results
        assert 'hit_rate@5' in results
        assert 'map@5' in results
        
        # All metrics should be non-negative
        for metric, value in results.items():
            if metric != 'model':
                assert value >= 0


class TestCoverageMetrics:
    """Test coverage metrics calculation."""
    
    def test_calculate_coverage_metrics(self):
        """Test coverage metrics calculation."""
        recommendations = [
            ['item1', 'item2', 'item3'],
            ['item2', 'item4', 'item5'],
            ['item1', 'item3', 'item6']
        ]
        all_items = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7']
        
        coverage_metrics = calculate_coverage_metrics(recommendations, all_items)
        
        # Catalog coverage: 6 unique items recommended out of 7 total
        assert abs(coverage_metrics['catalog_coverage'] - 6/7) < 0.01
        
        # User coverage: 3 users with recommendations out of 3 total
        assert coverage_metrics['user_coverage'] == 1.0


class TestDiversityMetrics:
    """Test diversity metrics calculation."""
    
    def test_calculate_diversity_metrics(self):
        """Test diversity metrics calculation."""
        recommendations = [
            ['item1', 'item2', 'item3'],  # All different
            ['item1', 'item1', 'item2'],  # Some duplicates
            ['item1']  # Single item
        ]
        
        diversity_metrics = calculate_diversity_metrics(recommendations)
        
        # Intra-list diversity should be positive
        assert diversity_metrics['intra_list_diversity'] >= 0
        assert diversity_metrics['intra_list_diversity'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
