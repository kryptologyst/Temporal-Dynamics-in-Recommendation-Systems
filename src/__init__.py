"""
Temporal Dynamics in Recommendation Systems

A comprehensive implementation of time-aware recommendation systems that capture
the temporal evolution of user preferences and item relevance over time.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models.temporal_recommender import (
    TemporalRecommender,
    TimeWeightedCollaborativeFiltering,
    TemporalMatrixFactorization,
    TemporalDataGenerator
)

from .data.data_pipeline import (
    TemporalDataLoader,
    TemporalSplitter,
    DataPreprocessor,
    create_sample_data
)

from .evaluation.metrics import (
    RecommendationMetrics,
    TemporalMetrics,
    ModelEvaluator,
    calculate_coverage_metrics,
    calculate_diversity_metrics
)

__all__ = [
    # Models
    "TemporalRecommender",
    "TimeWeightedCollaborativeFiltering", 
    "TemporalMatrixFactorization",
    "TemporalDataGenerator",
    
    # Data pipeline
    "TemporalDataLoader",
    "TemporalSplitter",
    "DataPreprocessor",
    "create_sample_data",
    
    # Evaluation
    "RecommendationMetrics",
    "TemporalMetrics", 
    "ModelEvaluator",
    "calculate_coverage_metrics",
    "calculate_diversity_metrics",
]
