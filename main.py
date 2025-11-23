"""
Main script for temporal recommendation system demo.

This script demonstrates the complete pipeline from data generation
to model evaluation and recommendation generation.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_pipeline import TemporalDataLoader, TemporalSplitter, create_sample_data
from models.temporal_recommender import (
    TimeWeightedCollaborativeFiltering,
    TemporalMatrixFactorization
)
from evaluation.metrics import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function."""
    logger.info("Starting Temporal Dynamics Recommendation System Demo")
    
    # Step 1: Create sample data if it doesn't exist
    data_dir = "data"
    if not Path(data_dir).exists() or not list(Path(data_dir).glob("*.csv")):
        logger.info("Creating sample data...")
        create_sample_data(data_dir)
    else:
        logger.info("Using existing data files")
    
    # Step 2: Load data
    logger.info("Loading data...")
    loader = TemporalDataLoader(data_dir)
    interactions = loader.load_interactions()
    items = loader.load_items()
    
    logger.info(f"Loaded {len(interactions)} interactions for {interactions['user_id'].nunique()} users and {interactions['item_id'].nunique()} items")
    
    # Step 3: Split data temporally
    logger.info("Splitting data temporally...")
    splitter = TemporalSplitter(test_ratio=0.2, validation_ratio=0.1)
    train_data, val_data, test_data = splitter.split_by_time(interactions)
    
    logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Step 4: Initialize and train models
    logger.info("Initializing models...")
    models = {
        "Time-Weighted Collaborative Filtering": TimeWeightedCollaborativeFiltering(
            decay_factor=0.1,
            min_interactions=3,
            n_factors=20
        ),
        "Temporal Matrix Factorization": TemporalMatrixFactorization(
            n_factors=20,
            learning_rate=0.01,
            n_epochs=30
        )
    }
    
    # Train models
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(train_data)
        logger.info(f"{name} training completed")
    
    # Step 5: Evaluate models
    logger.info("Evaluating models...")
    evaluator = ModelEvaluator(k_values=[5, 10, 20])
    evaluation_results = evaluator.compare_models(
        list(models.values()),
        test_data,
        items,
        n_recommendations=20
    )
    
    # Display results
    logger.info("Model Evaluation Results:")
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(evaluation_results.round(4))
    print("="*80)
    
    # Step 6: Generate sample recommendations
    logger.info("Generating sample recommendations...")
    
    # Get a sample user
    sample_user = train_data['user_id'].iloc[0]
    logger.info(f"Generating recommendations for user: {sample_user}")
    
    print(f"\nRecommendations for User: {sample_user}")
    print("-" * 50)
    
    for name, model in models.items():
        try:
            recommendations = model.recommend(sample_user, n_recommendations=5)
            
            print(f"\n{name}:")
            for i, (item_id, score) in enumerate(recommendations, 1):
                # Get item details if available
                item_info = items[items['item_id'] == item_id]
                if not item_info.empty:
                    item_title = item_info['title'].iloc[0]
                    item_category = item_info['category'].iloc[0]
                    print(f"  {i}. {item_title} ({item_id}) - Category: {item_category} - Score: {score:.3f}")
                else:
                    print(f"  {i}. {item_id} - Score: {score:.3f}")
                    
        except Exception as e:
            logger.error(f"Error generating recommendations with {name}: {e}")
    
    # Step 7: Show temporal patterns
    logger.info("Analyzing temporal patterns...")
    
    # Convert timestamp to datetime if needed
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    
    # Calculate some basic temporal statistics
    interactions['hour'] = interactions['timestamp'].dt.hour
    interactions['day_of_week'] = interactions['timestamp'].dt.dayofweek
    interactions['month'] = interactions['timestamp'].dt.month
    
    print(f"\nTemporal Analysis:")
    print("-" * 50)
    print(f"Date range: {interactions['timestamp'].min().date()} to {interactions['timestamp'].max().date()}")
    print(f"Most active hour: {interactions['hour'].mode().iloc[0]}:00")
    print(f"Most active day: {interactions['day_of_week'].mode().iloc[0]} (0=Monday)")
    print(f"Most active month: {interactions['month'].mode().iloc[0]}")
    print(f"Average rating: {interactions['rating'].mean():.2f}")
    
    logger.info("Demo completed successfully!")
    
    # Step 8: Instructions for running the demo
    print(f"\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("To explore the system interactively, run:")
    print("  streamlit run streamlit_demo.py")
    print("\nTo run tests:")
    print("  pytest tests/")
    print("\nTo generate new sample data:")
    print("  python -c \"from src.data.data_pipeline import create_sample_data; create_sample_data('data')\"")


if __name__ == "__main__":
    import pandas as pd
    main()
