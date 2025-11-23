"""
Utility scripts for temporal recommendation systems.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_pipeline import create_sample_data, TemporalDataLoader
from models.temporal_recommender import TimeWeightedCollaborativeFiltering, TemporalMatrixFactorization
from evaluation.metrics import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_data_command(args):
    """Generate sample data."""
    logger.info(f"Generating sample data in {args.output_dir}")
    create_sample_data(args.output_dir)
    logger.info("Sample data generated successfully!")


def train_model_command(args):
    """Train a specific model."""
    logger.info(f"Training {args.model} model...")
    
    # Load data
    loader = TemporalDataLoader(args.data_dir)
    interactions = loader.load_interactions()
    
    # Initialize model
    if args.model == "time-weighted-cf":
        model = TimeWeightedCollaborativeFiltering(
            decay_factor=args.decay_factor,
            min_interactions=args.min_interactions,
            n_factors=args.n_factors
        )
    elif args.model == "temporal-mf":
        model = TemporalMatrixFactorization(
            n_factors=args.n_factors,
            learning_rate=args.learning_rate,
            n_epochs=args.epochs
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Train model
    model.fit(interactions)
    
    # Save model (placeholder - would need proper serialization)
    logger.info(f"Model {args.model} trained successfully!")
    
    return model


def evaluate_command(args):
    """Evaluate models."""
    logger.info("Evaluating models...")
    
    # Load data
    loader = TemporalDataLoader(args.data_dir)
    interactions = loader.load_interactions()
    items = loader.load_items()
    
    # Split data
    from data.data_pipeline import TemporalSplitter
    splitter = TemporalSplitter(test_ratio=0.2, validation_ratio=0.1)
    train_data, val_data, test_data = splitter.split_by_time(interactions)
    
    # Initialize models
    models = {
        "Time-Weighted CF": TimeWeightedCollaborativeFiltering(
            decay_factor=0.1, min_interactions=3, n_factors=20
        ),
        "Temporal MF": TemporalMatrixFactorization(n_factors=20, n_epochs=30)
    }
    
    # Train models
    for model in models.values():
        model.fit(train_data)
    
    # Evaluate
    evaluator = ModelEvaluator(k_values=[5, 10, 20])
    results = evaluator.compare_models(list(models.values()), test_data, items)
    
    print("\nModel Evaluation Results:")
    print("=" * 50)
    print(results.round(4))
    
    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Temporal Dynamics Recommendation System CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate data command
    data_parser = subparsers.add_parser("generate-data", help="Generate sample data")
    data_parser.add_argument(
        "--output-dir", "-o", default="data",
        help="Output directory for generated data"
    )
    data_parser.set_defaults(func=generate_data_command)
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "model", choices=["time-weighted-cf", "temporal-mf"],
        help="Model to train"
    )
    train_parser.add_argument(
        "--data-dir", "-d", default="data",
        help="Data directory"
    )
    train_parser.add_argument(
        "--decay-factor", type=float, default=0.1,
        help="Time decay factor"
    )
    train_parser.add_argument(
        "--min-interactions", type=int, default=5,
        help="Minimum interactions per user/item"
    )
    train_parser.add_argument(
        "--n-factors", type=int, default=20,
        help="Number of latent factors"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.01,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    train_parser.set_defaults(func=train_model_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument(
        "--data-dir", "-d", default="data",
        help="Data directory"
    )
    eval_parser.add_argument(
        "--output", "-o",
        help="Output file for results"
    )
    eval_parser.set_defaults(func=evaluate_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
