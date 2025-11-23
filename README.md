# Temporal Dynamics in Recommendation Systems

A comprehensive implementation of time-aware recommendation systems that capture the temporal evolution of user preferences and item relevance over time.

## Overview

This project implements temporal recommendation systems that address the dynamic nature of user preferences. Unlike traditional recommendation systems that treat all interactions equally, temporal systems recognize that:

- User preferences change over time
- Items have seasonal relevance
- Recent interactions are more predictive of current preferences
- Temporal patterns exist in user behavior

## Features

### Models Implemented

1. **Time-Weighted Collaborative Filtering**
   - Applies exponential time decay to user-item interactions
   - Uses matrix factorization with temporal weighting
   - Captures recency bias in user preferences

2. **Temporal Matrix Factorization**
   - Incorporates temporal features into matrix factorization
   - Learns time-dependent user and item factors
   - Handles temporal regularization

### Key Capabilities

- **Temporal Data Generation**: Realistic synthetic data with seasonal patterns
- **Comprehensive Evaluation**: Multiple metrics including temporal-aware measures
- **Interactive Demo**: Streamlit web interface for exploration
- **Production-Ready Code**: Type hints, documentation, and testing

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Temporal-Dynamics-in-Recommendation-Systems.git
cd Temporal-Dynamics-in-Recommendation-Systems
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create data directory:
```bash
mkdir data
```

## Quick Start

### 1. Generate Sample Data

```python
from src.data.data_pipeline import create_sample_data

# Generate sample temporal recommendation data
create_sample_data("data")
```

### 2. Train Models

```python
from src.data.data_pipeline import TemporalDataLoader, TemporalSplitter
from src.models.temporal_recommender import TimeWeightedCollaborativeFiltering, TemporalMatrixFactorization
from src.evaluation.metrics import ModelEvaluator

# Load data
loader = TemporalDataLoader("data")
interactions = loader.load_interactions()
items = loader.load_items()

# Split data temporally
splitter = TemporalSplitter(test_ratio=0.2, validation_ratio=0.1)
train_data, val_data, test_data = splitter.split_by_time(interactions)

# Train models
models = {
    "Time-Weighted CF": TimeWeightedCollaborativeFiltering(decay_factor=0.1),
    "Temporal MF": TemporalMatrixFactorization(n_factors=20)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(train_data)
```

### 3. Evaluate Models

```python
# Evaluate models
evaluator = ModelEvaluator(k_values=[5, 10, 20])
results = evaluator.compare_models(list(models.values()), test_data, items)

print("Model Comparison Results:")
print(results)
```

### 4. Generate Recommendations

```python
# Generate recommendations for a user
user_id = "user_1"
recommendations = models["Time-Weighted CF"].recommend(user_id, n_recommendations=10)

print(f"Recommendations for {user_id}:")
for item_id, score in recommendations:
    print(f"  {item_id}: {score:.3f}")
```

## Interactive Demo

Launch the Streamlit demo to explore the system interactively:

```bash
streamlit run streamlit_demo.py
```

The demo provides:
- Temporal pattern analysis
- Model performance comparison
- Interactive recommendation generation
- Data exploration tools

## Data Schema

### Interactions Data (`interactions.csv`)

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Unique user identifier |
| item_id | string | Unique item identifier |
| rating | float | User rating (1-5 scale) |
| timestamp | datetime | When the interaction occurred |

### Items Data (`items.csv`)

| Column | Type | Description |
|--------|------|-------------|
| item_id | string | Unique item identifier |
| title | string | Item title/name |
| category | string | Item category |
| price | float | Item price |
| popularity | float | Item popularity score |

## Model Architecture

### Time-Weighted Collaborative Filtering

```python
class TimeWeightedCollaborativeFiltering:
    def __init__(self, decay_factor=0.1, min_interactions=5, n_factors=50):
        # Apply exponential time decay: exp(-decay_factor * days_diff / 365)
        # Filter sparse users/items
        # Matrix factorization with temporal weighting
```

### Temporal Matrix Factorization

```python
class TemporalMatrixFactorization:
    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.01):
        # Incorporate temporal features into factorization
        # SGD with temporal regularization
        # Learn time-dependent user/item factors
```

## Evaluation Metrics

### Standard Metrics
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **MAP@K**: Mean Average Precision

### Temporal Metrics
- **Temporal Precision@K**: Time-weighted precision considering recency
- **Recency Bias Score**: Measure of how recent recommended items are
- **Coverage**: Catalog and user coverage metrics
- **Diversity**: Intra-list diversity measures

## Project Structure

```
temporal-dynamics-recommendations/
├── src/
│   ├── data/
│   │   └── data_pipeline.py          # Data loading and preprocessing
│   ├── models/
│   │   └── temporal_recommender.py    # Temporal recommendation models
│   ├── evaluation/
│   │   └── metrics.py                 # Evaluation metrics
│   └── utils/
├── data/                              # Data directory
│   ├── interactions.csv              # Interaction data
│   └── items.csv                     # Item metadata
├── configs/                           # Configuration files
├── notebooks/                         # Jupyter notebooks
├── scripts/                          # Utility scripts
├── tests/                            # Unit tests
├── assets/                           # Static assets
├── streamlit_demo.py                # Interactive demo
├── requirements.txt                  # Dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## Configuration

### Model Parameters

```yaml
# configs/model_config.yaml
time_weighted_cf:
  decay_factor: 0.1
  min_interactions: 5
  n_factors: 50

temporal_mf:
  n_factors: 50
  learning_rate: 0.01
  regularization: 0.01
  n_epochs: 100

evaluation:
  k_values: [5, 10, 20]
  time_decay_factor: 0.1
```

## Advanced Usage

### Custom Data Loading

```python
from src.data.data_pipeline import TemporalDataLoader

# Load custom data
loader = TemporalDataLoader("path/to/data")
interactions = loader.load_interactions("custom_interactions.csv")
items = loader.load_items("custom_items.csv")
```

### Custom Temporal Splitting

```python
from src.data.data_pipeline import TemporalSplitter

# User-based temporal splitting
splitter = TemporalSplitter(test_ratio=0.2, validation_ratio=0.1)
train_data, val_data, test_data = splitter.split_by_user_time(
    interactions, n_test_interactions=5
)
```

### Adding New Models

```python
from src.models.temporal_recommender import TemporalRecommender

class CustomTemporalModel(TemporalRecommender):
    def __init__(self):
        super().__init__("Custom Temporal Model")
    
    def fit(self, interactions: pd.DataFrame) -> None:
        # Implement training logic
        pass
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        # Implement recommendation logic
        pass
```

## Performance Considerations

### Memory Usage
- Models use efficient sparse matrix representations
- Temporal features are computed on-demand
- Caching is implemented for repeated evaluations

### Scalability
- Matrix factorization scales to moderate datasets (10K+ users/items)
- For larger datasets, consider distributed implementations
- Use approximate nearest neighbor search for large item catalogs

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test categories:

```bash
pytest tests/test_models.py          # Model tests
pytest tests/test_data_pipeline.py   # Data pipeline tests
pytest tests/test_evaluation.py      # Evaluation tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for all classes and methods
- Use black for code formatting
- Use ruff for linting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{temporal_dynamics_recommendations,
  title={Temporal Dynamics in Recommendation Systems},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Temporal-Dynamics-in-Recommendation-Systems}
}
```

## Acknowledgments

- Inspired by temporal recommendation research
- Built with scikit-learn, pandas, and numpy
- Demo powered by Streamlit
- Visualization with Plotly

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and Python path is correct
2. **Memory Issues**: Reduce dataset size or model complexity
3. **Slow Training**: Use fewer factors or epochs for faster experimentation
4. **Demo Not Loading**: Check that Streamlit is installed and data files exist

### Getting Help

- Check the issues section for common problems
- Review the code documentation
- Run the demo to verify installation
- Test with sample data first

## Roadmap

- [ ] Add more temporal models (RNN-based, attention mechanisms)
- [ ] Implement distributed training for large datasets
- [ ] Add real-time recommendation capabilities
- [ ] Integrate with popular recommendation datasets
- [ ] Add more evaluation metrics and visualizations
- [ ] Implement model serving with FastAPI
- [ ] Add experiment tracking with MLflow/W&B
# Temporal-Dynamics-in-Recommendation-Systems
