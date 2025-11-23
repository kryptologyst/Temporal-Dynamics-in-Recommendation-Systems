"""
Streamlit demo for Temporal Dynamics Recommendation System.

This module provides an interactive web interface to explore temporal
recommendation models and their performance.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_pipeline import TemporalDataLoader, TemporalSplitter, DataPreprocessor
from evaluation.metrics import ModelEvaluator
from models.temporal_recommender import (
    TemporalDataGenerator,
    TimeWeightedCollaborativeFiltering,
    TemporalMatrixFactorization
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Temporal Dynamics in Recommendations",
    page_icon="⏰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-item {
        background-color: #ffffff;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cache the recommendation data."""
    try:
        loader = TemporalDataLoader("data")
        interactions = loader.load_interactions()
        items = loader.load_items()
        return interactions, items
    except FileNotFoundError:
        st.warning("Data files not found. Generating sample data...")
        generator = TemporalDataGenerator(n_users=200, n_items=100, n_interactions=2000)
        interactions = generator.generate_interactions()
        items = generator.generate_items()
        
        # Save generated data
        Path("data").mkdir(exist_ok=True)
        interactions.to_csv("data/interactions.csv", index=False)
        items.to_csv("data/items.csv", index=False)
        
        return interactions, items


@st.cache_data
def train_models(interactions: pd.DataFrame) -> Dict[str, any]:
    """Train and cache recommendation models."""
    # Split data
    splitter = TemporalSplitter(test_ratio=0.2, validation_ratio=0.1)
    train_data, val_data, test_data = splitter.split_by_time(interactions)
    
    # Initialize models
    models = {
        "Time-Weighted CF": TimeWeightedCollaborativeFiltering(
            decay_factor=0.1, min_interactions=3, n_factors=20
        ),
        "Temporal MF": TemporalMatrixFactorization(
            n_factors=20, learning_rate=0.01, n_epochs=30
        )
    }
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            model.fit(train_data)
            trained_models[name] = model
    
    return trained_models, test_data


def plot_temporal_patterns(interactions: pd.DataFrame) -> None:
    """Plot temporal patterns in the data."""
    st.subheader("📊 Temporal Patterns Analysis")
    
    # Convert timestamp to datetime if needed
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    
    # Create temporal features
    interactions['hour'] = interactions['timestamp'].dt.hour
    interactions['day_of_week'] = interactions['timestamp'].dt.dayofweek
    interactions['month'] = interactions['timestamp'].dt.month
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactions over time
        daily_interactions = interactions.groupby(interactions['timestamp'].dt.date).size()
        fig_time = px.line(
            x=daily_interactions.index,
            y=daily_interactions.values,
            title="Interactions Over Time",
            labels={'x': 'Date', 'y': 'Number of Interactions'}
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Hourly patterns
        hourly_interactions = interactions.groupby('hour').size()
        fig_hour = px.bar(
            x=hourly_interactions.index,
            y=hourly_interactions.values,
            title="Interactions by Hour of Day",
            labels={'x': 'Hour', 'y': 'Number of Interactions'}
        )
        fig_hour.update_layout(height=400)
        st.plotly_chart(fig_hour, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Day of week patterns
        dow_interactions = interactions.groupby('day_of_week').size()
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig_dow = px.bar(
            x=[dow_names[i] for i in dow_interactions.index],
            y=dow_interactions.values,
            title="Interactions by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Number of Interactions'}
        )
        fig_dow.update_layout(height=400)
        st.plotly_chart(fig_dow, use_container_width=True)
    
    with col4:
        # Monthly patterns
        monthly_interactions = interactions.groupby('month').size()
        fig_month = px.bar(
            x=monthly_interactions.index,
            y=monthly_interactions.values,
            title="Interactions by Month",
            labels={'x': 'Month', 'y': 'Number of Interactions'}
        )
        fig_month.update_layout(height=400)
        st.plotly_chart(fig_month, use_container_width=True)


def plot_model_comparison(evaluation_results: pd.DataFrame) -> None:
    """Plot model comparison results."""
    st.subheader("🏆 Model Performance Comparison")
    
    # Select metrics to display
    metric_options = [col for col in evaluation_results.columns if col != 'model']
    selected_metrics = st.multiselect(
        "Select metrics to compare:",
        metric_options,
        default=['precision@10', 'recall@10', 'ndcg@10']
    )
    
    if selected_metrics:
        # Create comparison chart
        fig = go.Figure()
        
        for metric in selected_metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=evaluation_results['model'],
                y=evaluation_results[metric],
                text=evaluation_results[metric].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed results table
        st.subheader("📋 Detailed Results")
        st.dataframe(evaluation_results.round(4), use_container_width=True)


def show_recommendations(
    models: Dict[str, any],
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    selected_user: str,
    n_recommendations: int
) -> None:
    """Show recommendations for a selected user."""
    st.subheader(f"🎯 Recommendations for User: {selected_user}")
    
    # Get user's interaction history
    user_history = interactions[interactions['user_id'] == selected_user].sort_values('timestamp')
    
    if not user_history.empty:
        st.write("**User's Interaction History:**")
        history_display = user_history[['item_id', 'rating', 'timestamp']].copy()
        history_display['timestamp'] = history_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(history_display, use_container_width=True)
    
    # Generate recommendations from each model
    col1, col2 = st.columns(2)
    
    for i, (model_name, model) in enumerate(models.items()):
        with col1 if i % 2 == 0 else col2:
            st.write(f"**{model_name}:**")
            
            try:
                recommendations = model.recommend(selected_user, n_recommendations)
                
                if recommendations:
                    for j, (item_id, score) in enumerate(recommendations, 1):
                        # Get item details
                        item_info = items[items['item_id'] == item_id]
                        item_title = item_info['title'].iloc[0] if not item_info.empty else item_id
                        item_category = item_info['category'].iloc[0] if not item_info.empty else "Unknown"
                        
                        st.markdown(f"""
                        <div class="recommendation-item">
                            <strong>{j}. {item_title}</strong><br>
                            <small>ID: {item_id} | Category: {item_category} | Score: {score:.3f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No recommendations available for this user.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")


def main() -> None:
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">⏰ Temporal Dynamics in Recommendation Systems</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases temporal recommendation systems that capture the evolution of user preferences over time.
    Explore different models, analyze temporal patterns, and see how time-aware algorithms improve recommendations.
    """)
    
    # Load data
    interactions, items = load_data()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Data overview
    st.sidebar.subheader("📊 Data Overview")
    st.sidebar.metric("Total Interactions", len(interactions))
    st.sidebar.metric("Unique Users", interactions['user_id'].nunique())
    st.sidebar.metric("Unique Items", interactions['item_id'].nunique())
    st.sidebar.metric("Date Range", f"{interactions['timestamp'].min().date()} to {interactions['timestamp'].max().date()}")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Configuration")
    decay_factor = st.sidebar.slider("Time Decay Factor", 0.01, 0.5, 0.1, 0.01)
    n_factors = st.sidebar.slider("Number of Factors", 10, 50, 20)
    
    # User selection
    st.sidebar.subheader("👤 User Selection")
    available_users = sorted(interactions['user_id'].unique())
    selected_user = st.sidebar.selectbox("Select User:", available_users)
    n_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Temporal Analysis", "🏆 Model Comparison", "🎯 Recommendations", "📋 Data Explorer"])
    
    with tab1:
        plot_temporal_patterns(interactions)
    
    with tab2:
        # Train models
        models, test_data = train_models(interactions)
        
        # Evaluate models
        evaluator = ModelEvaluator(k_values=[5, 10, 20])
        evaluation_results = evaluator.compare_models(
            list(models.values()),
            test_data,
            items,
            n_recommendations=20
        )
        
        plot_model_comparison(evaluation_results)
    
    with tab3:
        # Train models for recommendations
        models, _ = train_models(interactions)
        
        show_recommendations(models, interactions, items, selected_user, n_recommendations)
    
    with tab4:
        st.subheader("📋 Raw Data Explorer")
        
        # Data preview
        st.write("**Interactions Data:**")
        st.dataframe(interactions.head(100), use_container_width=True)
        
        st.write("**Items Data:**")
        st.dataframe(items.head(100), use_container_width=True)
        
        # Download buttons
        st.subheader("💾 Download Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_interactions = interactions.to_csv(index=False)
            st.download_button(
                label="Download Interactions CSV",
                data=csv_interactions,
                file_name="interactions.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_items = items.to_csv(index=False)
            st.download_button(
                label="Download Items CSV",
                data=csv_items,
                file_name="items.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
