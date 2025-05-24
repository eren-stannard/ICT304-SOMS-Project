# evaluate.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Model performance evaluation.

"""


# Libraries used
import numpy as np
import os
import pandas as pd
import plotly.express as px
import re
import streamlit as st
import time
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from stqdm import stqdm

# Files used
import ODS.config as config
from ODS.data_loader import get_data_loader
from ODS.autoencoder_model import AutoencoderModel
from ODS.cnn_model import CNNModel


def evaluate_model(model: AutoencoderModel | CNNModel, batch_size: int = config.BATCH_SIZE) -> dict[str, float | int]:
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : AutoencoderModel | CNNModel
        Trained model.
    batch_size : int, optional, default=BATCH_SIZE
        Batch size for evaluation.
    
    Returns
    -------
    results : dict[str, float | int]
        Dictionary containing evaluation metrics.
    
    See Also
    --------
    config.py : Default parameter configuration.
    """
    
    # Model name str
    model_name = " ".join(re.split(r"(?=Model)", model.__class__.__name__))
    
    # Set model to evaluation mode
    model.eval()
    
    # Display msg
    prog_msg = f"Evaluating :primary[**{model_name}**] model (using device: :primary[**{model.device}**])"

    # Load test data
    val_loader = get_data_loader(batch_size=batch_size, mode='evaluate')

    # Evaluation metrics
    true_counts = []
    predicted_counts = []
    inference_times = []

    # Evaluate model
    with torch.no_grad():
        
        for x, y in stqdm(val_loader, desc=prog_msg, unit='**batch**'):

            # Measure inference time
            start_time = time.time()
            outputs = model(x)
            end_time = time.time()

            # Record inference time
            batch_inference_time = (end_time - start_time) / x.size(0)  # time per image
            inference_times.extend([batch_inference_time] * x.size(0))

            # Record predictions and targets
            true_counts.extend(y.int().cpu().tolist())
            predicted_counts.extend(outputs.int().cpu().tolist())

    # Calculate metrics
    mse = mean_squared_error(true_counts, predicted_counts)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_counts, predicted_counts)
    r2 = r2_score(true_counts, predicted_counts)
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms

    # Results
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'avg_inference_time_ms': avg_inference_time,
        'true_counts': true_counts,
        'predicted_counts': predicted_counts,
    }
    
    # Plot results
    plot_results(results, model_name)

    return results


def plot_results(results: dict[str, float | int], model_name: str) -> None:
    """
    Plot evaluation results.

    Parameters
    ----------
    results : dict[str, float | int]
        Dictionary containing evaluation results.
    
    See Also
    --------
    config.py : Default parameter configuration.
    """

    # Extract data
    results_df = pd.DataFrame(
        data={
            'True Count': results['true_counts'],
            'Predicted Count': results['predicted_counts'],
        },
    )
    results_df['Error'] = abs(results_df['True Count'] - results_df['Predicted Count'])
    
    # Display metrics
    st.subheader(f"{model_name} Performance")
    cols = st.columns(5)
    cols[0].metric(
        "MSE", f"{results['mse']:.3f}",
        help="Mean Squared Error",
    )
    cols[1].metric(
        "RMSE", f"{results['rmse']:.3f}",
        help="Root Mean Squared Error",
    )
    cols[2].metric(
        "MAE", f"{results['mae']:.3f}",
        help="Mean Absolute Error",
    )
    cols[3].metric(
        "R²", f"{results['r2']:.3f}",
        help="Coefficient of Determination (R²-Score)",
    )
    cols[4].metric(
        "Inference Time", f"{results['avg_inference_time_ms']:.3f}",
        help="Average time (ms) to make prediction on image",
    )
    
    # Create scatter plot
    fig = px.scatter(
        results_df,
        x='True Count',
        y='Predicted Count',
        color='Error',
        color_continuous_scale=config.COLOUR_SCALE[::-1],
        opacity=0.35,
        trendline='ols',
        trendline_color_override=st.get_option("theme.primaryColor"),
        title=f"{model_name} Predicted Versus True Values",
    )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Save results to file
    results_df.to_csv(os.path.join(config.DATA_VIS_DIR, "evaluation_results.csv"), index=False)

    return