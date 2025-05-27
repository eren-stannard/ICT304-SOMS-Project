# dvs_page.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Page for Data Visualisation Sub-System (DVS).

"""


# Libraries used
import numpy as np
import os
import pandas as pd
import plotly.express as px
import streamlit as st

# Files used
import ODS.config as config
from ODS.data_augmentation import plot_data_distributions
from ODS.evaluate import plot_results
from ODS.train import plot_learning_curve


# Data visualisations paths
data_vis_dir = config.DATA_VIS_DIR
eval_results = os.path.join(data_vis_dir, "evaluation_results.csv")
optim_results = os.path.join(data_vis_dir, "optimisation_results.csv")

# Plot data distributions
if all([os.path.exists(f) for f in [config.TRAIN_LABELS_FILE, config.TEST_LABELS_FILE]]):
    
    train_labels = np.load(config.TRAIN_LABELS_FILE, mmap_mode='r')
    test_labels = np.load(config.TEST_LABELS_FILE, mmap_mode='r')
    
    st.plotly_chart(
        plot_data_distributions(
            [pd.Series(train_labels), pd.Series(test_labels)],
            ["Training", "Validation"],
            title="Data Distribution of Training and Validation Sets",
        ),
        use_container_width=True,
    )

# Plot learning curve
#if os.path.exists(optim_results):
#    plot_learning_curve(pd.read_csv(optim_results).to_dict(), model_name="CNN")

# Plot evaluation
if os.path.exists(eval_results):
    results_df = pd.read_csv(eval_results)
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
        title=f"CNN Model Predicted Versus True Values",
    )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)