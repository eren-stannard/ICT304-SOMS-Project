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


# Data visualisations paths
data_vis_dir = config.DATA_VIS_DIR
eval_results = os.path.join(data_vis_dir, "evaluation_results.csv")
optim_results = os.path.join(data_vis_dir, "optimisation_results.csv")

# Save data distributions
if all([os.path.exists(f) for f in [config.TRAIN_LABELS_FILE, config.TEST_LABELS_FILE]]):
    
    train_labels = np.load(config.TRAIN_LABELS_FILE, mmap_mode='r')
    test_labels = np.load(config.TEST_LABELS_FILE, mmap_mode='r')
    
    fig = plot_data_distributions(
        [pd.Series(train_labels), pd.Series(test_labels)],
        ["Training", "Validation"],
        title="Data Distribution of Training and Validation Sets",
    )
    st.plotly_chart(fig, use_container_width=True)