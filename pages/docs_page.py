# docs_page.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Page for Occupancy Detection AI Sub-System (ODS) Documentation.

"""


# Libraries used
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# Files used
import ODS.config as config
from ODS.autoencoder_model import AutoencoderModel
from ODS import data_loader


# To fix torch/streamlit conflict error. Adapted from Espen's (2025) solution at:
# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/5
torch.classes.__path__ = []


def main() -> None:

    ams, ods, dvs = st.tabs([
        ":primary[:material/video_camera_front:] AMS",
        ":primary[:material/detection_and_zone:] ODS",
        ":primary[:material/bar_chart_4_bars:] DVS",
    ])

    # AMS documentation
    with ams:
        st.info("TO DO")

    # ODS documentation
    with ods:
        
        dataset_tab, model_tab = st.tabs([
            ":primary[:material/dataset:] Dataset",
            ":primary[:material/network_intelligence:] Model",
        ])
        
        with dataset_tab:
            
            st.subheader("Data Loader")
            #dataset_loader = data_loader.get_data_loader()
            #st.help(dataset_loader)
            
            st.subheader("Dataset")
            #st.help(dataset_loader.dataset.dataset) # type: ignore
        
        with model_tab:
            model_docs()
        
    return


def model_docs() -> None:
            
    for i, v in AutoencoderModel().__class__.__dict__.items():
        
        if i in ['__module__', '__doc__']:
            
            if i == '__module__':
                st.header(f":primary[{v}]")
            
            else:
                st.write(f"{v}")
        
        else:
            
            st.subheader(f":primary[{i}]")
            st.help(getattr(AutoencoderModel(), i))