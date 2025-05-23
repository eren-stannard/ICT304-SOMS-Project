# streamlit_app.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Setup and execution for Streamlit application, pages, and navigation.

"""


# Libraries used
import os
import streamlit as st
import torch

# Files used
from src.page import create_page


# To fix torch/streamlit conflict error. Adapted from Espen's (2025) solution at:
# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/5
torch.classes.__path__ = []


# --- PAGE CONFIGURATION SETUP ---

# About dialog
about_dialog = """
    ### :primary[:material/monitoring:] SOMS :primary-badge[Smart Occupancy Monitoring System]
    
    **Created by:**  
    :small[:primary[:material/arrow_right:] Ervin Galas]
    [:material/id_card:](project_team_page#user-attributes-ervin-galas)  
    :small[:primary[:material/arrow_right:] Sofia Peeva]
    [:material/id_card:](project_team_page#user-attributes-sofia-peeva)  
    :small[:primary[:material/arrow_right:] Eren Stannard]
    [:material/id_card:](project_team_page#user-attributes-eren-stannard)  
    
    **ICT304: AI System Design**  
    :small[:primary[:material/arrow_right:]] :small[Murdoch University]
    [:material/school:](https://www.murdoch.edu.au/explore/about-murdoch)
"""

# Page title and config
st.set_page_config(
    page_title="SOMS",
    page_icon=":material/monitoring:",
    #layout='wide',
    menu_items={"About": about_dialog},
)


# --- PAGES SETUP ---

# SOMS project home page
home_page = create_page(
    page="pages/home_page.py",
    title="SOMS",
    subtitle="Smart Occupancy Monitoring System",
    icon=":material/monitoring:",
    default=True,
)

# Project team details
team_page = create_page(
    page="pages/team_page.py",
    title="Project Team",
    subtitle="Project Team Members Details",
    icon=":material/group:",
)

# AMS page
ams_page = create_page(
    page="pages/ams_page.py",
    title="AMS",
    subtitle="Attendance Monitoring Sub-System",
    icon=":material/video_camera_front:",
)

# ODS page
ods_page = create_page(
    page="pages/ods_page.py",
    title="ODS",
    subtitle="Occupancy Detection AI Sub-System",
    icon=":material/detection_and_zone:",
)

# DVS page
dvs_page = create_page(
    page="pages/dvs_page.py",
    title="DVS",
    subtitle="Data Visualisation Sub-System",
    icon=":material/bar_chart_4_bars:",
)

# Database page
db_page = create_page(
    page="pages/db_page.py",
    title="Database",
    subtitle="Occupancy Records",
    icon=":material/database:",
)

# Code documentation page
docs_page = create_page(
    page="pages/docs_page.py",
    title="Documentation",
    subtitle="System Functionality Reference",
    icon=":material/integration_instructions:",
)

# Datasets page
data_page = create_page(
    page="pages/data_page.py",
    title="Datasets",
    subtitle="Data Sources and Listing",
    icon=":material/dataset_linked:",
)

# References page
sources_page = create_page(
    page="pages/sources_page.py",
    title="References",
    subtitle="References and Acknowledgements",
    icon=":material/article_person:",
)


# --- NAVIGATION SETUP ---

# Page navigation sidebar
pg = st.navigation(
    {
        "About": [
            home_page,
            team_page,
        ],
        "Sub-Systems and Components": [
            ams_page,
            ods_page,
            dvs_page,
            db_page,
        ],
        "Resources": [
            docs_page,
            data_page,
            sources_page,
        ],
    },
)


# --- SIDEBAR SETUP ---

# Sidebar logo
st.logo(image=os.path.join("assets", "monitoring_icon.png"))

# Sidebar links to GitHub repositories
st.sidebar.link_button(
    label="GitHub Repository (SOMS)",
    url="https://github.com/eren-stannard/ICT304_SOMS",
    type='primary',
    icon=":material/folder_code:",
    use_container_width=True,
)
st.sidebar.link_button(
    label="GitHub Repository (Prototype)",
    url="https://github.com/sohazasoha16/ICT304_SOMS",
    icon=":material/folder_code:",
    use_container_width=True,
)

# Authors
st.sidebar.write(":primary[:material/copyright:] :grey[Ervin Galas, Sofia Peeva, Eren Stannard]")


# --- PAGE EXECUTION ---

# Run page
pg.header() # type: ignore
pg.run()