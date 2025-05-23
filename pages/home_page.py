# home_page.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Default home page and project/system information.

"""


# Libraries used
import streamlit as st


# Page links
option_map = {
    "pages/team_page.py": [":primary[:material/group:] Team", "Project team details and workload contribution"],
    "pages/ams_page.py": [":primary[:material/video_camera_front:] AMS", "Attendance Monitoring Sub-System"],
    "pages/ods_page.py": [":primary[:material/detection_and_zone:] ODS", "Occupancy Detection AI Sub-System"],
    "pages/dvs_page.py": [":primary[:material/bar_chart_4_bars:] DVS", "Data Visualisation Sub-System"],
    "pages/db_page.py": [":primary[:material/database:] DB", "Database"],
    "pages/docs_page.py": [":primary[:material/api:] Docs", "Documentation"],
    "pages/data_page.py": [":primary[:material/dataset_linked:] Data", "Datasets details"],
    "pages/sources_page.py": [":primary[:material/format_quote:] Sources", "References and acknowledgements"],
}
page = st.pills(
    label="Pages:",
    options=option_map.keys(),
    format_func=lambda option: option_map[option][0],
    help="  \n".join([": ".join(option) for option in option_map.values()]),
    label_visibility='collapsed',
)
if page is not None:
    st.switch_page(page)