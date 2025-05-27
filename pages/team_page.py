# team_page.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Page for project team and work contribution deails.

"""


# Libraries used
import streamlit as st


# --- PROJECT TEAM MEMBERS ---

# Ervin
with st.container(border=True):
    st.subheader(":primary[:material/user_attributes:] Ervin Galas :primary-badge[:material/id_card: 34276705]", divider='grey')
    st.write(
        """
            **:primary[:material/assured_workload:] Workload Contribution:**
            > * AI Techniques investigation
            > * Systems Engineering Product Life Cycle
            > * Database creation and adiministration
        """
    )

# Sofia
with st.container(border=True):
    st.subheader(":primary[:material/user_attributes:] Sofia Peeva :primary-badge[:material/id_card: 35133522]", divider='grey')
    st.write(
        """
            **:primary[:material/assured_workload:] Workload Contribution:**
            > * Prototype development
            > * Model optimisation and performance evaluation methods
            > * Data pre-processing
        """
    )

# Eren
with st.container(border=True):
    st.subheader(":primary[:material/user_attributes:] Eren Stannard :primary-badge[:material/id_card: 34189185]", divider='grey')
    st.write(
        """
            **:primary[:material/assured_workload:] Workload Contribution:**
            > * System architecture and requirements
            > * Client application
            > * Internal documentation
        """
    )