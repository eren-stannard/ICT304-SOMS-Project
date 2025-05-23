# data_page.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Page for dataset details.

"""


# Libraries used
import streamlit as st


# Mall Dataset
st.subheader("Mall Dataset [:material/open_in_new:]", divider='grey')
st.write("##### Dataset Structure")
st.code(
    """
        archive/
        | --- frames/
        | | --- frames/          # Images (.jpg)
        | | | --- seq_1.jpg
        | | | --- ∙∙∙
        | | | --- seq_2000.jpg
        | --- labels.csv         # Labels (.csv)
    """,
    #language=None,
)

# Human Crowd Detection Dataset
st.subheader("Human Crowd Detection Dataset", divider='grey')
st.write("##### Dataset Structure")
st.code(
    """
        NewData/
        | --- images/
        | | --- training/        # Images (.jpg), Labels (.txt)
        | | | --- <img_1>.jpg
        | | | --- <img_1>.txt
        | | | --- ∙∙∙
        | | | --- <img_542>.jpg
        | | | --- <img_542>.txt
    """,
    #language=None,
)

# Complex Indoor Environment Synthetic Dataset
st.subheader("Complex Indoor Environment Synthetic Dataset", divider='grey')
st.write("##### Dataset Structure")
st.code(
    """
        synthetic_no_people/
        | --- frames/            # Images (.jpg)
        | | --- synth_1.jpg
        | | --- ∙∙∙
        | | --- synth_42.jpg
        | --- labels.csv         # Labels (.csv)
    """,
    #language=None,
)