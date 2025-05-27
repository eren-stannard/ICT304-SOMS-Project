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


def subheader_link(body: str, url: str) -> None:
    """
    Display subheader for dataset and a link button to its Kaggle source page.
    
    Parameters
    ----------
    body : str
        Subheader body text (name of dataset).
    url : str
        Link to Kaggle source page for dataset.
    """
    
    st.subheader(body, divider='grey')
    col1, col2 = st.columns([3, 1], vertical_alignment='bottom')
    
    with col1:
        st.write("##### :primary[:material/folder_info:] Dataset Structure")
    with col2:
        st.link_button("Kaggle", url, icon=":material/dataset_linked:", use_container_width=True)
    
    return


# Mall Dataset
subheader_link(
    "Mall Dataset",
    "https://www.kaggle.com/datasets/chaozhuang/mall-dataset",
)
st.container(border=True).write(
    """
        > :material/folder_open: archive  
        >> :material/folder: images  
        >>> :material/image: seq_1.jpg  
        >>> :material/image: ∙∙∙  
        >>> :material/image: seq_2000.jpg\n
        >> :material/view_column: labels.csv
    """,
)

# Count the Number of Faces Present in an Image
subheader_link(
    "Count the Number of Faces Present in an Image",
    "https://www.kaggle.com/datasets/vin1234/count-the-number-of-faces-present-in-an-image",
)
st.container(border=True).write(
    """
        > :material/folder_open: count_faces  
        >> :material/folder: images  
        >>> :material/image: seq_1.jpg  
        >>> :material/image: ∙∙∙  
        >>> :material/image: seq_2000.jpg\n
        >> :material/view_column: labels.csv
    """,
)

# 3DHBD (3D Humanix Blender Dataset)
subheader_link(
    "3DHBD (3D Humanix Blender Dataset)",
    "https://www.kaggle.com/datasets/hamzaiqbal01/3dhbd-3d-humanix-blender-dataset",
)
st.container(border=True).write(
    """
        > :material/folder_open: Dataset  
        >> :material/folder: images  
        >>> :material/image: <img_1>.jpg  
        >>> :material/image: ∙∙∙  
        >>> :material/image: <img_542>.jpg\n
        >> :material/folder:  labels  
        >>> :material/article: <img_1>.txt  
        >>> :material/article: ∙∙∙  
        >>> :material/article: <img_542>.txt
    """,
)

# Classroom Images Or Hand Raised Detection Dataset
subheader_link(
    "Classroom Images Or Hand Raised Detection Dataset",
    "https://www.kaggle.com/datasets/piyushchakarborthy/classroom-images-or-hand-raised-detection-dataset",
)
st.container(border=True).write(
    """
        > :material/folder_open: handraised  
        >> :material/folder: images  
        >>> :material/image: <img_1>.jpg  
        >>> :material/image: ∙∙∙  
        >>> :material/image: <img_542>.jpg\n
        >> :material/folder:  labels  
        >>> :material/article: <img_1>.txt  
        >>> :material/article: ∙∙∙  
        >>> :material/article: <img_542>.txt
    """,
)

# Human Crowd Detection Dataset
subheader_link(
    "Human Crowd Detection Dataset",
    "https://www.kaggle.com/datasets/salmanshakil97/human-crowd-detection?resource=download",
)
st.container(border=True).write(
    """
        > :material/folder_open: NewData  
        >> :material/folder: images  
        >>> :material/image: <img_1>.jpg  
        >>> :material/image: ∙∙∙  
        >>> :material/image: <img_542>.jpg\n
        >> :material/folder:  labels  
        >>> :material/article: <img_1>.txt  
        >>> :material/article: ∙∙∙  
        >>> :material/article: <img_542>.txt
    """,
)

# Complex Indoor Environment Synthetic Dataset
subheader_link(
    "MIT Indoor Scenes",
    "https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019",
)
st.container(border=True).write(
    """
        > :material/folder_open: synthetic  
        >> :material/folder: images  
        >>> :material/image: <img_1>.jpg  
        >>> :material/image: ∙∙∙  
        >>> :material/image: <img_227>.jpg\n
        >> :material/view_column: labels.csv
    """,
)