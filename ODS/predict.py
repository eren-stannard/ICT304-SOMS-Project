# predict.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System

    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)

    ICT304: AI System Design
    Murdoch University

    Purpose of File:
    Predict the output of an image or real-time camera feed.

"""


# Libraries used
import cv2
import streamlit as st
import torch
import torchvision.transforms.v2 as T
from io import BytesIO
from streamlit.runtime import uploaded_file_manager as ufm
from torchvision.io import decode_image

# Files used
import ODS.config as config
from ODS.autoencoder_model import AutoencoderModel
from ODS.cnn_model import CNNModel


def predict_image(model: AutoencoderModel | CNNModel, image: str | torch.Tensor | ufm.UploadedFile) -> int:
    """
    Predict occupancy for a single image.

    Parameters
    ----------
    model : AutoencoderModel | CNNModel
        Model to use for prediction.
    image : str | Tensor | UploadedFile
        Path to input image or Streamlit image file to make prediction on.

    Returns
    -------
    pred : int
        Predicted label outcome for input image.
    """
    
    if isinstance(image, BytesIO):
        image = torch.frombuffer(image.getbuffer(), dtype=torch.uint8)
        
    # Load and transform image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToDtype(torch.float32),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = decode_image(image, mode='RGB').to(model.device) # type: ignore
    image = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(image)
        pred = round(output.item())

    return pred


def predict_camera(model: AutoencoderModel | CNNModel, camera_id: int = 0) -> int | None:
    """
    Real-time prediction from camera feed.

    Parameters
    model : AutoencoderModel | CNNModel
        Model to use for prediction.
    camera_id : int, optional, default=0
        Camera device ID.
    
    Returns
    -------
    pred : int | None
        Predicted occupancy label
    """
    
    # Open camera and capture frame
    enable = st.checkbox("Enable Camera")
    frame = st.camera_input("Capture image", disabled=not enable)
    
    if frame:
        
        st.toast("Camera opened successfully. Press 'q' to quit", icon=":material/linked_camera:")
        pred = predict_image(model, frame)
        
        return pred
    
    return