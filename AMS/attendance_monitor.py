# attendance_monitor.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Attendance Monitoring Sub-System (AMS) for real-time occupancy prediction.

"""


# Libraries used
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torchvision.transforms.v2 as T
from collections import deque
from datetime import datetime
from numpy.typing import NDArray
from torchvision import tv_tensors

# Files used
from ODS.autoencoder_model import AutoencoderModel
from ODS.cnn_model import CNNModel


class AttendanceMonitor:
    """Real-time attendance monitor class."""
    
    def __init__(self, model: AutoencoderModel | CNNModel, camera_id: int = 0, max_records: int = 100) -> None:
        """
        Attendance monitor constructor.
        
        Parameters
        ----------
        model : AutoencoderModel | CNNModel
            Model to use for prediction.
        camera_id : int, optional, default=0
            Camera device to use.
        max_records : int, optional, default=100
            Maximum number of records to show on plot.
        """
        
        self.model = model
        self.camera_id = camera_id
        self.cap = None
        
        # Occupancy data predictions and associated timestamps
        self.data = deque(maxlen=max_records)
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToDtype(torch.float32),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def start_camera(self) -> bool:
        """Open and start camera."""
        
        # Close camera if already open
        if self.cap:
            self.cap.release()
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
        
        # Camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 380)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return True
    
    def predict_frame(self) -> tuple[int | None, NDArray[np.number] | None]:
        """Capture frame and make prediction."""
        
        if self.cap is None:
            return None, None
        
        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        try:
            
            # Convert image to RGB, tensor, and transform
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = tv_tensors.Image(frame).permute(2, 0, 1)
            image = self.transform(image).unsqueeze(0).to(self.model.device)
            
            # Make prediction
            with torch.no_grad():
                pred = round(self.model(image).item())
            
            # Save current occupancy and time
            time = datetime.now()
            self.data.append((time, pred))
            
            return pred, frame
            
        except Exception as e:
            st.error(f"Error: {e}")
            return None, None
    
    def plot_occupancy(self) -> go.Figure:
        """Plot occupancy over time."""
        
        # Create figure
        fig = go.Figure()
        
        if self.data:
            times, preds = zip(*self.data)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=preds,
                mode='lines',
                line_color='#dd7878',
                name="Occupancy",
            ))
        
        fig.update_layout(
            title="Real-Time Occupancy Data",
            xaxis_title="Time",
            yaxis_title="Occupancy",
            margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
            height=300,
        )
        
        return fig
    
    def stop_camera(self) -> None:
        """Stop camera."""
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        return
    
    def clear_data(self) -> None:
        """Clear stored data."""
        
        self.data.clear()
        
        return
