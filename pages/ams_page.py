# ams_page.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Page for Attendance Monitoring Sub-System (AMS).

"""


# Libraries used
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import queue
import streamlit as st
import time
import threading
import torch
import torchvision.transforms.v2 as T
from collections import deque
from datetime import datetime
from numpy.typing import NDArray
from PIL import Image
from torchvision import tv_tensors
from typing import Literal

# Files used
import ODS.config as config
from ODS import autoencoder_model, cnn_model
from ODS.autoencoder_model import AutoencoderModel
from ODS.cnn_model import CNNModel


class StreamlitCameraPredictor:
    """Streamlit real-time camera predictor class."""
    
    def __init__(self, model: AutoencoderModel | CNNModel, camera_id: int = 0) -> None:
        """
        Streamlit camera predictor constructor.
        
        Parameters
        ----------
        model : Autoencoder | CNN
            Model to use for prediction.
        camera_id : int, optional, default=0
            Camera device ID.
        """
        
        self.model = model
        self.camera_id = camera_id
        
        # Data storage
        self.timestamps = deque(maxlen=150)
        self.predictions = deque(maxlen=150)
        
        # Camera and processing
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Transform for preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToDtype(torch.float32),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        return
    
    def initialise_camera(self) -> bool:
        """Initialise camera capture."""
        
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            
            st.error(f"Error: Could not open camera {self.camera_id}")
            
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 380)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def capture_frames(self) -> None:
        """Capture frames in a separate thread."""
        
        while self.running and self.cap is not None:
            
            ret, frame = self.cap.read()
            print(frame.dtype)
            if ret:
                
                # Keep only the latest frame
                if not self.frame_queue.empty():
                    
                    try:
                        self.frame_queue.get_nowait()
                    
                    except queue.Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait(frame)
                    
                except queue.Full:
                    pass
            
            time.sleep(0.01)
    
    def predict_frame(self, frame: NDArray[np.uint8]):
        """
        Process frame and make prediction.
        
        Parameters
        ----------
        frame 
        """
        
        try:
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # type: ignore
            
            # Convert to PIL image then to tensor and transform
            image = tv_tensors.Image(Image.fromarray(frame, mode='RGB'))
            image = self.transform(image).unsqueeze(0).to(self.model.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(image)
                pred = round(output.item())
            
            return pred, frame
        
        except Exception as e:
            
            st.error(f"Prediction error: {e}")
            
            return None, None
    
    def add_prediction(self, prediction: int) -> None:
        """
        Add new prediction to data storage.
        
        Parameters
        ----------
        prediction : int
            Occupancy prediction.
        """
        
        current_time = datetime.now()
        self.timestamps.append(current_time)
        self.predictions.append(prediction)
    
    def create_plot(self) -> go.Figure:
        """
        Create Plotly line chart.
        
        Returns : Figure
            Occupancy prediction plot.
        """
        
        # Create figure
        fig = go.Figure()
        
        if len(self.predictions) > 0:
            
            fig.add_trace(go.Scatter(
                x=list(self.timestamps),
                y=list(self.predictions),
                line={'color': '#1f77b4', 'width': 2},
                marker={'size': 6},
                mode='lines+markers',
                name="Occupancy Over Tims",
            ))
        
        fig.update_layout(
            margin={'b': 5, 'l': 5, 'r': 5, 't': 50},
            title="Real-time Occupancy Predictions",
            xaxis={'tickformat': '%H:%M:%S', 'showgrid': True},
            yaxis={'showgrid': True, 'zeroline': True},
            xaxis_title="Time",
            yaxis_title="Occupancy",
        )
        
        return fig
    
    def cleanup(self):
        """Clean up resources."""
        
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

def main() -> None:
    """Main entry point."""
    
    # Initialise session state variables
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'capture_thread' not in st.session_state:
        st.session_state.capture_thread = None
    
    # Load model
    model = load_trained_model('cnn')
    
    # Control buttons
    col1, col2 = st.columns([3, 1])
    
    with col2:
        
        if st.button("Start Camera", type="primary", icon=":material/videocam:", use_container_width=True):
            
            if st.session_state.predictor is None:
                st.session_state.predictor = StreamlitCameraPredictor(model, 0)
            
            if st.session_state.predictor.initialise_camera():
                st.session_state.predictor.running = True
                st.session_state.capture_thread = threading.Thread(
                    target=st.session_state.predictor.capture_frames,
                    daemon=True,
                )
                st.session_state.capture_thread.start()
                st.toast("Camera opened successfully!", icon=":material/videocam:")
            
            else:
                st.error("Failed to open camera")
        
        if st.button("Stop Camera", icon=":material/videocam_off:", use_container_width=True):
            
            if st.session_state.predictor is not None:
                st.session_state.predictor.cleanup()
                st.session_state.predictor = None
                st.session_state.capture_thread = None
                st.toast("Camera stopped", icon=":material/videocam_off:")
        
        if st.button("Clear Data", icon=":material/delete:", use_container_width=True):
            
            if st.session_state.predictor is not None:
                st.session_state.predictor.timestamps.clear()
                st.session_state.predictor.predictions.clear()
                st.toast("Data cleared", icon=":material/delete:")
    
    # Main display area placeholders
    video_placeholder = col1.empty()
    video_placeholder.container(height=152, border=True).write("Start camera stream to begin monitoring in real-time.")
    stats_placeholder = col2.empty()
    chart_placeholder = st.empty()
    
    # Real-time processing loop
    if st.session_state.predictor is not None and st.session_state.predictor.running:
        
        while st.session_state.predictor.running:
            
            try:
                
                # Get latest frame
                frame = st.session_state.predictor.frame_queue.get(timeout=1.0)
                
                # Make prediction
                prediction, frame = st.session_state.predictor.predict_frame(frame)
                
                if prediction and frame is not None:
                    
                    # Add prediction to data
                    st.session_state.predictor.add_prediction(prediction)
                    
                    # Display frame
                    video_placeholder.image(frame, channels='RGB', use_container_width=True)
                    
                    # Update chart
                    fig = st.session_state.predictor.create_plot()
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    # Update statistics
                    if len(st.session_state.predictor.predictions) > 0:
                        recent_preds = list(st.session_state.predictor.predictions)
                        stats_placeholder.metric(
                            label="Current Occupancy",
                            value=prediction,
                            delta=f"Avg: {np.mean(recent_preds):.1f}",
                        )
                
                time.sleep(0.5)
                
            except queue.Empty:
                continue
            
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                break


@st.cache_resource
def load_trained_model(model_type: Literal['autoencoder', 'cnn']) -> AutoencoderModel | CNNModel:
    """
    Model loading.
    
    Parameters
    ----------
    model_type : Literal['autoencoder', 'cnn']
        Type of model to load.
    
    Returns
    -------
    model : AutoencoderModel | CNNModel
        Model to use.
    """
    
    if model_type == 'autoencoder':
        return autoencoder_model.load_model(show_log=False)
    
    else:
        return cnn_model.load_model(show_log=False)


main()