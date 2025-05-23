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
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
from streamlit.runtime import uploaded_file_manager as ufm
from streamlit_webrtc import webrtc_streamer
from torchvision import tv_tensors
from torchvision.io import decode_image
from typing import Literal

# Files used
import ODS.config as config
from ODS import autoencoder_model, cnn_model
from ODS.autoencoder_model import AutoencoderModel
from ODS.cnn_model import CNNModel


class StreamlitCameraPredictor:
    """Streamlit real-time camera predictor class."""
    
    def __init__(self, model: AutoencoderModel | CNNModel, camera_id: int = 0, max_points: int = 100) -> None:
        """
        Streamlit camera predictor constructor.
        
        Parameters
        ----------
        model : Autoencoder | CNN
            Model to use for prediction.
        camera_id : int, optional, default=0
            Camera device ID.
        max_points : int, optional, default=100
            _description_.
        """
        
        self.model = model
        self.camera_id = camera_id
        self.max_points = max_points
        
        # Data storage
        self.timestamps = deque(maxlen=max_points)
        self.predictions = deque(maxlen=max_points)
        
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
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def capture_frames(self) -> None:
        """Capture frames in a separate thread."""
        
        while self.running and self.cap is not None:
            
            ret, frame = self.cap.read()
            
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
            time.sleep(0.033)  # ~30 FPS
    
    def predict_frame(self, frame):
        """Process frame and make prediction"""
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and preprocess
            image = tv_tensors.Image(Image.fromarray(frame_rgb, mode='RGB'))
            image_tensor = self.transform(image).unsqueeze(0).to(self.model.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                pred = round(output.item())
            
            return pred, frame_rgb
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None
    
    def add_prediction(self, prediction):
        """Add new prediction to data storage."""
        
        current_time = datetime.now()
        self.timestamps.append(current_time)
        self.predictions.append(prediction)
    
    def create_plot(self):
        """Create Plotly line chart."""
        
        # Create figure
        fig = go.Figure()
        
        if len(self.predictions) == 0:
            
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Predictions'))
        
        else:
            
            fig.add_trace(go.Scatter(
                x=list(self.timestamps),
                y=list(self.predictions),
                mode='lines+markers',
                name='Occupancy Count',
                line={'color': '#1f77b4', 'width': 2},
                marker={'size': 6},
            ))
        
        fig.update_layout(
            title="Real-time Occupancy Predictions",
            xaxis_title="Time",
            yaxis_title="Count",
            height=400,
            showlegend=True,
            xaxis={'tickformat': '%H:%M:%S', 'showgrid': True},
            yaxis={'showgrid': True, 'zeroline': True},
        )
        
        return fig
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

def main() -> None:
    """Main entry point."""
    
    # Sidebar controls
    st.sidebar.subheader("Settings")
    camera_id = st.sidebar.selectbox("Camera ID", [0, 1, 2], index=0)
    max_points = st.sidebar.slider("Max Data Points", 50, 500, 100)
    update_interval = st.sidebar.slider("Update Interval (Seconds)", 0.1, 2.0, 0.5)
    
    # Initialise session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'capture_thread' not in st.session_state:
        st.session_state.capture_thread = None
    
    # Load model
    #@st.cache_resource
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
            return autoencoder_model.load_model()
        
        else:
            return cnn_model.load_model()
    model = load_trained_model('cnn')
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(":material/videocam: Start Camera", type="primary"):
            if st.session_state.predictor is None:
                st.session_state.predictor = StreamlitCameraPredictor(
                    model, camera_id, max_points,
                )
            
            if st.session_state.predictor.initialise_camera():
                st.session_state.predictor.running = True
                st.session_state.capture_thread = threading.Thread(
                    target=st.session_state.predictor.capture_frames,
                    daemon=True,
                )
                st.session_state.capture_thread.start()
                st.success("Camera started successfully!")
            else:
                st.error("Failed to initialise camera")
    
    with col2:
        if st.button(":material/videocam_off: Stop Camera"):
            if st.session_state.predictor is not None:
                st.session_state.predictor.cleanup()
                st.session_state.predictor = None
                st.session_state.capture_thread = None
                st.success("Camera stopped")
    
    with col3:
        if st.button(":material/delete: Clear Data"):
            if st.session_state.predictor is not None:
                st.session_state.predictor.timestamps.clear()
                st.session_state.predictor.predictions.clear()
                st.success("Data cleared")
    
    # Main display area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Camera Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("Live Predictions Chart")
        chart_placeholder = st.empty()
    
    # Statistics display
    stats_placeholder = st.empty()
    
    # Real-time processing loop
    if st.session_state.predictor is not None and st.session_state.predictor.running:
        while st.session_state.predictor.running:
            try:
                # Get latest frame
                frame = st.session_state.predictor.frame_queue.get(timeout=1.0)
                
                # Make prediction
                prediction, frame_rgb = st.session_state.predictor.predict_frame(frame)
                
                if prediction and frame_rgb is not None:
                    
                    # Add prediction to data
                    st.session_state.predictor.add_prediction(prediction)
                    
                    # Add prediction text to frame
                    frame_display = frame_rgb.copy()
                    cv2.putText(
                        frame_display,
                        f"Count: {prediction}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    
                    # Display frame
                    video_placeholder.image(
                        frame_display,
                        channels="RGB",
                        use_container_width=True,
                    )
                    
                    # Update chart
                    fig = st.session_state.predictor.create_plot()
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    # Update statistics
                    if len(st.session_state.predictor.predictions) > 0:
                        recent_preds = list(st.session_state.predictor.predictions)
                        stats_placeholder.metric(
                            label="Current Count",
                            value=prediction,
                            delta=f"Avg: {np.mean(recent_preds):.1f}",
                        )
                
                time.sleep(update_interval)
                
            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                break


main()