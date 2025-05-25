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
import cv2.typing as cvt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import time
import threading
import torch
import torchvision.transforms.v2 as T
from collections import deque
from datetime import datetime
from PIL import Image
from torchvision import tv_tensors

# Files used
import ODS.config as config
from ODS import autoencoder_model, cnn_model
from ODS.autoencoder_model import AutoencoderModel
from ODS.cnn_model import CNNModel


class AttendanceMonitor:
    """Real-time attendance monitor."""
    
    def __init__(self, model: AutoencoderModel | CNNModel, camera_id: int = 0, max_points: int = 50) -> None:
        """
        Attendance monitor constructor.
        
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
        self.timestamps = deque(maxlen=max_points)
        self.predictions = deque(maxlen=max_points)
        
        # Camera and state
        self.cap = None
        
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
    
    def start_camera(self) -> bool:
        """Start camera feed capture."""
        
        # Release capture if already active
        if self.cap is not None:
            self.cap.release()
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
        
        # Set camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        return True
    
    def stop_camera(self) -> None:
        """Stop camera feed capture."""
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        return
    
    def get_predict_frame(self) -> tuple[int | None, cvt.MatLike | None]:
        """
        Capture frame and make prediction.
        
        Returns
        -------
        pred : int | None
            Predicted occupancy.
        frame : NDArray[uint8] | None
            Input image frame from camera feed.
        """
        
        # Return None if camera not open
        if not self.cap or not self.cap.isOpened():
            return None, None
        
        # Capture frame
        ret, frame = self.cap.read()
        
        # Return None if frame capture failed
        if not ret:
            return None, None
        
        try:
            
            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL image then to tensor
            image = tv_tensors.Image(Image.fromarray(frame, mode='RGB'))
            
            # Transform image
            image = self.transform(image).unsqueeze(0).to(self.model.device)
            
            # Make prediction
            with torch.no_grad():
                pred = round(self.model(image).item())
            
            # Update data
            self.timestamps.append(datetime.now())
            self.predictions.append(pred)
            
            return pred, frame
        
        except Exception as e:

            st.error(f"Prediction error: {e}")
            
            return None, None
    
    def plot_data_stream(self) -> go.Figure:
        """
        Plot occupancy over time.
        
        Returns : Figure
            Occupancy time-series plot.
        """
        
        # Create figure
        fig = go.Figure()
        
        if len(self.predictions) > 0:
            
            # Plot occupancy over time
            fig.add_trace(go.Scatter(
                x=list(self.timestamps),
                y=list(self.predictions),
                line_color=st.get_option("theme.primaryColor"),
                mode='lines+markers',
                name="Occupancy",
            ))
        
        fig.update_layout(
            margin={'b': 5, 'l': 5, 'r': 5, 't': 50},
            title="Real-Time Occupancy Data Stream",
            #xaxis={'tickformat': '%H:%M:%S', 'showgrid': True},
            #yaxis={'showgrid': True, 'zeroline': True},
            xaxis_title="Time (s)",
            yaxis_title="Occupancy",
        )
        
        return fig
    
    def clear_data(self) -> None:
        """Clear stored predictions."""
        
        self.timestamps.clear()
        self.predictions.clear()
        
        return


@st.cache_resource
def load_model(model_type: str = config.MODEL_TYPE) -> AutoencoderModel | CNNModel:
    """
    Model loading.
    
    Parameters
    ----------
    model_type : str, optional, default=MODEL_TYPE
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


# Auto-run every 0.5 seconds
@st.fragment(run_every=0.5)
def camera_feed() -> None:
    """Fragment that handles real-time camera updates."""
    
    if not st.session_state.get('camera_active', False):
        st.toast("Camera stopped")
        return
    
    monitor = st.session_state.get('monitor')
    if not monitor:
        st.error("No monitor available")
        return
    
    # Get frame and prediction
    frame, prediction = monitor.get_predict_frame()
    
    if frame is not None and prediction is not None:
        # Display video feed
        st.image(frame, channels='RGB', caption="Live Camera Feed", use_container_width=True)
        
        # Display current stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Occupancy", prediction)
        with col2:
            if monitor.predictions:
                avg = np.mean(list(monitor.predictions))
                st.metric("Average", f"{avg:.1f}")
    else:
        st.warning("No camera feed available")
    
    return


@st.fragment(run_every=1.0)
def occupancy_chart() -> None:
    """Fragment that handles chart updates."""
    
    monitor = st.session_state.get('monitor')
    
    if monitor and monitor.predictions:
        fig = monitor.create_plot()
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📊 Occupancy data will appear here once monitoring starts")

    return


def main() -> None:
    """Main entry point."""
    
    # Initialise session state variables
    if 'monitor' not in st.session_state:
        st.session_state.monitor = None
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = None
    
    # Load model
    model = load_model()
    
    # Control buttons
    col1, col2 = st.columns([3, 1])
    
    with col2:
        
        if st.button("Start", type="primary", icon=":material/videocam:", use_container_width=True):
            
            if st.session_state.monitor is None:
                st.session_state.monitor = AttendanceMonitor(model)
            
            if st.session_state.monitor.start_camera():
                st.session_state.camera_on = True
                st.toast("Camera opened successfully!", icon=":material/videocam:")
            
            else:
                st.error("Failed to open camera")
        
        if st.button("Stop", icon=":material/videocam_off:", use_container_width=True):
            
            if st.session_state.monitor is not None:
                st.session_state.monitor.stop_camera()
                st.session_state.camera_on = False
                st.toast("Camera closed", icon=":material/videocam_off:")
        
        if st.button("Clear", icon=":material/delete:", use_container_width=True):
            
            if st.session_state.monitor is not None:
                st.session_state.monitor.clear_data()
                st.toast("Data cleared", icon=":material/delete:")
    
    camera_feed()
    occupancy_chart()
    
    """
    # Main display area placeholders
    video_placeholder = col1.empty()
    chart_placeholder = st.empty()
    stats_col1, stats_col2 = st.columns(2)
    
    # Real-time processing loop
    if st.session_state.camera_on and st.session_state.monitor:
        
        pred, frame = st.session_state.monitor.get_predict_frame()
        
        if pred and frame is not None:
            
            # Display frame
            video_placeholder.image(frame, channels='RGB', use_container_width=True)
            
            # Update time-series plot
            fig = st.session_state.monitor.plot_data_stream()
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Display stats
            recent_preds = list(st.session_state.monitor.predictions)
            with stats_col1:
                st.metric("Current", pred)
            with stats_col2:
                st.metric("Average", f"{np.mean(recent_preds):.1f}")
            
            time.sleep(0.5)
            st.rerun()
            
        else:
            
            video_placeholder.write("Loading camera feed...")
    
    else:
        video_placeholder.info("Click 'Start' to begin real-time attendance monitoring")
    """
    
    return


if __name__ == '__main__':
    main()