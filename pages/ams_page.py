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
import ODS.config as config
from ODS import autoencoder_model, cnn_model
from ODS.autoencoder_model import AutoencoderModel
from ODS.cnn_model import CNNModel


class AttendanceMonitor:
    """Real-time attendance monitor class."""
    
    def __init__(self, model: AutoencoderModel | CNNModel, camera_id: int = 0) -> None:
        """
        Attendance monitor constructor.
        
        Parameters
        ----------
        model : AutoencoderModel | CNNModel
            Model to use for prediction.
        camera_id : int, optional, default=0
            Camera device to use.
        """
        
        self.model = model
        self.camera_id = camera_id
        self.cap = None
        
        # Occupancy data predictions and associated timestamps
        self.data = deque(maxlen=100)
        
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
            try:
                for id in range(-1, 10):
                    self.cap.release()
                    self.cap = None
                    print(f"Trying camera_id {id}...")
                    self.cap = cv2.VideoCapture(id)
                    if self.cap.isOpened():
                        print("Success!")
                        self.camera_id = id
                        break
            except:
                st.error("Failed to locate camera")
            
            return False
        
        # Camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
            self.data.append((datetime.now(), pred))
            
            return pred, frame
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
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


@st.cache_resource
def load_model(model_type: str = config.MODEL_TYPE) -> AutoencoderModel | CNNModel:
    """
    Load trained model.
    
    Parameters
    ----------
    model_type : str, optional, default=MODEL_TYPE
        Type of model to use.
    
    See Also
    --------
    ODS/config.py : Default ODS parameter configuration.
    """
    
    if model_type == 'autoencoder':
        return autoencoder_model.load_model(show_log=False)
    
    else:
        return cnn_model.load_model(show_log=False)


# Rerun every 0.5 seconds
@st.fragment(run_every=0.5)
def camera_stream() -> None:
    """Camera stream fragment. Updates independently from rest of script."""
    
    if 'monitor' not in st.session_state or st.session_state.camera_active == False:
        st.container(height=300, border=True).write(":grey[Start camera to begin monitoring]")
        return
    
    monitor = st.session_state.monitor
    pred, frame = monitor.predict_frame()
    
    if monitor is not None and frame is not None:
        col1, col2 = st.columns([3, 1])
        
        # Display frame
        with col1:
            st.image(frame, channels='RGB', use_container_width=True)
        
        # Display currrent occupancy
        with col2:
            _, recent_preds = zip(*monitor.data)
            avg_occupancy = np.mean(recent_preds) if recent_preds else 0
            st.metric("Current Occupancy", pred, delta=f"Avg: {avg_occupancy:.1f}")
        
        # Update time-series plot
        if monitor.data:
            fig = monitor.plot_occupancy()
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main entry point."""
    
    # Initialise session state variables
    if 'monitor' not in st.session_state:
        st.session_state.monitor = AttendanceMonitor(load_model())
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Camera", type="primary", icon=":material/video_camera_front:", use_container_width=True):
            if st.session_state.monitor.start_camera():
                st.session_state.camera_active = True
                st.toast("Camera started!", icon=":material/video_camera_front:")
                st.rerun()
            else:
                st.error("Failed to start camera")
    
    with col2:
        if st.button("Stop Camera", icon=":material/video_camera_front_off:", use_container_width=True):
            st.session_state.monitor.stop_camera()
            st.session_state.camera_active = False
            st.toast("Camera stopped", icon=":material/video_camera_front_off:")
            st.rerun()
    
    with col3:
        if st.button("Clear Data", icon=":material/delete_sweep:", use_container_width=True):
            st.session_state.monitor.clear_data()
            st.toast("Data cleared", icon=":material/delete_sweep:")
    
    # Camera streaming fragment
    camera_stream()


if __name__ == "__main__":
    main()