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
import numpy as np
import streamlit as st
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection

# Files used
import ODS.config as config
from AMS.attendance_monitor import AttendanceMonitor
from DB import database
from DB.database import Database
from ODS import autoencoder_model, cnn_model
from ODS.autoencoder_model import AutoencoderModel
from ODS.cnn_model import CNNModel


def main() -> None:
    """Main entry point."""
    
    # Initialise session state variables
    if 'monitor' not in st.session_state:
        st.session_state.monitor = AttendanceMonitor(load_model())
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'database' not in st.session_state:
        st.session_state.database = load_database()
    if st.session_state.database.conn is None or not st.session_state.database.conn.is_connected():
        st.session_state.database.reset_connection()
    
    # Insert current camera and room IDs into database
    init_camera_room()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Camera", type="primary", icon=":material/video_camera_front:", use_container_width=True):
            if st.session_state.monitor.start_camera():
                st.session_state.camera_active = True
                st.toast("Camera started!", icon=":material/video_camera_front:")
                st.rerun()
            else:
                st.error("Error: Failed to start camera.")
    
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
    
    # Database updating fragment
    update_database()


@st.cache_resource(ttl=3600)
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


@st.cache_resource(ttl=3600)
def init_connection() -> MySQLConnectionAbstract | PooledMySQLConnection:
    return database.init_connection()

@st.cache_resource(ttl=3600)
def load_database() -> Database:
    return Database(True)

@st.cache_data(ttl=3600)
def run_query(query: str, values: tuple = (), commit: bool = False) -> None:
    database.run_query(st.session_state.database.conn, query, values, commit)
    return


def init_camera_room() -> None:
    """Insert camera device ID and room ID into database."""
    
    db = st.session_state.database
    camera_id = str(st.session_state.monitor.camera_id + 1) # ID 0 triggers autoincrement. Fix in future.
    
    _, rooms = db.query_database("SELECT RoomID FROM rooms WHERE RoomId = 123;")[0]
    _, cameras = db.query_database(f"SELECT CameraID FROM cameras WHERE CameraID = {camera_id};")[0]
    
    if rooms.empty: # Insert arbitrary room for demo
        run_query(f"INSERT INTO rooms (RoomID, RoomName, Capacity) VALUES (123, 'NewRoom', 50);", commit=True)
    if cameras.empty:
        run_query(f"INSERT INTO cameras (CameraID, RoomID) VALUES ({camera_id}, 123);", commit=True)
    
    return


# Update every 0.5 seconds
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
    
    return


# Update every 5 seconds
@st.fragment(run_every=5)
def update_database(autocommit: bool = False) -> None:
    
    if 'database' not in st.session_state or not st.session_state.database.conn.is_connected():
        st.error("Error: Database not connected.")
        return
    
    if st.session_state.monitor.data:
        time, pred = st.session_state.monitor.data[-1]
    
        try:
            run_query(
                "INSERT INTO occupancyrecords (CameraID, RoomID, Timestamp, OccupancyCount) VALUES (%s, %s, %s, %s);",
                (st.session_state.monitor.camera_id + 1, 123, time, pred),
                commit=autocommit,
            )
        
        except:
            st.warning("Error: Could not insert occupancy record.")
        
        return


if __name__ == "__main__":
    main()