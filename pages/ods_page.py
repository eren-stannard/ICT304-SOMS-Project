# ods_page.py

"""
    Smart Occupancy Monitoring System - Client Application
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Page for Occupancy Detection AI Sub-System (ODS).

"""


# Libraries used
import streamlit as st
from typing import Literal

# Files used
import ODS.config as config
from ODS.main import main as forward_ods


def main() -> None:
    """Main entry point."""
    
    # Initialise session state variables
    if 'training_confirmed' not in st.session_state:
        st.session_state.training_confirmed = False
    
    # Get operation mode input
    mode = get_mode()
    
    if not mode:
        return
    
    # Arguments
    args = {
        'mode': mode,
        'model_path': config.MODEL_PATH,
        'batch_size': config.BATCH_SIZE,
        'epoch': config.NUM_EPOCHS,
        'lr': config.LEARNING_RATE,
        'train_ratio': config.TRAIN_RATIO,
        'image': None,
        'camera': False,
        'camera_id': 0,
    }
    
    # Execute based on mode
    if mode == 'train':
        training_mode(**args)
    elif mode == 'evaluate':
        forward_ods(**args)
    elif mode == 'predict':
        prediction_mode(**args)
    
    return


def get_mode() -> Literal['train', 'evaluate', 'predict'] | str | None:
    """
    Display segmented control widget for user input.
    Get input for system operation mode selection (train, evaluate, or predict).

    Returns
    -------
    mode : Literal['train', 'evaluate', 'predict'] | None
        Selected operation mode.
    """
    
    # Mode options
    option_map = {
        'train': ":material/model_training: Train Model",
        'evaluate': ":material/readiness_score: Evaluate Model",
        'predict': ":material/batch_prediction: Make Predictions",
    }

    # Help string
    help_str = (
        f"**{option_map['train']}**: Train model  \n" +
        f"**{option_map['evaluate']}**: Evaluate trained model performance  \n" +
        f"**{option_map['predict']}**: Predict occupancy from input image"
    )
    
    # Input selection widget
    return st.segmented_control(
        label="Select operation mode:",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        help=help_str,
    )


def training_mode(**kwargs) -> None:
    """
    Display widget for user input.
    Get input for model training parameters/arguments selection.

    Parameters
    ----------
    **kwargs : dict[str, float | int]
        Training parameters.
    
    See Also
    --------
    ODS/config.py : Default ODS parameter configuration.
    """
    
    # Get training parameter input
    st.write(":primary[:material/settings:] Enter model training parameters:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        kwargs['batch_size'] =st.number_input(
            label="Batch Size",
            min_value=1,
            max_value=1024,
            value=config.BATCH_SIZE,
            help="Number of data samples to load in each batch",
            disabled=st.session_state.training_confirmed,
            icon=":material/stacks:",
        )
    
    with col2:
        kwargs['epoch'] = st.number_input(
            label="Epoch",
            min_value=5,
            value=config.NUM_EPOCHS,
            help="Number of iterations for training model",
            disabled=st.session_state.training_confirmed,
            icon=":material/laps:",
        )
        
    with col3:
        kwargs['lr'] = st.number_input(
            label="Learning Rate",
            min_value=0.0001,
            value=config.LEARNING_RATE,
            step=0.0001,
            format='%0.4f',
            help="How much model parameters are updated after each training batch",
            disabled=st.session_state.training_confirmed,
            icon=":material/speed:",
        )
    
    with col4:
        kwargs['train_ratio'] = st.number_input(
            label="Train Size",
            min_value=0.5,
            max_value=0.99,
            value=config.TRAIN_RATIO,
            step=0.01,
            format='%0.2f',
            help="Proportion of data samples to use for training dataset",
            disabled=st.session_state.training_confirmed,
            icon=":material/percent:",
        )
    
    # Get training confirmation input
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            label="Confirm Parameters" if not st.session_state.training_confirmed else "Parameters Confirmed",
            icon=":material/save:",
            disabled=st.session_state.training_confirmed,
            use_container_width=True,
        ):
            st.session_state.training_confirmed = True
    
    with col2:
        start_training = st.button(
            label="Start Training",
            type='primary',
            icon=":material/play_circle:",
            disabled=not st.session_state.training_confirmed,
            use_container_width=True,
        )
    
    if start_training:
        with st.spinner("Training model...", show_time=True):
            forward_ods(**kwargs)
        
        st.session_state.training_confirmed = False
        st.toast("Training completed!", icon=":material/celebration:")
    
    return


def prediction_mode(**kwargs) -> None:
    """
    Display radio buttons widget for user input.
    Get input for prediction mode selection (image or camera feed).

    Parameters
    ----------
    **kwargs : dict[str, float | int]
        Prediction parameters.
    """
    
    # Mode options
    option_map = {
        'image': ":material/add_a_photo: Upload Image",
        'camera': ":material/video_camera_back: Use Camera",
    }
    
    # Get prediction mode selection input
    pred_mode = st.segmented_control(
        label="Select prediction input mode:",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
    )
    
    if not pred_mode:
        return
    
    # Predict occupancy from uploaded image file
    if pred_mode == 'image':
        
        img_file = st.file_uploader(
            label="Upload image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image file for prediction",
        )
        
        if img_file:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_file, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("Predict Occupancy", type='primary', icon=":material/frame_inspect:"):
                    kwargs['image'] = img_file
                    
                    with st.spinner("Predicting occupancy..."):
                        forward_ods(**kwargs)
    
    # Predict from camera capture frame
    elif pred_mode == 'camera':
        kwargs['camera'] = True
        forward_ods(**kwargs)

    return


if __name__ == "__main__":
    main()