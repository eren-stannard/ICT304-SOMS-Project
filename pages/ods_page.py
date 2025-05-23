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
    if 'save_train_params' not in st.session_state:
        st.session_state.save_train_params = False
    if 'start_training' not in st.session_state:
        st.session_state.start_training = False
    
    # Arguments
    args = {
        'mode': None,
        'model_path': config.MODEL_PATH,
        'batch_size': config.BATCH_SIZE,
        'epoch': config.NUM_EPOCHS,
        'lr': config.LEARNING_RATE,
        'train_ratio': config.TRAIN_RATIO,
        'image': None,
        'camera': False,
        'camera_id': 0,
    }
    
    # Get operation mode input
    args['mode'] = get_mode_input()
    
    # Execute mode according to input option
    run = False
    
    if args['mode'] == 'train':
        
        # Get training parameters/arguments input
        args.update(get_train_input())
        
        col1, _, col2 = st.columns(3, gap='large')
        
        # Save training params and disable widget
        with col1:
            if save_train_params_btn():
                st.session_state.save_train_params=True
        
        with col2:
            run = start_training_btn()
    
    elif args['mode'] == 'evaluate':
        
        run = True # st.button("Evaluate Model", icon=":material/play_circle:")
        
    elif args['mode'] == 'predict':
        
        pred_option = get_predict_input()
        
        if pred_option == 'image':
            
            # Upload image file for prediction
            args['image'] = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
            
            if args['image'] is not None:
                run = True
        
        elif pred_option == 'camera':
            
            args['camera'] = True
            run= True
    
    if run:
        forward_ods(**args)
        st.session_state.save_train_params = False
        st.session_state.start_training = False
    
    return


def get_mode_input() -> Literal['train', 'evaluate', 'predict'] | str | None:
    """
    Display segmented control widget for user input.
    Get input for system operation mode selection (train, evaluate, or predict).

    Returns
    -------
    mode : Literal['train', 'evaluate', 'predict'] | None
        Selected mode.
    """
    
    # Mode options
    option_map = {
        'train': ":material/model_training: Train",
        'evaluate': ":material/readiness_score: Evaluate",
        'predict': ":material/batch_prediction: Predict",
    }

    # Input selection widget
    help_str = (
        f"**{option_map['train']}**: Train model  \n" +
        f"**{option_map['evaluate']}**: Evaluate trained model  \n" +
        f"**{option_map['predict']}**: Predict occupancy from input image data"
    )
    mode = st.segmented_control(
        label="Select operation mode:",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        help=help_str,
    )
    
    return mode


def get_train_input() -> dict[str, float | int]:
    """
    Display widget for user input.
    Get input for model training parameters/arguments selection.

    Returns
    -------
    params : dict[str, float | int]
        Selected training parameters.
    
    See Also
    --------
    ODS/config.py : Default ODS parameter configuration.
    """
    
    # Training parameter arguments and config defaults
    params = {
        'batch_size': config.BATCH_SIZE,
        'epoch': config.NUM_EPOCHS,
        'lr': config.LEARNING_RATE,
        'train_ratio': config.TRAIN_RATIO,
    }
    
    st.write(":primary[:material/settings:] Enter model training parameters:")
    cols = st.columns(4)
    params['batch_size'] = cols[0].number_input(
        label="Batch Size",
        min_value=1,
        max_value=1024,
        value=params['batch_size'],
        help="Number of data samples to load per batch",
        disabled=st.session_state.save_train_params,
        icon=":material/stacks:",
    )
    params['epoch'] = cols[1].number_input(
        label="Epoch",
        min_value=1,
        value=params['epoch'],
        help="Number of iterations for training model",
        disabled=st.session_state.save_train_params,
        icon=":material/laps:",
    )
    params['lr'] = cols[2].number_input(
        label="Learning Rate",
        min_value=0.000001,
        value=params['lr'],
        step=0.000001,
        format='%0.6f',
        help="How much model parameters are updated after each training batch",
        disabled=st.session_state.save_train_params,
        icon=":material/speed:",
    )
    params['train_ratio'] = cols[3].number_input(
        label="Train Size",
        min_value=0.5,
        value=params['train_ratio'],
        step=0.05,
        format='%0.2f',
        help="Proportion of data samples to use for training dataset",
        disabled=st.session_state.save_train_params,
        icon=":material/percent:",
    )
    
    return params


def save_train_params_btn() -> bool:
    """
    Display button widget for user input.
    Click button to save model training parameters.

    Returns
    -------
    save : bool
        If True, save model training parameters.
    """
    
    # Disable parameter editing if saved. Adapted from Shawn_Pereira's (2023) solution at:
    # https://discuss.streamlit.io/t/how-to-disable-a-preceding-widget-using-a-button-without-clicking-it-twice/35367/2
    st.button(
        label="Selection Saved" if st.session_state.save_train_params else "Confirm Selection",
        key="launch",
        on_click=btn_callbk,
        kwargs={'key': 'save_train_params'},
        icon=":material/save:",
        disabled=st.session_state.save_train_params,
        use_container_width=True,
    )
    
    return st.session_state.save_train_params


def get_predict_input() -> Literal['image', 'camera'] | str | None:
    """
    Display radio buttons widget for user input.
    Get input for prediction mode selection (image or camera feed).

    Returns
    -------
    pred_mode : Literal['image', 'camera'] | None
        Selected prediction mode option.
    """
    
    # Mode options
    option_map = {
        'image': ":material/add_a_photo: Image",
        'camera': ":material/video_camera_back: Camera Feed",
    }
    
    # Get prediction mode selection input
    pred_mode = st.segmented_control(
        label="Prediction mode:",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
    )
    
    return pred_mode


def start_training_btn() -> bool:
    """
    Display button widget for user input.
    Click button to start model training with selected parameters.

    Returns
    -------
    train : bool
        If True, start model training.
    """
    
    st.button(
        label="Model Training" if st.session_state.start_training else "Train Model",
        on_click=btn_callbk,
        kwargs={'key': 'start_training'},
        type='primary',
        icon=":material/play_circle:",
        disabled=st.session_state.start_training or not st.session_state.save_train_params,
        use_container_width=True,
    )
    
    return st.session_state.start_training


def btn_callbk(key: str) -> None:
    """Button click callback function."""
    
    st.session_state[key] = not st.session_state[key]
    
    return


if __name__ == "__main__":
    
    # Get operation mode selection input
    main()