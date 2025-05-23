# main.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System

    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)

    ICT304: AI System Design
    Murdoch University

    Purpose of File:
    Main entry point.

"""


# Libraries used
import os
import streamlit as st
from typing import Literal

# Files used
import ODS.config as config
from ODS.cnn_model import load_model
from ODS.evaluate import evaluate_model, plot_results
from ODS.predict import predict_camera, predict_image
from ODS.train import train_model


def main(
    mode: Literal['train', 'evaluate', 'predict'],
    model_path: str = config.MODEL_PATH,
    batch_size: int = config.BATCH_SIZE,
    epoch: int = config.NUM_EPOCHS,
    lr: float = config.LEARNING_RATE,
    train_ratio: float = config.TRAIN_RATIO,
    image: str | None = None,
    camera: bool = False,
    camera_id: int = 0,
) -> None:
    """
    Main entry point.
    
    Parameters
    ----------
    mode : Literal['train', 'evaluate', 'predict']
        System operation mode (train, evaluate, or predict).
    model_path : str, optional, default=MODEL_PATH
        Path to saved trained model file.
    batch_size : int, optional, default=BATCH_SIZE
        Batch size for data loading.
    epoch : int, optional, default=NUM_EPOCHS
        Number of epochs.
    lr : float, optional, default=LEARNING_RATE
        Learning rate for updating model parameters.
    train_ratio : float, optional, default=TRAIN_RATIO
        Proportion of data samples to use for training dataset.
    image : str | None, optional, default=None
        Path to input image for prediction.
    camera : bool, optional, default=False
        If True, use camera for real-time prediction.
    camera_id : int, optional, default=0
        Camera device ID.
    
    See Also
    --------
    config.py : Default parameter configuration.
    """
    
    # Update config if args specified
    if batch_size:
        config.BATCH_SIZE = batch_size
    if epoch:
        config.NUM_EPOCHS = epoch
    if lr:
        config.LEARNING_RATE = lr
    if train_ratio:
        config.TRAIN_RATIO = train_ratio

    # Execute based on mode
    if mode == 'train':
        
        # Train model
        model_path = train_model(train_ratio)
        st.toast(f"Training complete! Model saved to: **{model_path}**", icon=":material/check:")
    
    else:

        # Check if model exists
        if not os.path.exists(model_path):
            
            st.error(
                f"Error: Model file not found at **{model_path}**\n" +
                "Please train a model first or specify a valid model path."
            )
        
        else:
            
            # Load model
            model = load_model(model_path)

            if mode == 'evaluate':
                
                # Evaluate model
                results = evaluate_model(
                    model,
                    config.DATA_DIR,
                    config.TEST_LABELS_FILE,
                    config.BATCH_SIZE,
                )

                # Plot results
                plot_results(results, model)
            
            else:

                if image:
                    
                    # Predict single image
                    pred = predict_image(model, image)
                    
                    if pred:
                        st.write(f"Prediction: :primary[**{pred}**] people")
                        st.image(image)

                elif camera:
                    
                    # Predict from camera feed
                    pred = predict_camera(model, camera_id)
                    
                    if pred:
                        st.write(f"Prediction: :primary[**{pred}**] people")
        
    return