# train.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)

    ICT304: AI System Design
    Murdoch University

    Purpose of File:
    Model training and optimisation.

"""


# Libraries used
import os
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from stqdm import stqdm
from torch import nn, optim

# Files used
import ODS.config as config
from ODS.autoencoder_model import AutoencoderModel
from ODS.cnn_model import CNNModel
from ODS.data_loader import get_data_loader


# To fix torch/streamlit conflict error. Adapted from Espen's (2025) solution at:
# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/5
torch.classes.__path__ = []


def train_model(train_ratio: float = config.TRAIN_RATIO) -> str:
    """
    Train model for occupancy detection.
    
    Parameters
    ----------
    train_ratio : float, optional, default=TRAIN_RATIO
        Proportion of samples to use for training set.

    Returns
    -------
    model_path : str
        Path to saved trained model.
    
    See Also
    --------
    config.py : Default configuration parameters.
    """
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Create model
    #model = AutoencoderModel(latent_dim=config.AUTOENCODER_LATENT_DIM)
    model = CNNModel()
    flatten = nn.Flatten()
    
    # Get data loaders
    dataset_status = st.status(label="Loading dataset...", state='running')
    train_loader, val_loader = get_data_loader(
        train_ratio=train_ratio,
        mode='train',
    )
    train_size, val_size = len(train_loader.dataset), len(val_loader.dataset) # type: ignore
    dataset_status.update(
        label=(
            f"Dataset loaded. " +
            f"Training set: **{train_size}** samples, " +
            f"Validation set: **{val_size}** samples"
        ),
        state='complete',
    )

    # Define loss functions for prediction loss and reconstruction loss and optimiser
    pred_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    losses: dict[str, list[float | int | str]] = {
        'Epoch': [],
        'Loss': [],
        'Loop': [],
    }
    best_val_loss = float('inf')
    model_path: str = config.MODEL_PATH
    
    # Training status
    c = st.container()
    prog_msg = f":primary[**{model.__class__.__name__}**] model (using device: :primary[**{model.device}**])"
    train_status = c.status(label=f"Training {prog_msg}", state='running')
    expander = st.expander("Epoch Results", icon=":material/query_stats:")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        
        # Training
        model.train()
        train_loss = 0.0

        epoch_str = f"Epoch **{epoch}/{config.NUM_EPOCHS}**"
        train_status.update(
            label=f"Training {prog_msg}... {epoch_str}",
            state='running',
        )
        
        # Use stqdm for progress bar (Streamlit tqdm)
        for x, y in stqdm(train_loader, desc=f"Training ({epoch_str})", unit='**batch**', st_container=c):
            
            # Forward pass
            #x = flatten(x)
            #reconstruction, _, count_pred = model.full_forward(x)
            count_pred = model(x)
            
            # Compute loss
            #recon_loss = recon_criterion(reconstruction, x)
            #pred_loss = 
            total_loss = pred_criterion(count_pred, y)#(recon_loss / x.numel()) + (pred_loss / y.numel())

            # Backward pass and optimise
            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()

            train_loss += total_loss.item() * x.size(0)
        
        # Calculate average loss
        train_loss = train_loss / len(train_loader.dataset) # type: ignore
        
        losses['Epoch'].append(epoch)
        losses['Loss'].append(train_loss)
        losses['Loop'].append('Training')

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            
            for x, y in stqdm(val_loader, desc=f"Validating ({epoch_str})", unit='**batch**', st_container=c):
                
                # Forward pass
                #x = flatten(x)
                #reconstruction, _, count_pred = model.full_forward(x)
                count_pred = model(x)
                
                # Compute loss
                #recon_loss = recon_criterion(reconstruction, x)
                #pred_loss = 
                total_loss = pred_criterion(count_pred, y)#(recon_loss / x.numel()) + (pred_loss / y.numel())

                val_loss += total_loss.item() * x.size(0)
        
        val_loss = val_loss / len(val_loader.dataset) # type: ignore
        
        losses['Epoch'].append(epoch)
        losses['Loss'].append(val_loss)
        losses['Loop'].append('Validation')
        
        # Display epoch results
        result = (
            f":primary-badge[:material/check: {epoch_str}] " +
            f"Training loss: **{train_loss:.3f}**, " +
            f"Validation loss: **{val_loss:.3f}**"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            expander.write(f"{result} :green-badge[**New best model saved!**]")
        
        else:
            expander.write(result)
    
    train_status.update(label=f":primary[**{model.__class__.__name__}**] training complete.", state='complete')
    
    # Plot learning curve
    plot_learning_curve(losses, model)
    
    return model_path


def plot_learning_curve(losses: dict[str, list[float | int | str]], model: AutoencoderModel | CNNModel) -> None:
    """
    Plot model learning curve (training vs. validation losses).

    Parameters
    losses : dict[str, float]
        Dict of model training and validation losses at each epoch.
    """
    
    losses_df: pd.DataFrame = pd.DataFrame(losses).sort_values(by=['Epoch', 'Loop'], ignore_index=True)
    
    fig= px.line(
        losses_df,
        x='Epoch',
        y='Loss',
        color='Loop',
        color_discrete_sequence=config.COLOUR_PAIR,
        title=f"{model.__class__.__name__} Learning Curve - Training Versus Validation Loss",
    )
    
    st.plotly_chart(fig, use_container_width=True)