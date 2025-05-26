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
import re
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


def train_model(
    model_type: str = config.MODEL_TYPE, batch_size: int = config.BATCH_SIZE, num_epochs: int = config.NUM_EPOCHS,
    lr: float = config.LEARNING_RATE, wd: float = config.WEIGHT_DECAY, train_ratio: float = config.TRAIN_RATIO,
) -> str:
    """
    Train model for occupancy detection.
    
    Parameters
    ----------
    model_type : str, optional, default=MODEL_TYPE
        Type of model to train.
    batch_size : int, optional, default=BATCH_SIZE
        Batch size to use.
    num_epochs : int, optional, default=NUM_EPOCHS
        Number of epochs.
    lr : float, optional, default=LEARNING_RATE
        How much model parameters are updated after each batch.
    wd : float, optional, default=WEIGHT_DECAY
        Weight decay.
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
    if model_type == 'autoencoder':
        model = AutoencoderModel(latent_dim=config.AUTOENCODER_LATENT_DIM)
    else:
        model = CNNModel()
    
    # Model name str
    model_name = " ".join(re.split(r"(?=Model)", model.__class__.__name__))
    
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
    #recon_criterion = nn.MSELoss() # For Autoencoder
    pred_criterion = nn.MSELoss()
    
    # Current learning rate
    current_lr = lr
    
    # Optimiser
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs)
    
    # Training loop
    losses: dict[str, list[float | int | str]] = {
        'Epoch': [],
        'Loss': [],
        'Loop': [],
    }
    best_val_loss = float('inf')
    best_model_epoch = 0
    model_path: str = config.MODEL_PATH
    
    # Training status
    c = st.container()
    prog_msg = f":primary[**{model_name}**] model (using device: :primary[**{model.device}**])"
    train_status = c.status(label=f"Training {prog_msg}", state='running')
    expander = st.expander("Epoch Results", icon=":material/query_stats:")
    
    for epoch in range(1, num_epochs + 1):
        
        # Training
        model.train()
        train_loss = 0.0

        epoch_str = f"Epoch **{epoch}/{num_epochs}**"
        train_status.update(
            label=f"Training {prog_msg}... {epoch_str}",
            state='running',
        )
        
        # Use stqdm for progress bar (Streamlit tqdm)
        for x, y in stqdm(train_loader, desc=f"Training ({epoch_str})", unit='**batch**', st_container=c):
            
            # Forward pass (Autoencoder)
            #x = flatten(x)
            #reconstruction, _, count_pred = model.full_forward(x)

            # Forward pass (CNN)
            count_pred = model(x)
            
            # Compute loss (Autoencoder)
            #recon_loss = recon_criterion(reconstruction, x)
            #pred_loss = pred_criterion(count_pred, y)
            #total_loss = (recon_loss / x.numel()) + (pred_loss / y.numel())

            # Compute loss (CNN)
            total_loss = pred_criterion(count_pred, y)

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

                # Forward pass (Autoencoder)
                #x = flatten(x)
                #reconstruction, _, count_pred = model.full_forward(x)
                
                # Forward pass (CNN)
                count_pred = model(x)

                # Compute loss (Autoencoder)
                #recon_loss = recon_criterion(reconstruction, x)
                #pred_loss = pred_criterion(count_pred, y)
                #total_loss = (recon_loss / x.numel()) + (pred_loss / y.numel())
                
                # Compute loss (CNN)
                total_loss = pred_criterion(count_pred, y)

                val_loss += total_loss.item() * x.size(0)

        val_loss = val_loss / len(val_loader.dataset) # type: ignore

        # Update LR scheduler
        scheduler.step()

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
            best_model_epoch = epoch
            torch.save(model.state_dict(), model_path)
            result += " :green-badge[:material/star: New best model saved!]"
        
        elif scheduler.get_last_lr()[0] < current_lr:
            current_lr = scheduler.get_last_lr()[0]
            result += f" :orange-badge[:material/arrow_cool_down: LR reduced: **{current_lr:.4f}**]"
        
        expander.write(result)
    
    train_status.update(label=f":primary[**{model_name}**] training complete.", state='complete')
    
    # Plot learning curve
    plot_learning_curve(losses, model_name, best_model_epoch)
    
    return model_path


def plot_learning_curve(
    losses: dict[str, list[float | int | str]], model_name: str, best_model_epoch: int | None = None,
) -> None:
    """
    Plot model learning curve (training vs. validation losses).

    Parameters
    losses : dict[str, float]
        Dict of model training and validation losses at each epoch.
    """
    
    losses_df: pd.DataFrame = pd.DataFrame(losses).sort_values(by=['Epoch', 'Loop'], ignore_index=True)
    
    # Create line plot
    fig= px.line(
        losses_df,
        x='Epoch',
        y='Loss',
        color='Loop',
        color_discrete_sequence=config.COLOUR_PAIR,
        title=f"{model_name} Learning Curve - Training Versus Validation Loss",
    )
    
    if best_model_epoch:
        
        # Get record of epoch with best model
        best_result = losses_df[losses_df['Epoch'] == best_model_epoch].set_index('Loop')['Loss']
        
        # Training and validation loss for best model
        annotation_text = (
            "Best Model:<br>" +
            f"Training Loss: {best_result['Training']:.3f}<br>" +
            f"Validation Loss: {best_result['Validation']:.3f}"
        )
        
        # Add line to show epoch of best model
        fig.add_vline(
            x=best_model_epoch,
            annotation={
                'align': 'right',
                'borderpad': 6,
                'font': {'color': config.COLOUR_PALETTE[0], 'size': 16},
                'text': annotation_text,
                'xanchor': 'right',
            },
            line={'color': config.COLOUR_PALETTE[0], 'dash': 'dot', 'width': 2},
        )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Save results to file
    losses_df.to_csv(os.path.join(config.DATA_VIS_DIR, "optimisation_results.csv"), index=False)
    
    return