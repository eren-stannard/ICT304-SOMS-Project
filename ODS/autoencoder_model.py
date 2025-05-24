# autoencoder_model.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System

    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)

    ICT304: AI System Design
    Murdoch University

    Purpose of File:
    Autoencoder model class.

"""


# Libraries used
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

# Files used
import ODS.config as config


class AutoencoderModel(nn.Module):
    """Autoencoder model for feature extraction and occupancy prediction."""

    def __init__(
        self, input_dim: int = (224 * 224 * 3), latent_dim: int = config.AUTOENCODER_LATENT_DIM,
    ) -> None:
        """
        Autoencoder model constructor.

        Parameters
        ----------
        input_dim : int, optional, default=(224 * 224 * 3)
            Input data dimensions.
        latent_dim : int, optional, default=AUTOENCODER_LATENT_DIM
            Latent data dimensions.
        
        See Also
        --------
        config.py : Default parameter configuration.
        """
        
        super(AutoencoderModel, self).__init__()
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToDtype(torch.float32),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        # Decoder (for training the autoencoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

        # Regressor for occupancy prediction
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2 if self.training else 0),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        
        # Set device
        self.device = config.DEVICE
        self.to(self.device)
        
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input data through model network and output prediction outcome.

        Parameters
        ----------
        x : Tensor
            Input data to make prediction on.

        Returns
        -------
        count : Tensor
            Prediction outcome.
        """
        
        # Get latent representation
        latent = self.encoder(x)

        # Get count prediction
        count = self.regressor(latent).squeeze()
        
        return count

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pass input data to encoder and decoder for training autoencoder.

        Parameters
        ----------
        x : Tensor
            Input data.

        Returns
        -------
        reconstruction : Tensor
            Reconstruction.
        latent : Tensor
            Latent representation.
        """
        
        # Get latent representation
        latent = self.encoder(x)
        
        # Reconstruct input
        reconstruction = self.decoder(latent)

        return reconstruction, latent

    def full_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass input data through model network and output prediction outcome.

        Parameters
        ----------
        x : Tensor
            Input data to make prediction on.

        Returns
        -------
        reconstruction : Tensor
            Reconstruction.
        latent : Tensor
            Latent representation.
        count : Tensor
            Prediction outcome.
        """
        
        # Get latent representation
        latent = self.encoder(x)
        
        # Reconstruct input
        reconstruction = self.decoder(latent)

        # Get count prediction
        count = self.regressor(latent).squeeze()
        
        return reconstruction, latent, count


def load_model(model_path: str = config.MODEL_PATH, show_log: bool = True) -> AutoencoderModel:
    """
    Load trained model.

    Parameters
    ----------
    model_path : str, optional, default=MODEL_PATH
        Path to trained model file.
    show_log : bool, optional, default=True
        Whether to show error log in Streamlit. If False, only prints to terminal.
        Used for compatibility with Streamlit caching.
    
    Returns
    -------
    model : AutoencoderModel
        Trained model.
    
    Raises
    ------
    e : RuntimeError
        Failed to load model.
    
    See Also
    --------
    config.py : Default parameter configuration.
    """
    
    # Create model
    model = AutoencoderModel(latent_dim=config.AUTOENCODER_LATENT_DIM)

    # Try to load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=model.device, mmap=True))
        model.eval()
        
    except RuntimeError as e:
        if show_log:
            st.error(f"Error: Failed to load model: {e}")
        else:
            print(f"Error: Failed to load model: {e}")
        raise e
    
    return model