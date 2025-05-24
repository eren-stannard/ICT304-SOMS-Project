# cnn_model.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System

    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)

    ICT304: AI System Design
    Murdoch University

    Purpose of File:
    CNN model for accurate people counting.
"""


# Libraries used
import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn

# Files used
import ODS.config as config


class CNNModel(nn.Module):
    """CNN model for occupancy prediction using pre-trained ResNet50 backbone."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        """
        CNN model constructor.

        Parameters
        ----------
        pretrained : bool, optional, default=True
            Whether to use pretrained backbone.
        freeze_backbone : bool, optional, default=True
            Whether to freeze backbone weights.
        """
        
        super(CNNModel, self).__init__()

        # Use pretrained weights
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            self.preprocess = weights.transforms()
            
        else:
            weights = None
            self.preprocess = None
        
        # Use a pretrained ResNet50 as backbone
        self.backbone = models.resnet50(weights=weights)

        # Freeze early layers to preserve general features and prevent overfitting
        if freeze_backbone:
            for param in list(self.backbone.parameters())[:-4]:
                param.requires_grad = False

        # Extract the number of features output by the backbone
        backbone_features = self.backbone.fc.in_features

        # Replace the final layer with our custom regressor
        self.backbone.fc = nn.Identity() # type: ignore

        # Create a regressor optimised for counting
        self.regressor = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularisation
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # No activation - we want direct count prediction
            # We can clamp to valid range during prediction
        )

        # Set device
        self.device = config.DEVICE
        self.to(self.device)

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.

        Returns
        -------
        count : torch.Tensor
            Predicted count.
        """
        
        # Transform
        if self.preprocess:
            x = self.preprocess(x)
        
        # Extract features using the backbone
        features = self.backbone(x)

        # Get count prediction
        count = self.regressor(features).squeeze()

        # We can clamp the output to valid ranges if needed
        # count = torch.clamp(count, min=0, max=config.MAX_COUNT)

        return count


class CNNEnsemble(nn.Module):
    """Ensemble of multiple CNN models for improved accuracy."""

    def __init__(self, model_paths: list[str], model_weights: list[float] | None = None) -> None:
        """
        Ensemble constructor.

        Parameters
        ----------
        model_paths : list[str]
            List of paths to trained model weights files.
        model_weights : list[float] | None, optional, default=None
            Weights for each model in the ensemble. If None, equal weighting is used.
        """
        
        super(CNNEnsemble, self).__init__()

        self.models = nn.ModuleList()

        # Load each model
        for path in model_paths:
            model = CNNModel()
            model.load_state_dict(torch.load(path, map_location=model.device, mmap=True))
            model.eval()  # Set to evaluation mode
            self.models.append(model)

        # Setup model weights
        if model_weights is None:
            # Equal weighting
            self.weights = torch.ones(len(model_paths)) / len(model_paths)
        else:
            # Normalise weights to sum to 1
            weights_sum = sum(model_weights)
            self.weights = torch.tensor([w / weights_sum for w in model_weights])

        # Move to device
        self.device = config.DEVICE
        self.to(self.device)
        self.weights = self.weights.to(self.device)
        
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.

        Returns
        -------
        weighted_pred : torch.Tensor
            Predicted count (weighted average of all models).
        """
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        # Stack predictions and take weighted average
        stacked_preds = torch.stack(predictions, dim=0)
        weighted_pred = torch.sum(stacked_preds * self.weights.view(-1, 1), dim=0)

        return weighted_pred


def load_model(model_path: str = config.MODEL_PATH, show_log: bool = True) -> CNNModel:
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
    model : CNNModel
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
    model = CNNModel()

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