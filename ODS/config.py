# config.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System

    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)

    ICT304: AI System Design
    Murdoch University

    Purpose of File:
    ODS Configuration parameters.

"""


# Libraries used
import os
import torch


# Data configuration
SRC_DATA_DIR = os.path.join('ODS', 'datasets')
APPLY_RESAMPLING = True
MAX_SAMPLES_PER_COUNT = 200

# Combined data cofigureation
COMBINED_DATA_DIR = os.path.join('ODS', 'dataset')
COMBINED_IMAGES_FILE = os.path.join(COMBINED_DATA_DIR, 'images.npy')
COMBINED_LABELS_FILE = os.path.join(COMBINED_DATA_DIR, 'labels.npy')

# Balanced data configuration
DATA_DIR = os.path.join('ODS', 'balanced_dataset')
TRAIN_IMAGES_FILE = os.path.join(DATA_DIR, 'train_images.npy')
TRAIN_LABELS_FILE = os.path.join(DATA_DIR, 'train_labels.npy')
TEST_IMAGES_FILE = os.path.join(DATA_DIR, 'test_images.npy')
TEST_LABELS_FILE = os.path.join(DATA_DIR, 'test_labels.npy')

# Model configuration
MODEL_TYPE = 'cnn'

# Training configuration
BATCH_SIZE = 128
NUM_EPOCHS = 30 # Changed from 20
LEARNING_RATE = 0.05
WEIGHT_DECAY = 0.00001
TRAIN_RATIO = 0.8

## Autoencoder configuration
AUTOENCODER_LATENT_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output configuration
OUTPUT_DIR = os.path.join('ODS', 'output')
DATA_VIS_DIR = os.path.join(OUTPUT_DIR, 'data_vis')
MODEL_PATH = os.path.join(OUTPUT_DIR, f'{MODEL_TYPE}_model.pth')

# Colour configuration using Catppuccin's Latte theme
COLOUR_PALETTE = [
    '#d20f39',  # Red
    '#fe640b',  # Peach
    '#df8e1d',  # Yellow
    '#40a02b',  # Green
    '#04a5e5',  # Sky
    '#1e66f5',  # Blue
    '#8839ef',  # Mauve
    '#ea76cb',  # Pink
]
COLOUR_PAIR = [
    '#dd7878',  # Flamingo
    '#7287fd',  # Lavender
]
COLOUR_SCALE = [
    '#d20f39',  # Red
    '#e64553',  # Maroon
    '#fe640b',  # Peach
    '#df8e1d',  # Yellow
    '#40a02b',  # Green
    '#179299',  # Teal
    '#1e66f5',  # Blue
]