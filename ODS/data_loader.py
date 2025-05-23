# data_loader.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Dataset loading.

"""


# Libraries used
import numpy as np
import os
import pandas as pd
import random
import streamlit as st
import torch
import torchvision.transforms.v2 as T
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Literal

# Files used
import ODS.config as config
from ODS.data_augmentation import balance_dataset, plot_data_distributions
from ODS.combine_datasets import combine_datasets


class OccupancyDataset(Dataset):
    """
    Dataset for occupancy detection.
    """

    def __init__(
        self, images_file: str, labels_file: str, indices: NDArray[np.int_],
        transform: T.Compose | None = None,
    ) -> None:
        """
        Occupancy detection dataset constructor.

        Parameters
        ----------
        images_file : str
            Path to .npy images file.
        labels_file : str
            Path to .npy labels file.
        indices : NDArray[int]
            Indices of data samples to use for dataset.
        transform : Compose | None, optional, default=None
            Transformations to apply to images.
        
        See Also
        --------
        config.py : Default parameter configuration.
        """
        
        super(OccupancyDataset, self).__init__()
        
        # Device
        self.device: torch.device = config.DEVICE
        
        # Data transformations
        self.transform: T.Compose | None = transform
        
        # Indices
        self.indices: NDArray[np.int_] = indices
        
        # Load NPY files
        self.images_file = images_file
        self.labels_file = labels_file
        #self.images: NDArray[np.uint8] = np.load(self.images_file, mmap_mode='r')
        #self.labels: NDArray[np.uint8] = np.load(self.labels_file, mmap_mode='r')
        
        # Load labels
        #self.labels_df: pd.Series[int] = pd.Series(self.labels)
        
        return


    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns
        -------
        n_samples : int
            Number of image samples in dataset.
        """
        
        return len(self.indices)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset at index location idx.

        Parameters
        ----------
        idx : int
            Index location of the sample.

        Returns
        -------
        image : Tensor
            Data sample image.
        label : Tensor
            Data sample label.
        """
        
        # Get image and label
        #image: torch.Tensor = torch.tensor(self.images[idx], dtype=torch.float32).to(self.device)
        #label: torch.Tensor = torch.tensor(self.labels[idx], dtype=torch.float32).to(self.device)
        image = torch.tensor(np.load(self.images_file, mmap_mode='r')[idx], dtype=torch.float32).to(self.device)
        label = torch.tensor(np.load(self.labels_file, mmap_mode='r')[idx], dtype=torch.float32).to(self.device)
        
        # Transform image
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_dataset(
    src_data_dir: str = config.SRC_DATA_DIR, combined_data_dir: str = config.COMBINED_DATA_DIR,
    images_file: str = config.COMBINED_IMAGES_FILE, labels_file: str = config.COMBINED_LABELS_FILE,
    transform: T.Compose | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Load dataset images and labels from JPG and CSV files into numpy arrays.
    
    Parameters
    ----------
    src_data_dir : str, optional, default=SRC_DATA_DIR
    combined_data_dir : str, optional, default=COMBINED_DATA_DIR
    images_file : str, optional, default=COMBINED_IMAGES_FILE
        Path to file containing images.
    labels_file : str, optional, default=COMBINED_LABELS_FILE
        Path to file containing labels.
    transform : Compose | Resize | None, optional, default=None
        Transformations to apply to images.
    
    Returns
    -------
    images : ndarray[int]
        Images array.
    labels : ndarray[int]
        Labels array.
    
    See Also
    --------
    config.py : Default parameter configuration.
    """
    
    images: NDArray[np.uint8]
    labels: NDArray[np.uint8]
    
    if not all([os.path.exists(file_path) for file_path in [images_file, labels_file]]):
        
        # Combined data directory
        images, labels = combine_datasets(
            src_data_dir,
            combined_data_dir,
            images_file,
            labels_file,
            transform,
        )
    
    else:
        
        images = np.load(images_file, mmap_mode='r')
        labels = np.load(labels_file, mmap_mode='r')
    
    return images, labels


def split_dataset(
    split_data_dir: str = config.DATA_DIR, train_images_file: str = config.TRAIN_IMAGES_FILE,
    train_labels_file: str = config.TRAIN_LABELS_FILE, test_images_file: str = config.TEST_IMAGES_FILE,
    test_labels_file: str = config.TEST_LABELS_FILE, train_ratio: float = config.TRAIN_RATIO,
) -> None:
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    split_data_dir : str, optional, default=DATA_DIR
        Path to directory containing split balanced dataset.
    train_images_file : str, optional, default=TRAIN_IMAGES_FILE
        Path to file containing training images.
    train_labels_file : str, optional, default=TRAIN_LABELS_FILE
        Path to file containing training labels.
    test_images_file : str, optional, default=TEST_IMAGES_FILE
        Path to file containing testing/validation images.
    test_labels_file : str, optional, default=TEST_LABELS_FILE
        Path to file containing testing/validation labels.
    train_ratio : float, optional, default=TRAIN_RATIO
        Proportion of data samples to use in training set.
    
    See Also
    --------
    config.py : Default parameter configuration.
    """
    
    # Define transform
    transform = T.Compose([T.Resize((224, 224))])

    # Load dataset
    images: NDArray[np.uint8]
    labels: NDArray[np.uint8]
    images, labels = load_dataset(transform=transform)
    
    # Get sample weights
    y_min = float(labels.min()) - 1
    y_max = float(labels.max()) + 1
    sample_weights = list(1 - ((labels - y_min) / (y_max - y_min))) # type: ignore
    train_size = int(len(sample_weights) * train_ratio)
    
    # Split into training and testing sets using weighted random sampling
    indices: NDArray[np.int_] = np.arange(len(images), dtype=np.int_)
    train_indices: NDArray[np.int_] = np.array(
        list(WeightedRandomSampler(sample_weights, train_size, replacement=False)),
        dtype=np.int_,
    )
    test_indices: NDArray[np.int_] = indices[np.isin(indices, train_indices, invert=True)]
    
    # Save testing set
    os.makedirs(split_data_dir, exist_ok=True)
    np.save(test_images_file, images[test_indices])
    np.save(test_labels_file, labels[test_indices])
    
    # Balance training set
    #train_images: NDArray[np.uint8]
    #train_labels: NDArray[np.uint8]
    #train_images, train_labels = balance_dataset(images[train_indices], labels[train_indices])
    
    # Save training set
    np.save(train_images_file, images[train_indices])
    np.save(train_labels_file, labels[train_indices])
    
    # Visualise data distribution
    fig = plot_data_distributions(
        [pd.Series(labels[train_indices]), pd.Series(labels[test_indices])],
        ["Training", "Validation"],
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return


def get_data_loader(
    batch_size: int = config.BATCH_SIZE, train_ratio: float = config.TRAIN_RATIO,
    mode: Literal['train', 'evaluate'] = 'train',
) -> DataLoader | tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.

    Parameters
    ----------
    batch_size : int, optional, default=BATCH_SIZE
        Batch size to use.
    train_ratio : float, optional, default=TRAIN_RATIO
        Size ratio of training set.
    mode : Literal['train', 'evaluate'], optional, default='train'
        Which data loader to create.

    Returns
    -------
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    
    See Also
    --------
    config.py : Default parameter configuration
    """
    
    # Training and testING data file paths
    train_images_file = config.TRAIN_IMAGES_FILE
    train_labels_file = config.TRAIN_LABELS_FILE
    test_images_file = config.TEST_IMAGES_FILE
    test_labels_file = config.TEST_LABELS_FILE
    
    if not all([os.path.exists(file_path) for file_path in [
        train_images_file, train_labels_file, test_images_file, test_labels_file,
    ]]):
        
        split_dataset(
            config.DATA_DIR,
            train_images_file,
            train_labels_file,
            test_images_file,
            test_labels_file,
            train_ratio,
        )
    
    # Define transforms
    train_transform = T.Compose([
        T.RandomAdjustSharpness(random.uniform(0.8, 1.2), 0.1),
        T.RandomAutocontrast(0.1),
        T.RandomHorizontalFlip(0.5),
        T.RandomInvert(0.1),
        T.Resize((224, 224)),
        T.ToDtype(torch.float32),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToDtype(torch.float32),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    if mode == 'train':
        
        # Get size of training and validation datasets
        train_size = len(np.load(train_labels_file, mmap_mode='r'))
        test_size = len(np.load(test_labels_file, mmap_mode='r'))
        
        # Create training and validation datasets
        train_dataset = OccupancyDataset(
            images_file=train_images_file,
            labels_file=train_labels_file,
            indices=np.arange(train_size, dtype=np.int_),
            transform=train_transform,
        )
        val_dataset = OccupancyDataset(
            images_file=test_images_file,
            labels_file=test_labels_file,
            indices=np.arange(test_size, dtype=np.int_),
            transform=val_transform,
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=True)

        return train_loader, val_loader
    
    else:
        
        # Get size of validation dataset
        test_size = len(np.load(test_labels_file, mmap_mode='r'))
        
        # Create only validation dataset
        val_dataset = OccupancyDataset(
            images_file=test_images_file,
            labels_file=test_labels_file,
            indices=np.arange(test_size, dtype=np.int_),
            transform=val_transform,
        )

        # Create only validation data loader
        val_loader = DataLoader(val_dataset, batch_size, shuffle=True)

        return val_loader