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
from ODS.data_augmentation import plot_data_distributions
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
        indices : NDArray[int_]
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
        
        # NPY files
        self.images_file = images_file
        self.labels_file = labels_file
        
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
            Index location of sample.

        Returns
        -------
        image : Tensor
            Data sample image.
        label : Tensor
            Data sample label.
        """
        
        # Get image and label
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
        Path to source dataset directory.
    combined_data_dir : str, optional, default=COMBINED_DATA_DIR
        Path to output combined dataset directory.
    images_file : str, optional, default=COMBINED_IMAGES_FILE
        Path to file containing images.
    labels_file : str, optional, default=COMBINED_LABELS_FILE
        Path to file containing labels.
    transform : Compose | None, optional, default=None
        Transformations to apply to images.
    
    Returns
    -------
    images : NDArray[uint8]
        Images array.
    labels : NDArray[uint8]
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


def simple_resample_dataset(
    images: NDArray[np.uint8], labels: NDArray[np.uint8], max_samples_per_count: int = config.MAX_SAMPLES_PER_COUNT,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Simple resampling to limit the number of samples per occupancy count.
    
    Parameters
    ----------
    images : NDArray[uint8]
        Dataset images.
    labels : NDArray[uint8]
        Dataset labels.
    max_samples_per_count : int, optional, default=MAX_SAMPLES_PER_COUNT
        Maximum number of samples to keep for each occupancy count.
    
    Returns
    -------
    resampled_images : NDArray[uint8]
        Resampled images.
    resampled_labels : NDArray[uint8]
        Resampled labels.
    """
    
    # Create dataframe for easier manipulation
    data_df = pd.DataFrame({
        'id': np.arange(len(labels)),
        'count': labels,
    })
    
    # Get counts per occupancy level
    count_distribution = data_df['count'].value_counts().sort_index()
    print("Original distribution:")
    print(count_distribution)
    
    # Resample each occupancy count
    resampled_indices = []
    for count_value in count_distribution.index:
        
        # Get all indices for this count
        count_indices = data_df[data_df['count'] == count_value]['id'].values
        
        # If we have more samples than max_samples_per_count, randomly sample
        if len(count_indices) > max_samples_per_count:
            
            selected_indices = np.random.choice(
                count_indices, 
                size=max_samples_per_count, 
                replace=False,
            )
            print(f"Count {count_value}: Reduced from {len(count_indices)} to {max_samples_per_count}")
            
        else:
            
            selected_indices = count_indices
            print(f"Count {count_value}: Kept all {len(count_indices)} samples")
            
        resampled_indices.extend(selected_indices)
    
    # Convert to numpy array and sort to maintain some order
    resampled_indices = np.array(resampled_indices)
    np.random.shuffle(resampled_indices)
    
    # Create resampled datasets
    resampled_images = images[resampled_indices]
    resampled_labels = labels[resampled_indices]
    
    # Print new distribution
    new_distribution = pd.Series(resampled_labels).value_counts().sort_index()
    print("\nNew distribution:")
    print(new_distribution)
    
    return resampled_images, resampled_labels


def split_dataset(
    split_data_dir: str = config.DATA_DIR, train_images_file: str = config.TRAIN_IMAGES_FILE,
    train_labels_file: str = config.TRAIN_LABELS_FILE, test_images_file: str = config.TEST_IMAGES_FILE,
    test_labels_file: str = config.TEST_LABELS_FILE, train_ratio: float = config.TRAIN_RATIO,
    apply_resampling: bool = True, max_samples_per_count: int = config.MAX_SAMPLES_PER_COUNT,
) -> None:
    """
    Split data into training and testing sets with optional resampling.
    
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
    apply_resampling : bool, optional, default=True
        Whether to apply simple resampling to reduce overrepresented counts.
    max_samples_per_count : int, optional, default=MAX_SAMPLES_PER_COUNT
        Maximum samples per occupancy count when resampling.
    
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
    
    # Apply resampling if requested
    if apply_resampling:
        st.info(f"Applying resampling with max {max_samples_per_count} samples per count...")
        images, labels = simple_resample_dataset(images, labels, max_samples_per_count)
    
    # Get sample weights (rest of the function remains the same)
    y_min = float(labels.min()) - 1
    y_max = float(labels.max()) + 1
    sample_weights = list(1 - ((labels - y_min) / (y_max - y_min)))
    train_size = int(len(sample_weights) * train_ratio)
    
    # Split into training and testing sets using weighted random sampling
    indices: NDArray[np.int_] = np.arange(len(images), dtype=np.int_)
    train_indices: NDArray[np.int_] = np.array(
        list(WeightedRandomSampler(sample_weights, train_size, replacement=False)),
        dtype=np.int_,
    )
    test_indices: NDArray[np.int_] = indices[np.isin(indices, train_indices, invert=True)]
    
    # Save trainng and testing sets
    os.makedirs(split_data_dir, exist_ok=True)
    np.save(train_images_file, images[train_indices])
    np.save(train_labels_file, labels[train_indices])
    np.save(test_images_file, images[test_indices])
    np.save(test_labels_file, labels[test_indices])
    
    # Visualise data distribution
    plot_data_distributions(
        [pd.Series(labels[train_indices]), pd.Series(labels[test_indices])],
        ["Training", "Validation"],
        title="Data Distribution of Training and Testing Sets",
        return_fig=False,
    )
    
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
        
        # Load training and validation datasets
        train_labels = np.load(train_labels_file, mmap_mode='r')
        test_labels = np.load(test_labels_file, mmap_mode='r')
        
        # Create data distribution directory
        data_vis_dir = os.path.join(config.OUTPUT_DIR, "data_vis")
        os.makedirs(data_vis_dir, exist_ok=True)
        
        pd.Series(
            data=train_labels,
            index=pd.RangeIndex(len(train_labels), name='id'),
            name='count',
        ).to_csv(os.path.join(data_vis_dir, "training_data_dist.csv"))
        
        pd.Series(
            data=test_labels,
            index=pd.RangeIndex(len(test_labels), name='id'),
            name='count',
        ).to_csv(os.path.join(data_vis_dir, "validation_data_dist.csv"))
        
        # Create training and validation datasets
        train_dataset = OccupancyDataset(
            images_file=train_images_file,
            labels_file=train_labels_file,
            indices=np.arange(len(train_labels), dtype=np.int_),
            transform=train_transform,
        )
        val_dataset = OccupancyDataset(
            images_file=test_images_file,
            labels_file=test_labels_file,
            indices=np.arange(len(test_labels), dtype=np.int_),
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