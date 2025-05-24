# combine_datasets.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Data augmentation functions for creating a uniformly distributed occupancy dataset.

"""


# Libraries used
import numpy as np
import os
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms.v2 as T
from numpy.typing import NDArray
from stqdm import stqdm
from torchvision.io import decode_image

# Libraries used
import ODS.config as config
from ODS.data_augmentation import plot_data_distributions


def combine_datasets(
    src_data_dir: str = config.SRC_DATA_DIR, dst_data_dir: str = config.COMBINED_DATA_DIR,
    dst_images_file: str = config.COMBINED_IMAGES_FILE, dst_labels_file: str = config.COMBINED_LABELS_FILE,
    transform: T.Compose | None = None, save_npy: bool = False,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Combine multiple datasets into NumPy arrays.
    
    Parameters
    ----------
    src_data_dir : str, optional, default=SRC_DATA_DIR
        Path to source directory containing all datasets to be combined.
    dst_data_dir : str, optional, default=COMBINED_DATA_DIR
        Path to directory to save .npy files if `save_npy` is True.
    dst_images_file : str, optional, default=COMBINED_IMAGES_FILE
        Path to file to save images .npy file if `save_npy` is True.
    dst_labels_file : str, optional, default=COMBINED_LABELS_FILE
        Path to file to save labels .npy file if `save_npy` is True.
    transform : T.Compose | None, optional, default=None
        Transformations to apply to images. Limit to resizing, mode formatting, and dtype conversion to avoid data leakage.
    save_npy : bool, optional, default=False
        Whether to save the resulting combined dataset to .npy files.
    
    Returns
    -------
    images : NDArray[uint8]
        Combined dataset images.
    labels : NDArray[np.uint8]
        Combined dataset labels.
    
    See Also
    --------
    config.py : Default parameter configuration.
    """
    
    # Dataset directories
    datasets: list[str] = os.listdir(src_data_dir)
    
    images: NDArray[np.uint8] | list[NDArray[np.uint8]] = []
    labels: NDArray[np.uint8] | list[NDArray[np.uint8]] = []
    
    labels_dfs: list[pd.Series[int]] = []
    
    for dataset in datasets:
        
        dir: str = os.path.join(src_data_dir, dataset)
        
        images_dir: str = os.path.join(dir, "images")
        labels_dir: str = os.path.join(dir, "labels")
        labels_file: str = os.path.join(dir, "labels.csv")
        
        new_images: NDArray[np.uint8]
        new_labels: NDArray[np.uint8]
        
        # No labels files => 0 occupancy images => create synthetic labels
        if len(os.listdir(dir)) == 1:
            
            st.toast(
                f"Synthetic empty dataset found. Creating synthetic labels for {dir}...",
                icon=":material/data_info_alert:",
            )
            create_synthetic_labels(dir, images_dir, labels_file)
        
        if "labels.csv" in os.listdir(dir):
            
            print(f"Loading files from {dir}...")
            new_images, new_labels = load_img_csv(dir, images_dir, labels_file, transform)
        
        else:
            
            print(f"Loading files from {dir}...")
            new_images, new_labels = load_img_txt(dir, images_dir, labels_dir, transform)
        
        images.append(new_images)
        labels.append(new_labels)
        
        # Append dataset labels to list
        labels_dfs.append(pd.Series(
            data=new_labels,
            index=pd.RangeIndex(len(new_labels), name='id'),
            name='count',
        ))
    
    # Visualise data distribution
    plot_data_distributions(labels_dfs, datasets, return_fig=False)
    
    images = np.concatenate(images, dtype=np.uint8)
    labels = np.concatenate(labels, dtype=np.uint8)
    
    if save_npy:
        os.makedirs(dst_data_dir, exist_ok=True)
        np.save(dst_images_file, images)
        np.save(dst_labels_file, labels)
    
    return images, labels


def load_img_csv(
    data_dir: str, images_dir: str, labels_file: str, transform: T.Compose | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Load data from directory structured as:  
    > `| <data_dir>/`  
    > `| | --- images/`  
    > `| | | --- <sample1>.jpg`  
    > `| | | --- <sample2>.jpg`  
    > `| | | --- <...>`  
    > `| | --- labels.csv`
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing dataset.
    images_dir : str
        Path to directory inside `data_dir` containing .jpg images named 'images'.
    labels_file : str
        Path to .csv file containing labels named 'labels.csv'.
    transform : T.Compose | None, optional, default=None
        Transformations to apply to images.
    
    Returns
    -------
    images : NDArray[uint8]
        Dataset images
    labels : NDArray[uint8]
        Dataset labels.
    """
    
    images: NDArray[np.uint8] | list[NDArray[np.uint8]] = []
    labels: NDArray[np.uint8] | list[int] = []
    
    # Load labels from csv
    labels_df: pd.Series[int] = pd.read_csv(labels_file, index_col='id')['count']
    
    if np.issubdtype(labels_df.index.dtype, np.number): # type: ignore
        
        image_files: list[str] = os.listdir(images_dir)
        
        for image_file in stqdm(image_files, desc=f"Loading files from **{data_dir}**", unit='**file**'):
            
            # Get image file path
            image_path = os.path.join(images_dir, image_file) # type: ignore
            
            # Load image
            image: torch.Tensor = decode_image(image_path, mode='RGB') # type: ignore
            if transform:
                image = transform(image)
            
            # Get label
            id = int(os.path.splitext(image_file)[0].split('_')[-1])
            count = labels_df.at[id]
            
            # Append new image and label
            images.append(image.numpy())
            labels.append(count)
        
    else:
    
        for id in stqdm(labels_df.index, desc=f"Loading files from **{data_dir}**", unit='**file**'):
            
            # Get image file path
            image_path = os.path.join(images_dir, id) # type: ignore
            
            # Load image
            image: torch.Tensor = decode_image(image_path, mode='RGB') # type: ignore
            if transform:
                image = transform(image)
            
            # Append new image and label
            images.append(image.numpy())
            labels.append(labels_df.at[id])
    
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    
    return images, labels


def load_img_txt(
    data_dir: str, images_dir: str, labels_dir: str, transform: T.Compose | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Load data from directory structured as:  
    > `| <data_dir>/`  
    > `| | --- images/`  
    > `| | | --- <sample1>.jpg`  
    > `| | | --- <sample2>.jpg`  
    > `| | | --- <...>`  
    > `| | --- labels/`  
    > `| | | --- <sample1>.txt`  
    > `| | | --- <sample2>.txt`  
    > `| | | --- <...>`
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing dataset.
    images_dir : str
        Path to directory inside `data_dir` containing .jpg images named 'images'.
    labels_dir : str
        Path to directory containing .txt labels named 'labels'.
    transform : T.Compose | None, optional, default=None
        Transformations to apply to images.
    
    Returns
    -------
    images : NDArray[uint8]
        Dataset images
    labels : NDArray[uint8]
        Dataset labels.
    """
    
    images: NDArray[np.uint8] | list[NDArray[np.uint8]] = []
    labels: NDArray[np.uint8] | list[int] = []
    
    # Get image file paths
    image_files: list[str] = os.listdir(images_dir)
    
    for image_file in stqdm(image_files, desc=f"Loading files from **{data_dir}**", unit='**file**'):
        
        # Get image and label file paths
        image_path: str = os.path.join(images_dir, image_file)
        label_path: str = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")
        
        # Load image
        image: torch.Tensor = decode_image(image_path, mode='RGB') # type: ignore
        if transform:
            image = transform(image)
        
        # Load label
        with open(label_path) as f:
            count: int = len(f.readlines())
        
        # Append new image and label
        images.append(image.numpy())
        labels.append(count)
    
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    
    return images, labels


def create_synthetic_labels(data_dir: str, images_dir: str, labels_file: str) -> None:
    """
    Create synthetic labels in a .csv file for images with 0 occupancy when directory structured as:
    > `| <data_dir>/`  
    > `| | --- images/`  
    > `| | | --- <sample1>.jpg`  
    > `| | | --- <sample2>.jpg`  
    > `| | | --- <...>`
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing dataset.
    images_dir : str
        Path to directory inside `data_dir` containing .jpg images named 'images'.
    labels_file : str
        Path to file where synthetic labels file will be output named 'labels.csv'.
    """
    
    # Get image file paths
    image_files: list[str] = os.listdir(images_dir)
    
    # Create synthetic labels for 0 occupancy
    pd.Series(
        data=np.zeros(len(image_files), dtype=int),
        index=pd.Index(image_files, name='id'),
        name='count',
    ).to_csv(labels_file)
    
    return