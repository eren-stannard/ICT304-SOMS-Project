# data_augmentation.py

"""
    Smart Occupancy Monitoring System - Occupancy Detection AI Sub-System
    
    Authors:
        Ervin Galas    (34276705)
        Sofia Peeva    (35133522)
        Eren Stannard  (34189185)
    
    ICT304: AI System Design
    Murdoch University
    
    Purpose of File:
    Data augmentation and resampling functions for uniformly distributing dataset.

"""


# Libraries used
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import random
import streamlit as st
import torch
import torchvision.transforms.v2 as T
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from stqdm import stqdm

# Files used
import ODS.config as config


def augment_image(
    image: NDArray[np.uint8], adjustment_range: tuple[float, float] = (0.8, 1.2),
) -> NDArray[np.uint8] | None:
    """
    Apply brightness and contrast augmentation to an image.
    
    Parameters
    ----------
    image : NDArray[uint8]
        Input image.
    adjustment_range : tuple[float, float], optional, default=(0.8, 1.2)
        Range for augmentation adjustment factors.
    
    Returns
    -------
    image : NDArray[uint8] | None
        Augmented image.
    """
    
    try:
        
        # Convert numpy array image to tensor
        img = torch.tensor(image, dtype=torch.uint8)

        # Augmentation adjustment factor
        factor = random.uniform(adjustment_range[0], adjustment_range[1])
        
        # Define augmentation transforms
        augment = T.Compose([
            T.RandomAdjustSharpness(factor),
            T.RandomAutocontrast(),
            T.RandomChannelPermutation(),
            T.RandomHorizontalFlip(),
            T.RandomInvert(),
        ])
        
        # Transform image
        img = augment(img)
        
        # Convert tensor image back to numpy array
        image = img.numpy()
        
        return image
    
    except Exception as e:
        
        return None


def create_uniform_distribution(min_val: int = 0, max_val: int = 50, total_samples: int = 5000) -> dict[int, int]:
    """
    Create a uniformly distributed set of count values.
    
    Parameters
    ----------
    min_val : int, optional, default=0
        Minimum value.
    max_val : int, optional, default=50
        Maximum value.
    total_samples : int, optional, default=5000
        Total number of samples.
    
    Returns
    -------
    target_distribution : dict[int, int]
        Count values as keys, target frequencies as values.
    """
    
    # Calculate the number of unique values in the range
    num_values: int = max_val - min_val + 1

    # Calculate the samples per value (floor to ensure we don't exceed total)
    samples_per_value = total_samples // num_values

    # Initialise distribution with equal counts
    target_distribution = {val: samples_per_value for val in range(min_val, max_val + 1)}

    # Distribute any remaining samples (due to integer division)
    remaining = total_samples - (samples_per_value * num_values)
    for val in range(min_val, min_val + remaining):
        target_distribution[val] += 1
    
    return target_distribution


def balance_dataset(train_images: NDArray[np.uint8], train_labels: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Resample training dataset by balancing distribution of data.
    
    Parameters
    ----------
    train_images : NDArray[uint8]
        Training dataset images.
    train_labels : NDArray[uint8]
        Training dataset labels.
    
    Returns
    -------
    new_train_images : NDArray[uint8]
        Balanced training dataset images.
    new_train_labels : NDArray[uint8]
        Balanced training dataset labels.
    """
    
    # Load data subset labels
    labels_df = pd.Series(train_labels)
    
    # Create a dictionary of current counts
    current_counts = labels_df.value_counts().to_dict()
    min_val: int = labels_df.min()
    max_val: int = int((labels_df.max() + labels_df.quantile(0.75)) / 2)

    # Get uniform distribution
    target_distribution = create_uniform_distribution(
        min_val=min_val,
        max_val=max_val,
        total_samples=train_images.shape[0],
    )

    # Initialise a new labels dataframe
    new_images = []
    new_labels = {'id': [], 'count': []}
    image_counter = 1

    # Process each count value
    for count_value, target_count in stqdm(target_distribution.items(), desc="Resampling **training** data"):
        
        available_images = labels_df[labels_df == count_value]

        # If we have original images with this count, include them first
        if len(available_images) > 0:
            
            # Take up to the target count of original images
            use_count = min(len(available_images), target_count)
            use_images = available_images.sample(use_count) if use_count < len(available_images) else available_images
            
            for row in use_images.index:
                
                # Add to new images and labels
                image, label = train_images[row], train_labels[row]
                new_images.append(image)
                new_labels['id'].append(image_counter)
                new_labels['count'].append(count_value)
                image_counter += 1

            # Update how many more we need
            target_count -= use_count

        # Now augment images if we still need more
        if target_count > 0:
            
            # Find the closest count value that has images
            available_counts = sorted(current_counts.keys())
            closest_count = min(available_counts, key=lambda x: abs(x - count_value))
            donor_images = labels_df[labels_df == closest_count]

            # Augment images to fill the gap
            for i in range(target_count):
                
                # Select a random image from the donor set
                random_idx = random.randint(0, len(donor_images) - 1)
                row = donor_images.iloc[random_idx]

                # Load and augment the original image
                image = train_images[row]
                augmented_img = augment_image(image)

                if augmented_img is not None:
                    
                    # Add to new images and labels
                    new_images.append(augmented_img)
                    new_labels['id'].append(image_counter)
                    new_labels['count'].append(count_value)
                    
                    image_counter += 1
    
    # Save new labels
    new_labels_df = pd.Series(new_labels['count'], index=new_labels['id'])
    
    # Convert new images and labels to numpy
    new_train_images: NDArray[np.uint8] = np.array(new_images, dtype=np.uint8)
    new_train_labels: NDArray[np.uint8] = np.array(new_labels['count'], dtype=np.uint8)
    
    # Visualise data distributions
    plot_data_distributions(
        [labels_df, new_labels_df],
        ["Original", "Balanced"],
        title="Data Distribution of Original and Balanced Training Data",
        return_fig=False,
    )

    return new_train_images, new_train_labels


def plot_data_distributions(
    data: list[pd.Series], titles: list[str], title: str | None = None,
    xbins_range: tuple[int, int] | None = None, return_fig: bool = True,
) -> go.Figure | None:
    """
    Plot data distributions of given datasets.
    
    Parameters
    ----------
    data : list[Series]
        List of datasets' labels.
    titles : list[str]
        List of dataset names.
    title : str | None, optional, default=None
        Title of figure.
    xbins_range : tuple[int, int] | None, optional, default=None
        Range of bins to use for distribution plot.
    return_fig : bool, optional, default=True
        Whether to return the figure without displaying it. If False, figure is displayed.
    
    Returns
    -------
    fig : Figure | None
        Figure of distribution plots if `return_fig` is True, otherwise None.
    """
    
    # Create data distribution directory
    data_vis_dir = config.DATA_VIS_DIR
    os.makedirs(data_vis_dir, exist_ok=True)
    
    # Plot heights
    box_height: float = 0.1 * len(data) if len(data) < 5 else 0.4
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[box_height, 1 - box_height],
        shared_xaxes=True,
        horizontal_spacing=0,
        vertical_spacing=0,
        specs=[[{'type': 'box'}], [{'type': 'xy'}]],
        )
    
    # Get colour palette
    colours = config.COLOUR_PAIR if len(data) <= 2 else config.COLOUR_PALETTE
    
    for d, t, c in zip(data, titles, colours):
        
        # Save distribution
        d.to_csv(os.path.join(data_vis_dir, f"dataset_{t}_data_dist.csv"))
        
        # Add distribution histogram and box plot to figure
        dist, dist_box = add_distribution(d, t, c)
        fig.add_trace(dist, row=2, col=1)
        fig.add_trace(dist_box, row=1, col=1)
    
    # Update axis titles
    fig.update_xaxes(title_text="Image Occupancy", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    # Update figure layout
    fig.update_layout(
        barmode='overlay',
        legend={'x': 1, 'xanchor': 'right', 'y': 1 - box_height, 'yanchor': 'top'},
        margin={'b': 10, 'l': 5, 'r': 5, 't': 50},
        title=title or "Comparison of Data Distributions",
    )
    
    # Set xbins range to bin all x-axis samples if not specified otherwise
    if xbins_range is None:
        xbins_range = (min([d.min() for d in data]), max([d.max() for d in data]) + 1)
    
    # Update xbins
    fig.update_traces(xbins={'start': xbins_range[0], 'end': xbins_range[1] + 1, 'size': 1}, row=2, col=1)
    
    if return_fig:
        return fig
    
    # Display figure without returning it
    st.plotly_chart(fig, use_container_width=True)
    
    return


def add_distribution(data: pd.Series, title: str, colour: str) -> tuple[go.Histogram, go.Box]:
    """
    Plot a histogram and box plot for a single dataset distribution.
    
    Parameters
    ----------
    data : Series
        Dataset to plot distribution.
    title : str
        Dataset name.
    colour : str
        Colour to use for plots.
    
    Returns
    -------
    dist : Histogram
        Histogram of dataset distribution.
    dist_box : Box
        Box plot of dataset distribution.
    """
    
    # Plot distribution histogram
    dist = go.Histogram(
        x=data,
        marker={
            'color': colour,
            'line': {'color': colour, 'width': 0.5},
            'opacity': 0.6,
        },
        name=title,
    )
    
    # Plot distribution box plot
    dist_box = go.Box(
        x=data,
        boxmean='sd',
        marker={
            'color': colour,
            'line': {'color': colour, 'width': 1},
        },
        line={'width': 1},
        name=title,
        showlegend=False,
    )
    
    return dist, dist_box