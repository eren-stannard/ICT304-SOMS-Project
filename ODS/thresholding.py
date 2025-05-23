# thresholding.py


# Libraries used
import cv2 as cv
import torch
from torchvision import tv_tensors
from typing import Any, Literal


def apply_thresholding(img: Any, method: Literal['mean', 'gaussian'] = 'mean') -> torch.Tensor:
    """
    Apply thresholding to input image.
    
    Parameters
    ----------
    img : Any
        Input image.
    method : Literal['mean', 'gaussian'], optional, default='mean'
        Thresholding method to apply.
    
    Returns
    -------
    image : Tensor
        Processed image.
    """

    # Apply median blur to images to reduce noise
    img = cv.medianBlur(img, 5)

    # Apply thresholding
    match method:

        case 'mean':
            img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

        case 'gaussian':
            img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    image = tv_tensors.Image(data=img, dtype=torch.float32) # type: ignore
    
    return image