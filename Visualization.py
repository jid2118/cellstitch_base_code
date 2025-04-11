# from output import make_contour_mask, make_rgb_mask
import os
import numpy as np
import tifffile
import tensorflow as tf
from deepcell.applications import Mesmer
import matplotlib.pyplot as plt
from skimage import measure
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from deepcell.utils.plot_utils import make_outline_overlay, create_rgb_image
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

#This only returns the contours
def make_contour_mask(predictions):
    """Create a binary mask of segmentation boundaries (contours) only.

    Args:
        predictions: 4D array of segmentation predictions of shape (batch, height, width, channel)

    Returns:
        numpy.array: binary mask of contours with shape (batch, height, width, channel)

    Raises:
        ValueError: If predictions are not 4D
    """
    
    if len(predictions.shape) != 4:
        raise ValueError(f'Predictions must be 4D, got {predictions.shape}')

    batch_size, height, width, _ = predictions.shape
    contour_mask = np.zeros((batch_size, height, width, 1), dtype=np.uint8)

    for img in range(batch_size):
        boundary = find_boundaries(predictions[img, ..., 0], connectivity=1, mode='inner')
        contour_mask[img, ..., 0] = boundary.astype(np.uint8)

    return contour_mask

def make_rgb_mask(predictions):
    """Create a color map of segmentation predictions.

    Args:
        predictions: 4D array of segmentation predictions of shape (batch, height, width, channel)

    Returns:
        numpy.array: binary mask of contours with shape (batch, height, width, channel)

    Raises:
        ValueError: If predictions are not 4D
    """
    # Isolate (0, height, width, 0)
    label_mask = predictions[0, ..., 0]
    rgb_mask = label2rgb(label_mask, bg_label=0)

    return rgb_mask

#Returns the contours ON TOP of the original image
def overlay_countors(image, mask):
  # create overlay of predictions
  from deepcell.utils.plot_utils import make_outline_overlay
  image_overlay = make_outline_overlay(rgb_data=image, predictions=prediction)
  return image_overlay
