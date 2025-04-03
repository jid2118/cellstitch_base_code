#common imports
import os
import numpy as np
import torch
import tifffile
import pandas as pd

from skimage import io
import matplotlib.pyplot as plt

from collections import Counter
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from tqdm import tqdm

#cellpose imports
from cellpose import core, utils, io, models, metrics, plot
from cellpose import models, io
from cellpose.models import Cellpose


import h5py

from cellstitch.pipeline import full_stitch
filename = "notebooks/Test_images/BFP_60.tif"

def process_image(filename):
    if filename[-3:] == 'npy':  # image in .npy format
        img = np.load(filename)
    elif filename[-3:] == 'tif': # imagge in TIFF format
        img = tifffile.imread(filename)
    else:
        try:
            img = io.imread(filename)
        except:
            raise IOError('Failed to load image {}'.format(filename))
    return img

def run_cellstitch(image):
    return 1
def run_mesmer(image):
    return 1


def run_cellpose(img):    
    channels = [[2,3], [0,0], [0,0]]
    model = models.Cellpose(gpu=False, model_type='cyto3')
    files = "output"
    masks, flows, styles, diams = model.eval(img, diameter=None, channels=channels)
    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, img, masks, flows[0], channels=channels)
    plt.tight_layout()
    plt.show()

    
    from scipy.ndimage import gaussian_filter

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the stained image with smoothing
    smoothed_image = gaussian_filter(img, sigma=1)
    ax[0].imshow(smoothed_image, cmap='gray')
    ax[0].set_title('Smoothed Stain Image')
    
    # Plot the segmentation mask
    ax[1].imshow(masks, cmap='jet', alpha=0.6)
    ax[1].set_title('Segmentation Mask')
    
    plt.tight_layout()
    plt.show()
    return masks
    
def run_stardist(image):
    return 1
def get_stats(mask):

    labels, counts = np.unique(mask, return_counts=True)
    cell_pixel_counts = {label: count for label, count in zip(labels, counts) if label != 0}
    print(cell_pixel_counts)
    print(len(labels)-1)


image = process_image(filename)
def run_all():
    cellstitch = run_cellstitch(image)
    stardist = run_stardist(image)
    cellpose = run_cellpose(image)
    mesmer = run_cellstitch(image)
    all_masks = [cellstitch, stardist, cellpose, mesmer]
    for mask in all_masks:
        get_stats(mask)
    
