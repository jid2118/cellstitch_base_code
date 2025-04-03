import os
import numpy as np
import torch
import tifffile
from cellpose.models import Cellpose
from skimage import io
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics, plot
from cellpose import models, io

import h5py

from cellstitch.pipeline import full_stitch

def run_cellstitch(image):
    pass
def run_mesmer(image):
    pass
def run_cellpose(image):    
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
    
def run_stardist(image):
    pass

def process_image(image_path):
    pass
def stat

def run_all():
    pass
