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

# stardist imports
from stardist.models import StarDist2D

#mesmer imports
import tensorflow as tf
from deepcell.applications import Mesmer
from deepcell.applications.mesmer import mesmer_preprocess, mesmer_postprocess


from scipy.ndimage import gaussian_filter

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

def run_cellstitch(img):
    flow_threshold = 1
    use_gpu = True if torch.cuda.is_available() else False
    # print(use_gpu)
    model = Cellpose(model_type='cyto3', gpu=use_gpu)
    flow_threshold = 0.4
    xy_masks, _, _, _ = model.eval([img], flow_threshold=flow_threshold, channels = [0,0])
    xy_masks = np.array(xy_masks)
    return xy_masks
    #honestly irrelevant since cellstitch 2d is any 2d segmentation tbh
def run_mesmer(image):
    image_copy = image
    #intialize model
    app = mesmer()

    #Image needs to be in the following format Rank 4: (batch, X, Y, channels)
    #Usually has (X, Y, channels), need to add the batch dimension
    image_copy = np.expand_dims(image_copy, axis=0)
    image_copy = np.moveaxis(image_copy, 1, -1)
    image_copy = mesmer_preprocess(image_copy)

    #Generate Segmentations, returns a numpy.ndarray Object
    prediction = app.predict(RFP, image_mpp=0.5, batch_size=1, compartment="whole-cell")

    return prediction


def run_stardist(img):
    # Loading Stardist model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Predicting segmentation
    labels, details = model.predict(img)
    
    # Visualization
    # fig, ax = plt.subplots(1, 2, figsize=(10,5))

    # Plotting smoothed image
    # smoothed_image = gaussian_filter(img, sigma=1)
    # ax[0].imshow(smoothed_image, cmap='gray')
    # ax[0].set_title('Smoothed Stain Image')
    
    # # Plot the segmentation mask
    # ax[1].imshow(masks, cmap='jet', alpha=0.6)
    # ax[1].set_title('Segmentation Mask')
    
    # plt.tight_layout()
    # plt.show()
    return labels


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


def get_stats(mask):

    labels, counts = np.unique(mask, return_counts=True)
    cell_pixel_counts = {label: count for label, count in zip(labels, counts) if label != 0}
    print(cell_pixel_counts)
    print(f"there are {len(labels)-1} unique cells.")

def cellpose_plotting(img, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the stained image with smoothing
    smoothed_image = gaussian_filter(img, sigma=1)
    ax[0].imshow(smoothed_image, cmap='gray')
    ax[0].set_title('Smoothed Stain Image')
    
    # Plot the segmentation mask
    ax[1].imshow(mask, cmap='jet', alpha=0.6)
    ax[1].set_title('Segmentation Mask')
    
    plt.tight_layout()
    return fig

image = process_image(filename)
def run_all():
    cellstitch = run_cellstitch(image)
    stardist = run_stardist(image)
    cellpose = run_cellpose(image)
    mesmer = run_cellstitch(image)
    all_masks = [cellstitch, stardist, cellpose, mesmer]
    mask_names = ["cellstitch", "stardist", "cellpose", "mesmer"]
    for i, mask in enumerate(all_masks):
        my_fig = cellpose_plotting(image, mask)
        my_fig.savefig(f"notebooks/{mask_names[i]}_segmentation.png", dpi=300)
        get_stats(mask)

if __name__ == '__main__':
    run_all()
    
