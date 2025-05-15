import matplotlib.pyplot as plt
import numpy as np
from skimage.data import gravel
from skimage.filters import difference_of_gaussians, window
from scipy.fft import fftn, fftshift
import os
import numpy as np
import torch
import tifffile
from cellpose.models import Cellpose
from skimage import io
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import scipy.ndimage as ndi
from skimage import filters, morphology, measure, segmentation
from skimage.morphology import disk
from skimage.segmentation import watershed
import h5py

from cellstitch.pipeline import full_stitch
from cellpose import core, utils, io, models, metrics, plot
import cv2
from skimage.util import img_as_ubyte
from skimage.measure import label, regionprops

import h5py

from cellstitch.pipeline import full_stitch
#this runs cellpose 2d on all tif files in a folder
#you should change the output path, output filename probably to fit your own file


# Fill in on the path you would like to store the stitched mask
output_path = 'notebooks/output/' #
output_filename = 'BFP_60.npy' #these too

def get_files(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# log_file = open("notebooks/log.txt", "w")
flow_threshold = 1
use_gpu = True if torch.cuda.is_available() else False
    # print(use_gpu)
model = Cellpose(model_type='cyto2', gpu=use_gpu)
flow_threshold = 0.4

def get_mask(img):
    xy_masks, _, _, _ = model.eval([img], flow_threshold=flow_threshold, channels = [0,0])
    xy_masks = np.array(xy_masks)
    return xy_masks

def mask_outline(*masks, base, line_thick=1, overlap_color=(255, 255, 255)):
    """
    Draws colored contours for multiple masks and highlights overlaps.

    Parameters:
    - base: base image (2D or 3D)
    - *masks: variable number of 2D masks
    - line_thick: contour line thickness
    - overlap_color: RGB tuple for overlapping regions

    Returns:
    - RGB image with contours and highlighted overlaps
    """
    # cellpose function to convert image to rgb
    img_rgb = plot.image_to_rgb(base.copy(), channels=[0, 0])
    outline_image = img_rgb.copy()

    # possible colors, so technically limited to 10 masks
    color_palette = [
        (36, 255, 12),   # green
        (255, 128, 0),   # orange
        (0, 255, 0),     # lime
        (0, 128, 255),   # sky blue
        (0, 0, 255),     # blue
        (255, 0, 255),   # magenta
        (255, 0, 0),     # red
        (0, 255, 255),   # cyan
        (128, 0, 255),   # purple
        (128, 128, 0)    # olive
    ]

    # overlap tracker
    overlap_accumulator = np.zeros_like(masks[0], dtype=np.uint8)

    # iterate over masks
    for idx, mask in enumerate(masks):
        viz = mask
        gray = cv2.cvtColor(plot.image_to_rgb(viz, channels=[0, 0]), cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        color_choice = color_palette[idx % len(color_palette)]

        for c in cnts:
            cv2.drawContours(outline_image, [c], -1, color_choice, thickness=line_thick)

        # update overlap counter
        overlap_accumulator[mask > 0] += 1

    # detect overlapping regions (i.e., >1 mask)
    overlap_regions = (overlap_accumulator > 1).astype(np.uint8) * 255
    overlap_cnts = cv2.findContours(overlap_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlap_cnts = overlap_cnts[0] if len(overlap_cnts) == 2 else overlap_cnts[1]

    # draw in overlap contours in "overlap_color"
    for c in overlap_cnts:
        cv2.drawContours(outline_image, [c], -1, overlap_color, thickness=line_thick + 1)

    return img_as_ubyte(outline_image)

file_list= get_files("w1_images") #put your folder name here!!
num_cells_dict = dict()
for file in file_list:
    print(f"doing {file}")
    pathName = f"w1_images/{file}" #here too
    with tifffile.TiffFile(pathName, use_ome=False) as tif:
        if tif.is_ome:
            print(f"{file} is an OME-TIFF with {len(tif.series[0].pages)} pages")
            # Read the first series (common in OME-TIFFs)
            test_img = tif.series[0].asarray()
        else:
            print(f"{file} is a regular TIFF with {len(tif.pages)} pages")
            # Stack all pages manually
            test_img = tif.asarray()
    # tif = tifffile.TiffFile(pathName)
    # test_img = tif.asarray()
    print(test_img.shape)
    img = tifffile.imread(pathName)
    print(img.shape)
    max_proj = np.max(img, axis=0)
    print(max_proj.shape)
    plt.imshow(max_proj, cmap='gray')
    plt.show()
    tifffile.imsave(f"output/MaxProjections/max_proj_{file}.tif", max_proj)
    p_high = np.percentile(max_proj, 98.5)
    p_low = np.percentile(max_proj, 3)
    max_proj[max_proj > p_high] = p_low

    filtered_image = difference_of_gaussians(max_proj, 1, 90) 

   

    # default_mask = get_mask(image)
    filtered_mask = get_mask(filtered_image)
    image_outline = mask_outline(max_proj, filtered_mask)


plt.show()
