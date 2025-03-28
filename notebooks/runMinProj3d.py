import os
import numpy as np
import torch
import tifffile
from cellpose.models import Cellpose
from skimage import io
import matplotlib.pyplot as plt


import h5py

from cellstitch.pipeline import full_stitch

# filename = "embryo/Image_060.tif" #change this!!
filename = 'Test_images/BFP_60.tif'

# maskname = '<path>/<filename>'

# Fill in on the path you would like to store the stitched mask
output_path = 'notebooks/output/'
output_filename = 'BFP_60.npy' #these too

if filename[-3:] == 'npy':  # image in .npy format
    img = np.load(filename)
elif filename[-3:] == 'tif': # imagge in TIFF format
    img = tifffile.imread(filename)
else:
    try:
        img = io.imread(filename)
    except:
        raise IOError('Failed to load image {}'.format(filename))
    
mt= []
for i in range(0, 21, 3):
    # print(test_arr[i:i+3])
    # print(i)
    min_proj = np.min(img[i:i+3], axis=0)
    print(len(min_proj))
    mt+=[ min_proj]
img= np.array(mt)

flow_threshold = 1
use_gpu = True if torch.cuda.is_available() else False
print(use_gpu)
model = Cellpose(model_type='cyto2', gpu=use_gpu)
flow_threshold = 0.4
xy_masks, _, _, _ = model.eval([img], flow_threshold=flow_threshold, channels = [0,0])
yz_masks, _, _, _ = model.eval([img.transpose(1,0,2)], flow_threshold=flow_threshold, channels = [0,0])
yz_masks = np.array(yz_masks).transpose(1,0,2)
print("done yz")

xz_masks, _, _, _ = model.eval([img.transpose(2,1,0)], flow_threshold=flow_threshold, channels = [0,0])
xz_masks = np.array(xz_masks).transpose(2,1,0)
print("done xz")

cellstitch_masks = full_stitch(xy_masks, yz_masks, xz_masks)
np.save(os.path.join(output_path, output_filename), cellstitch_masks)
image_mask = np.load("output/tcell_T010.tif.npy")
plt.figure(figsize=(30, 15))

for i in range(3):  # Show first 3 slices
    # plt.subplot(1, 8, i + 1)
    plt.imshow(image_mask[i], cmap="gray")
    plt.title(f"Slice {i}")
    plt.axis("off")
# plt.savefig(f"output/{image_mask}.png")
plt.show()