import os
import numpy as np
import torch
import tifffile
from cellpose.models import Cellpose
from skimage import io
import matplotlib.pyplot as plt


import h5py

from cellstitch.pipeline import full_stitch
#this runs cellpose 2d on all tif files in a folder
#you should change the output path, output filename probably to fit your own file


# Fill in on the path you would like to store the stitched mask
output_path = 'output/' #
output_filename = 'BFP_60.npy' #these too

def get_files(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


file_list= get_files("Test_images") #put your folder name here!!

for file in file_list:
    print(f"doing {file}")
    pathName = f"Test_images/{file}" #here too
    img = tifffile.imread(pathName)

    flow_threshold = 1
    use_gpu = True if torch.cuda.is_available() else False
    # print(use_gpu)
    model = Cellpose(model_type='cyto3', gpu=use_gpu)
    flow_threshold = 0.4

    xy_masks, _, _, _ = model.eval([img], flow_threshold=flow_threshold, channels = [0,0])
    xy_masks = np.array(xy_masks)
    print(np.unique(xy_masks))
    output_filename = f'{file}.npy'
    print(output_filename)


    np.save(os.path.join(output_path, output_filename), xy_masks)
    image_mask = np.load(f"output/{output_filename}")

    plt.figure(figsize=(10, 5))

    for i in range(1):  # Show first 3 slices
    # plt.subplot(1, 8, i + 1)
        plt.imshow(image_mask[i], cmap="gray")
        plt.title(f"{output_filename}")
        plt.axis("off")

    plt.show()