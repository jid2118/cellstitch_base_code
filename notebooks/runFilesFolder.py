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
output_path = 'notebooks/output/' #
output_filename = 'BFP_60.npy' #these too

def get_files(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

log_file = open("notebooks/log.txt", "w")

file_list= get_files("notebooks/Test_images") #put your folder name here!!
num_cells_dict = dict()
for file in file_list:
    print(f"doing {file}")
    pathName = f"notebooks/Test_images/{file}" #here too
    img = tifffile.imread(pathName)

    flow_threshold = 1
    use_gpu = True if torch.cuda.is_available() else False
    # print(use_gpu)
    model = Cellpose(model_type='cyto3', gpu=use_gpu)
    flow_threshold = 0.4

    xy_masks, _, _, _ = model.eval([img], flow_threshold=flow_threshold, channels = [0,0])
    xy_masks = np.array(xy_masks)
    # print(np.unique(xy_masks))
    labels, counts = np.unique(xy_masks, return_counts=True)
    cell_pixel_counts = {label: count for label, count in zip(labels, counts) if label != 0}
    # print(cell_pixel_counts)
    print(file, file=log_file)
    print(cell_pixel_counts, file=log_file)


    output_filename = f'{file}.npy'
    num_cells_dict[output_filename] = len(np.unique(xy_masks)-1)
    # print(output_filename)


    np.save(os.path.join(output_path, output_filename), xy_masks)
    image_mask = np.load(f"notebooks/output/{output_filename}")

log_file.close()

# plt.figure(figsize=(10, 5))
print(num_cells_dict)
#plots all outputs on one subdiagram
outputs = get_files("notebooks/output")
num_files = len(outputs)
cols = min(num_files, 5)  # max 5 columns
rows = (num_files + cols - 1) // cols  # compute number of rows

plt.figure(figsize=(4 * cols, 4 * rows))

for i, file in enumerate(outputs):
    plt.subplot(rows, cols, i + 1)
    data = np.load(f"notebooks/output/{file}")[0]  # assuming shape (1, H, W)
    plt.imshow(data, cmap="flag")
    plt.title(f"Unique Cells in {file}: {num_cells_dict[file]}", fontsize=8)
    plt.axis("off")

plt.savefig("notebooks/all_masks.png", dpi=300)
# plt.tight_layout()
plt.show()
