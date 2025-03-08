import os
import numpy as np
import torch
import tifffile
from cellpose.models import Cellpose
from skimage import io
print(np.__version__)

from cellstitch.pipeline import full_stitch
# from matplotlib import rcParams
# from IPython.display import display
# rcParams.update({'font.size': 10})