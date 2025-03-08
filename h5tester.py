import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('test.h5', 'r') as f:
    # Access a dataset
    # data = f['dataset_name'][:] 

    # # Access a group
    # group = f['group_name']

    # List all keys in the file
    array = f["raw"][:]
    print(array)
    # print(array.shape)
    # print(list(f.keys()))
    # print(type(f['label']))
    x = np.array(f['raw'])
    print(x.shape)
    print(x[0].shape)
    # y = np.load("outpupt.npy")
    # # y = np.array(x[0])
    # plt.imshow(y, cmap="gray")
    # plt.axis("off")
    # plt.savefig('ou2t.png', bbox_inches='tight', pad_inches=0)
    # plt.show()
  