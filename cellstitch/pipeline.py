from cellpose.metrics import _label_overlap
from cellpose.utils import fill_holes_and_remove_small_masks

from .alignment import *


def relabel_layer(masks, z, lbls):
    """
    Relabel the label in LBLS in layer Z of MASKS.
    """
    layer = masks[z]
    if z != 0:
        reference_layer = masks[z - 1]
    else:
        reference_layer = masks[z + 1]

    overlap = _label_overlap(reference_layer, layer)

    for lbl in lbls:
        lbl0 = np.argmax(overlap[:, lbl])
        layer[layer == lbl] = lbl0


def overseg_correction(masks):
    lbls = np.unique(masks)[1:]

    # get a list of labels that need to be corrected
    layers_lbls = {}

    for lbl in lbls:
        existing_layers = np.sum(masks == lbl, axis=(1, 2), dtype="bool")
        depth = existing_layers.sum()

        if depth == 1:
            z = np.where(existing_layers != 0)[0][0]
            if z in layers_lbls.keys():
                layers_lbls[z].append(lbl)
            else:
                layers_lbls[z] = [lbl]

    for z, lbls in layers_lbls.items():
        relabel_layer(masks, z, lbls)

def full_stitch(xy_masks_prior, yz_masks, xz_masks, verbose=False):
    """
    Stitch masks in-place (top -> bottom).
    """
    xy_masks = xy_masks_prior.copy()
    num_frame = xy_masks.shape[0]
    prev_index = 0

    while Frame(xy_masks[prev_index]).is_empty():
        prev_index += 1

    curr_index = prev_index + 1

    while curr_index < num_frame:
        if Frame(xy_masks[curr_index]).is_empty():
            # if frame is empty, skip
            curr_index += 1
        else:
            if verbose:
                print("===Stitching frame %s with frame %s ...===" % (curr_index, prev_index))
            
            yz_not_stitched = (yz_masks[prev_index] != 0) * (yz_masks[curr_index] != 0) * (yz_masks[prev_index] != yz_masks[curr_index])
            xz_not_stitched = (xz_masks[prev_index] != 0) * (xz_masks[curr_index] != 0) * (xz_masks[prev_index] != xz_masks[curr_index])
     
            fp = FramePair(xy_masks[prev_index], xy_masks[curr_index], max_lbl=xy_masks.max())
            fp.stitch(yz_not_stitched, xz_not_stitched)
            xy_masks[curr_index] = fp.frame1.mask

            prev_index = curr_index
            curr_index += 1

    xy_masks = fill_holes_and_remove_small_masks(xy_masks)
    overseg_correction(xy_masks)
    return xy_masks
