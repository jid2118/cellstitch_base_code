import os
import ot
import ot.plot
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import ndimage as ndi

from cellpose import utils as cp_utils
from cellpose import plot as cp_plot


#-------------------------------
# interpolation helper functions
#-------------------------------

def comp_match_plan(pc1, pc2, dist='sqeuclidean'):
    """Compute optimal matching plans between 2 sets of point clouds"""
    # compute cost matrix
    C = ot.dist(pc1, pc2, metric=dist).astype(np.float64)
    C /= C.max()

    # convert point clouds to uniform distributions
    n_pts1, n_pts2 = pc1.shape[0], pc2.shape[0]
    mu1, mu2 = np.ones(n_pts1) / n_pts1, np.ones(n_pts2) / n_pts2

    # compute transport plan
    plan = ot.emd(mu1, mu2, C)

    return plan


def interpolate(pc1, pc2, dist='sqeuclidean', anisotropy=2):
    """
    Calculate interpolated predictions

    Parameters
    ----------
    pc1 : np.ndarray
        Point cloud representing cell boundary in frame 1

    pc2 : np.ndarray
        Point cloud representing cell boundary in frame 2

    anisotropy : int
        Ratio of sampling rate between different axes

    Returns
    -------
    interp_pcs : list
        Smoothed boundary locations along interpolated layers
    """
    alphas = np.linspace(0, 1, anisotropy + 1)[1:-1]
    plan = comp_match_plan(pc1, pc2, dist=dist)
    normalized_plan = plan / plan.sum(axis=1, keepdims=1)  # normalize so that the row sum is 1

    interp_pcs = []

    for alpha in alphas:
        n_pts = pc1.shape[0]
        avg_pc = np.zeros((n_pts, 2), dtype=int)

        for i in range(n_pts):
            point = pc1[i]
            target_weights = normalized_plan[i]

            weighted_target = np.array([np.sum(target_weights * pc2[:, 0]),
                                        np.sum(target_weights * pc2[:, 1])])

            avg_pc[i, :] = point * (1 - alpha) + alpha * weighted_target

        interp_pcs.append(avg_pc)

    return interp_pcs


#-----------------------------------------
# Util functions for interp reconstruction
#-----------------------------------------

def get_lbls(mask):
    """Get unique labels from the predicted masks"""
    return np.unique(mask)[1:]


def min_size_filter(res, thld=100):
    """Filter out all masks with area below threshold"""
    assert len(res) == 4, "Invalid Cellpose 4-tuple result"
    preds = res[0]

    for i in range(len(preds)):
        lbls = get_lbls(preds[i])
        for lbl in lbls:
            msk = (preds[i] == lbl)
            if msk.sum() < thld:
                coords = np.nonzero(msk)
                preds[i][coords] = 0

    res_filtered = (preds, res[1], res[2], res[3])
    return res_filtered


def get_contours(masks):
    """Transfer solid mask predictions to non-overlapping contours w/ distinct integers"""
    masks_new = masks.copy()
    outlines = cp_utils.masks_to_outlines(masks)
    masks_new[~outlines] = 0

    return masks_new


def get_mask_perimeter(masks, lbl, is_contour=False):
    assert lbl in get_lbls(masks), \
        "Label {} doesn't in current mask predictions".format(lbl)

    if is_contour:
        p = (masks == lbl).sum()
    else:
        mask_lbl = (masks == lbl).astype(np.uint8)
        p = cp_utils.masks_to_outlines(mask_lbl).sum()

    return p

def calc_vols(pred):
    """
    Calculate volumes of each mask from predictions
    """
    lbls = get_lbls(pred)
    vols = [(pred == lbl).sum() for lbl in lbls]
    return vols


def calc_depth(masks):
    """
    Calculate z-layer depth of predictions
    """
    assert masks.ndim == 3, "Mask predictions must be 3D to calculate depth"

    lbls = get_lbls(masks)
    depths = np.vectorize(
        lambda lbl:
        np.diff(np.nonzero(masks == lbl)[0][[0, -1]])[0]
    )(lbls)

    return depths


def mask_to_coord(mask):
    """ Return (n, 2) coordinates from masks """
    coord = np.asarray(np.nonzero(mask)).T
    return coord


def coord_to_mask(coord, size, lbl):
    """ Convert from coordinates to original labeled masks """
    mask = np.zeros(size)
    mask[tuple(coord.T)] = lbl
    return mask


def contour_to_mask(contour):
    lbl = get_lbls(contour)[0]
    """ Convert contour to solid masks with fill-in labels"""
    binary_contour = (contour > 0)
    binary_mask = ndi.binary_fill_holes(binary_contour)

    mask = np.zeros_like(binary_contour)
    mask[binary_mask] = lbl

    return mask


def connect(coord1, coord2, mask):
    """
    Modify the mask by connecting the two given coordinates.

    Parameters
    ----------
    coord1 : [x1, y1]
        Coordinate of the first pixel.

    coord2 : [x2, y2]
        Coordinate of the second pixel.

    mask : binary np.narray
        Binary mask of the boundary.
    """
    x1, y1 = coord1
    x2, y2 = coord2

    x_offset, y_offset = x2 - x1, y2 - y1

    # skip if the two coordinates are already connected
    if x_offset ** 2 + y_offset ** 2 <= 2:
        return

    diag_length = min(abs(x_offset), abs(y_offset))

    # initialize at coord1
    added_x, added_y = x1, y1

    # first, add diagonal pixels
    for i in range(1, diag_length + 1):
        added_x = x1 + i * np.sign(x_offset)
        added_y = y1 + i * np.sign(y_offset)

        mask[added_x, added_y] = 1

        # need to walk vertically
    if added_x == x2 and added_y != y2:
        offset = abs(added_y - y2)
        for i in range(1, offset + 1):
            mask[added_x, added_y + i * np.sign(y_offset)] = 1

            # or, now need to walk horizonally
    if added_y == y2 and added_x != x2:
        offset = abs(added_x - x2)
        for i in range(1, offset + 1):
            mask[added_x + i * np.sign(x_offset), added_y] = 1


def calc_angles(sc_pt, tg_pts, eps=1e-20):
    """
    Calculate angle (rad) between source point (sc_pt) & list of target points(tg_pts, dim=(n, 2))
    """
    sc_pts = np.tile(sc_pt, (tg_pts.shape[0], 1))
    diff = tg_pts - sc_pts
    angles = np.apply_along_axis(
        lambda x: np.arctan2(x[1], x[0]+eps),
        axis=1,
        arr=diff
    )

    return angles


def connect_boundary(coords, size, lbl=1):
    """
    Connect interpolation coordinates to generate close-loop mask contours

    Parameters
    ----------
    coords : ndarray, shape (ns,2)
        Boundary coordinates (might be disconnected).

    size: (n1, n2)
        Shape of the final mask.

    lbl: int
        Label of the original mask.

    Returns
    -------
    mask: ndarray, shape (n1, n2)
        Connected boundary mask.

    """
    # Sort boundary labels by angle to mask's mass center
    mass_center = np.round(coords.mean(0)).astype(np.int64)
    angles = calc_angles(mass_center, coords)
    sorted_coords = coords[angles.argsort()]

    mask = coord_to_mask(coords, size, lbl)

    for i, (x, y) in enumerate(sorted_coords[:-1]):
        next_x, next_y = sorted_coords[i+1]
        connect((x, y), (next_x, next_y), mask)

    connect(tuple(sorted_coords[-1]), tuple(sorted_coords[0]), mask)

    return mask


#-----------------------------
# Core Interpolation functions
#-----------------------------

def interp_layers(sc_mask, tg_mask, dist='sqeuclidean', anisotropy=2):
    """
    Interpolating adjacent z-layers
    """

    def _dilation(coords, lims):
        y, x = coords
        ymax, xmax = lims
        dy, dx = np.meshgrid(np.arange(y-2, y+3), np.arange(x-2, x+3), indexing='ij')
        dy, dx = dy.flatten(), dx.flatten()
        mask = np.logical_and(
            np.logical_and(dy >= 0, dx >= 0),
            np.logical_and(dy < ymax, dx < xmax)
        )
        return dy[mask], dx[mask]

    shape = sc_mask.shape
    sc_contour = get_contours(sc_mask)
    tg_contour = get_contours(tg_mask)

    # Boundary condition: if empty on source / target label
    # align the empty slice w/ mass centers to represent instance endings
    sc_dummy = np.zeros_like(sc_mask)
    tg_dummy = np.zeros_like(tg_mask)
    if not np.intersect1d(get_lbls(sc_contour), get_lbls(tg_contour)).size:
        if (sc_contour.sum() == tg_contour.sum() == 0) or (np.logical_and(sc_mask, tg_mask).sum() > 0):
            return np.zeros(shape)
        get_mask_center = lambda x: (
            np.round(np.nonzero(x)[0].sum() / x.sum()).astype(np.uint16),
            np.round(np.nonzero(x)[1].sum() / x.sum()).astype(np.uint16)
        )
        for lbl in get_lbls(sc_contour):
            yc, xc = _dilation(get_mask_center(sc_mask == lbl), sc_mask.shape)
            tg_dummy[yc, xc] = lbl
        for lbl in get_lbls(tg_contour):
            yc, xc = _dilation(get_mask_center(tg_mask == lbl), tg_mask.shape)
            sc_dummy[yc, xc] = lbl
        sc_contour += sc_dummy
        tg_contour += tg_dummy

    joint_lbls = np.intersect1d(
        get_lbls(sc_contour),
        get_lbls(tg_contour)
    )

    interp_masks = np.zeros((
        anisotropy+1,      # num. interpolated layers
        len(joint_lbls),   # num. individual masks
        shape[0],          # x
        shape[1]           # y
    ))

    for i, lbl in enumerate(joint_lbls):
        sc_ct = (sc_contour== lbl).astype(np.uint8)
        tg_ct = (tg_contour == lbl).astype(np.uint8)

        sc_coord = mask_to_coord(sc_ct)
        tg_coord = mask_to_coord(tg_ct)

        interp_coords = interpolate(sc_coord, tg_coord, dist=dist, anisotropy=anisotropy)
        interps = [
            ndi.binary_fill_holes(connect_boundary(interp, shape)) * lbl
            for interp in interp_coords
        ]

        interp_masks[1:-1, i,...] = interps

    interp_masks = interp_masks.max(1)
    interp_masks[0] = sc_mask
    interp_masks[-1] = tg_mask

    return interp_masks


def full_interpolate(masks, anisotropy=2, dist='sqeuclidean', verbose=False):
    """
    Interpolating between all adjacent z-layers

    Parameters
    ----------
    masks : np.ndarray
        layers of 2D predictions
        (dim: (Depth, H, W))

    anisotropy : int
        Ratio of sampling rate between xy-axes & z-axis

    Returns
    -------|
    interp_masks : np.ndarray
        interpolated masks
        (dim: (Depth * anisotropy - (anisotropy-1), H, W))

    """
    interp_masks = np.zeros((
        len(masks) + (len(masks)-1)*(anisotropy-1),
        masks.shape[1],
        masks.shape[2]
    ))

    idx = 0
    for i, sc_mask in enumerate(masks[:-1]):
        if verbose and i % 20 == 0:
            print('Interpolating layer {} & {}...'.format(i, i+1))
        tg_mask = masks[i+1]
        interps = interp_layers(sc_mask, tg_mask, dist=dist, anisotropy=anisotropy)
        interp_masks[idx:idx+anisotropy+1] = interps
        idx += anisotropy

    return interp_masks
