import numpy as np
import scipy.ndimage as ndi
from skimage import morphology
from numba import jit, prange
import torch


@jit(nopython=True)
def _local_thick_2d(mask, med_axis, distance, search_extent):
    out = np.zeros_like(mask, dtype=np.float32)

    nonzero_x = np.nonzero(mask)
    ii = nonzero_x[0]
    jj = nonzero_x[1]

    nonzero_med_axis = np.nonzero(med_axis)
    mm = nonzero_med_axis[0]
    nn = nonzero_med_axis[1]

    for e in range(len(ii)):
        i = ii[e]
        j = jj[e]

        best_val = 0

        if search_extent is not None:
            r0 = max(i - search_extent[0], 0)
            r1 = min(i + search_extent[0], mask.shape[0] - 1)
            c0 = max(j - search_extent[1], 0)
            c1 = min(j + search_extent[1], mask.shape[1] - 1)

        for w in range(len(mm)):
            m = mm[w]
            n = nn[w]

            if search_extent is not None:
                if m < r0 or m > r1 or n < c0 or n > c1:
                    continue

            if ((i - m) ** 2 + (j - n) ** 2) < (distance[m, n] ** 2):
                if distance[m, n] > best_val:
                    best_val = distance[m, n]

        out[i, j] = best_val
    return out


@jit(nopython=True, parallel=True)
def _local_thick_3d(mask, med_axis, distance, search_extent):
    out = np.zeros_like(mask, dtype=np.float32)

    nonzero_mask = np.nonzero(mask)
    ii = nonzero_mask[0]
    jj = nonzero_mask[1]
    kk = nonzero_mask[2]
    num_pts_mask = len(ii)

    nonzero_med_axis = np.nonzero(med_axis)
    mm = nonzero_med_axis[0]
    nn = nonzero_med_axis[1]
    oo = nonzero_med_axis[2]
    num_pts_med_axis = len(mm)

    best_vals = np.zeros((num_pts_mask, ))

    for e in prange(num_pts_mask):
        i = ii[e]
        j = jj[e]
        k = kk[e]

        if search_extent is not None:
            r0 = max(i - search_extent[0], 0)
            r1 = min(i + search_extent[0], mask.shape[0] - 1)
            c0 = max(j - search_extent[1], 0)
            c1 = min(j + search_extent[1], mask.shape[1] - 1)
            p0 = max(k - search_extent[2], 0)
            p1 = min(k + search_extent[2], mask.shape[2] - 1)

        for w in range(num_pts_med_axis):
            m = mm[w]
            n = nn[w]
            o = oo[w]

            if search_extent is not None:
                if m < r0 or m > r1 or n < c0 or n > c1 or o < p0 or o > p1:
                    continue

            if ((i - m) ** 2 + (j - n) ** 2 + (k - o) ** 2) <= (distance[m, n, o] ** 2):
                if distance[m, n, o] > best_vals[e]:
                    best_vals[e] = distance[m, n, o]

        out[i, j, k] = best_vals[e]
    return out


def _local_thickness(mask, *, mode='med2d_dist3d_lth3d',
                     spacing_mm=None, stack_axis=None,
                     thickness_max_mm=None,
                     return_med_axis=False, return_distance=False):
    """
    Inspired by https://imagej.net/Local_Thickness .

    Args:
        mask: (D0, D1[, D2]) ndarray
        mode: One of {'straight_skel_3d', 'stacked_2d',
               'med2d_dist2d_lth3d', 'med2d_dist3d_lth3d'} or None
            Implementation mode for 3D ``mask``. Ignored for 2D.
        spacing_mm: tuple of ``mask.ndim`` elements
            Size of ``mask`` voxels in mm.
        stack_axis: None or int
            Index of axis to perform slice selection along. Ignored for 2D.
        thickness_max: None or int
            Hypothesised maximum thickness in absolute values.
            Used to constrain local ROIs to speed up best candidate search.
        return_med_axis: bool
            Whether to return the medial axis.
        return_distance: bool
            Whether to return the distance transform.

    Returns:
        out: ndarray
            Local thickness.
        med_axis: ndarray
            Medial axis. Returned only if ``return_med_axis`` is True.
        distance: ndarray
            Distance transform. Returned only if ``return_distance`` is True.
    """
    # 1. Compute the distance transform
    # 2. Find the distance ridge (/ exclude the redundant points)
    # 3. Compute local thickness
    if spacing_mm is None:
        spacing_mm = (1,) * mask.ndim

    if thickness_max_mm is None:
        search_extent = None
    else:
        # Distance to the closest surface point is half of the thickness
        distance_max_mm = thickness_max_mm / 2.
        search_extent = np.ceil(distance_max_mm / np.asarray(spacing_mm)).astype(np.uint)

    if mask.ndim == 2:
        med_axis = morphology.medial_axis(mask)
        distance = ndi.distance_transform_edt(mask, sampling=spacing_mm)
        out = _local_thick_2d(mask=mask, med_axis=med_axis, distance=distance,
                              search_extent=search_extent)

    elif mask.ndim == 3:
        if mode == 'straight_skel_3d':
            from warnings import warn
            msg = 'Straight skeleton is not suitable for local thickness'
            warn(msg)
            if thickness_max_mm is not None:
                msg = f'`thickness_max_mm` is not supported in mode {mode}'
                raise NotImplementedError()
            skeleton = morphology.skeletonize_3d(mask)
            distance = ndi.distance_transform_edt(mask, sampling=spacing_mm)
            out = _local_thick_3d(mask=mask, med_axis=skeleton, distance=distance)
            med_axis = skeleton

        elif mode == 'stacked_2d':
            if thickness_max_mm is not None:
                msg = f'`thickness_max_mm` is not supported in mode {mode}'
                raise NotImplementedError()

            acc_med = []
            acc_dist = []
            acc_out = []

            for idx_slice in range(mask.shape[stack_axis]):
                sel_idcs = [slice(None), ] * mask.ndim
                sel_idcs[stack_axis] = idx_slice
                sel_idcs = tuple(sel_idcs)

                if spacing_mm is None:
                    sel_spacing = None
                else:
                    sel_spacing = (list(spacing_mm[:stack_axis]) +
                                   list(spacing_mm[stack_axis+1:]))
                sel_mask = mask[sel_idcs]
                sel_res = _local_thickness(sel_mask, spacing_mm=sel_spacing,
                                           return_med_axis=True, return_distance=True)
                acc_med.append(sel_res[1])
                acc_dist.append(sel_res[2])
                acc_out.append(sel_res[0] / 2)

            med_axis = np.stack(acc_med, axis=stack_axis)
            distance = np.stack(acc_dist, axis=stack_axis)
            out = np.stack(acc_out, axis=stack_axis)

        elif mode == 'med2d_dist2d_lth3d':
            if thickness_max_mm is not None:
                msg = f'`thickness_max_mm` is not supported in mode {mode}'
                raise NotImplementedError(msg)

            acc_med = []
            acc_dist = []

            for idx_slice in range(mask.shape[stack_axis]):
                sel_idcs = [slice(None), ] * mask.ndim
                sel_idcs[stack_axis] = idx_slice
                sel_idcs = tuple(sel_idcs)

                sel_med = morphology.medial_axis(mask[sel_idcs])
                sel_dist = ndi.distance_transform_edt(mask[sel_idcs], sampling=spacing_mm)
                acc_med.append(sel_med)
                acc_dist.append(sel_dist)

            med_axis = np.stack(acc_med, axis=stack_axis)
            distance = np.stack(acc_dist, axis=stack_axis)

            out = _local_thick_3d(mask=mask, med_axis=med_axis, distance=distance)

        elif mode == 'med2d_dist3d_lth3d':
            acc_med = []

            for idx_slice in range(mask.shape[stack_axis]):
                sel_idcs = [slice(None), ] * mask.ndim
                sel_idcs[stack_axis] = idx_slice
                sel_idcs = tuple(sel_idcs)

                sel_res = morphology.medial_axis(mask[sel_idcs])
                acc_med.append(sel_res)

            med_axis = np.stack(acc_med, axis=stack_axis)
            distance = ndi.distance_transform_edt(mask, sampling=spacing_mm)
            out = _local_thick_3d(mask=mask, med_axis=med_axis, distance=distance,
                                  search_extent=search_extent)

        elif mode == 'exact_3d':
            raise NotImplementedError(f'Mode {mode} is not yet supported')

        else:
            raise ValueError(f'Invalid mode: {mode}')

    else:
        msg = 'Only 2D and 3D arrays are supported'
        raise ValueError(msg)

    # Thickness is twice the distance to the closest surface point
    out = 2 * out

    if return_med_axis:
        if return_distance:
            return out, med_axis, distance
        else:
            return out, med_axis
    else:
        if return_distance:
            return out, distance
        else:
            return out


def local_thickness(input_, num_classes, stack_axis, spacing_mm=(1, 1, 1),
                    skip_classes=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        stack_axis: int
            Index of axis to perform slice selection along. Ignored for 2D.
        spacing_mm: 3-tuple
            Pixel spacing in mm, one per each spatial dimension of `input_`.
        skip_classes: None or tuple of ints
    Returns:
        out: (b, d0, ..., dn) ndarray
            Thickness map for each class in each batch sample.

    """
    if skip_classes is None:
        skip_classes = tuple()

    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
        dims = tuple(input_.size())[1:]
    else:
        num_samples = input_.shape[0]
        dims = input_.shape[1:]

    th_maps = np.zeros((num_samples, *dims))

    for sample_idx in range(num_samples):
        th_map = np.zeros_like(input_[sample_idx])

        for class_idx in range(num_classes):
            if class_idx in skip_classes:
                continue

            sel_input_ = input_[sample_idx] == class_idx

            th_map_class = _local_thickness(
                sel_input_, mode='med2d_dist3d_lth3d',
                spacing_mm=spacing_mm, stack_axis=stack_axis,
                return_med_axis=False, return_distance=False)

            th_map[sel_input_] = th_map_class[sel_input_]
        th_maps[sample_idx, :] = th_map

    return th_maps
