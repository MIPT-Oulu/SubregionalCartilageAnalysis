import numpy as np
import scipy.ndimage as ndi
import torch


def distance_transform(input_, num_classes, spacing_mm=(1, 1, 1), skip_classes=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
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

            th_map_class = ndi.distance_transform_edt(sel_input_, sampling=spacing_mm)

            th_map[sel_input_] = th_map_class[sel_input_]
        th_maps[sample_idx, :] = th_map

    return th_maps
