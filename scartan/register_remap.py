from pathlib import Path
import shutil
import logging
import tempfile

import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
from joblib import Parallel, delayed
import pandas as pd
import torch
import click
from tqdm import tqdm

from scartan.datasets.constants import (valid_mappings_zp3n_61om,
                                        valid_mappings_zp3n_w02n,
                                        valid_mappings_zp3n_mh53)
from scartan.various import png_to_nifti, nifti_to_numpy
from scartan.registration import Elastix


logging.basicConfig()
logger = logging.getLogger('register')
logger.setLevel(logging.DEBUG)


def multiatlas_from_path(path_root, suffix):
    """

    Args:
        path_root: str
            Path to multiatlas root directory.
        suffix: str
            Suffix used for naming of the atlas masks.

    Returns:
        multiatlas: list
            Multiatlas of dict-like atlases.
    """
    paths_scans = Path(path_root).glob('*/*/*')

    multiatlas = []

    for path_scan in paths_scans:
        fields = str(path_scan).rsplit('/')[-3:]
        atlas = {
            'path_image': Path(path_scan, 'image_in.nii'),
            'path_mask': Path(path_scan, 'mask_in.nii'),
            f'path_mask_{suffix}': Path(path_scan, f'mask_in_{suffix}.nii'),
            'patient': fields[0],
            'release': fields[1],
            'sequence': fields[2],
        }
        multiatlas.append(atlas)
    return multiatlas


def remap_to_closest_labels(map_input, map_reference, mappings,
                            labels_retain=None, spacings=None):
    """

    Args:
        map_input: [D0, D1, D2] uint array

        map_reference: [D0, D1, D2] uint array

        mappings: dict
            Valid mappings from `map_input` to `map_reference` labels.
        labels_retain: tuple of int
            Labels of `map_input` to retain in place. If None, set to (0, ).
        spacings: 3-tuple of float
            (pixel spacing in r, pixel spacing in c, slice thickness).

    Returns:
        map_result: [D0, D1, D2] uint array

    """
    if labels_retain is None:
        labels_retain = (0, )

    map_result = np.zeros_like(map_input)

    labels_input = np.unique(map_input).astype(np.uint)

    for li in labels_input:
        sel_input = (map_input == li)

        if li in labels_retain:
            # Transfer ignored labels as-is
            map_result[sel_input] = li
        else:
            if li not in mappings:
                raise ValueError(f'Label {li} is not found in the mapping')
            # Consider only valid mappings (i.e. "femoral" -> "posterior femoral")
            labels_reference_sel = mappings[li]

            # Build the proximity map
            tmp_bg = np.isin(map_reference, labels_reference_sel, invert=True)
            tmp_dt = distance_transform_edt(tmp_bg, sampling=spacings,
                                            return_distances=False, return_indices=True)
            # Transfer the values
            map_result[sel_input] = map_reference[tmp_dt[0][sel_input],
                                                  tmp_dt[1][sel_input],
                                                  tmp_dt[2][sel_input]]

    return map_result


def fuse_labels(labels):
    """Perform majority voting"""
    tmp = np.stack(labels, axis=0)
    res = torch.mode(torch.Tensor(tmp), dim=0).values.numpy()
    # res, _ = scipy.stats.mode(tmp, axis=0)  # Do not use SciPy as it is slow
    return np.squeeze(res).astype(np.uint8)


@click.command()
@click.option('--path_experiment_root', type=click.Path(exists=True),
              default='../../results/temporary')
@click.option('--path_atlas_root', type=click.Path(exists=True),
              default='../../data/atlas')
@click.option('--atlas_suffix')
@click.option('--path_elastix_root', type=click.Path(exists=True),
              default='/home/egor/Software/elastix-5.0.0-linux')
@click.option('--path_config_elastix', type=click.Path(exists=True),
              default='./registration/config_fs4_similarity.txt')
@click.option('--dataset', type=click.Choice(['oai_imo', 'oai_prj_22', ]))
@click.option('--no_prep', is_flag=True)
@click.option('--num_workers', default=12, type=int)
@click.option('--num_threads_elastix', default=12, type=int)
def main(**config):
    config['path_experiment_root'] = Path(config['path_experiment_root']).resolve()

    config['path_logs'] = Path(config['path_experiment_root'], 'logs_register')
    config['path_logs'].mkdir(exist_ok=True)

    logging_fh = logging.FileHandler(Path(config['path_logs'], 'main.log'))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)

    atlas_suffix = config['atlas_suffix']
    multiatlas = multiatlas_from_path(config['path_atlas_root'], suffix=atlas_suffix)
    if atlas_suffix == 'imo':
        valid_mappings = valid_mappings_zp3n_mh53
    elif atlas_suffix == 'biomediq':
        valid_mappings = valid_mappings_zp3n_61om
    elif atlas_suffix == 'chondr75n':
        valid_mappings = valid_mappings_zp3n_w02n
    else:
        msg = f'Unknown atlas suffix {atlas_suffix}'
        raise ValueError(msg)

    elastix = Elastix(path_root=config['path_elastix_root'])

    path_meta = Path(config['path_experiment_root'],
                     f"predicts_{config['dataset']}_test",
                     'meta_dynamic.csv')
    df_meta = pd.read_csv(path_meta,
                          dtype={'patient': str,
                                 'release': str,
                                 'sequence': str,
                                 'side': str,
                                 'slice_idx': int,
                                 'pixel_spacing_0': float,
                                 'pixel_spacing_1': float,
                                 'slice_thickness': float,
                                 'KL': int,
                                 'has_mask': int},
                          index_col=False)

    groupers_stack = ['patient', 'release', 'sequence', 'side']

    gb_iter = df_meta.groupby(groupers_stack)
    for gb_name, gb_df in tqdm(gb_iter, total=gb_iter.ngroups):
        patient, release, sequence, side = gb_name

        dir_scan_base = Path(config['path_experiment_root'],
                             f"predicts_{config['dataset']}_test",
                             patient, release, sequence)
        dir_tmp = tempfile.mkdtemp()

        # Flip the scans to the left side to make the registration feasible
        if side == 'RIGHT':
            reverse = True
        else:
            reverse = False

        spacings_in = (gb_df['pixel_spacing_0'].iloc[0],
                       gb_df['pixel_spacing_1'].iloc[0],
                       gb_df['slice_thickness'].iloc[0])

        # Convert the image to NIfTI
        patt = Path(dir_scan_base, 'image_prep', '*.png')
        path_image_in = Path(dir_tmp, 'image_in.nii')
        png_to_nifti(pattern_fname_in=patt,
                     fname_out=path_image_in,
                     spacings=spacings_in,
                     reverse=reverse,
                     rcp_to_ras=True)

        # Convert the mask to NIfTI
        if not config['no_prep']:
            patt = Path(dir_scan_base, 'mask_prep', '*.png')
            path_mask_prep = Path(dir_tmp, 'mask_prep_in.nii')
            png_to_nifti(pattern_fname_in=patt,
                         fname_out=path_mask_prep,
                         spacings=spacings_in,
                         reverse=reverse,
                         rcp_to_ras=True)

        patt = Path(dir_scan_base, 'mask_foldavg', '*.png')
        path_mask_foldavg = Path(dir_tmp, 'mask_foldavg_in.nii')
        png_to_nifti(pattern_fname_in=patt,
                     fname_out=path_mask_foldavg,
                     spacings=spacings_in,
                     reverse=reverse,
                     rcp_to_ras=True)

        masks_prep_remapped = []
        masks_foldavg_remapped = []

        # Register with multiatlas
        for idx, atlas in enumerate(multiatlas):
            # Estimate the warping
            logger.info(f'Registering atlas {idx} to {gb_name}')
            dir_transf = Path(dir_tmp, f'transf_{idx}')
            ret = elastix.fit(path_scan_fixed=path_image_in,
                              path_scan_moving=atlas['path_image'],
                              path_param_file=config['path_config_elastix'],
                              path_root_out=dir_transf,
                              num_threads=config['num_threads_elastix'])
            if ret != 0:
                continue

            # Read the transformation parameters file
            path_transf_in = Path(dir_transf, 'TransformParameters.0.txt')
            with open(path_transf_in, 'r') as f:
                str_transf_in = f.readlines()

            # Adapt the transformation file for mask warping:
            # use NN interpolation, etc
            replacements = {
                '(ResampleInterpolator "FinalBSplineInterpolatorFloat")':
                    '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
                '(FinalBSplineInterpolationOrder 1)': ''
            }
            str_transf_out = []
            for l_in in str_transf_in:
                l_out = l_in
                for k, v in replacements.items():
                    if l_in.startswith(k):
                        l_out = l_in.replace(k, v)
                str_transf_out.append(l_out)

            # Write the transformation parameters files
            path_transf_out = Path(dir_transf, 'TransformParameters.x.txt')
            with open(path_transf_out, 'w') as f:
                f.writelines(str_transf_out)

            # Warp the atlas mask
            logger.info(f'Warping atlas {idx} mask')
            dir_res = Path(dir_tmp, f'res_{idx}')
            ret = elastix.predict(path_scan=atlas[f'path_mask_{atlas_suffix}'],
                                  path_root_out=dir_res,
                                  path_transf=path_transf_out,
                                  num_threads=config['num_threads_elastix'])
            if ret != 0:
                continue

            # Read the masks into ndarrays
            if not config['no_prep']:
                mask_prep_in, spacing_in = nifti_to_numpy(path_mask_prep,
                                                          ras_to_rcp=True)
            mask_foldavg_in, spacing_in = nifti_to_numpy(path_mask_foldavg,
                                                         ras_to_rcp=True)
            path_mask_atlas_warp = Path(dir_res, 'result.nii')
            mask_atlas_warp, spacing_warp = nifti_to_numpy(path_mask_atlas_warp,
                                                           ras_to_rcp=True)

            # Update the masks based on the warped atlas
            if not config['no_prep']:
                mask_prep_remap = delayed(remap_to_closest_labels)(
                    map_input=mask_prep_in,
                    map_reference=mask_atlas_warp,
                    mappings=valid_mappings,
                    spacings=spacing_warp)
            mask_foldavg_remap = delayed(remap_to_closest_labels)(
                map_input=mask_foldavg_in,
                map_reference=mask_atlas_warp,
                mappings=valid_mappings,
                spacings=spacing_warp)

            if not config['no_prep']:
                masks_prep_remapped.append(mask_prep_remap)
            masks_foldavg_remapped.append(mask_foldavg_remap)

        # Compute the masks
        logger.info(f'Remapping the masks')
        if not config['no_prep']:
            masks_prep_remapped = \
                Parallel(n_jobs=config['num_workers'])(masks_prep_remapped)
        masks_foldavg_remapped = \
            Parallel(n_jobs=config['num_workers'])(masks_foldavg_remapped)

        # Fuse the mappings
        logger.info(f'Fusing the masks')
        if not config['no_prep']:
            mask_prep_final = fuse_labels(masks_prep_remapped)
        mask_foldavg_final = fuse_labels(masks_foldavg_remapped)

        # Reverse to the original orientation
        if reverse:
            if not config['no_prep']:
                mask_prep_final = mask_prep_final[..., ::-1]
            mask_foldavg_final = mask_foldavg_final[..., ::-1]

        # Save the results
        logger.info(f'Saving the results')
        if not config['no_prep']:
            path_prep_final = Path(dir_scan_base, f'mask_prep_{atlas_suffix}')
            path_prep_final.mkdir(exist_ok=True)
        path_foldavg_final = Path(dir_scan_base, f'mask_foldavg_{atlas_suffix}')
        path_foldavg_final.mkdir(exist_ok=True)

        if not config['no_prep']:
            for idx in range(mask_prep_final.shape[-1]):
                fname = Path(path_prep_final, '{:>03d}.png'.format(idx))
                cv2.imwrite(str(fname), mask_prep_final[..., idx])
        for idx in range(mask_foldavg_final.shape[-1]):
            fname = Path(path_foldavg_final, '{:>03d}.png'.format(idx))
            cv2.imwrite(str(fname), mask_foldavg_final[..., idx])

        # Save also to NIfTI
        if not config['no_prep']:
            patt = Path(path_prep_final, '*.png')
            path_out = Path(dir_scan_base, f'mask_prep_{atlas_suffix}.nii')
            png_to_nifti(pattern_fname_in=patt,
                         fname_out=path_out,
                         spacings=spacings_in,
                         reverse=False,
                         rcp_to_ras=True)
        patt = Path(path_foldavg_final, '*.png')
        path_out = Path(dir_scan_base, f'mask_foldavg_{atlas_suffix}.nii')
        png_to_nifti(pattern_fname_in=patt,
                     fname_out=path_out,
                     spacings=spacings_in,
                     reverse=False,
                     rcp_to_ras=True)

        shutil.rmtree(dir_tmp)


if __name__ == '__main__':
    main()
