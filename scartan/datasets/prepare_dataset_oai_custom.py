import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import click
import pydicom
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

from scartan.datasets.meta_oai import (side_code_to_str, release_to_prefix_var,
                                       release_to_visit_month)


cv2.ocl.setUseOpenCL(False)


def read_dicom(fname, ignore_missing_meta=False):
    data = pydicom.read_file(fname)
    image = np.frombuffer(data.PixelData, dtype=np.uint16).astype(float)

    if data.PhotometricInterpretation == 'MONOCHROME1':
        image = image.max() - image
    image = image.reshape((data.Rows, data.Columns))

    if 'RIGHT' in data.SeriesDescription:
        side = 'RIGHT'
    elif 'LEFT' in data.SeriesDescription:
        side = 'LEFT'
    else:
        print(data)
        msg = f'DICOM {fname} does not contain side info'
        raise ValueError(msg)

    if hasattr(data, 'ImagerPixelSpacing'):
        spacing = [float(e) for e in data.ImagerPixelSpacing[:2]]
    elif hasattr(data, 'PixelSpacing'):
        spacing = [float(e) for e in data.PixelSpacing[:2]]
    else:
        msg = f'DICOM {fname} does not contain spacing info'
        if ignore_missing_meta:
            spacing = (0, 0)
        else:
            raise AttributeError(msg)

    return {'image': image,
            'pixel_spacing_0': spacing[0],
            'pixel_spacing_1': spacing[1],
            'slice_thickness': float(data.SliceThickness),
            'side': side}


def worker_xz9(path_root_output, path_stack, margin, meta_only):
    meta = defaultdict(list)

    release, patient = path_stack.split('/')[-4:-2]
    prefix_var = release_to_prefix_var[release]
    visit = release_to_visit_month[release]
    sequence = 'sag_3d_dess_we'

    num_slices = len(list(Path(path_stack).glob('*')))

    for slice_idx in range(num_slices):
        # Indexing of slices in OAI dataset starts with 001
        fname_src = Path(path_stack, '{:>03}'.format(slice_idx + 1))
        dicom_image = read_dicom(fname_src, ignore_missing_meta=False)
        image = dicom_image['image']

        if margin != 0:
            image = image[margin:-margin, margin:-margin]

        fname_pattern = '{slice_idx:>03}.{ext}'

        # Save image and mask data
        dir_rel_image = Path(patient, release, sequence, 'images')
        dir_abs_image = Path(path_root_output, dir_rel_image)
        dir_abs_image.mkdir(exist_ok=True)

        fname_image = fname_pattern.format(slice_idx=slice_idx, ext='png')
        path_abs_image = Path(dir_abs_image, fname_image)
        if not meta_only:
            cv2.imwrite(str(path_abs_image), image)

        path_rel_image = Path(dir_rel_image, fname_image)

        meta['patient'].append(patient)
        meta['release'].append(release)
        meta['prefix_var'].append(prefix_var)
        meta['visit'].append(visit)
        meta['sequence'].append(sequence)
        meta['side'].append(dicom_image['side'])
        meta['slice_idx'].append(slice_idx)
        meta['pixel_spacing_0'].append(dicom_image['pixel_spacing_0'])
        meta['pixel_spacing_1'].append(dicom_image['pixel_spacing_1'])
        meta['slice_thickness'].append(dicom_image['slice_thickness'])
        meta['path_rel_image'].append(str(path_rel_image))
    return meta


def read_compose_assessments(fpaths):
    print(fpaths)
    dfs = []

    for i, fpath in enumerate(fpaths):
        df = pd.read_csv(fpath, sep='|', index_col=False,
                         dtype={'ID': str})

        # Find release prefix and add corresponding column
        prefix_var = 'VXX'
        for c in df.columns:
            if re.match("V\d\d.*$", c):
                prefix_var = c[:3]
                break
        df.loc[:, 'PREFIX_VAR'] = prefix_var

        # Remove prefix from column names
        columns = []
        for c in df.columns:
            if c.startswith(prefix_var):
                columns.append(c[3:])
            else:
                columns.append(c)
        df.columns = columns

        print(f'df idx: {i} num: {len(df)}')
        dfs.append(df)

    # Standardize the column names and drop repeated assessments
    for i, df in enumerate(dfs):
        df = df.rename(lambda x: x.upper(), axis=1)
        df = df.rename({'ID': 'patient',
                        'PREFIX_VAR': 'prefix_var',
                        'SIDE': 'side',
                        'XRKL': 'KL'}, axis=1)
        df.loc[:, 'side'] = df['side'].apply(lambda s: side_code_to_str[s])

        dfs[i] = df.drop_duplicates(subset=['patient', 'side', 'READPRJ'])
        print(f'df idx: {i} num w/o dup: {len(dfs[i])}')

    if len(dfs) > 1:
        out = pd.concat(dfs, axis=0)
    else:
        out = dfs[0]
    print(f'num total: {len(out)}')
    return out


@click.command()
@click.argument('path_root_oai_mri', type=click.Path(exists=True))
@click.argument('path_root_output')
@click.option('--num_threads', default=12, type=click.IntRange(0, 16))
@click.option('--margin', default=0, type=int)
@click.option('--meta_only', is_flag=True)
def main(**config):
    config['path_root_oai_mri'] = config['path_root_oai_mri'].resolve()
    config['path_root_output'] = config['path_root_output'].resolve()

    # OAI data path structure:
    #   root / examination / release / patient / date / barcode (/ slices)
    paths_stacks = [str(p) for p in config['path_root_oai_mri'].glob('*/*/*/*/*')]
    paths_stacks.sort(key=lambda x: int(x.split('/')[-3]))

    metas = Parallel(config['num_threads'])(delayed(worker_xz9)(
        *[config['path_root_output'], path_stack,
          config['margin'], config['meta_only']]
    ) for path_stack in tqdm(paths_stacks))

    # Merge meta information from different stacks
    tmp = defaultdict(list)
    for d in metas:
        for k, v in d.items():
            tmp[k].extend(v)
    df_out = pd.DataFrame.from_dict(tmp)

    # Find semi-quantitative data
    fpaths_sq = sorted(Path(config['path_root_oai_mri']).glob('*/*_sq_*.txt'))

    df_sq = read_compose_assessments(fpaths_sq)

    # Select the subset for which the assessments are available
    indexers = ['patient', 'side', 'prefix_var']
    sel = df_out.set_index(indexers).index.unique()
    df_sq = (df_sq
             .drop_duplicates(subset=indexers)
             .set_index(indexers)
             .loc[sel, :]
             .reset_index())

    df_out = pd.merge(df_out, df_sq, on=indexers, how='left')
    # Fill n/a in assessment info
    df_out.loc[:, 'KL'] = df_out['KL'].fillna(-1).astype(int)

    # Select only the necessary columns
    df_out = df_out.loc[:, ['patient', 'release', 'prefix_var', 'visit', 'sequence',
                            'side', 'slice_idx', 'pixel_spacing_0', 'pixel_spacing_1',
                            'slice_thickness', 'path_rel_image', 'KL']]

    path_output_meta = Path(config['path_root_output'], 'meta_base.csv')
    df_out.to_csv(path_output_meta, index=False)


if __name__ == '__main__':
    main()
