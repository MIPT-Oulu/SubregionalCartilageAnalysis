import logging
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from torch.utils.data.dataset import Dataset

from scartan.datasets.constants import locations_mh53


logging.basicConfig()
logger = logging.getLogger('dataset')
logger.setLevel(logging.DEBUG)


def index_from_path_oai_custom(path_root, force=False):
    fname_meta_dyn = Path(path_root, 'meta_dynamic.csv')
    fname_meta_base = Path(path_root, 'meta_base.csv')

    if not fname_meta_dyn.exists() or force:
        fnames_image = list(Path(path_root).glob('**/images/*.png'))
        logger.info('{} images found'.format(len(fnames_image)))

        df_meta = pd.read_csv(fname_meta_base,
                              dtype={'patient': str,
                                     'release': str,
                                     'prefix_var': str,
                                     'sequence': str,
                                     'side': str,
                                     'slice_idx': int,
                                     'pixel_spacing_0': float,
                                     'pixel_spacing_1': float,
                                     'slice_thickness': float,
                                     'KL': int},
                              index_col=False)

        if len(fnames_image) != len(df_meta):
            raise ValueError("Number of images doesn't match with the metadata")

        df_meta['path_image'] = [Path(path_root, e)
                                 for e in df_meta['path_rel_image']]

        # Sort the records
        df_meta_sorted = (df_meta
                          .sort_values(['patient', 'release', 'sequence',
                                        'side', 'slice_idx'])
                          .reset_index()
                          .drop('index', axis=1))

        df_meta_sorted.to_csv(fname_meta_dyn, index=False)
    else:
        df_meta_sorted = pd.read_csv(fname_meta_dyn,
                                     dtype={'patient': str,
                                            'release': str,
                                            'prefix_var': str,
                                            'sequence': str,
                                            'side': str,
                                            'slice_idx': int,
                                            'pixel_spacing_0': float,
                                            'pixel_spacing_1': float,
                                            'slice_thickness': float,
                                            'KL': int},
                                     index_col=False)
    return df_meta_sorted


def read_image(path_file):
    image = cv2.imread(str(path_file), cv2.IMREAD_GRAYSCALE)
    return image.reshape((1, *image.shape))


class DatasetOAICustomSagittal2d(Dataset):
    def __init__(self, df_meta, mask_mode=None, name=None, transforms=None,
                 sample_mode='x_y', **kwargs):
        logger.warning('Redundant dataset init arguments:\n{}'
                       .format(repr(kwargs)))

        self.df_meta = df_meta
        self.mask_mode = mask_mode
        self.name = name
        self.transforms = transforms
        self.sample_mode = sample_mode

    def __len__(self):
        return len(self.df_meta)

    def _getitem_x_y(self, idx):
        image = read_image(self.df_meta['path_image'].iloc[idx])
        mask = np.zeros_like(image)

        # Apply transformations
        if self.transforms is not None:
            for t in self.transforms:
                if hasattr(t, 'randomize'):
                    t.randomize()
                image, mask = t(image, mask)

        tmp = dict(self.df_meta.iloc[idx])
        tmp['image'] = image
        tmp['mask'] = mask

        tmp['xs'] = tmp['image']
        tmp['ys'] = tmp['mask']
        return tmp

    def __getitem__(self, idx):
        if self.sample_mode == 'x_y':
            return self._getitem_x_y(idx)
        else:
            raise ValueError('Invalid `sample_mode`')

    def read_image(self, path_file):
        return read_image(path_file=path_file)

    def read_mask(self, path_file):
        return None

    def describe(self, *args, **kwargs):
        raise NotImplementedError()
