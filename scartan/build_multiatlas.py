"""
Script to finalize the multi-atlas initialisation. Steps:
- Pack the scans into NIfTI format
- Flip the scan when needed
"""
import click
from pathlib import Path

import pandas as pd

from scartan.various import png_to_nifti


@click.command()
@click.argument('path_root_atlas')
def main(**config):

    path_meta = Path(config['path_root_atlas'], 'meta_base.csv')
    df_meta = pd.read_csv(path_meta)

    groupers_stack = ['patient', 'release', 'sequence', 'side']

    for gb_name, gb_df in df_meta.groupby(groupers_stack):
        patient, release, sequence, side = gb_name

        # Flip the scans to the left side to make the registration feasible
        if side == 'RIGHT':
            reverse = True
        else:
            reverse = False

        spacings = (gb_df['pixel_spacing_0'].iloc[0],
                    gb_df['pixel_spacing_1'].iloc[0],
                    gb_df['slice_thickness'].iloc[0])

        # Convert the image to NIfTI
        path_atlas = Path(config['path_root_atlas'],
                          str(patient), release, sequence)
        pattern = Path(path_atlas, 'images', '*.png')
        path_image = Path(path_atlas, 'image_in.nii')

        png_to_nifti(pattern_fname_in=pattern, fname_out=path_image,
                     spacings=spacings, reverse=reverse, rcp_to_ras=True)

        # Convert the mask to NIfTI
        path_atlas = Path(config['path_root_atlas'],
                          str(patient), release, sequence)
        pattern = Path(path_atlas, 'masks', '*.png')
        path_mask = Path(path_atlas, 'mask_in.nii')

        png_to_nifti(pattern_fname_in=pattern, fname_out=path_mask,
                     spacings=spacings, reverse=reverse, rcp_to_ras=True)


if __name__ == "__main__":
    main()
