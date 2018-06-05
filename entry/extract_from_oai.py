from pathlib import Path
import shutil
import click
import logging

import pandas as pd


logging.basicConfig()
logger = logging.getLogger('extract')
logger.setLevel(logging.DEBUG)


EXAMINATIONS = ["00m", ]
PATIENTS = []
SCAN_KINDS = ['SAG_3D_DESS_LEFT', 'SAG_3D_DESS_RIGHT']


@click.command()
@click.argument('path_root_oai')
@click.argument('path_root_results')
@click.option('--examinations', '-e', multiple=True, default=EXAMINATIONS)
@click.option('--patients', '-p', multiple=True, default=PATIENTS)
@click.option('--scan_kinds', '-s', multiple=True, default=SCAN_KINDS)
def main(**config):

    for examination in config['examinations']:
        df = pd.read_csv(
            Path(config['path_root_oai'], examination, 'contents.csv'))

        logger.info(f'Entries, total: {len(df)}')

        df = df[df['ParticipantID'].isin(config['patients'])]
        df = df[df['SeriesDescription'].isin(config['scan_kinds'])]

        logger.info(f'Entries, selected: {len(df)}')

        for _, row in df.iterrows():
            tmp = row['Folder']
            logger.info(f'... {tmp}')

            path_from = Path(config['path_root_oai'], examination, tmp)
            path_to = Path(config['path_root_results'], examination, tmp)

            path_to.mkdir(exist_ok=True)
            for fn in path_from.iterdir():
                shutil.copy2(Path(path_from, fn), path_to)


@click.command()
@click.argument('path_root_oai')
@click.argument('path_root_results')
@click.argument('path_file_meta')
def temp(**config):

    with open(config['path_file_meta']) as f:
        entries = f.readlines()

    logger.info(f'Entries, total: {len(entries)}')

    for entry in entries:
        patient_id, path_obj = entry.strip().split(': ')

        logger.info(f'... {path_obj}')

        path_from = Path(config['path_root_oai'], '00m', path_obj)
        path_to = Path(config['path_root_results'], '00m', path_obj)

        path_to.mkdir(exist_ok=True)
        for fn in path_from.iterdir():
            shutil.copy2(Path(path_from, fn), path_to)


if __name__ == '__main__':
    # main()
    temp()
