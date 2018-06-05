import pickle
import logging
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import click
from tqdm import tqdm
from joblib import Parallel, delayed

from scartan.metrics import (dice_score, volume_error,
                             avg_symm_surf_dist, robust_hausdorff_dist,
                             volume_total, local_thickness, distance_transform,
                             jaccard_score, sensitivity_score)
from scartan.various import png_to_numpy
from scartan.datasets.constants import (atlas_to_locations, valid_mappings_zp3n_mh53,
                                        valid_mappings_mh53_zp3n,
                                        valid_mappings_61om_mh53,
                                        valid_mappings_mh53_61om,
                                        valid_mappings_w02n_mh53,
                                        valid_mappings_mh53_w02n)


logging.basicConfig()
logger = logging.getLogger('quantify')
logger.setLevel(logging.INFO)


def _extract_thickness_features(*, stack, spacings, suffix, num_classes,
                                mappings_to_tissues, mappings_from_tissues,
                                mappings_merge, method, cache_file):
    """
    Args:
        stack: ndarray
        suffix: {"true", "pred"}
        num_classes: int
        mappings_to_tissues: dict
        mappings_from_tissues: dict
        mappings_merge: dict or None
        method: {"thick_local", "dist_transf"}
        cache_file: Path

    Returns:
        res: dict

    """
    res = dict()

    if cache_file.exists():
        tmp_map = np.load(cache_file)['arr']
    else:
        # Remap from compartments to tissues, such that thickness is meaningful
        tmp_mappings = {k: v[0] for k, v in mappings_to_tissues.items()}
        tmp_stack = np.vectorize(tmp_mappings.get)(stack)

        if method == "thick_local":
            tmp_map = local_thickness(
                tmp_stack,
                num_classes=len(mappings_from_tissues),
                stack_axis=2,
                spacing_mm=spacings,
                skip_classes=(0,))
        elif method == "dist_transf":
            tmp_map = distance_transform(
                tmp_stack,
                num_classes=len(mappings_from_tissues),
                spacing_mm=spacings,
                skip_classes=(0,))
        else:
            raise ValueError(f"Unknown method {method}")

        # Cache the thickness map
        np.savez_compressed(cache_file.with_suffix(''), arr=tmp_map)

    # Atlas classes
    tmp_stats = _labelwise_stats(
        labels=stack[0],
        values=tmp_map[0],
        skip_classes=(0,),
        # stat_names=('mean',),
        stat_names=('mean', 'percentile_01', 'percentile_99'),
        num_classes=num_classes)
    res[f"{method}_mean_{suffix}"] = np.squeeze(tmp_stats[:, 0])
    res[f"{method}_perc01_{suffix}"] = np.squeeze(tmp_stats[:, 1])
    res[f"{method}_perc99_{suffix}"] = np.squeeze(tmp_stats[:, 2])

    # Merged classes
    if mappings_merge is not None:
        res[f"{method}_merged_mean_{suffix}"] = []
        res[f"{method}_merged_perc01_{suffix}"] = []
        res[f"{method}_merged_perc99_{suffix}"] = []

        for ks, v in mappings_merge.items():
            merged = np.zeros_like(stack[0])
            tmp_sel = np.isin(stack[0], ks)
            merged[tmp_sel] = 1

            tmp_stats = _labelwise_stats(
                labels=merged,
                values=tmp_map[0],
                skip_classes=(0,),
                # stat_names=('mean',),
                stat_names=('mean', 'percentile_01', 'percentile_99'),
                num_classes=2)
            res[f"{method}_merged_mean_{suffix}"].append(np.squeeze(tmp_stats[1, 0]))
            res[f"{method}_merged_perc01_{suffix}"].append(np.squeeze(tmp_stats[1, 1]))
            res[f"{method}_merged_perc99_{suffix}"].append(np.squeeze(tmp_stats[1, 2]))

    return res


def _scan_to_metrics_single(*, path_root, name_pred, num_classes,
                            mappings_to_tissues, mappings_from_tissues, meta,
                            mappings_merge=None):
    scan_dir = Path(path_root, meta['patient'], meta['release'], meta['sequence'])

    # Read the data
    patt_pred = Path(scan_dir, name_pred, '*.png')
    stack_pred = png_to_numpy(patt_pred)

    # Add batch dimension
    stack_pred = stack_pred[None, ...]

    # Compute the metrics
    res = meta
    spacings = (meta['pixel_spacing_0'],
                meta['pixel_spacing_1'],
                meta['slice_thickness'])

    # Measurements
    if True:
        for kind, stack, artef in (("pred", stack_pred, name_pred),):
            # Volume
            res[f"volume_total_{kind}"] = volume_total(stack,
                                                       num_classes=num_classes,
                                                       spacing_mm=spacings)[0]
            # Thickness
            for method in ("thick_local", "dist_transf"):
                cache_file = Path(scan_dir, f"{method}_{artef}.npz")

                tmp = _extract_thickness_features(
                    stack=stack, spacings=spacings,
                    suffix=kind, num_classes=num_classes,
                    mappings_to_tissues=mappings_to_tissues,
                    mappings_from_tissues=mappings_from_tissues,
                    mappings_merge=mappings_merge,
                    method=method,
                    cache_file=cache_file)
                res.update(tmp)

    return res


def _scan_to_metrics_paired(*, path_root, name_true, name_pred, num_classes,
                            mappings_to_tissues, mappings_from_tissues, meta,
                            mappings_merge=None):
    scan_dir = Path(path_root, meta['patient'], meta['release'], meta['sequence'])

    # Read the data
    patt_true = Path(scan_dir, name_true, '*.png')
    stack_true = png_to_numpy(patt_true)

    patt_pred = Path(scan_dir, name_pred, '*.png')
    stack_pred = png_to_numpy(patt_pred)

    # Add batch dimension
    stack_true = stack_true[None, ...]
    stack_pred = stack_pred[None, ...]

    # Compute the metrics
    res = meta
    spacings = (meta['pixel_spacing_0'],
                meta['pixel_spacing_1'],
                meta['slice_thickness'])

    # Volume
    res['dice_score'] = dice_score(stack_pred, stack_true,
                                   num_classes=num_classes)
    res['volume_error'] = volume_error(stack_pred, stack_true,
                                       num_classes=num_classes)
    res['jaccard_score'] = jaccard_score(stack_pred, stack_true,
                                         num_classes=num_classes)
    res['sensitivity_score'] = sensitivity_score(stack_pred, stack_true,
                                                 num_classes=num_classes)

    # Surface
    res['avg_symm_surf_dist'] = avg_symm_surf_dist(stack_pred, stack_true,
                                                   num_classes=num_classes,
                                                   spacing_mm=spacings)
    res['robust_hausdorff_dist'] = robust_hausdorff_dist(stack_pred, stack_true,
                                                         num_classes=num_classes,
                                                         spacing_mm=spacings,
                                                         percent=100)

    # Measurements
    if True:
    # if False:
        for kind, stack, artef in (("true", stack_true, name_true),
                                   ("pred", stack_pred, name_pred)):
            # Volume
            res[f"volume_total_{kind}"] = volume_total(stack,
                                                       num_classes=num_classes,
                                                       spacing_mm=spacings)[0]
            # Thickness
            for method in ("thick_local", "dist_transf"):
                cache_file = Path(scan_dir, f"{method}_{artef}.npz")

                tmp = _extract_thickness_features(
                    stack=stack, spacings=spacings,
                    suffix=kind, num_classes=num_classes,
                    mappings_to_tissues=mappings_to_tissues,
                    mappings_from_tissues=mappings_from_tissues,
                    mappings_merge=mappings_merge,
                    method=method,
                    cache_file=cache_file)
                res.update(tmp)

    return res


def _labelwise_stats(*, labels, values, stat_names, num_classes, skip_classes=None):
    """Extract specified statistics of image label-wise.
    """
    if skip_classes is None:
        skip_classes = ()
    tmp_stats = np.zeros((num_classes, len(stat_names)))

    for class_idx in range(num_classes):
        if class_idx in skip_classes:
            tmp_stats[class_idx, :] = np.nan
            continue

        sel_label = labels == class_idx
        is_empty_sel = ~np.any(sel_label)

        if is_empty_sel:
            tmp_stats[class_idx, :] = np.nan
            continue

        sel_values = values[sel_label]

        for stat_idx, stat_name in enumerate(stat_names):
            if stat_name == 'min':
                stat_val = np.min(sel_values)
            elif stat_name == 'max':
                stat_val = np.max(sel_values)
            elif stat_name == 'mean':
                stat_val = np.mean(sel_values)
            elif stat_name == 'median':
                stat_val = np.median(sel_values)
            elif stat_name.startswith('percentile'):
                percentile = int(stat_name.split('_')[-1])
                stat_val = np.percentile(sel_values, percentile)
            else:
                msg = f'Unknown statistics `{stat_name}`'
                raise ValueError(msg)

            tmp_stats[class_idx, stat_idx] = stat_val
    return tmp_stats


def metrics_single_volumew(*, path_pred, dirname_pred, df, num_classes,
                           mappings_to_tissues, mappings_from_tissues,
                           mappings_merge=None, num_workers=1):
    acc_l = []
    groupers_stack = ['patient', 'release', 'sequence', 'side']

    for name_gb, df_gb in tqdm(df.groupby(groupers_stack)):
        patient, release, sequence, side = name_gb

        acc_l.append(delayed(_scan_to_metrics_single)(
            path_root=path_pred,
            name_pred=dirname_pred,
            num_classes=num_classes,
            mappings_to_tissues=mappings_to_tissues,
            mappings_from_tissues=mappings_from_tissues,
            mappings_merge=mappings_merge,
            meta={
                'patient': patient,
                'release': release,
                'sequence': sequence,
                'side': side,
                **df_gb.to_dict("records")[0],
            }
        ))

    acc_l = Parallel(n_jobs=num_workers, verbose=10)(
        t for t in tqdm(acc_l, total=len(acc_l)))
    # Convert from list of dicts to dict of lists
    acc_d = {k: [d[k] for d in acc_l] for k in acc_l[0]}
    return acc_d


def metrics_paired_volumew(*, path_pred, dirname_true, dirname_pred, df, num_classes,
                           mappings_to_tissues, mappings_from_tissues,
                           mappings_merge=None, num_workers=1):
    acc_l = []
    groupers_stack = ['patient', 'release', 'sequence', 'side']

    for name_gb, df_gb in tqdm(df.groupby(groupers_stack)):
        patient, release, sequence, side = name_gb

        acc_l.append(delayed(_scan_to_metrics_paired)(
            path_root=path_pred,
            name_true=dirname_true,
            name_pred=dirname_pred,
            num_classes=num_classes,
            mappings_to_tissues=mappings_to_tissues,
            mappings_from_tissues=mappings_from_tissues,
            mappings_merge=mappings_merge,
            meta={
                'patient': patient,
                'release': release,
                'sequence': sequence,
                'side': side,
                **df_gb.to_dict("records")[0],
            }
        ))

    acc_l = Parallel(n_jobs=num_workers, verbose=10)(
        t for t in tqdm(acc_l, total=len(acc_l)))
    # Convert from list of dicts to dict of lists
    acc_d = {k: [d[k] for d in acc_l] for k in acc_l[0]}
    return acc_d


def print_metrics_paired_volumew_b46e(acc, metric_names):
    """Average and print the metrics with respect to KL grades.

    Args:
        acc: dict of lists
        metric_names: tuple of str

    """
    kl_vec = np.asarray(acc['KL'])
    kl_groups = [(0, 1, 2, 3, 4), (0, ), (1, ), (2, ), (3, ), (4, )]

    for kl_vals in kl_groups:
        kl_sel = np.isin(kl_vec, kl_vals)
        if not np.any(kl_sel):
            continue

        for metric_name in metric_names:
            if metric_name not in acc:
                logger.error(f'`{metric_name}` is not presented in `acc`')
                continue
            metric_values = acc[metric_name]
            tmp = np.asarray(metric_values)[kl_sel]

            tmp_means, tmp_stds = [], []
            for class_idx, class_scores in enumerate(
                    np.moveaxis(tmp[..., 1:], -1, 0), start=1):
                class_scores = class_scores[class_scores != 0]

                tmp_means.append(np.mean(class_scores))
                tmp_stds.append(np.std(class_scores))

            tmp_values = ', '.join(
                [f'{m:.03f}({s:.03f})' for m, s in zip(tmp_means, tmp_stds)])
            logger.info(f"KL{str(kl_vals)}, {metric_name}\n{tmp_values}\n")


@click.command()
@click.option('--path_experiment_root', default='../../results/temporary')
@click.option('--dirname_pred', required=True)
@click.option('--dirname_true', default=None)
@click.option('--dataset', required=True, type=click.Choice(['oai_imo', 'oai_prj_22']))
@click.option('--atlas', required=True, type=click.Choice(['imo', 'segm', 'biomediq', 'chondr75n']))
@click.option('--ignore_cache', is_flag=True)
@click.option('--interactive', is_flag=True)
@click.option('--num_workers', default=1, type=int)
def main(**config):

    path_pred = Path(config['path_experiment_root'], f"predicts_{config['dataset']}_test")
    path_logs = Path(config['path_experiment_root'], f"logs_{config['dataset']}_test")
    path_logs.mkdir(exist_ok=True)

    if config['dirname_true'] is not None:
        true_is_avail = True
    else:
        true_is_avail = False

    # Get the information on object classes
    locations = atlas_to_locations[config['atlas']]
    class_names = [k for k in locations]
    num_classes = max(locations.values()) + 1

    if config['atlas'] == 'segm':
        mappings_to_tissues = valid_mappings_zp3n_mh53
        mappings_from_tissues = valid_mappings_mh53_zp3n
    elif config['atlas'] == 'imo':
        mappings_to_tissues = {k: (k, ) for k in range(7)}
        mappings_from_tissues = mappings_to_tissues
    elif config['atlas'] == 'biomediq':
        mappings_to_tissues = valid_mappings_61om_mh53
        mappings_from_tissues = valid_mappings_mh53_61om
    elif config['atlas'] == 'chondr75n':
        mappings_to_tissues = valid_mappings_w02n_mh53
        mappings_from_tissues = valid_mappings_mh53_w02n
    else:
        msg = f"Atlas {config['atlas']} is not supported or invalid"
        raise ValueError(msg)

    if config['atlas'] == 'biomediq':
        mappings_merge = OrderedDict([
            ((1, 2), 1),  # F
        ])
        logger.info("Selected mappings merge: Biomediq")
    elif config['atlas'] == 'chondr75n':
        mappings_merge = OrderedDict([
            ((2, 3, 4), 1),  # cLF
            ((7, 8, 9), 2),  # cMF
            ((11, 12, 13, 14, 15), 3),  # LT
            ((16, 17, 18, 19, 20), 4),  # MT
            ((2, 13), 5),  # cLFTC: ccLF + cLT
            ((7, 18), 6),  # cMFTC: ccMF + cMT
            ((2, 3, 4, 11, 12, 13, 14, 15), 7),  # LFTC: cLF + LT
            ((7, 8, 9, 16, 17, 18, 19, 20), 8),  # MFTC: cMF + MT
        ])
        logger.info("Selected mappings merge: Chondrometrics")
    else:
        logger.info("Selected mappings merge: None")

    # Get the index of image files and the corresponding metadata
    path_meta = Path(path_pred, 'meta_dynamic.csv')
    df_meta = pd.read_csv(path_meta,
                          dtype={'patient': str,
                                 'release': str,
                                 'prefix_var': str,
                                 'sequence': str,
                                 'side': str,
                                 'slice_idx': int,
                                 'pixel_spacing_0': float,
                                 'pixel_spacing_1': float,
                                 'slice_thickness': float,
                                 'KL': int,
                                 'has_mask': int},
                          index_col=False)

    df_sel = df_meta.sort_values(['patient', 'release', 'sequence', 'side', 'slice_idx'])

    if true_is_avail:
        fname_pkl = Path(path_logs,
                         f"cache_{config['dataset']}_test_"
                         f"{config['atlas']}_"
                         f"volumew_paired.pkl")

        logger.info('Volumetric scores')
        if fname_pkl.exists() and not config['ignore_cache']:
            logger.info('Loading from the cache')
            with open(fname_pkl, 'rb') as f:
                acc_volumew = pickle.load(f)
        else:
            logger.info('Computing')
            acc_volumew = metrics_paired_volumew(
                path_pred=path_pred,
                dirname_true=config['dirname_true'],
                dirname_pred=config['dirname_pred'],
                df=df_sel,
                num_classes=num_classes,
                mappings_to_tissues=mappings_to_tissues,
                mappings_from_tissues=mappings_from_tissues,
                mappings_merge=mappings_merge,
                num_workers=config['num_workers']
            )
            logger.info('Caching the results into file')
            with open(fname_pkl, 'wb') as f:
                pickle.dump(acc_volumew, f)

        print_metrics_paired_volumew_b46e(
            acc=acc_volumew,
            metric_names=(
                'dice_score',
                'volume_error', 'jaccard_score',
                'avg_symm_surf_dist', 'robust_hausdorff_dist',
                'sensitivity_score',

                'volume_total_true', 'volume_total_pred',
                'thick_local_mean_true', 'thick_local_mean_pred',
                'dist_transf_mean_true', 'dist_transf_mean_pred',
            ),
        )

    else:
        fname_pkl = Path(path_logs,
                         f"cache_{config['dataset']}_test_"
                         f"{config['atlas']}_"
                         f"volumew_single.pkl")

        logger.info('Volumetric scores')
        if fname_pkl.exists() and not config['ignore_cache']:
            logger.info('Loading from the cache')
            with open(fname_pkl, 'rb') as f:
                acc_volumew = pickle.load(f)
        else:
            logger.info('Computing')
            acc_volumew = metrics_single_volumew(
                path_pred=path_pred,
                dirname_pred=config['dirname_pred'],
                df=df_sel,
                num_classes=num_classes,
                mappings_to_tissues=mappings_to_tissues,
                mappings_from_tissues=mappings_from_tissues,
                mappings_merge=mappings_merge,
                num_workers=config['num_workers']
            )
            logger.info('Caching the results into file')
            with open(fname_pkl, 'wb') as f:
                pickle.dump(acc_volumew, f)


if __name__ == '__main__':
    main()
