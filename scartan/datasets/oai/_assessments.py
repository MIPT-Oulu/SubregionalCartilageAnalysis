import re
import logging
import numpy as np
import pandas as pd

from scartan.datasets.oai._constants import (side_code_to_str,
                                             prefix_var_to_visit_month)


logging.basicConfig()
logger = logging.getLogger('datasets:OAI')
logger.setLevel(logging.INFO)


def read_compose_asmts_mri(fpaths, verbose=False):
    """ """
    if verbose:
        print(fpaths)
    dfs = []

    for i, fpath in enumerate(fpaths):
        df = pd.read_csv(fpath, sep='|', index_col=False)

        # Find release info
        prefix_var = 'VXX'
        for c in df.columns:
            if re.match("V\d\d.*$", c):
                prefix_var = c[:3]
                break
        # Remove prefix from column names and add corresponding column
        columns = []
        for c in df.columns:
            if c.startswith(prefix_var):
                columns.append(c[3:])
            else:
                columns.append(c)
        df.columns = columns
        df.loc[:, 'PREFIX_VAR'] = prefix_var

        if verbose:
            print(f'df idx: {i} num: {len(df)}')
        dfs.append(df)

    if len(dfs) > 1:
        out = pd.concat(dfs, axis=0)
    else:
        out = dfs[0]
    if verbose:
        print(f"num total: {len(out)}")
    return out


def preproc_asmts_biomediq(df, projects=('22',), verbose=False):
    """See `kmri_fnih_qcart_biomediq_descrip.pdf` for the details.

    """
    df_tmp = df.copy()
    if verbose:
        print(f'num total: {len(df_tmp)}')

    # Select records from the specified projects only
    df_tmp.loc[:, 'READPRJ'] = df_tmp['READPRJ'].astype(str)
    df_tmp = df_tmp[df_tmp['READPRJ'].isin(projects)]
    if verbose:
        print(f'num in projects: {len(df_tmp)}')

    # Remove incomplete records
    if verbose:
        nan_counts = np.sum(df_tmp.isna(), axis=0)
        print(f"NaN records by columns: {nan_counts}")
    df_tmp = df_tmp.dropna(axis=0)
    if verbose:
        print(f'num w/o n/a: {len(df_tmp)}')
    df_tmp.reset_index(inplace=True)

    # Rename the unique selectors
    df_tmp = df_tmp.rename({
        'ID': 'patient',
        'SIDE': 'side',
        'PREFIX_VAR': 'prefix_var',
    }, axis=1)

    df_tmp.loc[:, 'side'] = df_tmp['side'].replace(
        {'1: Right': 'RIGHT', '2: Left': 'LEFT'})

    df_tmp = df_tmp.astype({
        'patient': str,
        'prefix_var': str,
        'side': str,
    })

    df_tmp.loc[:, 'visit'] = df_tmp['prefix_var'].apply(
        lambda p: prefix_var_to_visit_month[p])

    # Assessments
    df_tmp.loc[:, 'F.VC'] = (df_tmp['MedialFemoralCartilage'] +
                             df_tmp['LateralFemoralCartilage'])

    mapping_rename = [
        ('LateralTibialCartilage', 'LT.VC'),
        ('MedialTibialCartilage', 'MT.VC'),
        ('PatellarCartilage', 'P.VC'),
        ('LateralMeniscus', 'LM.VC'),
        ('MedialMeniscus', 'MM.VC'),
        ('LateralFemoralCartilage', 'LF.VC'),
        ('MedialFemoralCartilage', 'MF.VC'),
    ]
    df_tmp = df_tmp.rename(dict(mapping_rename), axis=1)

    # Compose the output
    selection = [
        'F.VC',
        *[x[1] for x in mapping_rename]
    ]
    df_tmp = df_tmp[['patient', 'side', 'prefix_var', 'visit',
                     *selection]]
    return df_tmp


def preproc_asmts_chondrometrics(df, projects=('22', '22b', '66'), verbose=False):
    """See `kmri_qcart_eckstein_descrip.pdf` for the details.

    """
    df_tmp = df.copy()
    if verbose:
        print(f"num total: {len(df_tmp)}")

    # Select records from the specified projects only
    df_tmp.loc[:, 'READPRJ'] = df_tmp['READPRJ'].astype(str)
    df_tmp = df_tmp[df_tmp['READPRJ'].isin(projects)]
    if verbose:
        print(f"num in projects: {len(df_tmp)}")

    # Remove incomplete records
    if verbose:
        nan_counts = np.sum(df_tmp.isna(), axis=0)
        print(f"NaN records by columns: {nan_counts}")
    df_tmp = df_tmp.dropna(axis=0)
    if verbose:
        print(f"num w/o n/a: {len(df_tmp)}")
    df_tmp.reset_index(inplace=True)

    # Rename the unique selectors
    df_tmp = df_tmp.rename({
        'ID': 'patient',
        'SIDE': 'side',
        'PREFIX_VAR': 'prefix_var',
    }, axis=1)

    df_tmp.loc[:, 'side'] = df_tmp['side'].replace(
        {1: 'RIGHT', 2: 'LEFT',
         '1: Right': 'RIGHT', '2: Left': 'LEFT',
         '1': 'RIGHT', '2': 'LEFT'})

    df_tmp = df_tmp.astype({
        'patient': str,
        'prefix_var': str,
        'side': str,
    })

    df_tmp.loc[:, 'visit'] = df_tmp['prefix_var'].apply(
        lambda p: prefix_var_to_visit_month[p])

    # Assessments
    mapping_rename = [
        ('WMTVCL', 'MT.VC'),
        ('WMTSBA', 'MT.tAB'),
        ('WMTVCN', 'MT.VCtAB'),
        ('WMTMTH', 'MT.ThCtAB.MEAN'),
        ('WMTACS', 'MT.AC'),
        ('WMTPD', 'MT.dAB%'),
        ('WMTCAAB', 'MT.cAB'),
        ('WMTMTC', 'MT.ThCcAB.MEAN'),
        ('WMTMAV', 'MT.ThCtAB.MAX'),
        ('WMTCTS', 'MT.ThCtAB.SD'),
        ('WMTACV', 'MT.ThCtAB.CV'),
        ('CMTMAT', 'cMT.ThCtAB.MIN'),
        ('CMTMTH', 'cMT.ThCtAB.MEAN'),
        ('EMTMTH', 'eMT.ThCtAB.MEAN'),
        ('IMTMTH', 'iMT.ThCtAB.MEAN'),
        ('AMTMTH', 'aMT.ThCtAB.MEAN'),
        ('PMTMTH', 'pMT.ThCtAB.MEAN'),
        ('CMTPD', 'cMT.dAB%'),
        ('EMTPD', 'eMT.dAB%'),
        ('IMTPD', 'iMT.dAB%'),
        ('AMTPD', 'aMT.dAB%'),
        ('PMTPD', 'pMT.dAB%'),
        ('BMFVCL', 'cMF.VC'),
        ('BMFSBA', 'cMF.tAB'),
        ('BMFVCN', 'cMF.VCtAB'),
        ('BMFMTH', 'cMF.ThCtAB.MEAN'),
        ('BMFACS', 'cMF.AC'),
        ('BMFPD', 'cMF.dAB%'),
        ('BMFCAAB', 'cMF.cAB'),
        ('BMFMTC', 'cMF.ThCcAB.MEAN'),
        ('BMFMAV', 'cMF.ThCcAB.MAX'),
        ('BMFCTS', 'cMF.ThCcAB.SD'),
        ('BMFACV', 'cMF.ThCcAB.CV'),
        ('CBMFMAT', 'ccMF.ThCtAB.MIN'),
        ('CBMFMTH', 'ccMF.ThCtAB.MEAN'),
        ('EBMFMTH', 'ecMF.ThCtAB.MEAN'),
        ('IBMFMTH', 'icMF.ThCtAB.MEAN'),
        ('CBMFPD', 'ccMF.dAB%'),
        ('EBMFPD', 'ecMF.dAB%'),
        ('IBMFPD', 'icMF.dAB%'),
        ('WMTFVCL', 'MFTC.VC'),
        ('WMTFVCN', 'MFTC.VCtAB'),
        ('WMTFMTH', 'MFTC.ThCtAB.MEAN'),
        ('WMTFMAV', 'MFTC.ThCtAB.MAX'),
        ('BMTFMAT', 'cMFTC.ThCtAB.MIN'),
        ('BMTFMTH', 'cMFTC.ThCtAB.MEAN'),
        ('WLTVCL', 'LT.VC'),
        ('WLTSBA', 'LT.tAB'),
        ('WLTVCN', 'LT.VCtAB'),
        ('WLTMTH', 'LT.ThCtAB.MEAN'),
        ('WLTACS', 'LT.AC'),
        ('WLTPD', 'LT.dAB%'),
        ('WLTCAAB', 'LT.cAB'),
        ('WLTMTC', 'LT.ThCcAB.MEAN'),
        ('WLTMAV', 'LT.ThCcAB.MAX'),
        ('WLTCTS', 'LT.ThCcAB.SD'),
        ('WLTACV', 'LT.ThCcAB.CV'),
        ('CLTMAT', 'cLT.ThCtAB.MIN'),
        ('CLTMTH', 'cLT.ThCtAB.MEAN'),
        ('ELTMTH', 'eLT.ThCtAB.MEAN'),
        ('ILTMTH', 'iLT.ThCtAB.MEAN'),
        ('ALTMTH', 'aLT.ThCtAB.MEAN'),
        ('PLTMTH', 'pLT.ThCtAB.MEAN'),
        ('CLTPD', 'cLT.dAB%'),
        ('ELTPD', 'eLT.dAB%'),
        ('ILTPD', 'iLT.dAB%'),
        ('ALTPD', 'aLT.dAB%'),
        ('PLTPD', 'pLT.dAB%'),
        ('BLFVCL', 'cLF.VC'),
        ('BLFSBA', 'cLF.tAB'),
        ('BLFVCN', 'cLF.VCtAB'),
        ('BLFMTH', 'cLF.ThCtAB.MEAN'),
        ('BLFACS', 'cLF.AC'),
        ('BLFPD', 'cLF.dAB%'),
        ('BLFCAAB', 'cLF.cAB'),
        ('BLFMTC', 'cLF.ThCcAB.MEAN'),
        ('BLFMAV', 'cLF.ThCcAB.MAX'),
        ('BLFCTS', 'cLF.ThCtAB.SD'),
        ('BLFACV', 'cLF.ThCtAB.CV'),
        ('CBLFMAT', 'ccLF.ThCtAB.MIN'),
        ('CBLFMT', 'ccLF.ThCtAB.MEAN'),
        ('EBLFMT', 'ecLF.ThCtAB.MEAN'),
        ('IBLFMT', 'icLF.ThCtAB.MEAN'),
        ('CBLFPD', 'ccLF.dAB%'),
        ('EBLFPD', 'ecLF.dAB%'),
        ('IBLFPD', 'icLF.dAB%'),
        ('WLTFVCL', 'LFTC.VC'),
        ('WLTFVCN', 'LFTC.VCtAB'),
        ('WLTFMTH', 'LFTC.ThCtAB.MEAN'),
        ('WLTFMAV', 'LFTC.ThCtAB.MAX'),
        ('BLTFMAT', 'cLFTC.ThCtAB.MIN'),
        ('BLTFMTH', 'cLFTC.ThCtAB.MEAN'),
    ]
    df_tmp = df_tmp.rename(dict(mapping_rename), axis=1)

    # Compose the output
    selection = [
        *[x[1] for x in mapping_rename]
    ]
    df_tmp = df_tmp[['patient', 'side', 'prefix_var', 'visit',
                     *selection]]
    return df_tmp


def read_info_proj_22(fpath, verbose=False):
    """OA Biomarkers Consortium FNIH Project.
    """
    if verbose:
        print(fpath)
    df = pd.read_csv(fpath, sep='|', index_col=False)
    if verbose:
        print(f'num total: {len(df)}')
    return df


def preproc_info_proj_22(df):
    """OA Biomarkers Consortium FNIH Project.

    See `OAI_CompleteData_ASCII/Clinical_FNIH_Descrip.pdf` for the details.
    """
    df_tmp = df.copy()

    # Rename the unique selectors
    df_tmp["KL"] = df_tmp['V00XRKL']
    df_tmp['age'] = df_tmp['V00AGE']
    df_tmp['sex'] = df_tmp['P02SEX']
    df_tmp['BMI'] = df_tmp['P01BMI']

    df_tmp = df_tmp.rename({
        'ID': 'patient',
        'SIDE': 'side',
    }, axis=1)

    df_tmp.loc[:, 'side'] = df_tmp['side'].replace(
        {'1: Right': 'RIGHT', '2: Left': 'LEFT'})
    df_tmp.loc[:, 'sex'] = df_tmp['sex'].replace(
        {'1: Male': 'MALE', '2: Female': 'FEMALE'})
    for field_kl in ["V00XRKL", "KL"]:
        df_tmp.loc[:, field_kl] = df_tmp[field_kl].replace(
            {'1: 1': 1, '2: 2': 2, '3: 3': 3})

    df_tmp = df_tmp.astype({
        'patient': str,
        'side': str,
        'age': int,
        'sex': str,
        'BMI': float,
        'KL': int,
    })

    return df_tmp
