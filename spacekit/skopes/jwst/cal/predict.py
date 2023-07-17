# Program (PID) of exposures comes into pipeline
# create list of L1B exposures
# scrape refpix vals from scihdrs + any additional metadata
# determine potential L3 products based on obs, filters, detectors, etc
# calculate sky separation / reference pixel offset statistics
# preprocessing: create dataframe of all input values, encode categoricals
# load model
# run inference


import os
import glob
from astropy.io import fits
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd
from spacekit.logger.log import Logger
from spacekit.skopes.jwst.cal.config import GENKEYS, SCIKEYS, COLUMN_ORDER
from spacekit.preprocessor.scrub import apply_nandlers
# from spacekit.preprocessor.prep import JwstCalPrep
from spacekit.preprocessor.transform import arrays_to_tensors
from spacekit.builder.architect import BuilderMLP

# create list of L1B exposures

# fnames = [
#     'jw01032001001_02101_00001_mirimage_uncal.fits',
#     'jw01032002001_02101_00001_mirimage_uncal.fits',
#     'jw01032002001_02101_00002_mirimage_uncal.fits',
#     'jw01032002001_02101_00003_mirimage_uncal.fits',
#     'jw01032002001_02101_00004_mirimage_uncal.fits',
# ]
def get_input_exposures(input_path):
    return glob.glob(f"{input_path}/*_uncal.fits")

# scrape refpix vals from scihdrs + any additional metadata

def scrape_fits_headers(fpaths, gen_keys=GENKEYS, sci_keys=SCIKEYS):
    exp_headers = {}
    for fpath in fpaths:
        fname = str(os.path.basename(fpath))
        sfx = fname.split("_")[-1] # _uncal.fits
        name = fname.replace(f"_{sfx}", "")
        exp_headers[name] = dict()
        genhdr = fits.getheader(fpath, ext=0)
        scihdr = fits.getheader(fpath, ext=1)
        for g in gen_keys:
            exp_headers[name][g] = genhdr[g] if g in genhdr else "NaN"
        for s in sci_keys:
            exp_headers[name][s] = scihdr[s] if s in scihdr else "NaN"
    return exp_headers


# determine potential L3 products based on obs, filters, detectors, etc
# group by target+obs num+filter (+pupil)

def get_potential_l3_products(exp_headers):
    #exp_headers = scrape_fits_headers()
    targetnames = list(set([v['TARGNAME'] for v in exp_headers.values()]))
    tnums = [f"t{i+1}" for i, t in enumerate(targetnames)]
    targs = dict(zip(targetnames, tnums))
    products = dict()

    for k, v in exp_headers.items():
        tnum = targs.get(v['TARGNAME'])
        if v['PUPIL'] == "CLEAR":
            p = f"jw{v['PROGRAM']}-o{v['OBSERVTN']}-{tnum}_{v['INSTRUME']}_{v['PUPIL']}-{v['FILTER']}".lower()
        elif v['PUPIL'] != "NaN":
            p = f"jw{v['PROGRAM']}-o{v['OBSERVTN']}-{tnum}_{v['INSTRUME']}_{v['FILTER']}-{v['PUPIL']}".lower()
        else:
            p = f"jw{v['PROGRAM']}-o{v['OBSERVTN']}-{tnum}_{v['INSTRUME']}_{v['FILTER']}".lower()
        if p in products:
            products[p]
            products[p][k] = v
        else:
            products[p] = {k:v}
    return products


# calculate sky separation / reference pixel offset statistics
def image_pixel_scales():
    return dict(
        NIRCAM=dict(
            SHORT=0.03,
            LONG=0.06,
        ),
        MIRI=0.11,
        NIRISS=0.06,
        NIRSPEC=0.1,
    )

def get_scale(instr, channel=None):
    p_scales = image_pixel_scales()
    if channel not in [None, "NONE", "NaN"]:
        return p_scales[instr][channel]
    else:
        return p_scales[instr]


def pixel_sky_separation(ra, dec, p_coords, scale):
    coords = SkyCoord(ra, dec, unit="deg")
    skysep_angle = p_coords.separation(coords)
    arcsec = skysep_angle.arcsecond
    pixel = arcsec / scale
    return pixel


def offset_statistics(refpix, offsets, k, pfx=''):
    offsets = np.asarray(offsets)
    refpix[k][f'{pfx}max_offset'] = np.max(offsets)
    refpix[k][f'{pfx}mean_offset'] = np.mean(offsets)
    refpix[k][f'{pfx}sigma_offset'] = np.std(offsets)
    refpix[k][f'{pfx}err_offset'] = np.std(offsets) / np.sqrt(len(offsets))
    sigma1_idx = np.where(offsets > np.mean(offsets)+np.std(offsets))[0]
    if len(sigma1_idx) > 0:
        refpix[k][f'{pfx}sigma1_mean'] = np.mean(offsets[sigma1_idx])
        refpix[k][f'{pfx}frac'] = len(offsets[sigma1_idx]) / len(offsets)
    else:
        refpix[k][f'{pfx}sigma1_mean'] = 0.0
        refpix[k][f'{pfx}frac'] = 0.0
    return refpix


def get_pixel_offsets(products, p):
    exp_headers = products.get(p)
    offsets = []
    refpix = {p:dict(nexposur=len(list(exp_headers.keys())))}
    min_offset = None
    detectors = []
    for k, v in exp_headers.items():
        scale = get_scale(v['INSTRUME'], channel=v['CHANNEL'])
        t_coords = SkyCoord(v['TARG_RA'], v['TARG_DEC'], unit="deg")
        pixel = pixel_sky_separation(v['CRVAL1'], v['CRVAL2'], t_coords, scale=scale)
        exp_headers[k]['offset'] = pixel
        offsets.append(pixel)
        if min_offset is None:
            min_offset = pixel
        elif pixel < min_offset:
            min_offset = pixel
        else:
            continue
        if v['DETECTOR'] not in detectors:
            detectors.append(v['DETECTOR'])
    ref_exp = [k for k,v in exp_headers.items() if v['offset'] == min_offset][0]
    refpix[p].update(exp_headers[ref_exp])

    if len(detectors) > 1:
        refpix[p]['DETECTOR'] = '|'.join([d for d in detectors])
    else:
        refpix[p]['DETECTOR'] = detectors[0]

    refpix = offset_statistics(refpix, offsets, p)
    return refpix

# refpix = {
#     'jw01153-o011-t1_nircam_clear-f212n': {
#         'nexposur': 2,
#         'PROGRAM': '01153',
#         'OBSERVTN': '011',
#         'BKGDTARG': False,
#         'VISITYPE': 'PRIME_WFSC_SENSING_ONLY',
#         'TSOVISIT': False,
#         'TARGNAME': 'TYC 4212-1079-1',
#         'TARG_RA': 268.9183320833334,
#         'TARG_DEC': 65.85770833333332,
#         'INSTRUME': 'NIRCAM',
#         'DETECTOR': 'NRCA2',
#         'FILTER': 'F212N',
#         'PUPIL': 'CLEAR',
#         'EXP_TYPE': 'NRC_IMAGE',
#         'CHANNEL': 'SHORT',
#         'SUBARRAY': 'FULL',
#         'NUMDTHPT': 2,
#         'GS_RA': 269.048483,
#         'GS_DEC': 65.855498,
#         'RA_REF': 268.8796579753335,
#         'DEC_REF': 65.87927918306961,
#         'CRVAL1': 268.8796579753335,
#         'CRVAL2': 65.87927918306961,
#         'offset': 3209.403065384182,
#         'max_offset': 3213.1090393674435,
#         'mean_offset': 3211.256052375813,
#         'sigma_offset': 1.8529869916308144,
#         'err_offset': 1.3102596672326092,
#         'sigma1_mean': 0.0,
#         'frac': 0.0,
#     }
# }

# preprocessing

def preprocess_inputs(refpix, xcols=COLUMN_ORDER['asn']):
    df = pd.DataFrame.from_dict(refpix, orient='index')
    cols = list(df.columns)
    df.rename(dict(zip(cols, [c.lower() for c in cols])), axis=1, inplace=True)
    df.rename({'instrume':'instr'}, axis=1, inplace=True)
    df = df[xcols]
    nandler_keys = dict(
        continuous=[
            'nexposur',
            'numdthpt',
            'offset',
            'max_offset',
            'mean_offset',
            'sigma_offset',
            'err_offset',
            'sigma1_mean',
            'frac',
        ],
        boolean=['bkgdtarg','tsovisit'],
        categorical=[
            'instr',
            'detector',
            'filter',
            'pupil',
            'exp_type',
            'channel',
            'subarray',
            'visitype',
        ],
    )
    df = apply_nandlers(df, nandler_keys)

    # TODO
    # df, enc_pairs = encode_categories(df, categorical, **encoding_kwargs)

    # jp = JwstCalPrep(
    #     df,
    #     y_target='imagesize',
    #     X_cols=xcols,
    #     norm_cols=[],
    #     rename_cols=[],
    #     tensors=True,
    #     normalize=False,
    #     random=42,
    #     tsize=0.2,
    #     encode_targets=False,
    # )

# load model


# run inference

