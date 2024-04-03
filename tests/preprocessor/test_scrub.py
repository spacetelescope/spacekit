from pytest import mark
from spacekit.preprocessor.scrub import HstSvmScrubber, JwstCalScrubber
from spacekit.skopes.jwst.cal.config import KEYPAIR_DATA
import os

SCRUBBED_COLS = [
    "detector",
    "dataset",
    "targname",
    "ra_targ",
    "dec_targ",
    "numexp",
    "imgname",
    "point",
    "segment",
    "gaia",
]

FINAL_COLS = [
    "numexp",
    "rms_ra",
    "rms_dec",
    "nmatches",
    "point",
    "segment",
    "gaia",
    "det",
    "wcs",
    "cat",
]

JWST_EXPECTED = {
    'jw02732-o001-t2_nircam_clear-f150w': [
        'jw02732001005_02103_00005_nrcb1',
        'jw02732001005_02103_00005_nrcb2',
        'jw02732001005_02103_00005_nrcb3',
        'jw02732001005_02103_00005_nrcb4'
    ],
    "jw02732-o005-t1_miri_f1130w": [
        'jw02732005001_02105_00001_mirimage',
        'jw02732005001_02105_00002_mirimage'
    ],
    'jw01018-o006-t1_niriss_clear-f150w': [
        'jw01018006001_02101_00002_nis_uncal.fits',
        'jw01018006001_02101_00003_nis_uncal.fits',
        'jw01018006001_02101_00004_nis_uncal.fits',
        'jw02732001005_02103_00005_nrcb2_uncal.fits',
    ],
}

JWST_SCRUBBED_COLS = [
    'instr',
    'detector',
    'exp_type',
    'visitype',
    'filter',
    'pupil',
    'grating',
    'channel',
    'subarray',
    'bkgdtarg',
    'is_imprt',
    'tsovisit',
    'nexposur',
    'numdthpt',
    'targ_max_offset',
    'offset',
    'max_offset',
    'mean_offset',
    'sigma_offset',
    'err_offset',
    'sigma1_mean',
    'frac',
    'targ_frac',
    'gs_mag',
    'crowdfld'
]


@mark.hst
@mark.svm
@mark.preprocessor
@mark.scrub
def test_svm_scrubber(raw_svm_data, single_visit_path):
    scrubber = HstSvmScrubber(
        single_visit_path,
        data=raw_svm_data,
        output_path="tmp",
        output_file="scrubbed",
        crpt=0,
    )
    assert scrubber.df.shape[1] == 9
    scrubber.preprocess_data()
    assert scrubber.df.shape[1] == 10
    assert list(scrubber.df.columns) == FINAL_COLS
    assert os.path.exists(scrubber.data_path)
    base_path = os.path.dirname(scrubber.data_path)
    raw_file = "raw_" + os.path.basename(scrubber.data_path)
    raw_fpath = os.path.join(base_path, raw_file)
    assert os.path.exists(raw_fpath)


# TEST SCRUBCOLS
@mark.hst
@mark.svm
@mark.preprocessor
@mark.scrub
def test_scrub_cols(raw_svm_data, single_visit_path):
    scrubber = HstSvmScrubber(
        single_visit_path,
        data=raw_svm_data,
        output_path="tmp",
        output_file="scrubbed",
        crpt=0,
    )
    scrubber.scrub_columns()
    assert scrubber.df.shape == (1, 10)
    for col in SCRUBBED_COLS:
        if col in list(scrubber.df.columns):
            assert True
        else:
            assert False


@mark.jwst
@mark.preprocessor
@mark.scrub
def test_jwst_cal_scrubber(jwstcal_input_path):
    scrubber = JwstCalScrubber(jwstcal_input_path, encoding_pairs=KEYPAIR_DATA)
    assert len(scrubber.fpaths) == 10
    assert len(scrubber.imgpix) == 3
    imgpix_products = list(scrubber.imgpix.keys())
    for product in imgpix_products:
        assert len(scrubber.imgpix[product].keys()) == 48
    image_inputs = scrubber.scrub_inputs(exp_type="IMAGE")
    assert len(image_inputs) == 3
    assert list(image_inputs.columns) == JWST_SCRUBBED_COLS
