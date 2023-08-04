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
    # 'jw01018-o006-t1_niriss_clear-f150w': [],
}



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


def test_jwst_cal_scrubber(jwstcal_input_path):
    scrubber = JwstCalScrubber(jwstcal_input_path, encoding_pairs=KEYPAIR_DATA)
    assert len(scrubber.fpaths) == 6
    nrc_product = 'jw02732-o001-t2_nircam_clear-f150w'
    miri_product = 'jw02732-o005-t1_miri_f1130w'
    for prod in list(scrubber.products.keys()):
        assert prod in list(JWST_EXPECTED.keys())

    nrc_exposures = sorted(list(scrubber.products[nrc_product].keys()))
    assert nrc_exposures == JWST_EXPECTED[nrc_product]
    miri_exposures = sorted(list(scrubber.products[miri_product].keys()))
    assert miri_exposures == JWST_EXPECTED[miri_product]

    scrubber.scrub_inputs()
    assert len(scrubber.df) > 0
