from pytest import mark
from spacekit.preprocessor.scrub import HstSvmScrubber
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

@mark.hst
@mark.svm
@mark.preprocessor
@mark.scrub
def test_svm_scrubber(raw_svm_data, single_visit_path):
    scrubber = HstSvmScrubber(
        single_visit_path, data=raw_svm_data, output_path="tmp", output_file="scrubbed", crpt=0
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
        single_visit_path, data=raw_svm_data, output_path="tmp", output_file="scrubbed", crpt=0
    )
    scrubber.scrub_columns()
    assert scrubber.df.shape == (1, 10)
    for col in SCRUBBED_COLS:
        if col in list(scrubber.df.columns):
            assert True
        else:
            assert False
