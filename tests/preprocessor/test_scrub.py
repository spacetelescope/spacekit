from pytest import mark
from spacekit.preprocessor.scrub import SvmScrubber
import pandas as pd
import os

SCRUBBED_COLS = [
    'detector',
    'dataset',
    'targname',
    'ra_targ',
    'dec_targ',
    'numexp',
    'imgname',
    'point',
    'segment',
    'gaia'
]

@mark.preprocessor
@mark.scrub
def test_svm_scrubber(raw_csv_file, svm_visit_data):
    data = pd.read_csv(raw_csv_file, index_col="index")
    scrub = SvmScrubber(
        data, svm_visit_data, output_path="tmp", output_file="scrubbed", crpt=0
    )
    scrub.preprocess_data()
    assert os.path.exists(scrub.fname)


# TEST SCRUBCOLS
@mark.preprocessor
@mark.scrub
def test_scrub_cols(raw_csv_file, svm_visit_data):
    data = pd.read_csv(raw_csv_file, index_col="index")
    scrubber = SvmScrubber(
        svm_visit_data, df=data, output_path="tmp", output_file="scrubbed", crpt=0
    )
    df = scrubber.scrub_columns()
    assert df.shape == (1, 10)
    for col in SCRUBBED_COLS:
        if col in list(df.columns):
            assert True
        else:
            assert False


# @mark.preprocessor
# @mark.scrub
# def test_alt_svm_scrubber(svm_visit_data):
#     #data_path = "tests/data/svm/prep/singlevisits/"
#     scb = SvmScrubber(
#         svm_visit_data, df=None, output_path="tmp", output_file="scrubbed", crpt=0
#     )
#     scb.scrub()
#     assert os.path.exists(scb.fname)
