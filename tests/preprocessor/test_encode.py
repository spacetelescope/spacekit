from pytest import mark
from spacekit.preprocessor.encode import SvmEncoder
import pandas as pd

ENCODED_COL_EXPECTED = [ 
    'detector',
    'dataset',
    'targname',
    'ra_targ',
    'dec_targ',
    'numexp',
    'imgname',
    'point',
    'segment',
    'gaia',
    'rms_ra',
    'rms_dec',
    'nmatches',
    'wcstype',
    'category',
    'cat'
]

@mark.preprocessor
@mark.encode
def test_scrape_mast(scraped_mast_file):
    data = pd.read_csv(scraped_mast_file, index_col="index")
    enc = SvmEncoder(data)
    enc.encode_features()
    assert enc.df.shape == (1, 16)
    for col in ENCODED_COL_EXPECTED:
        if col in list(enc.df.columns):
            assert True
        else:
            assert False
