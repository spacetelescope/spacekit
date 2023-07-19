from pytest import mark
from spacekit.preprocessor.encode import PairEncoder, HstSvmEncoder
import pandas as pd
from numpy import asarray

ENCODED_COL_EXPECTED = [
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
    "rms_ra",
    "rms_dec",
    "nmatches",
    "wcstype",
    "category",
    "cat",
    "wcs",
    "det",
]


@mark.svm
@mark.preprocessor
@mark.encode
def test_svm_encoder(scraped_mast_file):
    data = pd.read_csv(scraped_mast_file, index_col="index")
    enc = HstSvmEncoder(data)
    enc.encode_features()
    assert enc.df.shape == (1, 18)
    for col in ENCODED_COL_EXPECTED:
        if col in list(enc.df.columns):
            assert True
        else:
            assert False
    assert enc.df.cat[0] == 3
    assert enc.df.det[0] == 1
    assert enc.df.wcs[0] == 0


@mark.svm
@mark.preprocessor
@mark.encode
def test_pair_encoder_unknown_val(scraped_mast_file):
    data = pd.read_csv(scraped_mast_file, index_col="index")
    data.loc["hst_12286_38_wfc3_ir_total_ibl738", "wcstype"] = "NaN"
    keypairs = {"a posteriori": 0, "a priori": 1, "default a": 2, "not aligned": 3}
    enc = PairEncoder()
    enc.fit(data, keypairs, axiscol="wcstype", handle_unknowns=False)
    try:
        enc.transform()
    except KeyError:
        assert True


@mark.svm
@mark.preprocessor
@mark.encode
def test_svm_encoder_handle_unknown(scraped_mast_file):
    data = pd.read_csv(scraped_mast_file, index_col="index")
    data.loc["hst_12286_38_wfc3_ir_total_ibl738", "wcstype"] = "NaN"
    keypairs = {"a posteriori": 0, "a priori": 1, "default a": 2, "not aligned": 3}
    enc = PairEncoder()
    enc.fit(data, keypairs, axiscol="wcstype", handle_unknowns=True)
    enc.transform()
    assert enc.transformed[0] == 4


@mark.svm
@mark.preprocessor
@mark.encode
def test_pair_encoder_unspecified_column(scraped_mast_file):
    data = pd.read_csv(scraped_mast_file, index_col="index")
    keypairs = {"a posteriori": 0, "a priori": 1, "default a": 2, "not aligned": 3}
    enc = PairEncoder()
    enc.fit(data, keypairs)
    assert enc.arr is None
    try:
        enc.keypairs
    except AttributeError:
        assert True


@mark.svm
@mark.preprocessor
@mark.encode
def test_pair_encoder_array_1d():
    keypairs = {"a posteriori": 0, "a priori": 1, "default a": 2, "not aligned": 3}
    arr1d = asarray(["a priori"], dtype=object)
    enc = PairEncoder()
    enc.fit(arr1d, keypairs)
    enc.transform()
    assert enc.transformed[0] == 1


@mark.svm
@mark.preprocessor
@mark.encode
def test_pair_encoder_array_2d():
    keypairs = {"a posteriori": 0, "a priori": 1, "default a": 2, "not aligned": 3}
    arr2d = asarray(
        [
            [
                "ir",
                "ibl738",
                "ANY",
                262.46,
                52.32,
                2,
                "myfile.fits",
                284,
                185,
                7,
                5.87,
                13.38,
                5,
                "default a",
                "UNIDENTIFIED;PARALLEL FIELD",
            ]
        ]
    )
    enc = PairEncoder()
    enc.fit(arr2d, keypairs, axiscol=13)
    enc.transform()
    assert enc.transformed[0] == 2


@mark.svm
@mark.preprocessor
@mark.encode
def test_pair_encoder_array_2d_unspecified_axis():
    keypairs = {"a posteriori": 0, "a priori": 1, "default a": 2, "not aligned": 3}
    arr2d = asarray(
        [
            [
                "ir",
                "ibl738",
                "ANY",
                262.46,
                52.32,
                2,
                "myfile.fits",
                284,
                185,
                7,
                5.87,
                13.38,
                5,
                "default a",
                "UNIDENTIFIED;PARALLEL FIELD",
            ]
        ]
    )
    enc = PairEncoder()
    enc.fit(arr2d, keypairs)
    try:
        enc.classes_
    except AttributeError:
        assert True


@mark.svm
@mark.preprocessor
@mark.encode
def test_pair_encoder_inverse_transform():
    data = asarray(["ir", "ir", "uvis", "wfc"], dtype=object)
    detector_keys = {"hrc": 0, "ir": 1, "sbc": 2, "uvis": 3, "wfc": 4}
    enc = PairEncoder()
    enc.fit(data, detector_keys)
    enc.transform()
    assert enc.transformed == [1, 1, 3, 4]
    enc.inverse_transform()
    assert enc.inversed == list(data)
