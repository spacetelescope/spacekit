from pytest import mark
from spacekit.extractor.load import load_datasets, SVMImageIO
from spacekit.preprocessor.transform import PowerX
import numpy as np

EXPECTED_COLS = [
    'numexp_scl',
    'rms_ra_scl',
    'rms_dec_scl',
    'nmatches_scl',
    'point_scl',
    'segment_scl',
    'gaia_scl',
    'det',
    'wcs',
    'cat'
]


@mark.preprocessor
@mark.transform
def test_powerX_transform(svm_labeled_dataset):
    output_path = "tmp"
    df = load_datasets([svm_labeled_dataset])
    cols = ["numexp", "rms_ra", "rms_dec", "nmatches", "point", "segment", "gaia"]
    ncols = [i for i, c in enumerate(df.columns) if c in cols]
    Px = PowerX(df, cols=cols, ncols=ncols, save_tx=True, output_path=output_path)
    assert type(Px.normalized) == np.ndarray
    
    img_path = "tests/data/svm/train/img_data.npz"
    (X, _), _ = SVMImageIO(img_path, w=128, h=128, d=3 * 3, inference=False, data=df, v=0.85).load()
    X_train = PowerX(X[0], cols=cols, ncols=ncols, tx_data=Px.tx_data).Xt
    assert list(X_train.columns) == EXPECTED_COLS
    for col in EXPECTED_COLS[:6]:
        assert np.abs(np.round(np.mean(X_train[col]))) == 0.0
        assert np.abs(np.round(np.std(X_train[col]))) == 1.0

    X_test_array = np.asarray(X[1])
    X_test_norm = PowerX(X_test_array, cols=cols, ncols=ncols, tx_data=Px.tx_data).Xt
    for i in ncols:
        assert np.abs(np.round(np.mean(X_test_norm[:, i]))) == 0.0
        assert np.abs(np.round(np.std(X_test_norm[:, i]))) == 1.0
