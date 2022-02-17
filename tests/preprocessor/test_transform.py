from pytest import mark
from spacekit.extractor.load import load_datasets, SVMImageIO
from spacekit.preprocessor.transform import PowerX
import numpy as np
import os

RENAMED_COLS = {
    "svm": [
        "numexp_scl",
        "rms_ra_scl",
        "rms_dec_scl",
        "nmatches_scl",
        "point_scl",
        "segment_scl",
        "gaia_scl",
    ],
    "cal": [
        "x_files",
        "x_size",
    ],
}


@mark.hst
@mark.preprocessor
@mark.transform
def test_transform_arrays_from_file(cfg, df_ncols):
    (df, ncols) = df_ncols
    X_arr = np.asarray(df)
    Px = PowerX(X_arr, cols=ncols, tx_file=cfg.tx_file)
    assert list(Px.tx_data.keys()) == ["lambdas", "mu", "sigma"]
    X_norm = Px.Xt
    for i in ncols:
        assert np.abs(np.round(np.mean(X_norm[:, i]))) == 0.0
        assert np.abs(np.round(np.std(X_norm[:, i]))) == 1.0


@mark.hst
@mark.preprocessor
@mark.transform
def test_powerx_from_df(cfg, df_ncols):
    (df, ncols) = df_ncols
    Px = PowerX(
        df,
        cols=cfg.norm_cols,
        ncols=ncols,
        save_tx=True,
        output_path="tmp",
        rename=cfg.rename_cols,
        join_data=1,
    )
    assert os.path.exists("tmp/tx_data.json")
    df_norm = Px.Xt
    assert list(df_norm.columns) == RENAMED_COLS[cfg.env] + cfg.enc_cols
    for col in RENAMED_COLS[cfg.env]:
        assert np.abs(np.round(np.mean(df_norm[col]))) == 0.0
        assert np.abs(np.round(np.std(df_norm[col]))) == 1.0

    X_norm = Px.normalized
    for i in ncols:
        assert np.abs(np.round(np.mean(X_norm[:, i]))) == 0.0
        assert np.abs(np.round(np.std(X_norm[:, i]))) == 1.0


@mark.hst
@mark.preprocessor
@mark.transform
def test_transform_join_options(cfg, df_ncols):
    (df, ncols) = df_ncols
    Px0 = PowerX(df, cols=cfg.norm_cols, ncols=ncols, rename=None, join_data=0)
    norm0 = Px0.Xt
    assert norm0.shape[1] == len(cfg.norm_cols)
    assert list(norm0.columns) == cfg.norm_cols

    Px1 = PowerX(df, cols=cfg.norm_cols, ncols=ncols, rename=None, join_data=1)
    norm1 = Px1.Xt
    assert norm1.shape[1] == len(cfg.norm_cols + cfg.enc_cols)
    assert norm1.shape[1] == len(df.columns)
    assert list(norm1.columns) == cfg.norm_cols + cfg.enc_cols

    Px2 = PowerX(
        df, cols=cfg.norm_cols, ncols=ncols, rename=cfg.rename_cols, join_data=2
    )
    norm2 = Px2.Xt
    assert norm2.shape[1] == len(df.columns) + len(cfg.norm_cols)
    assert list(norm2.columns) == RENAMED_COLS[cfg.env] + list(df.columns)


@mark.hst
@mark.preprocessor
@mark.transform
def test_transform_1d_series(cfg, df_ncols):
    (df, _) = df_ncols
    x = df.iloc[0]
    x_norm = PowerX(x, cols=cfg.norm_cols, tx_file=cfg.tx_file).Xt
    assert len(x_norm.shape) == 2
    assert x_norm.shape[0] == 1


@mark.hst
@mark.preprocessor
@mark.transform
def test_transform_1d_array(cfg, df_ncols):
    (df, ncols) = df_ncols
    x = np.asarray(df.iloc[0])
    x_norm = PowerX(x, cols=ncols, tx_file=cfg.tx_file).Xt
    assert len(x_norm.shape) == 2
    assert x_norm.shape[0] == 1


@mark.svm
@mark.preprocessor
@mark.transform
def test_svm_normalize_training(svm_labeled_dataset, svm_train_npz):
    df = load_datasets([svm_labeled_dataset])
    (X, _), _ = SVMImageIO(
        svm_train_npz, w=128, h=128, d=3 * 3, inference=False, data=df, v=0.85
    ).load()
    cols = ["numexp", "rms_ra", "rms_dec", "nmatches", "point", "segment", "gaia"]
    ncols = [i for i, c in enumerate(X[0].columns) if c in cols]
    Px = PowerX(df, cols=cols, ncols=ncols, rename="_scl")
    X_train = PowerX(X[0], cols=cols, ncols=ncols, tx_data=Px.tx_data).Xt
    expected = RENAMED_COLS["svm"] + ["det", "wcs", "cat"]
    assert list(X_train.columns) == expected
    for col in RENAMED_COLS["svm"]:
        assert np.abs(np.round(np.mean(X_train[col]))) == 0.0
        assert np.abs(np.round(np.std(X_train[col]))) == 1.0

    X_test_array = np.asarray(X[1])
    X_test_norm = PowerX(X_test_array, cols=cols, ncols=ncols, tx_data=Px.tx_data).Xt
    for i in ncols:
        assert np.abs(np.round(np.mean(X_test_norm[:, i]))) == 0.0
        assert np.abs(np.round(np.std(X_test_norm[:, i]))) == 1.0
