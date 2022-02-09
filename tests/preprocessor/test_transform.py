from pytest import mark
from spacekit.extractor.load import load_datasets, SVMFileIO
from spacekit.preprocessor.transform import PowerX

#filename, img_path, img_size=128, dim=3, ch=3, norm=0, v=0.85, output_path=None

def test_powerX_transform(svm_labeled_dataset):
    output_path = "tmp"
    df = load_datasets([svm_labeled_dataset])
    (X, y), (train, test, val) = SVMFileIO(
        "tmp/img", w=128, h=128, d=3 * 3, inference=False, data=df, v=0.85
    ).load()

    cols = ["numexp", "rms_ra", "rms_dec", "nmatches", "point", "segment", "gaia"]

    ncols = [i for i, c in enumerate(df.columns) if c in cols]
    Px = PowerX(df, cols=cols, ncols=ncols, save_tx=True, output_path=output_path)
    X_train = PowerX(train[1], cols=cols, ncols=ncols, tx_data=Px.tx_data).Xt
    X_test = PowerX(test[1], cols=cols, ncols=ncols, tx_data=Px.tx_data).Xt
    X_val = PowerX(val[1], cols=cols, ncols=ncols, tx_data=Px.tx_data).Xt
    assert True
