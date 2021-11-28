import os
import argparse
import pandas as pd
import time
import datetime as dt
from sklearn.model_selection import train_test_split
from spacekit.preprocessor.augment import training_data_aug, training_img_aug
from spacekit.preprocessor.transform import (
    apply_power_transform,
    power_transform_matrix,
)
from spacekit.builder.cnn import Ensemble
from spacekit.analyzer.compute import ComputeTest, ComputeVal
from spacekit.extractor.load_images import SVMImages
from spacekit.analyzer.track import stopwatch

DIM = 3
CH = 3
WIDTH = 128
HEIGHT = 128
DEPTH = DIM * CH
SHAPE = (DIM, WIDTH, HEIGHT, CH)
TF_CPP_MIN_LOG_LEVEL = 2


def split_datasets(df):
    print("Splitting Data ---> X-y ---> Train-Test-Val")
    y = df["label"]
    X = df.drop("label", axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train
    )
    return X_train, X_test, X_val, y_train, y_test, y_val


def normalize_data(df, X_train, X_test, X_val):
    print("Applying Normalization (Leo-Johnson PowerTransform)")
    _, px = apply_power_transform(df)
    X_train = power_transform_matrix(X_train, px)
    X_test = power_transform_matrix(X_test, px)
    X_val = power_transform_matrix(X_val, px)
    return X_train, X_test, X_val


def make_image_sets(X_train, X_test, X_val, img_path=".", w=128, h=128, d=9, exp=None):
    """
    Read in train/test files and produce X-y data splits.
    y labels are encoded as 0=valid, 1=compromised
    returns X_train, X_test, y_train, y_test, y_val
    d=9: 3x3 rgb images (9 channels total)
    """
    start = time.time()
    stopwatch("LOADING IMAGES", t0=start)

    print("\n*** Training Set ***")
    svm_img = SVMImages(img_path, w, h, d)
    train = svm_img.detector_training_images(X_train, exp=exp)  # (idx, X, y)
    print("\n*** Test Set ***")
    test = svm_img.detector_training_images(X_test, exp=exp)
    print("\n*** Validation Set ***")
    val = svm_img.detector_training_images(X_val, exp=exp)

    end = time.time()
    stopwatch("LOADING IMAGES", t0=start, t1=end)

    print("\n[i] Length of Splits:")
    print(f"X_train={len(train[1])}, X_test={len(test[1])}, X_val={len(val[1])}")

    return train, test, val


def make_ensembles(
    train_img,
    test_img,
    val_img,
    train_data,
    test_data,
    val_data,
    y_train,
    y_test,
    y_val,
):
    print("Stacking mixed inputs (DATA + IMG)")
    XTR = [train_data, train_img]
    XTS = [test_data, test_img]
    XVL = [val_data, val_img]
    YTR = y_train.reshape(-1, 1)
    YTS = y_test.reshape(-1, 1)
    YVL = y_val.reshape(-1, 1)
    return XTR, YTR, XTS, YTS, XVL, YVL


def prep_ensemble_data(filename, img_path, synth=None, norm=False):
    print("[i] Importing Regression Test Data")
    df = pd.read_csv(filename, index_col="index")
    print("\tREG DATA: ", df.shape)
    if synth:
        print("\nAdding artificial corruption dataset")
        syn = pd.read_csv(synth, index_col="index")
        print(f"\tSYNTH DATA: {syn.shape}")
        df = pd.concat([df, syn], axis=0)
        print(f"\tTOTAL: {df.shape}")
    print(f"\nClass Labels (0=Aligned, 1=Misaligned)\n{df['label'].value_counts()}")

    X_train, X_test, X_val, y_train, y_test, y_val = split_datasets(df)

    # IMG DATA
    image_sets = [X_train, X_test, X_val]
    train, test, val = make_image_sets(
        *image_sets, img_path=img_path, w=WIDTH, h=HEIGHT, d=DEPTH
    )
    # MLP DATA
    print("\nPerforming Data Augmentation on X_train DATA")
    X_train, _ = training_data_aug(X_train, X_test, X_val, y_train, y_test, y_val)
    if norm:
        X_train, X_test, X_val = normalize_data(df, X_train, X_test, X_val)
    print("\nPerforming Data Augmentation on X_train IMAGES")
    img_idx, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl = training_img_aug(train, test, val)
    XTR, YTR, XTS, YTS, XVL, YVL = make_ensembles(
        X_tr, X_ts, X_vl, X_train, X_test, X_val, y_tr, y_ts, y_vl
    )
    tv_idx = [y_test, y_val, img_idx]
    return tv_idx, XTR, YTR, XTS, YTS, XVL, YVL


def train_model(XTR, YTR, XTS, YTS, model_name, params=None):
    if params is None:
        params = dict(
            batch_size=32,
            epochs=60,
            lr=1e-4,
            decay=[100000, 0.96],
            early_stopping=None,
            verbose=1,
            ensemble=True,
        )
    ens = Ensemble(XTR, YTR, XTS, YTS, params=params)
    name, outpath = os.path.basename(model_name), os.path.dirname(model_name)
    ens.name = name
    ens.build_ensemble(lr_sched=True)
    ens.fit_cnn()
    ens.save_model(weights=True, output_path=outpath)
    return ens.model, ens.history


def compute_results(
    model, history, model_name, tv_idx, res_path, XTR, YTR, XTS, YTS, XVL, YVL
):
    com = ComputeTest(
        model_name,
        ["aligned", "misaligned"],
        model,
        history,
        XTR,
        YTR,
        XTS,
        YTS,
        tv_idx[0],
    )
    com.res_path = res_path
    com.calculate_results()
    com.draw_plots()
    com.download()
    val = ComputeVal(
        model_name, ["aligned", "misaligned"], model, XTS, YTS, XVL, YVL, tv_idx[1]
    )
    val.res_path = res_path
    val.calculate_results()
    val.draw_plots()
    val.download()


def main(training_data, img_path, synth_data, norm, model_name, params, output_path):
    os.makedirs(output_path, exist_ok=True)
    tv_idx, XTR, YTR, XTS, YTS, XVL, YVL = prep_ensemble_data(
        training_data, img_path, synth=synth_data, norm=norm
    )
    ens_model, ens_history = train_model(XTR, YTR, XTS, YTS, model_name, params)
    res_path = os.path.join(output_path, "results")
    compute_results(
        ens_model,
        ens_history,
        model_name,
        tv_idx,
        res_path,
        XTR,
        YTR,
        XTS,
        YTS,
        XVL,
        YVL,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="spacekit",
        usage="python -m spacekit.skopes.hst.svm.train training_data.csv path/to/img"
    )
    parser.add_argument(
        "training_data", type=str, help="path to training data csv file(s)"
    )
    parser.add_argument("img_path", type=str, help="path to training image directory")
    parser.add_argument(
        "-m", "--model_name", type=str, default="ensembleSVM", help="name to give model"
    )
    parser.add_argument(
        "-s",
        "--synthetic_data",
        type=str,
        default=None,
        help="path to synthetic/corruption csv file (if saved separately)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=os.getcwd(),
        help="path to synthetic/corruption csv file (if saved separately)",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        type=str,
        default=0,
        help="apply normalization and scaling to regression test data",
    )
    parser.add_argument("-b", "--batchsize", type=int, default=32, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=60, help="number of epochs")
    parser.add_argument(
        "-y", "--early_stopping", type=str, default=None, help="early stopping"
    )
    parser.add_argument("-v", "--verbose", type=int, default=2, help="verbosity level")
    args = parser.parse_args()
    training_data = args.training_data
    img_path = args.img_path
    model_name = args.model_name
    timestamp = str(int(dt.datetime.now().timestamp()))
    output_path = os.path.join(args.output_path, f"mml_{timestamp}")
    synth_data = args.synthetic_data
    norm = args.normalize
    verbose = args.verbose
    # SET MODEL FIT PARAMS
    BATCHSIZE = args.batchsize
    EPOCHS = args.epochs
    EARLY = args.early_stopping
    params = dict(
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        lr=1e-4,
        decay=[100000, 0.96],
        early_stopping=EARLY,
        verbose=verbose,
        ensemble=True,
    )

    main(training_data, img_path, synth_data, norm, model_name, params, output_path)
