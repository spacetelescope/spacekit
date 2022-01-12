import os
import argparse
import pandas as pd
import time
import datetime as dt
from sklearn.model_selection import train_test_split
from spacekit.generator.augment import training_data_aug, training_img_aug
from spacekit.preprocessor.transform import (
    apply_power_transform,
    power_transform_matrix,
)
from spacekit.builder.networks import Ensemble
from spacekit.analyzer.compute import ComputeBinary
from spacekit.extractor.load import SVMImages
from spacekit.analyzer.track import stopwatch

DIM = 3
CH = 3
WIDTH = 128
HEIGHT = 128
DEPTH = DIM * CH
SHAPE = (DIM, WIDTH, HEIGHT, CH)
TF_CPP_MIN_LOG_LEVEL = 2


def import_dataset(filename, synth=None):
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
    return df


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
    svm_img = SVMImages(img_path, w=w, h=h, d=d)
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


def load_ensemble_data(filename, img_path, synth=None, norm=False):
    df = import_dataset(filename, synth=synth)

    X_train, X_test, X_val, y_train, y_test, y_val = split_datasets(df)

    # IMG DATA
    image_sets = [X_train, X_test, X_val]
    train, test, val = make_image_sets(
        *image_sets, img_path=img_path, w=WIDTH, h=HEIGHT, d=DEPTH
    )
    # MLP DATA
    print("\nPerforming Regression Data Augmentation")
    X_train, _ = training_data_aug(X_train, X_test, X_val, y_train, y_test, y_val)
    if norm:
        X_train, X_test, X_val = normalize_data(df, X_train, X_test, X_val)
        # TODO: normalize images: img / 255.
    print("\nPerforming Image Data Augmentation")
    img_idx, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl = training_img_aug(train, test, val)
    XTR, YTR, XTS, YTS, XVL, YVL = make_ensembles(
        X_tr, X_ts, X_vl, X_train, X_test, X_val, y_tr, y_ts, y_vl
    )
    tv_idx = [y_test, y_val, img_idx]
    return tv_idx, XTR, YTR, XTS, YTS, XVL, YVL


def train_ensemble(
    XTR, YTR, XTS, YTS, model_name="ensembleSVM", params=None, output_path=None
):
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
    ens = Ensemble(
        XTR,
        YTR,
        XTS,
        YTS,
        params=params,
        input_name="svm_mixed_inputs",
        output_name="svm_output",
        name="ensemble_svm",
    )
    ens.build_ensemble()
    ens.batch_fit()
    if output_path is None:
        output_path = os.getcwd()
    model_outpath = os.path.join(output_path, os.path.dirname(model_name))
    ens.save_model(weights=True, output_path=model_outpath)
    return ens


def compute_results(ens, tv_idx, output_path=None):
    if output_path is None:
        output_path = os.getcwd()
    res_path = os.path.join(output_path, "results")
    # test set
    ens.test_idx = tv_idx[0]
    com = ComputeBinary(builder=ens, res_path=f"{res_path}/test")
    com.calculate_results()
    _ = com.make_outputs()
    if ens.X_val is not None:
        # validation set
        ens.test_idx = tv_idx[1]
        val = ComputeBinary(builder=ens, res_path=f"{res_path}/val", validation=True)
    val.calculate_results()
    _ = val.make_outputs()
    return com, val


def run_training(
    data_file,
    img_path,
    synth_data=None,
    norm=False,
    model_name=None,
    params=None,
    output_path=None,
):
    """Main calling function to load and prep the data, train the model, compute results and save to disk.

    Parameters
    ----------
    data_file : str (path)
        path to preprocessed dataframe csv file
    img_path : str (path)
        path to png images parent directory
    synth_data : str (path), optional
        path to additional (synthetic/corrupted) dataframe csv file, by default None
    norm : bool, optional
        apply normalization step, by default False
    model_name : str, optional
        custom name to assign to model, by default None
    params : dict, optional
        custom training hyperparameters dictionary, by default None
    output_path : str (path), optional
        custom path for saving model, results, by default None (current working directory)

    Returns
    -------
    builder.networks.Ensemble, analyzer.compute.BinaryCompute, analyzer.compute.BinaryCompute
        ensemble builder object, binary compute object, validation compute object
    """
    tv_idx, XTR, YTR, XTS, YTS, XVL, YVL = load_ensemble_data(
        data_file, img_path, synth=synth_data, norm=norm
    )
    ens = train_ensemble(
        XTR,
        YTR,
        XTS,
        YTS,
        model_name=model_name,
        params=params,
        output_path=output_path,
    )
    ens.X_val, ens.y_val = XVL, YVL
    com, val = compute_results(ens, tv_idx, output_path=output_path)
    return ens, com, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spacekit",
        usage="python -m spacekit.skopes.hst.svm.train svm_train.csv path/to/img",
    )
    parser.add_argument("data_file", type=str, help="path to training data csv file(s)")
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
        default=None,
        help="filepath location to save outputs",
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
    parser.add_argument(
        "-p", "--plots", type=int, default=0, help="draw model performance plots"
    )
    args = parser.parse_args()
    model_name = args.model_name
    timestamp = str(int(dt.datetime.now().timestamp()))
    if args.output_path is None:
        output_path = os.path.join(os.getcwd(), f"mml_{timestamp}")
    else:
        output_path = args.output_path
    # SET MODEL FIT PARAMS
    params = dict(
        batch_size=args.batchsize,
        epochs=args.epochs,
        lr=1e-4,
        decay=[100000, 0.96],
        early_stopping=args.early_stopping,
        verbose=args.verbose,
        ensemble=True,
    )
    ens, com, val = run_training(
        args.data_file,
        args.img_path,
        synth_data=args.synthetic_data,
        norm=args.normalize,
        model_name=args.model_name,
        params=params,
        output_path=output_path,
    )
    if args.plots is True:
        com.draw_plots()
        val.draw_plots()
