"""
This module builds, trains, and evaluates an ensemble model for labeled and preprocessed SVM regression test data and alignment images. The ensemble model is a combination of two neural networks: a MultiLayerPerceptron (for regression test data) and a 3D Image Convolutional Neural Network (CNN). The script includes functions for the following steps:

1. load and prep the data and images for ML
2. build and train the model
3. compute results and save to disk

This script (and/or its functions) should be used in conjunction with spacekit.skopes.hst.svm.prep if using raw data (since both the regression test dataframe for MLP and the png images for the CNN need to be created first). Once a model has been trained using this script, it is saved to disk and can be loaded again later for use with the predict script (spacekit.skopes.hst.svm.predict).
"""

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
    """Import prerocessed regression test dataset from csv file. Optionally combine with synthetic data (if saved in a separate file).

    Parameters
    ----------
    filename : str (path)
        path to csv file containing preprocessed regression test data.
    synth : str (path), optional
        path to csv file containing synthetic regression test data, by default None

    Returns
    -------
    DataFrame
        Labeled dataframe loaded from csv file.
    """
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


def split_datasets(df, target="label", val=True):
    """Splits Pandas dataframe into feature (X) and target (y) train, test and validation sets.

    Parameters
    ----------
    df : Pandas dataframe
        preprocessed SVM regression test dataset
    target : str, optional
        target class label for alignment model predictions, by default "label"
    val : bool, optional
        create a validation set separate from train/test, by default True

    Returns
    -------
    Pandas dataframes
        features (X) and targets (y) split into train, test, and validation sets
    """
    print("Splitting Data ---> X-y ---> Train-Test-Val")
    y = df[target]
    X = df.drop(target, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )
    if val is True:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train
        )
        return X_train, X_test, X_val, y_train, y_test, y_val
    else:
        return X_train, X_test, y_train, y_test


def normalize_data(df, X_train, X_test, X_val):
    """Apply Leo-Johnson PowerTransform (via scikit learn) normalization and scaling to the input features.

    Parameters
    ----------
    df : pandas dataframe
        SVM regression test dataset
    X_train : numpy array
        training set feature inputs array
    X_test : numpy array
        test set feature inputs array
    X_val : numpy array
        validation set inputs array

    Returns
    -------
    numpy arrays
        normalized and scaled training, test, and validation sets
    """
    print("Applying Normalization (Leo-Johnson PowerTransform)")
    _, px = apply_power_transform(df)
    X_train = power_transform_matrix(X_train, px)
    X_test = power_transform_matrix(X_test, px)
    X_val = power_transform_matrix(X_val, px)
    return X_train, X_test, X_val


def make_image_sets(
    X_train, X_test, X_val, img_path="img", w=128, h=128, d=9, exp=None
):
    """
    Read in train/test files and produce X-y data splits. y labels are encoded as 0=valid, 1=compromised.
    By default, the ImageCNN3D model expects RGB image inputs as 3x3 arrays, for a total of 9 channels. This can of course be customized for other image arrangements using other classes in the spacekit.builder.networks module.

    Parameters
    ----------
    X_train : numpy array
        training image inputs
    X_test : [type]
        test image inputs
    X_val : [type]
        validation image inputs
    img_path : str, optional
        path to png images parent directory, by default "img"
    w : int, optional
        width of image, by default 128
    h : int, optional
        height of image, by default 128
    d : int, optional
        dimensions of image (determined by channels (rgb=3) multipled by depth (num image frames), by default 9
    exp : int, optional
        "expand" dimensions: (exp, w, h, 3). Set to 3 for predictions, None for training, by default None

    Returns
    -------
    nested lists
        train, test, val nested lists each containing an index of the visit names and png image data as numpy arrays.
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
    """Creates tupled pairs of regression test (MLP) data and image (CNN) array inputs for an ensemble model.

    Parameters
    ----------
    train_img : numpy array
        training set image inputs
    test_img : numpy array
        test set image inputs
    val_img : numpy array
        validation set image inputs
    train_data : numpy array
        training set feature data inputs
    test_data : numpy array
        test set feature data inputs
    val_data : numpy array
        validation set feature data inputs
    y_train : numpy array
        training set target values
    y_test : numpy array
        test set target values
    y_val : numpy array
        validation set target values

    Returns
    -------
    numpy arrays
        XTR, YTR, XTS, YTS, XVL, YVL
        List/tuple of feature input arrays (data, img) and target values for train-test-val sets
    """
    print("Stacking mixed inputs (DATA + IMG)")
    XTR = [train_data, train_img]
    XTS = [test_data, test_img]
    XVL = [val_data, val_img]
    YTR = y_train.reshape(-1, 1)
    YTS = y_test.reshape(-1, 1)
    YVL = y_val.reshape(-1, 1)
    return XTR, YTR, XTS, YTS, XVL, YVL


def load_ensemble_data(filename, img_path, synth=None, norm=False):
    """Loads regression test data from a csv file and image data from png files. Splits the data into train, test and validation sets, applies normalization (if norm=1), creates a maste index of the original dataset input names, and stacks the features and class targets for both data types into lists which can be used as inputs for an ensemble model.

    Parameters
    ----------
    filename : str
        path to preprocessed dataframe csv file
    img_path : str
        path to png images parent directory
    synth : str, optional
        path to additional (synthetic/corrupted) dataframe csv file, by default None
    norm : bool, optional
        apply normalization step, by default False

    Returns
    -------
    list, numpy arrays
        tv_idx, XTR, YTR, XTS, YTS, XVL, YVL
        list of test-validation indices, train-test feature (X) and target (y) numpy arrays.
    """
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
    """Build, compile and fit an ensemble model with regression test data and image input arrays.

    Parameters
    ----------
    XTR : tuple/list
        training set feature (X) tuple of regression data and image data numpy arrays.
    YTR : numpy array
        training set target values
    XTS : tuple/list
        test set feature (X) tuple of regression data and image data numpy arrays.
    YTS : numpy array
        test set target values
    model_name : str, optional
        name of model, by default "ensembleSVM"
    params : dict, optional
        custom parameters for model fitting, by default None
    output_path : str, optional
        custom path for saving model, results, by default None (current working directory)

    Returns
    -------
    spacekit.builder.networks.Ensemble model object
        Builder ensemble subclass model object trained on the inputs
    """
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
    """Creates Compute objects of test and validation sets for model evaluation and saves calculated results to disk for later analysis.

    Parameters
    ----------
    ens : builder.networks.Ensemble
        ensemble model builder object
    tv_idx : list of Pandas Series
        test-validation indices (used for FNFP analysis)
    output_path : str, optional
        custom path for saving model, results, by default None (current working directory)

    Returns
    -------
    spacekit.analyzer.compute.Computer objects
        Test and Validation computer objects
    """
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
    parser.add_argument(
        "img_path", type=str, help="path to png images parent directory"
    )
    parser.add_argument(
        "-m", "--model_name", type=str, default="ensembleSVM", help="name to give model"
    )
    parser.add_argument(
        "-s",
        "--synth_data",
        type=str,
        default=None,
        help="path to synthetic/corruption csv file (if saved separately)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="custom path for saving model, results, by default None (current working directory)",
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
        synth_data=args.synth_data,
        norm=args.normalize,
        model_name=args.model_name,
        params=params,
        output_path=output_path,
    )
    if args.plots is True:
        com.draw_plots()
        val.draw_plots()
