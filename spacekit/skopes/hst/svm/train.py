"""
This module builds, trains, and evaluates an ensemble model for labeled and preprocessed SVM regression test data and alignment images. The ensemble model is a combination of two neural networks: a MultiLayerPerceptron (for regression test data) and a 3D Image Convolutional Neural Network (CNN). The script includes functions for the following steps:

1. load and prep the data and images for ML
2. build and train the model
3. compute results and save to disk

This script (and/or its functions) should be used in conjunction with spacekit.skopes.hst.svm.prep if using raw data (since both the regression test dataframe for MLP and the png images for the CNN need to be created first). Once a model has been trained using this script, it is saved to disk and can be loaded again later for use with the predict script (spacekit.skopes.hst.svm.predict).
"""
import os
import argparse
import datetime as dt
from spacekit.extractor.load import load_datasets, SVMImageIO
from spacekit.generator.augment import training_data_aug, training_img_aug
from spacekit.preprocessor.transform import (
    normalize_training_data,
    normalize_training_images,
)
from spacekit.builder.architect import BuilderEnsemble
from spacekit.analyzer.compute import ComputeBinary

DIM = 3
CH = 3
WIDTH = 128
HEIGHT = 128
DEPTH = DIM * CH
SHAPE = (DIM, WIDTH, HEIGHT, CH)
TF_CPP_MIN_LOG_LEVEL = 2


def make_ensembles(
    train_data,
    train_img,
    train_label,
    test_data,
    test_img,
    test_label,
    val_data=None,
    val_img=None,
    val_label=None,
):
    """Creates tupled pairs of regression test (MLP) data and image (CNN) array inputs for an ensemble model.

    Parameters
    ----------
    train_data : numpy array
        training set feature data inputs
    train_img : numpy array
        training set image inputs
    train_label : numpy array
        training set target values
    test_data : numpy array
        test set feature data inputs
    test_img : numpy array
        test set image inputs
    test_label : numpy array
        test set target values
    val_data : numpy array, optional
        validation set feature data inputs
    val_img : numpy array, optional
        validation set image inputs
    val_label : numpy array, optional
        validation set target values

    Returns
    -------
    tuples of 6 ndarrays (only 4 if validation kwargs are None)
        XTR, YTR, XTS, YTS, XVL, YVL
        List/tuple of feature input arrays (data, img) and target values for train-test-val sets
    """
    print("Stacking mixed inputs (DATA + IMG)")
    XTR = [train_data, train_img]
    YTR = train_label.reshape(-1, 1)
    XTS = [test_data, test_img]
    YTS = test_label.reshape(-1, 1)
    if val_data is not None:
        XVL = [val_data, val_img]
        YVL = val_label.reshape(-1, 1)
        return XTR, YTR, XTS, YTS, XVL, YVL
    else:
        return XTR, YTR, XTS, YTS


def load_ensemble_data(
    filename, img_path, img_size=128, dim=3, ch=3, norm=0, v=0.85, output_path=None
):
    """Loads regression test data from a csv file and image data from png files. Splits the data into train, test and validation sets, applies normalization (if norm=1), creates a maste index of the original dataset input names, and stacks the features and class targets for both data types into lists which can be used as inputs for an ensemble model.

    Parameters
    ----------
    filename : str
        path to preprocessed dataframe csv file
    img_path : str
        path to png images parent directory
    img_size: int, optional
        image size (single value assigned to width and height), by default 128
    dim: int, optional
        dimensions (or volume) of image frames per image (for 3D CNN), by default 3
    ch: int, optional
        channels (rgb is 3, grayscale is 1), by default 3
    norm : bool, optional
        apply normalization step, by default 0
    v: float, optional
        validation set ratio for evaluating model, by default 0.85
    output_path: str, optional
        where to save the outputs (defaults to current working directory), by default None

    Returns
    -------
    list, ndarrays
        tv_idx, XTR, YTR, XTS, YTS, XVL, YVL
        list of test-validation indices, train-test feature (X) and target (y) numpy arrays.
    """
    # LOAD MLP and CNN DATA
    print("[i] Importing Regression Test Data")
    df = load_datasets([filename])
    print("\tREG DATA: ", df.shape)
    print(f"\nClass Labels (0=Aligned, 1=Misaligned)\n{df['label'].value_counts()}")

    (X, y), (train, test, val) = SVMImageIO(
        img_path, w=img_size, h=img_size, d=dim * ch, inference=False, data=df, v=v
    ).load()

    # DATA AUGMENTATION
    print("\nPerforming Regression Data Augmentation")
    X_train, _ = training_data_aug(X[0], y[0])

    # IMAGE AUGMENTATION
    print("\nPerforming Image Data Augmentation")
    img_idx, (X_tr, y_tr), (X_ts, y_ts), (X_vl, y_vl) = training_img_aug(
        train, test, val=val
    )

    # NORMALIZATION and SCALING
    if norm:
        cols = ["numexp", "rms_ra", "rms_dec", "nmatches", "point", "segment", "gaia"]
        X_train, X_test, X_val = normalize_training_data(
            df, cols, X_train, X[1], X_val=X[2], output_path=output_path, rename=None
        )
        X_tr, X_ts, X_vl = normalize_training_images(X_tr, X_ts, X_vl=X_vl)
    else:
        X_test, X_val = X[1], X[2]
    # JOIN INPUTS: MLP + CNN
    XTR, YTR, XTS, YTS, XVL, YVL = make_ensembles(
        X_train,
        X_tr,
        y_tr,
        X_test,
        X_ts,
        y_ts,
        val_data=X_val,
        val_img=X_vl,
        val_label=y_vl,
    )
    tv_idx = [y[1], y[2], img_idx]
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
    ens = BuilderEnsemble(
        XTR,
        YTR,
        XTS,
        YTS,
        params=params,
        input_name="svm_mixed_inputs",
        output_name="svm_output",
        name=model_name,
    )
    ens.build()
    ens.batch_fit()
    if output_path is None:
        output_path = os.getcwd()
    model_outpath = os.path.join(output_path, os.path.dirname(model_name))
    ens.save_model(weights=True, output_path=model_outpath)
    return ens


def compute_results(ens, tv_idx, val_set=(), output_path=None):
    """Creates Compute objects of test and validation sets for model evaluation and saves calculated results to disk for later analysis. Validation set is a subset of data that has not been seen by the model and is necessary for measuring robustness.

    Parameters
    ----------
    ens : builder.networks.Ensemble
        ensemble model builder object
    tv_idx : tuple or list of Pandas Series
        test and validation indices (used for FNFP analysis)
    val_set: tuple or list of arrays
        validation set (X_val, y_val) of features and target arrays.
    output_path : str, optional
        custom path for saving model, results, by default None (current working directory)

    Returns
    -------
    spacekit.analyzer.compute.Computer objects
        Test and Validation computer objects (if val_set is left empty, returns only a single Com obj)
    """
    if output_path is None:
        output_path = os.getcwd()
    res_path = os.path.join(output_path, "results")
    # test set
    ens.test_idx = tv_idx[0]
    com = ComputeBinary(builder=ens, res_path=f"{res_path}/test")
    com.calculate_results()
    _ = com.make_outputs()
    # validation set
    if len(val_set) == 2 and val_set[0][0].shape[0] > 2:  # temp (ignores test data)
        (ens.X_val, ens.y_val), ens.test_idx = val_set, tv_idx[1]
        val = ComputeBinary(builder=ens, res_path=f"{res_path}/val", validation=True)
        val.calculate_results()
        _ = val.make_outputs()
    else:
        val = None
    return com, val


def run_training(
    data_file,
    img_path,
    img_size=128,
    norm=0,
    v=0.85,
    model_name="ensembleSVM",
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
    img_size: int, optional
        image size (single value assigned to width and height)
    norm : int, optional
        apply normalization step (1=True, 0=False), by default 0
    v: float, optional
        validation set ratio for evaluating model, by default 0.85
    model_name : str, optional
        custom name to assign to model, by default "ensembleSVM"
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
        data_file, img_path, img_size=img_size, norm=norm, v=v, output_path=output_path
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
    com, val = compute_results(ens, tv_idx, val_set=(XVL, YVL), output_path=output_path)
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
        "-s",
        "--image_size",
        type=int,
        default=128,
        help="image pixel size (single value assigned to width and height)",
    )
    parser.add_argument(
        "-m", "--model_name", type=str, default="ensembleSVM", help="name to give model"
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
        "-y",
        "--early_stopping",
        type=str,
        default=None,
        choices=["val_accuracy", "val_loss"],
        help="early stopping",
    )
    parser.add_argument(
        "-v",
        "--validate",
        type=int,
        default=1,
        help="evaluate model with validation sample",
    )
    parser.add_argument(
        "-p", "--plots", type=int, default=0, help="draw model performance plots"
    )
    args = parser.parse_args()
    if args.validate == 1:
        v = 0.85
    else:
        v = 0
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
        img_size=args.image_size,
        norm=args.normalize,
        v=v,
        model_name=args.model_name,
        params=params,
        output_path=output_path,
    )
    if args.plots is True:
        com.draw_plots()
        val.draw_plots()
