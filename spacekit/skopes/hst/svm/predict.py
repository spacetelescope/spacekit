"""
This module generates predictions using a pre-trained ensemble neural network for unlabeled SVM regression test data and alignment images. The ensemble model is a combination of two neural networks: a MultiLayerPerceptron (for regression test data) and a 3D Image Convolutional Neural Network (CNN). The script includes functions for the following steps:

1. load and prep the data and images for ML
2. load the saved model and generate predictions
3. save predictions and summary statistics to disk

This script (and/or its functions) should be used in conjunction with spacekit.skopes.hst.svm.prep if using raw data (since both the regression test dataframe for MLP and the png images for the CNN need to be created first). Once a model has been trained using the spacekit.skopes.hst.svm.train script, it is saved to disk and can be loaded for use here to generate predictions on unlabeled data.
"""
# from zipfile import ZipFile
# import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import os
import sys
import datetime as dt
from spacekit.extractor.load import load_datasets, SVMFileIO
from spacekit.builder.architect import Builder

# from spacekit.builder.blueprints import Blueprint

DIM = 3
CH = 3
SIZE = 128
DEPTH = DIM * CH
SHAPE = (DIM, SIZE, SIZE, CH)

TF_CPP_MIN_LOG_LEVEL = 2


def load_mixed_inputs(data_file, img_path, size=128):
    """Load the regression test data and image input data, then stacks the arrays into a single combined input (list) for the ensemble model.

    Parameters
    ----------
    data_file : str
        path to preprocessed mosaic data csv file
    img_path : str
        path to png images parent directory
    size : int, optional
        image size (width and height), by default 128

    Returns
    -------
    list
        regression test data (MLP inputs) and image data (CNN inputs) joined as a list
    """
    cols = [
        "numexp",
        "rms_ra",
        "rms_dec",
        "nmatches",
        "point",
        "segment",
        "gaia",
        "det",
        "wcs",
        "cat",
    ]
    X_data = load_datasets([data_file], column_order=cols)
    # idx, X_img = load_image_data(X_data, img_path, size=size)
    print("Loading images into arrays...")
    idx, X_img = SVMFileIO(img_path, w=size, h=size, d=9, data=X_data).load()
    diff = X_data.shape[0] - X_img.shape[0]
    if diff > 0:
        X_data = X_data.loc[X_data.index.isin(idx)]
        print(f"{diff} missing images removed from index")
        print(f"X_data: {X_data.shape}\nX_img:  {X_img.shape}")
    print("Joining regression data and image arrays")
    return [X_data, X_img]


def classification_report(df, output_path):
    """Generates a scikit learn classification report with model evaluation metrics and saves to disk.

    Parameters
    ----------
    df : Pandas dataframe
        Feature inputs for which the model will generate predictions.
    output_path : str
        location to store prediction output files
    """
    P, T = df["y_pred"], df["det"].value_counts()
    C = df.loc[P == 1.0]
    cmp = C["det"].value_counts()
    dets = ["HRC", "IR", "SBC", "UVIS", "WFC"]
    separator = "---" * 5
    out = sys.stdout
    with open(f"{output_path}/clf_report.txt", "w") as f:
        sys.stdout = f
        print("CLASSIFICATION REPORT - ", dt.datetime.now())
        print(separator)
        print("Mean Probability Score: ", df["y_proba"].mean())
        print("Standard Deviation: ", df["y_proba"].std())
        print(separator)
        print("Aligned ('0.0') vs Misaligned ('1.0')")
        cnt_pct = pd.concat(
            [P.value_counts(), P.value_counts(normalize=True)],
            axis=1,
            keys=["cnt", "pct"],
        )
        print(cnt_pct)
        print(separator)
        print("Misalignment counts by Detector")
        for i, d in enumerate(dets):
            if i in cmp:
                print(f"{d}\t{cmp[i]} \t ({T[i]}) \t {np.round((cmp[i]/T[i])*100, 1)}%")
            elif i in T:
                print(f"{d}\t0 \t ({T[i]}) \t 0%")
            else:
                print(f"{d}\t0 \t (0) \t 0%")
        sys.stdout = out
    print(f"\nClassification Report created: {output_path}/clf_report.txt")
    with open(f"{output_path}/compromised.txt", "w") as file:
        for line in list(C["y_pred"].index):
            file.writelines(f"{line}\n")
    print(f"\nSuspicious/Compromised List created: {output_path}/compromised.txt")


def classify_alignments(X, model_path=None, output_path=None):
    """Returns classifier predictions and probability scores

    Parameters
    ----------
    X : numpy array
        input features
    model_path : str, optional
        saved model directory path, by default None
    output_path : str
        location to store prediction output files

    Returns
    -------
    Pandas dataframe
        prediction values, probability scores for target, merged with original input features
    """
    if output_path is None:
        output_path = os.getcwd()
    output_path = os.path.join(output_path, "predictions")
    os.makedirs(output_path, exist_ok=True)
    model = Builder(model_path=model_path).load_saved_model()
    y_proba = model.predict(X)
    y_pred = np.round(y_proba[:, 0]).reshape(-1, 1)
    # y_proba = proba[:, 0].reshape(-1, 1)
    preds = np.concatenate([y_pred, y_proba], axis=1)
    pred_proba = pd.DataFrame(preds, index=X[0].index, columns=["y_pred", "y_proba"])
    preds = X[0].join(pred_proba)
    preds["index"] = preds.index
    output_file = f"{output_path}/predictions.csv"
    preds.to_csv(output_file, index=False)
    print("Y_PRED + Probabilities added. Dataframe saved to: ", output_file)
    classification_report(preds, output_path)
    return preds


def predict_alignment(data_file, img_path, model_path=None, output_path=None, size=128):
    """Main calling function to load the data and model, generate predictions, and save results to disk.

    Parameters
    ----------
    data_file : str
        path to preprocessed mosaic data csv file
    img_path : str
        path to png images parent directory
    model_path : str, optional
        saved model directory path, by default None
    output_path : str, optional
        location to store prediction output files, by default None
    size : int, optional
        image size (width and height), by default None (128)
    """
    X = load_mixed_inputs(data_file, img_path, size=size)
    preds = classify_alignments(X, model_path=model_path, output_path=output_path)
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spacekit",
        usage="spacekit.skopes.hst.svm.predict svm_data.csv img",
    )
    parser.add_argument(
        "data_file",
        type=str,
        default="svm_data.csv",
        help="path to preprocessed mosaic data csv file",
    )
    parser.add_argument(
        "img_path", type=str, help="path to png images parent directory"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=None,
        help="saved model path",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="location to store prediction output files",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=128,
        help="image size (width and height). Default is None (128).",
    )
    args = parser.parse_args()
    data_file = args.data_file
    img_path = args.img_path
    model_path = args.model_path
    output_path = args.output_path
    size = args.size
    predict_alignment(
        data_file, img_path, model_path=model_path, output_path=output_path, size=size
    )
