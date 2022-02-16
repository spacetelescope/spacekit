from sklearn.model_selection import StratifiedKFold
from spacekit.builder.architect import BuilderEnsemble
from spacekit.analyzer.compute import ComputeBinary
from spacekit.skopes.hst.svm.train import make_ensembles
from spacekit.generator.augment import training_data_aug, training_img_aug
from spacekit.preprocessor.transform import (
    normalize_training_images,
)
from spacekit.preprocessor.transform import PowerX
from spacekit.extractor.load import load_datasets
import pandas as pd
import numpy as np
import os

HOME = os.path.abspath(os.curdir)
DATA = os.path.join(HOME, "data")
SVMDIR = os.path.join(DATA, "2021-07-28")

# fname = os.path.join(SVMDIR, "svm_combined_v3.csv")
fname = os.path.join(SVMDIR, "svm_combined_v4.csv")
# paths to labeled image directories
# img_path = os.path.join(SVMDIR, 'img')
img_path_npz = os.path.join(DATA, "training_img.npz")
img_path_png = os.path.join(SVMDIR, "img")

DIM = 3
CH = 3
WIDTH = 128
HEIGHT = 128
DEPTH = DIM * CH
SHAPE = (DIM, WIDTH, HEIGHT, CH)
TF_CPP_MIN_LOG_LEVEL = 2

params = dict(
    batch_size=32,
    epochs=200,
    lr=1e-4,
    decay=[100000, 0.96],
    early_stopping=None,
    verbose=2,
    ensemble=True,
)

import datetime as dt

timestring = dt.datetime.now().isoformat()[:-7]  # 2022-01-19T21:31:21
timestamp = int(dt.datetime.fromisoformat(timestring).timestamp())
dtstring = timestring.split("T")[0] + "-" + str(timestamp)
output_path = f"{DATA}/{dtstring}"  # 2022-02-14-1644850390

os.makedirs(output_path, exist_ok=True)

img_size = 128
dim = 3
ch = 3
depth = dim * ch

cols = ["numexp", "rms_ra", "rms_dec", "nmatches", "point", "segment", "gaia"]
ncols = list(range(len(cols)))
model_name = "svm_ensemble"

df = load_datasets([fname])

from spacekit.extractor.load import SVMFileIO

((X_train, X_test, X_val), (y_train, y_test, y_val)), (train, test, val) = SVMFileIO(
    img_path_npz, w=img_size, h=img_size, d=dim * ch, inference=False, data=df, v=0.85
).load()

X_train = pd.concat([X_train, X_val], axis=0)
y_train = pd.concat([y_train, y_val], axis=0)

image_index = np.concatenate([train[0], test[0], val[0]], axis=0)
X_img = np.concatenate([train[1], test[1], val[1]], axis=0)
y_img = np.concatenate([train[2], test[2], val[2]], axis=0)

Px = PowerX(df, cols=cols, save_tx=False, output_path=output_path)

# Define the K-fold Cross Validator
num_folds = 5
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)

# sep targets and inputs
y = df["label"]
X = df.drop("label", axis=1, inplace=False)

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
roc_per_fold = []
fn_per_fold = []
fp_per_fold = []

df_aug = pd.DataFrame()
# K-fold Cross Validation model evaluation
fold_no = 1
for train_idx, test_idx in kfold.split(X, y):
    nfold = f"split_{fold_no}"
    df[nfold] = "train"
    test_idx_str = df.iloc[test_idx].index
    df.loc[test_idx_str, nfold] = "test"
    # mlp inputs
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # image inputs
    img_idx_train, img_idx_test = image_index[train_idx], image_index[test_idx]
    train_lab, test_lab = y_img[train_idx], y_img[test_idx]
    train_img, test_img = X_img[train_idx], X_img[test_idx]
    train = (img_idx_train, train_img, train_lab)
    test = (img_idx_test, test_img, test_lab)
    # augmentation
    X_train, _ = training_data_aug(X_train, y_train)
    img_idx, (X_tr, y_tr), (X_ts, y_ts), _ = training_img_aug(train, test, val=None)
    # normalization
    X_train = PowerX(X_train, cols=ncols, tx_data=Px.tx_data).Xt
    X_test = PowerX(X_test, cols=cols, tx_data=Px.tx_data).Xt
    X_tr, X_ts = normalize_training_images(X_tr, X_ts, X_vl=None)
    # merge inputs
    XTR, YTR, XTS, YTS = make_ensembles(X_train, X_tr, y_tr, X_test, X_ts, y_ts)
    tv_idx = [y_test, img_idx]
    split_out = f"{output_path}/{nfold}"
    os.makedirs(split_out, exist_ok=True)
    # build and compile model
    ens = BuilderEnsemble(
        XTR,
        YTR,
        XTS,
        YTS,
        params=params,
        input_name=f"svm_{nfold}",
        output_name="svm_output",
        name=model_name,
    )
    ens.build()
    # fit to the training data
    print(f"\nTraining for fold {fold_no} ...")
    ens.batch_fit()
    # evaluate
    ens.test_idx = tv_idx[0]
    com = ComputeBinary(builder=ens, res_path=f"{split_out}/results/test")
    _ = com.calculate_results()
    _ = com.make_outputs()
    # com, _ = compute_results(ens, tv_idx, val_set=(), output_path=split_out)
    acc, loss = com.acc_loss["test_acc"], com.acc_loss["test_loss"]
    roc = com.roc_auc
    fn, fp = len(com.fnfp["fn_idx"]), len(com.fnfp["fp_idx"])
    print(f"\nScore for fold {fold_no}:")
    print(f"\nLOSS={loss}")
    print(f"\nACC={acc*100}%")
    print(f"\nROC={roc}%")
    print(f"\nFN={fn}")
    print(f"\nFP={fp}")
    print("\n-------" * 9)
    print("\n")

    acc_per_fold.append(acc)
    loss_per_fold.append(loss)
    roc_per_fold.append(roc)
    fn_per_fold.append(fn)
    fp_per_fold.append(fp)

    mid = len(train_idx)
    xtr = [XTR[0][:mid], XTR[1][:mid]]
    aug = [XTR[0][mid:], XTR[1][mid:]]
    ptrain = ens.model.predict(xtr)
    ptest = ens.model.predict(XTS)
    train_idx_str = df.iloc[train_idx].index
    df.loc[train_idx_str, f"proba_{fold_no}"] = ptrain
    df.loc[test_idx_str, f"proba_{fold_no}"] = ptest

    p_aug = ens.model.predict(aug)
    aug_pred = pd.DataFrame(p_aug, index=list(range(len(p_aug))), columns={nfold})
    if len(df_aug) == 0:
        df_aug = aug_pred
    else:
        df_aug = df_aug.join(aug_pred, how="left")

    fold_no += 1

T = len(y_test)
# == Provide average scores ==
print("--------" * 11)
print("Score per fold")
for i in range(0, len(acc_per_fold)):
    print("--------" * 11)
    print(
        f"> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% - ROC: {np.round(roc_per_fold[i], 4)} - FP: {np.round((fp_per_fold[i]/T)*100, 4)} - FN: {np.round((fn_per_fold[i]/T)*100, 4)}"
    )
print("--------" * 11)
print("Averages:")
print(f"> Accuracy: {np.mean(acc_per_fold)} (+/- {np.round(np.std(acc_per_fold), 4)})")
print(f"> Loss: {np.mean(loss_per_fold)}")
print(f"> AUC: {np.round(np.mean(roc_per_fold), 4)}")
print("--------" * 11)

df["index"] = df.index
df.to_csv(f"{output_path}/data/training_splits.csv", index=False)
df_aug.to_csv(f"{output_path}/data/augmented_splits.csv")
