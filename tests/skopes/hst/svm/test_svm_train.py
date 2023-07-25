import os
from pytest import mark
from spacekit.skopes.hst.svm.train import load_ensemble_data, run_training

EXPECTED_RES = [
    "cmx",
    "test_idx",
    "y_scores",
    "fnfp",
    "acc_loss",
    "roc_auc",
    "history",
    "cmx_norm",
    "report",
    "y_pred",
    "y_onehot",
]

PARAMS = dict(
    batch_size=6,
    epochs=1,
    lr=1e-4,
    decay=[100000, 0.96],
    early_stopping=None,
    verbose=2,
    ensemble=True,
)


@mark.svm
@mark.train
@mark.parametrize("norm", [(1), (0)])
def test_svm_training(labeled_dataset, svm_train_npz, norm):
    output_path = os.path.join("tmp", "2021-11-04-1636048291")
    ens, com, _ = run_training(
        labeled_dataset,
        svm_train_npz,
        img_size=128,
        norm=norm,
        v=0.85,
        model_name="ensembleSVM",
        params=PARAMS,
        output_path=output_path,
    )
    # model arch
    assert str(type(ens.mlp)) == "<class 'spacekit.builder.architect.BuilderMLP'>"
    assert str(type(ens.cnn)) == "<class 'spacekit.builder.architect.BuilderCNN3D'>"
    assert str(type(ens)) == "<class 'spacekit.builder.architect.BuilderEnsemble'>"
    # results
    assert com.y_onehot.values.shape == (3, 2)
    assert com.roc_auc is not None
    assert list(com.acc_loss.keys()) == [
        "train_acc",
        "train_loss",
        "test_acc",
        "test_loss",
    ]
    assert com.cmx.shape == (2, 2)
    assert com.cmx_norm.shape == (2, 2)
    assert list(com.fnfp.keys()) == [
        "pred_proba",
        "conf_idx",
        "conf_proba",
        "fn_idx",
        "fp_idx",
    ]
    res_actual = os.listdir(com.res_path)
    assert [r in res_actual for r in EXPECTED_RES]


@mark.svm
@mark.train
def test_load_training_data(labeled_dataset, svm_train_img):
    tv_idx, XTR, YTR, XTS, YTS, XVL, YVL = load_ensemble_data(
        labeled_dataset,
        svm_train_img,
        img_size=128,
        norm=0,
        v=0.85,
        output_path="tmp",
    )
    assert len(tv_idx) == 3
    # check input features
    assert XTR[0].shape == (14, 10)
    assert XTR[1].shape == (14, 3, 128, 128, 3)
    assert XTS[0].shape == (3, 10)
    assert XTS[1].shape == (3, 3, 128, 128, 3)
    assert XVL[0].shape == (2, 10)
    assert XVL[1].shape == (2, 3, 128, 128, 3)
    # check target labels
    assert YTR.shape == (14, 1)
    assert YTS.shape == (3, 1)
    assert YVL.shape == (2, 1)
