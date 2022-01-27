from pytest import mark
from spacekit.skopes.hst.svm.train import load_ensemble_data, run_training

params = dict(
    batch_size=7,
    epochs=1,
    lr=1e-4,
    decay=[100000, 0.96],
    early_stopping=None,
    verbose=2,
    ensemble=True,
)

@mark.skip
@mark.svm
@mark.train
@mark.parametrize("norm", [(0), (1)])
def test_svm_training(svm_labeled_dataset, svm_train_npz, norm):
    ens, com, val = run_training(
        svm_labeled_dataset, 
        svm_train_npz,
        img_size=128,
        norm=norm,
        val_size=0.2,
        model_name="ensembleSVM",
        params=params,
        output_path="tmp"
        )
    # model arch
    assert str(type(ens.mlp)) == "<class 'spacekit.builder.architect.BuilderMLP'>"
    assert str(type(ens.cnn)) == "<class 'spacekit.builder.architect.BuilderCNN3D'>"
    assert str(type(ens)) == "<class 'spacekit.builder.architect.BuilderEnsemble'>"


@mark.svm
@mark.train
def test_load_training_data(svm_labeled_dataset, svm_train_img, norm):
    tv_idx, XTR, YTR, XTS, YTS, XVL, YVL = load_ensemble_data(
        svm_labeled_dataset, svm_train_img, img_size=128, norm=norm, val_size=0.2, output_path="tmp"
    )
    assert len(tv_idx) == 3
    # check input features
    assert XTR[0].shape == (14, 10)
    assert XTR[1].shape == (12, 3, 128, 128, 3)
    assert XTS[0].shape == (3, 10)
    assert XTS[1].shape == (3, 3, 128, 128, 3)
    assert XVL[0].shape == (2, 10)
    assert XVL[1].shape == (2, 3, 128, 128, 3)
    # check target labels
    assert YTR.shape == (12, 1)
    assert YTS.shape == (3, 1)
    assert YVL.shape == (2, 1)
