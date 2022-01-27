import os
from pytest import mark
from spacekit.builder.architect import Builder
from spacekit.skopes.hst.svm.train import load_ensemble_data

params = dict(
    batch_size=6,
    epochs=1,
    lr=1e-4,
    decay=[100000, 0.96],
    early_stopping=None,
    verbose=2,
    ensemble=True,
)
@mark.skip
def test_ensemble_builder(svm_labeled_dataset, svm_train_npz):
    tv_idx, XTR, YTR, XTS, YTS, XVL, YVL = load_ensemble_data(
        svm_labeled_dataset, svm_train_npz, img_size=128, norm=0, v=0.85, output_path="tmp"
    )
    params = dict(
        batch_size=14,
        epochs=1,
        lr=1e-4,
        decay=[100000, 0.96],
        early_stopping=None,
        verbose=2,
        ensemble=True,
    )
    ens = Builder.BuilderEnsemble(
        XTR,
        YTR,
        XTS,
        YTS,
        params=params,
        input_name="svm_mixed_inputs",
        output_name="svm_output",
        name="ens_build_test",
    )
    assert ens.steps_per_epoch > 0
    ens.build()
