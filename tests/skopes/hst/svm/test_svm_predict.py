import os
from pytest import mark
from conftest import check_skope
from spacekit.builder.architect import BuilderEnsemble
from spacekit.skopes.hst.svm.predict import (
    predict_alignment,
    load_mixed_inputs,
    classify_alignments,
)


@mark.hst
@mark.svm
@mark.predict
def test_predict_alignment(skope, unlabeled_dataset, svm_pred_img, tmp_path):
    check_skope(skope, "svm")
    preds = predict_alignment(
        unlabeled_dataset, svm_pred_img, output_path=tmp_path, size=128
    )
    assert len(preds) > 0
    pred_files = os.path.join(tmp_path, "predictions")
    expected = ["clf_report.txt", "compromised.txt", "predictions.csv"]
    actual = os.listdir(pred_files)
    for e in expected:
        assert e in actual


@mark.hst
@mark.svm
@mark.predict
def test_load_predict(skope, unlabeled_dataset, svm_pred_img, tmp_path):
    check_skope(skope, "svm")
    ens = BuilderEnsemble()
    ens.load_saved_model(arch="svm_align")
    ens.find_tx_file()
    X = load_mixed_inputs(
        unlabeled_dataset, svm_pred_img, size=128, tx=ens.tx_file, norm=0
    )
    assert X[0].shape == (2, 10)
    assert X[1].shape == (2, 3, 128, 128, 3)

    preds = classify_alignments(X, ens.model, output_path=tmp_path)
    assert len(preds) > 0
