import os
from pytest import mark, parametrize
from spacekit.skopes.hst.svm.predict import predict_alignment, load_mixed_inputs, classify_alignments

@mark.svm
@mark.predict
# @parametrize(["norm"], [(0), (1)])
def test_predict_alignment(svm_unlabeled_dataset, svm_pred_img):
    preds = predict_alignment(svm_unlabeled_dataset, svm_pred_img, output_path="tmp", size=128)
    pred_files = os.path.join("tmp", "predictions")
    expected = ["clf_report.txt", "compromised.txt", "predictions.csv"]
    actual = os.listdir(pred_files)
    for e in expected:
        assert e in actual

@mark.svm
@mark.predict
def test_load_predict(svm_unlabeled_dataset, svm_pred_img):
    X = load_mixed_inputs(svm_unlabeled_dataset, svm_pred_img, size=128)
    assert X[0].shape == (2, 10)
    assert X[1].shape ==  (2, 3, 128, 128, 3)

    preds = classify_alignments(X, model_path=None, output_path="tmp")
    assert len(preds) > 0
