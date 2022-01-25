from pytest import mark
import os
from spacekit.skopes.hst.svm.predict import get_model, load_regression_data, load_image_data, classify_alignments, predict_alignment
from spacekit.skopes.hst.svm.train import load_ensemble_data, train_ensemble, compute_results
from spacekit.extractor.load import load_datasets, load_npz

from spacekit.skopes.hst.svm.prep import run_preprocessing


def test_prep():
    fname = run_preprocessing("singlevisits", fname="test_prep")
    assert os.path.exists(fname)

    img_dir = os.path.join(os.path.dirname(fname), "img")
    visit = os.listdir(img_dir)
    assert len(visit) == 1

    images = os.listdir(os.path.join(img_dir, visit))
    assert len(images) == 3

def test_predict(svm_data, img_path):
    predict_alignment(svm_data, img_path)
    preds = os.path.join(os.getcwd(), "predictions")
    expected = ["clf_report.txt", "compromised.txt", "predictions.csv"]
    actual = os.listdir(preds)
    for e in expected:
        assert e in actual




# @mark.svm
# @mark.predict
# class SvmPredictTests:
    
#     def test_get_model(self):
#         self.ens = get_model()
#         assert len(self.ens.layers) == 28
    
#     def test_import_data(self, svm_data):
#         self.X_data = load_regression_data(data_file=svm_data)
#         assert self.df.shape == (12, 10)

#     def test_load_from_png(self, img_path):
#         self.idx, self.X_img = load_image_data(self.X_data, img_path, size=None)
#         assert True
    
#     def test_svm_classifier(self):
#         self.X = [self.X_data, self.X_img]
#         self.y_pred, self.y_proba = classify_alignments(self.ens, self.X)

    # def test_predict_alignment(self, svm_data, img_path):
    #     predict_alignment(svm_data, img_path)

    # ens_clf = get_model(model_path=model_path)
    # X_data, X_img = load_mixed_inputs(data_file, img_path, size=size)
    # X = make_ensemble_data(X_data, X_img)
    # y_pred, y_proba = classify_alignments(ens_clf, X)
    # if output_path is None:
    #     output_path = os.getcwd()
    # output_path = os.path.join(output_path, "predictions")
    # os.makedirs(output_path, exist_ok=True)
    # preds = save_preds(X_data, y_pred, y_proba, output_path)
    # classification_report(preds, output_path)
    # load_mixed_inputs
    # make_ensemble_data
    # classify_alignments
    # save_preds
    # classification_report
    # predict_alignment

# @mark.skip
# @mark.svm
# @mark.train
# class SvmTrainTests:

#     def __init__(self):
#         self.data = None
#         self.ens = None
#         self.com = None
#         self.val = None
    
#     def test_load_dataframe(self, svm_dataset):
#         self.X_data = load_datasets([svm_dataset])
    
#     def test_load_npz_images(self, img_file):
#         self.X_img = load_npz(npz_file=img_file)
#         assert len(self.X_img) == 3

#     def test_load_ensemble_training(self, svm_dataset, img_file):
#         tv_idx, XTR, YTR, XTS, YTS, XVL, YVL = load_ensemble_data(
#             svm_dataset, img_file, img_size=128, dim=3, ch=3, norm=False, output_path=None
#         )
#         self.data = tv_idx, XTR, YTR, XTS, YTS, XVL, YVL
#         assert True
    
#     def test_train_ensemble_model(self):
#         self.ens = train_ensemble(
#             self.data[1], self.data[2], self.data[3], self.data[4], model_name="ensembleSVM", params=None, output_path=None
#             )
#         assert True
    
#     def test_compute_ensemble_model(self):
#         self.com, self.val = compute_results(self.ens, self.data[0], val_set=(self.data[-2], self.data[-1]), output_path=None)
#         assert True
