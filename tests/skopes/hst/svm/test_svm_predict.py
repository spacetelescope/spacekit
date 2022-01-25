# import os
# from spacekit.skopes.hst.svm.predict import predict_alignment

# def test_predict(svm_data, img_path):
#     predict_alignment(svm_data, img_path)
#     preds = os.path.join(os.getcwd(), "predictions")
#     expected = ["clf_report.txt", "compromised.txt", "predictions.csv"]
#     actual = os.listdir(preds)
#     for e in expected:
#         assert e in actual

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
