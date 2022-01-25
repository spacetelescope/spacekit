# from spacekit.skopes.hst.svm.train import load_ensemble_data, train_ensemble, compute_results
# from spacekit.extractor.load import load_datasets, load_npz

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
