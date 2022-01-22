import os
from pytest import mark
import numpy as np
from spacekit.skopes.hst.svm.predict import get_model, load_regression_data, load_image_data

test_img = np.array([])
test_label = []


@mark.svm
class SvmTests:

    @mark.predict
    def test_get_model(self):
        ens = get_model()
        assert len(ens.layers) == 28
    
    def test_import_data(self, svm_dataset):
        X_data = load_regression_data(data_file=svm_dataset)
        assert X_data.shape == (12, 10)

#     def test_svm_predict(self):
#         data_file = None #data_file
#         img_path = None #img_path
#         model_path = None #model_path
#         output_path = None #output_path
#         main(model_path, data_file, img_path, output_path)
#         assert True