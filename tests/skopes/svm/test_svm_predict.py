import os
import pytest
from pytest import mark
import numpy as np
from spacekit.skopes.hst.svm.predict import get_model

test_img = np.array([])
test_label = []

@mark.svm
@mark.predict
def test_get_model():
    ens = get_model()
    assert len(ens.layers) == 28
    


# class SvmPredictTests:
#     def test_svm_predict(self):
#         data_file = None #data_file
#         img_path = None #img_path
#         model_path = None #model_path
#         output_path = None #output_path
#         main(model_path, data_file, img_path, output_path)
#         assert True