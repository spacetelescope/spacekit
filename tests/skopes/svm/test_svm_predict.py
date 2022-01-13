import os
import pytest
from pytest import mark

# from spacekit.skopes.hst.svm.predict import get_model
import importlib.resources
from zipfile import ZipFile

@mark.svm
@mark.predict
def test_get_model():
    # model = get_model()
    # print(vars(model))
    # if model is not None:
    #     assert True
    model_path = None
    with importlib.resources.path(
        "spacekit.skopes.trained_networks", "ensembleSVM.zip"
    ) as M:
        model_path = M
    os.makedirs("models", exist_ok=True)
    model_base = os.path.basename(model_path).split(".")[0]
    with ZipFile(model_path, "r") as zip_ref:
        zip_ref.extractall("models")
    model_path = os.path.join("models", model_base)
    print("Loading saved model: ", model_path)
    if model_path is not None:
        assert True


# class SvmPredictTests:
#     def test_svm_predict(self):
#         data_file = None #data_file
#         img_path = None #img_path
#         model_path = None #model_path
#         output_path = None #output_path
#         main(model_path, data_file, img_path, output_path)
#         assert True