import os
from pytest import fixture
# try:
#     from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
#                                                TESTED_VERSIONS)
# except ImportError:
#     PYTEST_HEADER_MODULES = {}
#     TESTED_VERSIONS = {}
PYTEST_HEADER_MODULES = {}
TESTED_VERSIONS = {}

try:
    from spacekit import __version__ as version
except ImportError:
    version = 'unknown'

# The following line treats all DeprecationWarnings as exceptions.
from astropy.tests.helper import enable_deprecations_as_exceptions
enable_deprecations_as_exceptions()

# Uncomment and customize the following lines to add/remove entries
# from the list of packages for which version numbers are displayed
# when running the tests.
# PYTEST_HEADER_MODULES['astropy'] = 'astropy'
# PYTEST_HEADER_MODULES.pop('Matplotlib')
# PYTEST_HEADER_MODULES.pop('Pandas')
# PYTEST_HEADER_MODULES.pop('h5py')

TESTED_VERSIONS['spacekit'] = version


@fixture(scope='function') # session
def svm_labeled_dataset():
    svm_labeled_data = "tests/data/svm/train/dataset.csv"
    return svm_labeled_data


@fixture(scope='function')
def svm_unlabeled_dataset():
    svm_unlabeled_data = "tests/data/svm/predict/unlabeled.csv"
    return svm_unlabeled_data


@fixture(params=["img", "images.npz"])
def svm_train_img(request):
    img_path = os.path.join("tests/data/svm/train", request.param)
    return img_path

@fixture(scope='function')
def svm_train_png():
    img_path = "tests/data/svm/train/img"
    return img_path

@fixture(scope='function')
def svm_train_npz():
    img_path = "tests/data/svm/train/images.npz"
    return img_path


@fixture(params=["img"])
def svm_pred_img(request):
    img_path = os.path.join("tests/data/svm/predict", request.param)
    return img_path


@fixture(scope='function')
def svm_visit_data():
    visit_data = "tests/data/svm/prep/singlevisits"
    return visit_data