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

# SVM PREP
@fixture(scope='function')
def svm_visit_data():
    visit_data = "tests/data/svm/prep/singlevisits"
    return visit_data

# SVM PREDICT
@fixture(scope='function')
def svm_unlabeled_dataset():
    svm_unlabeled_data = "tests/data/svm/predict/unlabeled.csv"
    return svm_unlabeled_data

@fixture(params=["img", "img_pred.npz"])
def svm_pred_img(request):
    img_path = os.path.join("tests/data/svm/predict", request.param)
    return img_path

# SVM TRAIN
@fixture(scope='function') # session
def svm_labeled_dataset():
    svm_labeled_data = "tests/data/svm/train/training.csv"
    return svm_labeled_data

@fixture(params=["img", "img_data.npz"])
def svm_train_img(request):
    img_path = os.path.join("tests/data/svm/train", request.param)
    return img_path

@fixture(scope='function')
def svm_train_png():
    img_path = "tests/data/svm/train/img"
    return img_path

@fixture(scope='function')
def svm_train_npz():
    img_path = "tests/data/svm/train/img_data.npz"
    return img_path

@fixture(params=["single_reg.csv"])
def draw_mosaic_fname(request):
    fname = os.path.join("tests/data/svm/prep", request.param)
    return fname

@fixture(params=["ibl738"])
def draw_mosaic_visit(request):
    visit = os.path.join("tests/data/svm/prep/singlevisits", request.param)

@fixture(params=["*", "ibl*", ""])
def draw_mosaic_pattern(request):
    return request.param

@fixture(scope='function')
def raw_csv_file():
    return "tests/data/svm/prep/single_scrub.csv"

@fixture(scope='function')
def h5_data():
    return "tests/data/svm/prep/single_reg.h5"

@fixture(scope='function')
def scrubbed_cols_file():
    return "tests/data/svm/prep/scrubbed_cols.csv"

@fixture(scope='function')
def scraped_fits_file():
    return "tests/data/svm/prep/scraped_fits.csv"



# @fixture(scope='class', params=["svm", "cal"])
# def scanner(request):
#     if request.param == "svm":
#         scanner = SvmScanner(perimeter="data/20??-*-*-*", primary=-1)
#     elif request.param == "cal":
#         scanner = CalScanner(perimeter="data/20??-*-*-*", primary=-1)
#     return scanner

# @fixture(scope='class', params=["svm", "cal"])
# def scanner(request):
#     if request.param == "svm":
#         scanner = SvmScanner(perimeter="data/20??-*-*-*", primary=-1)
#     elif request.param == "cal":
#         scanner = CalScanner(perimeter="data/20??-*-*-*", primary=-1)
#     return scanner


# @fixture(scope='class', params=["svm", "cal"])
# def exp_scan(request):
#     return EXPECTED[request.param]
