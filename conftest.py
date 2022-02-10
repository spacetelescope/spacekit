import os
from pytest import fixture
import tarfile
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


# @fixture(scope='session')
# def svm_visit_data(tmp_path_factory):
#     basepath = tmp_path_factory.getbasetemp()
#     visit_data = os.path.join(basepath, "singlevisits")
#     return visit_data

# SVM PREP
@fixture(scope='session') # "ibl738.tgz"
def single_visit_path(tmp_path_factory):
    visit_path = os.path.abspath("tests/data/svm/prep/singlevisits.tgz")
    basepath = tmp_path_factory.getbasetemp()
    with tarfile.TarFile.open(visit_path) as tar:
        tar.extractall(basepath)
        dname = os.path.basename(visit_path.split(".")[0])
        visit_path = os.path.join(basepath, dname)
    return visit_path


@fixture(scope='function')
def img_outpath(tmp_path):
    return os.path.join(tmp_path, "img")


# SVM PREDICT
@fixture(scope='function')
def svm_unlabeled_dataset():
    return "tests/data/svm/predict/unlabeled.csv"


@fixture(scope='session', params=["img.tgz", "img_pred.npz"])
def svm_pred_img(request, tmp_path_factory):
    img_path = os.path.join("tests/data/svm/predict", request.param)
    if img_path.split(".")[-1] == "tgz":
        basepath = tmp_path_factory.getbasetemp()
        with tarfile.TarFile.open(img_path) as tar:
            tar.extractall(basepath)
            fname = os.path.basename(img_path.split(".")[0])
            img_path = os.path.join(basepath, fname)
    return img_path


# SVM TRAIN
@fixture(scope='function') # session
def svm_labeled_dataset():
    return "tests/data/svm/train/training.csv"
    

@fixture(scope='session', params=["img.tgz", "img_data.npz"])
def svm_train_img(request, tmp_path_factory):
    img_path = os.path.join("tests/data/svm/train", request.param)
    if img_path.split(".")[-1] == "tgz":
        basepath = tmp_path_factory.getbasetemp()
        with tarfile.TarFile.open(img_path) as tar:
            tar.extractall(basepath)
            fname = os.path.basename(img_path.split(".")[0])
            img_path = os.path.join(basepath, fname)
    return img_path


@fixture(scope='function')
def svm_train_npz():
    return "tests/data/svm/train/img_data.npz"


# GENERATOR: DRAW
@fixture(params=["single_reg.csv"])
def draw_mosaic_fname(request):
    return os.path.join("tests/data/svm/prep", request.param)


@fixture(params=["*", "ibl*", ""])
def draw_mosaic_pattern(request):
    return request.param


# PREPROCESSOR: SCRUB
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
