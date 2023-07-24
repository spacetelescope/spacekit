import os
import pytest
from pytest import fixture
from zipfile import ZipFile
import pandas as pd
from spacekit.analyzer.explore import HstCalPlots, HstSvmPlots
from spacekit.analyzer.scan import HstSvmScanner, HstCalScanner, import_dataset
from spacekit.extractor.load import load_datasets, extract_file


TESTED_VERSIONS = {}

try:
    from spacekit import __version__ as version
except ImportError:
    version = "unknown"


TESTED_VERSIONS["spacekit"] = version
# Uncomment and customize the following lines to add/remove entries
# from the list of packages for which version numbers are displayed
# when running the tests.
# PYTEST_HEADER_MODULES = {}
# PYTEST_HEADER_MODULES['astropy'] = 'astropy'
# PYTEST_HEADER_MODULES.pop('Matplotlib')
# PYTEST_HEADER_MODULES.pop('Pandas')
# PYTEST_HEADER_MODULES.pop('h5py')


pytest_plugins = [
    'tests.data_plugin'
]


class Config:
    def __init__(self, env):
        SUPPORTED_ENVS = ["svm", "cal"]
        self.env = env

        if env.lower() not in SUPPORTED_ENVS:
            raise Exception(
                f"{env} is not a supported environment (supported envs: {SUPPORTED_ENVS})"
            )

        self.data_path = {
            "svm": os.path.join(f"tests/data/{env}/data.zip"),
            "cal": os.path.join(f"tests/data/{env}/data.zip"),
        }[env]

        self.kwargs = {"svm": dict(index_col="index"), "cal": dict(index_col="ipst")}[
            env
        ]

        self.decoder = {
            "svm": {"det": {0: "hrc", 1: "ir", 2: "sbc", 3: "uvis", 4: "wfc"}},
            "cal": {"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}},
        }[env]

        self.labeled = {
            "svm": "tests/data/svm/train/training.csv",
            "cal": "tests/data/cal/train/training.csv",
        }[env]

        self.unlabeled = {
            "svm": "tests/data/svm/predict/unlabeled.csv",
            "cal": "tests/data/cal/predict/unlabeled.csv",
        }[env]

        self.norm_cols = {
            "svm": [
                "numexp",
                "rms_ra",
                "rms_dec",
                "nmatches",
                "point",
                "segment",
                "gaia",
            ],
            "cal": ["n_files", "total_mb"],
        }[env]
        self.rename_cols = {"svm": "_scl", "cal": ["x_files", "x_size"]}[env]

        self.enc_cols = {
            "svm": ["det", "wcs", "cat"],
            "cal": [
                "drizcorr",
                "pctecorr",
                "crsplit",
                "subarray",
                "detector",
                "dtype",
                "instr",
            ],
        }[env]

        self.tx_file = {
            "svm": "tests/data/svm/tx_data.json",
            "cal": "tests/data/cal/tx_data.json",
        }[env]

        self.visits = {
            "svm": [],
            "cal": {
                "asn": ["j8zs05020", "ic0k06010", "la8mffg5q", "oc3p011i0"],
                "svm": []
            }
        }[env]


def pytest_addoption(parser):
    # parser.addoption("--env", action="store", default="cal", help="Environment to run tests against")
    parser.addoption("--env", action="store", default=None, help="Environment to run tests against")

# def pytest_configure(config):
#     config.addinivalue_line("markers", "skopes: check env against param")

# def pytest_collection_modifyitems(config, items):
#     env_param = config.getoption("--env")
#     if env_param:
#         skope_param = pytest.mark.parametrize("skope", [(env_param)], indirect=True)
#     else:
#         skope_param = pytest.mark.parametrize("skope", [("cal", "svm")], indirect=True)
#         # skip_param = pytest.mark.skipif(reason="skip params based on --env")
#     for item in items:
#         if "skopes" in item.keywords:
#             item.add_marker(skope_param)


@fixture(scope="session")
def env(request):
    return request.config.getoption("--env")

# @fixture(scope="session")
# def cfg(env):
#     cfg = Config(env)
#     return cfg

@fixture(scope="session", params=["cal", "svm"])
def skope(request):
    env_param = request.config.getoption("--env")
    if env_param is not None and request.param != env_param:
        pytest.skip(reason="skipping param based on --env")
    else:
        return Config(request.param)

@fixture(scope="session")
def test_data(request):
    return request.config.getoption("data_path")


@fixture(scope="session")
def res_data_path(test_data, skope):
    skope_data = os.path.join(test_data, skope.env, "data")
    if not os.path.exists(skope_data):
        with ZipFile(skope.data_path, "r") as z:
            z.extractall(os.path.dirname(skope_data))
    return skope_data 


@fixture(scope="session")
def df_ncols(skope):
    fname = skope.labeled
    X_cols = skope.norm_cols + skope.enc_cols
    df = load_datasets([fname], index_col=skope.kwargs["index_col"], column_order=X_cols)
    ncols = [i for i, c in enumerate(df.columns) if c in skope.norm_cols]
    return (df, ncols)


@fixture(scope="session")
def scanner(skope, res_data_path):
    if skope.env == "svm":
        scanner = HstSvmScanner(perimeter=f"{res_data_path}/20??-*-*-*", primary=-1)
    elif skope.env == "cal":
        scanner = HstCalScanner(perimeter=f"{res_data_path}/20??-*-*-*", primary=-1)
    scanner.exp = skope.env
    return scanner


@fixture(scope="session")
def explorer(skope, res_data_path):
    fname = res_data_path
    df = import_dataset(filename=fname, kwargs=skope.kwargs, decoder=skope.decoder)
    if skope.env == "svm":
        hst = HstSvmPlots(df)
    elif skope.env == "cal":
        hst = HstCalPlots(df)
    hst.env = skope.env
    return hst


# SVM PREP
@fixture(scope="session")  # "ibl738.tgz"
def single_visit_path(tmp_path_factory):
    visit_path = os.path.abspath("tests/data/svm/prep/singlevisits.tgz")
    basepath = tmp_path_factory.getbasetemp()
    extract_file(visit_path, dest=basepath)
    dname = os.path.basename(visit_path.split(".")[0])
    visit_path = os.path.join(basepath, dname)
    return visit_path


@fixture(scope="function")
def img_outpath(tmp_path):
    return os.path.join(tmp_path, "img")


# SVM PREDICT
@fixture(scope="function")
def svm_unlabeled_dataset():
    return "tests/data/svm/predict/unlabeled.csv"


@fixture(scope="session", params=["img.tgz", "img_pred.npz"])
def svm_pred_img(request, tmp_path_factory):
    img_path = os.path.join("tests/data/svm/predict", request.param)
    if img_path.split(".")[-1] == "tgz":
        basepath = tmp_path_factory.getbasetemp()
        extract_file(img_path, dest=basepath)
        fname = os.path.basename(img_path.split(".")[0])
        img_path = os.path.join(basepath, fname)
    return img_path


# SVM TRAIN
@fixture(scope="function")  # session
def svm_labeled_dataset():
    return "tests/data/svm/train/training.csv"


@fixture(scope="session", params=["img.tgz", "img_data.npz"])
def svm_train_img(request, tmp_path_factory):
    img_path = os.path.join("tests/data/svm/train", request.param)
    fname = os.path.basename(img_path.split(".")[0])
    if img_path.split(".")[-1] == "tgz":
        basepath = tmp_path_factory.getbasetemp()
        extract_file(img_path, dest=basepath)
        img_path = os.path.join(basepath, fname)
    return img_path


@fixture(scope="function")
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
@fixture(scope="module")
def raw_csv_file():
    return "tests/data/svm/prep/single_scrub.csv"


@fixture(scope="module")
def raw_svm_data(raw_csv_file):
    data = pd.read_csv(raw_csv_file, index_col="index")
    return data


@fixture(scope="function")
def h5_data():
    return "tests/data/svm/prep/single_reg"


@fixture(scope="function")
def scrubbed_svm_file():
    return "tests/data/svm/prep/scrubbed_cols.csv"


@fixture(scope="function")
def scrubbed_svm_data(scrubbed_svm_file):
    data = pd.read_csv(scrubbed_svm_file, index_col="index")
    return data


@fixture(scope="function")
def scraped_fits_file():
    return "tests/data/svm/prep/scraped_fits.csv"


@fixture(scope="function")
def scraped_fits_data(scraped_fits_file):
    data = pd.read_csv(scraped_fits_file, index_col="index")
    return data


@fixture(scope="function")
def scraped_mast_file():
    return "tests/data/svm/prep/scraped_mast.csv"


# CAL
@fixture(scope="function")
def cal_labeled_dataset():
    return "tests/data/cal/train/training.csv"

# @fixture(scope="function")
# def training_data_file(skope):
#     return skope.labeled

@fixture(scope="function")
def cal_predict_visits():
    return {
            "asn": ["j8zs05020", "ic0k06010", "la8mffg5q", "oc3p011i0"]
    }
# def predict_visits(skope):
    # if skope.env == "svm":
    #     pytest.skip(reason="SVM test data not yet available")
    # else:
    #     return skope.visits
