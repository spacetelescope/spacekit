import os
import pytest
from pytest import fixture
from zipfile import ZipFile
import pandas as pd
from spacekit.analyzer.explore import HstCalPlots, HstSvmPlots
from spacekit.analyzer.scan import HstSvmScanner, HstCalScanner, import_dataset
from spacekit.extractor.load import load_datasets, extract_file
from spacekit.skopes.jwst.cal.config import KEYPAIR_DATA


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
        SUPPORTED_ENVS = ["svm", "hstcal", "jwstcal"]
        self.env = env

        if env.lower() not in SUPPORTED_ENVS:
            raise Exception(
                f"{env} is not a supported environment (supported envs: {SUPPORTED_ENVS})"
            )

        self.data_path = {
            "svm": os.path.join(f"tests/data/{env}/data.zip"),
            "hstcal": os.path.join(f"tests/data/{env}/data.zip"),
            "jwstcal": os.path.join(f"tests/data/{env}/data.zip"),
        }[env]

        self.kwargs = {
            "svm": dict(index_col="index"), 
            "hstcal": dict(index_col="ipst"),
            "jwstcal": dict(index_col="img_name")
        }[env]

        self.decoder = {
            "svm": {"det": {0: "hrc", 1: "ir", 2: "sbc", 3: "uvis", 4: "wfc"}},
            "hstcal": {"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}},
            "jwstcal": KEYPAIR_DATA,
        }[env]

        self.labeled = {
            "svm": "tests/data/svm/train/training.csv",
            "hstcal": "tests/data/hstcal/train/training.csv",
            "jwstcal": "tests/data/jwstcal/train/training.csv",
        }[env]

        self.unlabeled = {
            "svm": "tests/data/svm/predict/unlabeled.csv",
            "hstcal": "tests/data/hstcal/predict/unlabeled.csv",
            "jwstcal" : "tests/data/jwstcal/predict/unlabeled.csv"
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
            "hstcal": ["n_files", "total_mb"],
            "jwstcal": [
                'offset',
                'max_offset',
                'mean_offset',
                'sigma_offset',
                'err_offset',
                'sigma1_mean'
            ],
        }[env]

        self.rename_cols = {
            "svm": "_scl",
            "hstcal": ["x_files", "x_size"],
            "jwstcal": "_scl"
        }[env]

        self.enc_cols = {
            "svm": ["det", "wcs", "cat"],
            "hstcal": [
                "drizcorr",
                "pctecorr",
                "crsplit",
                "subarray",
                "detector",
                "dtype",
                "instr",
            ],
            "jwstcal": [
                "instr",
                "detector",
                "visitype",
                "filter",
                "pupil",
                "channel",
                "subarray"
            ],
        }[env]

        self.tx_file = {
            "svm": "tests/data/svm/tx_data.json",
            "hstcal": "tests/data/hstcal/tx_data.json",
            "jwstcal": "tests/data/jwstcal/tx_data.json"
        }[env]

        self.visits = {
            "svm": [],
            "hstcal": {
                "asn": ["j8zs05020", "ic0k06010", "la8mffg5q", "oc3p011i0"],
                "svm": [],
            },
            "jwstcal": ["jw02732"],
        }[env]


def pytest_addoption(parser):
    # parser.addoption("--env", action="store", default="hstcal", help="Environment to run tests against")
    parser.addoption("--env", action="store", default=None, help="Environment to run tests against")

# def pytest_configure(config):
#     config.addinivalue_line("markers", "skope_svm: only run in svm skope")
#     config.addinivalue_line("markers", "skope_cal: only run in cal skope")

# def pytest_collection_modifyitems(config, items, skope):
#     if skope.env == "hstcal"
    # env_param = config.getoption("--env")
    # if env_param:
    #     skope_param = pytest.mark.parametrize("skope", [(env_param)], indirect=True)
    # else:
    #     skope_param = pytest.mark.parametrize("skope", [("hstcal", "svm")], indirect=True)
    #     # skip_param = pytest.mark.skipif(reason="skip params based on --env")
    # for item in items:
    #     if "skopes" in item.keywords:
    #         item.add_marker(skope_param)


@fixture(scope="session")
def env(request):
    return request.config.getoption("--env")


@fixture(scope="session", params=["hstcal", "svm"])
def skope(request):
    env_param = request.config.getoption("--env")
    if env_param is not None and request.param != env_param:
        pytest.skip(reason="skipping param based on --env")
    else:
        return Config(request.param)


def check_skope(skope, param):
    if skope.env != param:
        pytest.skip(reason="skipping based on skope param")


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
    elif skope.env == "hstcal":
        scanner = HstCalScanner(perimeter=f"{res_data_path}/20??-*-*-*", primary=-1)
    else:
        pytest.skip(reason="JWST does not yet have a scanner.")
    scanner.exp = skope.env
    return scanner


@fixture(scope="session")
def explorer(skope, res_data_path):
    fname = res_data_path
    df = import_dataset(filename=fname, kwargs=skope.kwargs, decoder=skope.decoder)
    if skope.env == "svm":
        hst = HstSvmPlots(df)
    elif skope.env == "hstcal":
        hst = HstCalPlots(df)
    elif skope.env == "jwstcal":
        pytest.skip(reason="TODO")
    hst.env = skope.env
    return hst

@fixture(scope="session")
def labeled_dataset(skope):
    return skope.labeled


@fixture(scope="session")  # session
def unlabeled_dataset(skope):
    return skope.unlabeled


# SVM PREP
@fixture(scope="session")  # "ibl738.tgz"
def single_visit_path(tmp_path_factory):
    visit_path = os.path.relpath("tests/data/svm/prep/singlevisits.tgz")
    basepath = tmp_path_factory.getbasetemp()
    extract_file(visit_path, dest=basepath)
    dname = os.path.basename(visit_path.split(".")[0])
    visit_path = os.path.join(basepath, dname)
    return visit_path


@fixture(scope="function")
def img_outpath(tmp_path):
    return os.path.join(tmp_path, "img")


# # SVM PREDICT

@fixture(scope="session", params=["img.tgz", "img_pred.npz"])
def svm_pred_img(request, tmp_path_factory):
    img_path = os.path.join("tests/data/svm/predict", request.param)
    if img_path.split(".")[-1] == "tgz":
        basepath = tmp_path_factory.getbasetemp()
        extract_file(img_path, dest=basepath)
        fname = os.path.basename(img_path.split(".")[0])
        img_path = os.path.join(basepath, fname)
    return img_path


# # SVM TRAIN

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


@fixture(scope="function")
def hst_cal_predict_visits():
    return {
            "asn": ["j8zs05020", "ic0k06010", "la8mffg5q", "oc3p011i0"]
    }


@fixture(scope="function")
def jwstcal_input_path():
    return "tests/data/jwstcal/predict/inputs"
