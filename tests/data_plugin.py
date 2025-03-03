import requests
import hashlib
import os
import shutil
from pytest import TempPathFactory
from spacekit.extractor.load import extract_file
from spacekit.datasets.meta import ZID

RETENTION_COUNT=1
RETENTION_POLICY='none'

def pytest_addoption(parser):
    parser.addoption(
        "--data_path",
        action="store",
        dest="data_path",
        default=os.path.abspath("tests/data"),
        help="Default pytest data path",
    )
    parser.addoption(
        "--cleanup",
        action="store_true",
        dest="cleanup",
        default=False,
        help="Remove test data saved in tests/data",
    )


def pytest_configure(config):
    config.option.disable_warnings = True
    data_path = config.getoption("data_path")
    
    # only retrieve from zenodo if tests/data/{env}/data.zip DNE
    if not os.path.exists(data_path):
        try:
            tmp_path_factory = TempPathFactory(
                config.option.basetemp, 
                RETENTION_COUNT, 
                RETENTION_POLICY,
                trace=config.trace.get("tmpdir"),
                _ispytest=True
            )
        except Exception: # pytest >= 7.3
            tmp_path_factory = TempPathFactory(
                config.option.basetemp, trace=config.trace.get("tmpdir"), _ispytest=True
            )
        data_uri = f"https://zenodo.org/record/{ZID}/files/pytest_data.tgz?download=1"
        basepath = tmp_path_factory.getbasetemp()
        target_path = os.path.join(basepath, "pytest_data.tgz")
        with open(target_path, "wb") as f:
            response = requests.get(data_uri, stream=True)
            if response.status_code == 200:
                f.write(response.raw.read())
        chksum = "3b4f8759b32ae76007988073d63156c3995f69a77925ff686a0e3fde785e2421"
        with open(target_path, "rb") as f:
            digest = hashlib.sha256(f.read())
            if digest.hexdigest() == chksum:
                extract_file(target_path, dest=os.path.abspath("tests"))

def pytest_unconfigure(config):
    cleanup = config.getoption("cleanup")
    if cleanup is True:
        tmp = os.path.abspath("tmp")
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        data_path = config.getoption("data_path")
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
