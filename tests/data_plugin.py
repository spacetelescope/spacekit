import requests
import hashlib
import tarfile
import os
import shutil
from pytest import TempPathFactory

def pytest_addoption(parser):
    parser.addoption(
        "--data_path",
        action="store",
        dest="data_path",
        default=os.path.abspath("tests/data"),
        help="Default pytest data path"
    )
    parser.addoption(
        "--cleanup",
        action="store_true",
        dest="cleanup",
        default=False,
        help="Remove test data saved in tests/data"
    )


def pytest_configure(config):
    config.option.disable_warnings = True
    data_path = config.getoption("data_path")
    # only retrieve from zenodo if tests/data/{env}/data.zip DNE
    if not os.path.exists(data_path):
        #tmp_path_factory = TempPathFactory(config.option.basetemp, retention_count, retention_policy, trace=config.trace.get("tmpdir"), _ispytest=True)
        tmp_path_factory = TempPathFactory(config.option.basetemp, trace=config.trace.get("tmpdir"), _ispytest=True)
        data_uri = "https://zenodo.org/record/8185020/files/pytest_data.tgz?download=1"
        basepath = tmp_path_factory.getbasetemp()
        target_path = os.path.join(basepath, "pytest_data.tgz")
        with open(target_path, 'wb') as f:
            response = requests.get(data_uri, stream=True)
            if response.status_code == 200:
                f.write(response.raw.read())
        chksum = "1a5204f3a5f5c42173e1da389a420ab2d079ecbd27323d50b5fc4155da643399"
        with open(target_path, "rb") as f:
            digest = hashlib.sha256(f.read())
            if digest.hexdigest() == chksum:
                with tarfile.TarFile.open(target_path) as tar:
                    tar.extractall(os.path.abspath("tests"))

def pytest_unconfigure(config):
    cleanup = config.getoption("cleanup")
    if cleanup is True:
        tmp = os.path.abspath("tmp")
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        data_path = config.getoption("data_path")
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
