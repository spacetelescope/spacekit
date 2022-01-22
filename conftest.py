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

# scope='function' each test function gets its own object
# @fixture(scope='function') # function, session, module (class?)
# def chrome_browser():
#     browser = webdriver.Chrome()
#     return browser

# @fixture(scope='function') # session
# def svm_dataset():
#     svm_data = 'test_files/svm/test_img_data.csv'
#     # data = os.path.join(
#     # os.path.dirname(os.path.realpath(__file__)),
#     # 'test_files/svm/test_img_data.csv',
#     # )
#     yield svm_data

# @fixture(scope='function')
# def svm_images():
#     img_file = os.path.join(
#     os.path.dirname(os.path.realpath(__file__)),
#     'test_files/svm/test_images.npz',
#     )
#     yield img_file

# @fixture(scope='function')
# def svm_png_images():
#     img_path = os.path.join(
#     os.path.dirname(os.path.realpath(__file__)),
#     'test_files/svm/img',
#     )
#     yield img_path

