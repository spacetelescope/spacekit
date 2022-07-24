#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright: (c) 2021 by Ru Keïn
:license: MIT, see LICENSE for more details.
"""
import sys
from setuptools import setup


TEST_HELP = """
Note: Run cal and svn tests separately using the --env flag:

    pytest --env cal
    pytest --env svm

"""

if "test" in sys.argv:
    print(TEST_HELP)
    sys.exit(1)

DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py build_docs'. Instead you will need to run:

    cd docs
    make html

"""

if "build_docs" in sys.argv or "build_sphinx" in sys.argv:
    print(DOCS_HELP)
    sys.exit(1)


# setup(use_scm_version={'write_to': 'spacekit/_version.py'})
setup(version="0.3.2")
