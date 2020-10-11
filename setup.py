#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright: (c) 2020 by Ru Keïn
:license: MIT, see LICENSE for more details.
"""

import os
import sys

from setuptools import setup
from setuptools.command.install import install

# circleci.py version
VERSION = "0.0.5"

with open("README.md", "r") as fh:
    long_description = fh.read()

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setup(
    name="spacekit",
    version=VERSION,
    description="Python wrapper for the CircleCI API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hakkeray/spacekit",
    author="Ru Keïn",
    author_email="hakkeray@gmail.com",
    license="MIT",
    keywords='spacekit spkt ml ai api sdk',
    packages=['spacekit'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'requests==2.23.0',
        'pandas==1.1.2',
        'numpy==1.16.5',
        'scikit-learn==0.23.2',
        'scipy==1.4.1',
        'tensorflow==2.3.0'
    ],
    python_requires='>=3.6',
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)

