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

# spacekit version
VERSION = "0.2.1"

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
    description="Machine Learning Tools for Astronomical Data Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alphasentaurii/spacekit",
    author="Ru Keïn",
    author_email="rkein@stsci.edu",
    license="MIT",
    keywords='spacekit spkt ml ai api sdk',
    packages=['spacekit'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pandas>=1.1.2',
        'matplotlib==3.2.2',
        'numpy>=1.16.5',
        'sklearn',
        'tensorflow>=2.3.0',
        'keras==2.4.3',
        'astropy>=4.0.1',
        'boto3>=1.15.16',
        'astroquery>=0.4.1',
        'wget==3.2'
    ],
    python_requires='>=3.7',
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)

