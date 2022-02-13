"""Utilities for loading datasets from the spacekit archive.
"""
from ._base import scrape_archives
from ._base import load_from_archive
from ._base import load_cal
from ._base import load_svm
from ._base import load_k2
from ._base import import_collection
from ._base import load

__all__ = [
    "scrape_archives",
    "load_from_archive",
    "load_cal",
    "load_svm",
    "load_k2",
    "import_collection",
    "load",
]
