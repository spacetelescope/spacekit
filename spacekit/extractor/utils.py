import os
from zipfile import ZipFile
from tarfile import TarFile


def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory

def safe_extract(tar, expath=".", members=None, *, numeric_owner=False):
    directory = os.path.dirname(tar)
    for member in tar.getmembers():
        member_path = os.path.join(directory, member.name)
        if not is_within_directory(directory, member_path):
            raise Exception("WARNING: Attempted Path Traversal in Tar File")
    
    tar.extractall(expath, members, numeric_owner=numeric_owner)

def extract_file(fpath, dest="."):
    fp = fpath.split(".")
    if len(fp) > 2:
        mode = f"r:{fp[-1]}"
        kind = fp[-2]
    else:
        kind = fp[-1]
        mode = "r"
    if kind == "zip":
        with ZipFile(fpath, mode) as zip_ref:
            zip_ref.extractall(dest)
    elif kind == "tar":
        with TarFile.open(fpath, mode) as tar:
            safe_extract(tar, expath=dest)
    else:
        raise Exception(f"Could not extract file of type {kind}")


    