"""
CRVAL1 and CRVAL2 give the center coordinate as right ascension and declination or longitude and latitude in decimal degrees.

CRPIX1 and CRPIX2 are the pixel coordinates of the reference point to which the projection and the rotation refer.

The default corruption method is on CRVAL1 and/or CRVAL2 only. Corruption can also be performed on CRPIX values; this would allow a machine learning algorithm to be trained on an alternative (albeit more rare) cause of single visit mosaic misalignment.

"""
import os
import sys
import argparse
import shutil
import glob
import numpy as np
from astropy.io import fits
import time
from tqdm import tqdm
from progressbar import ProgressBar

from spacekit.extractor.draw_mosaics import (
    generate_total_images,
    generate_filter_images,
)
from spacekit.analyzer.track import stopwatch

SVM_QUALITY_TESTING = "on"


def pick_random_exposures(dataset):
    hapfiles = glob.glob(f"{dataset}/*.fits")
    n_corruptions = np.random.randint(2, 4)
    if len(hapfiles) < n_corruptions:
        n_corruptions = max(len(hapfiles), 2)
    print(f"Selecting {n_corruptions} out of {len(hapfiles)} files")
    np.random.shuffle(hapfiles)
    cor_idx = []
    for _ in list(range(n_corruptions)):
        cor_idx.append(np.random.randint(len(hapfiles)))
    print(f"Shuffled random index: {cor_idx}")
    selected_files = []
    for i, j in enumerate(hapfiles):
        if i in cor_idx:
            selected_files.append(j)
    print("\nFiles selected for corruption: ", selected_files)
    return selected_files


def pick_random_filter(dataset):
    drizzle_dct = find_filter_files(dataset)
    i = np.random.randint(0, len(drizzle_dct))
    f = list(drizzle_dct.keys())[i]
    print(f"\nRANDOM FILTER SELECTED: {f}")
    drz_files = drizzle_dct[f]
    print(f"\nFILES SELECTED: {drz_files}")
    return drz_files


def find_filter_files(dataset):
    drz_file = glob.glob(f"{dataset}/*.out")[0]
    drizzle_dct = {}
    with open(drz_file, "r") as f:
        input_strings = f.readlines()
        for s in input_strings:
            tokens = s.split(",")
            drizzle_file = f"{dataset}/{tokens[0]}"
            fltr = tokens[-3].replace(";", "_")
            if fltr not in drizzle_dct:
                drizzle_dct[fltr] = [drizzle_file]
            else:
                drizzle_dct[fltr].append(drizzle_file)
    return drizzle_dct


def modify_paths(drizzle_dct, name):
    drizzle_mod = {}
    for flt, paths in drizzle_dct.items():
        drizzle_mod[flt] = []
        for p in paths:
            filename = p.split("/")[-1]
            new = f"{name}/{filename}"
            drizzle_mod[flt].append(new)
    return drizzle_mod


def pick_random_subset(filter_files):
    n_files = len(filter_files)
    if n_files > 1:
        n_corruptions = np.random.randint(1, n_files)
    else:
        print(f"WARNING - {n_files} exposure for this filter.")
        n_corruptions = n_files
    print(f"\nSelecting {n_corruptions} out of {n_files} files")
    np.random.shuffle(filter_files)
    cor_idx = []
    for _ in list(range(n_corruptions)):
        cor_idx.append(np.random.randint(n_files))
    print(f"\nShuffled random index: {cor_idx}")
    selected_files = []
    for i, j in enumerate(filter_files):
        if i in cor_idx:
            selected_files.append(j)
    print("\nFiles selected for corruption: ", selected_files)
    return selected_files


def set_lambda_threshold(thresh):
    if thresh == "major":
        return np.random.uniform(10, 20)
    elif thresh == "standard":
        return np.random.uniform(0.5, 10)
    elif thresh == "minor":
        return np.random.uniform(0, 1)
    else:
        return np.random.uniform(0, 20)


def static_augment(thresh):
    lamb = set_lambda_threshold(thresh)
    delta = (lamb * 0.04) / 3600
    p, r = np.random.uniform(0, 1), np.random.uniform(0, 1)
    print("PIXEL offset: ", lamb)
    if r < p:
        print("DEGREE offset: +", delta)
        return delta
    else:
        print("DEGREE offset: -", delta)
        return -delta


def static_corruption(fits_file, delta):
    print("\nApplying static augment: ", fits_file)
    with fits.open(fits_file, "update") as hdu:
        wcs_valid = {
            "CRVAL1": hdu[1].header["CRVAL1"],
            "CRVAL2": hdu[1].header["CRVAL2"],
        }
        wcs_corrupt = wcs_valid.copy()
        wcs_corrupt["CRVAL1"] += delta
        wcs_corrupt["CRVAL2"] += delta
        hdu[1].header["CRVAL1"] = wcs_corrupt["CRVAL1"]
        hdu[1].header["CRVAL2"] = wcs_corrupt["CRVAL2"]
    return wcs_valid, wcs_corrupt


def stochastic_augment(x, thresh):
    lamb = set_lambda_threshold(thresh)
    delta = (lamb * 0.04) / 3600
    p, r = np.random.uniform(0, 1), np.random.uniform(0, 1)
    print("\nCRVAL: ", x)
    print("PIXEL offset: ", lamb)
    if r < p:
        print("DEGREE offset: +", delta)
        return x + delta
    else:
        print("DEGREE offset: -", delta)
        return x - delta


def stochastic_corruption(fits_file, thresh):
    print("\nApplying stochastic augment: ", fits_file)
    with fits.open(fits_file, "update") as hdu:
        wcs_valid = {
            "CRVAL1": hdu[1].header["CRVAL1"],
            "CRVAL2": hdu[1].header["CRVAL2"],
        }
        wcs_corrupt = wcs_valid.copy()
        wcs_corrupt["CRVAL1"] = stochastic_augment(wcs_corrupt["CRVAL1"], thresh)
        wcs_corrupt["CRVAL2"] = stochastic_augment(wcs_corrupt["CRVAL2"], thresh)
        hdu[1].header["CRVAL1"] = wcs_corrupt["CRVAL1"]
        hdu[1].header["CRVAL2"] = wcs_corrupt["CRVAL2"]
    return wcs_valid, wcs_corrupt


def print_corruptions(wcs_valid, wcs_corrupt):
    separator = "---!@#$%^&*()_+---" * 3
    print("\nCRVAL1-old: ", wcs_valid["CRVAL1"])
    print("CRVAL1-new: ", wcs_corrupt["CRVAL1"])
    print("\nCRVAL2-old: ", wcs_valid["CRVAL2"])
    print("CRVAL2-new: ", wcs_corrupt["CRVAL2"])
    print(f"\n{separator}")


def run_header_corruption(selected_files, mode="stoc", thresh="any"):
    if mode == "stat":
        print("\nStarting static corruption\n")
        delta = static_augment(thresh)
        for fits_file in selected_files:
            wcs_valid, wcs_corrupt = static_corruption(fits_file, delta)
            print_corruptions(wcs_valid, wcs_corrupt)
    else:
        print("\nStarting stochastic corruption\n")
        for fits_file in selected_files:
            wcs_valid, wcs_corrupt = stochastic_corruption(fits_file, thresh)
            print_corruptions(wcs_valid, wcs_corrupt)


def artificial_misalignment(dataset, outputs, palette):
    dname = dataset.split("/")[-1]
    name = f"{outputs}/{dname}_{palette}"
    shutil.copytree(dataset, name)
    if palette == "rex":
        selected_files = pick_random_exposures(name)
    elif palette == "rfi":
        selected_files = pick_random_filter(name)
    run_header_corruption(selected_files)


def multiple_permutations(dataset, outputs, expos, mode, thresh="any"):
    drizzle_dct = find_filter_files(dataset)
    filters = list(drizzle_dct.keys())
    separator = "---" * 5
    bar = ProgressBar().start()

    for x, f in zip(bar(range(len(filters))), filters):
        dname = dataset.split("/")[-1]
        name = f"{outputs}/{dname}_{f.lower()}_{expos}_{mode}"
        if not os.path.exists(name):
            shutil.copytree(dataset, name)
        drizzle_mod = modify_paths(drizzle_dct, name)
        out = sys.stdout
        err = 0
        with open(f"{name}/corruption.txt", "w") as logfile:
            sys.stdout = logfile
            print(separator)
            print("\nFILTER: ", f)
            if expos == "all":
                selected_files = drizzle_mod[f]
                print("\nALL FILES: ", selected_files)
            else:
                filter_files = drizzle_mod[f]
                if len(filter_files) == 1:
                    err += 1
                selected_files = pick_random_subset(filter_files)
            run_header_corruption(selected_files, mode=mode, thresh=thresh)
            sys.stdout = out
        if err == 1:
            with open(f"{name}/warning.txt", "w") as warning:
                sys.stdout = warning
                print("WARNING: only 1 exposure but you requested a subset")
                sys.stdout = out
        bar.update(x)
    bar.finish()


def run_svm(dataset, outputs):
    from drizzlepac import runsinglehap

    os.environ.get("SVM_QUALITY_TESTING", "on")
    mutations = glob.glob(f"{outputs}/{dataset}_*")
    for m in mutations:
        warning = f"{m}/warning.txt"
        if os.path.exists(warning):
            print(f"Skipping {m} - see warning file")
        else:
            drz_file = glob.glob(f"{m}/*.out")[0]
            if drz_file:
                runsinglehap.perform(drz_file, log_level="info")
            # cmd = ["runsinglehap", drz_file]
            # err = subprocess.call(cmd)
            # if err:
            #     print(f"SVM failed to run for {m}")


def generate_images(dataset, outputs, filters=False):
    input_path = outputs
    output_path = os.path.join(os.path.dirname(outputs), "img/1")
    generate_total_images(input_path, datasets=[dataset], output_img=output_path)
    if filters is True:
        generate_filter_images(
            input_path,
            dataset=dataset,
            outpath="./img/filter",
            figsize=(24, 24),
            crpt=0,
        )


def all_permutations(dataset, outputs):
    multiple_permutations(dataset, outputs, "all", "stat")
    multiple_permutations(dataset, outputs, "all", "stoc")
    multiple_permutations(dataset, outputs, "sub", "stat")
    multiple_permutations(dataset, outputs, "sub", "stoc")


def run_blocks(datasets, outputs, prc, cfg):
    start_block = time.time()
    stopwatch("Block Workflow", t0=start_block, out=outputs)
    if prc["crpt"]:
        prcname = "CORRUPTION"
        start = time.time()
        stopwatch(prcname, t0=start, out=outputs)
        for dataset in tqdm(datasets):
            if cfg["palette"] == "multi":
                all_permutations(dataset, outputs)
            elif cfg["palette"] == "mfi":
                multiple_permutations(
                    dataset, outputs, cfg["expos"], cfg["mode"], cfg["thresh"]
                )
            elif cfg["palette"] in ["rex", "rfi"]:
                artificial_misalignment(dataset, outputs, cfg["palette"])
        end = time.time()
        stopwatch(prcname, t0=start, t1=end, out=outputs)

    if prc["runsvm"]:
        prcname = "ALIGNMENT"
        start = time.time()
        stopwatch(prcname, t0=start, out=outputs)
        for dataset in tqdm(datasets):
            run_svm(datasets, outputs)
        end = time.time()
        stopwatch(prcname, t0=start, t1=end, out=outputs)

    if prc["imagegen"]:
        prcname = "IMAGE GENERATION"
        start = time.time()
        stopwatch(prcname, t0=start, out=outputs)
        for dataset in tqdm(datasets):
            generate_images(dataset, outputs)
        end = time.time()
        stopwatch(prcname, t0=start, t1=end, out=outputs)
    end_block = time.time()
    stopwatch("Block Workflow", t0=start_block, t1=end_block, out=outputs)


def run_pipes(datasets, outputs, prc, cfg):
    start = time.time()
    stopwatch("Pipe Workflow", t0=start, out=outputs)
    for dataset in tqdm(datasets):
        t0 = time.time()
        stopwatch(dataset, t0=t0, out=outputs)
        if prc["crpt"]:
            if cfg["palette"] == "multi":
                all_permutations(dataset, outputs)
            elif cfg["palette"] == "mfi":
                multiple_permutations(
                    dataset, outputs, cfg["expos"], cfg["mode"], thresh=cfg["thresh"]
                )
            elif cfg["palette"] in ["rex", "rfi"]:
                artificial_misalignment(dataset, outputs, cfg["palette"])
        if prc["runsvm"]:
            run_svm(dataset, outputs)
        if prc["imagegen"]:
            generate_images(dataset, outputs)
        t1 = time.time()
        stopwatch(dataset, t0=t0, t1=t1, out=outputs)

    end = time.time()
    stopwatch("Pipe Workflow", t0=start, t1=end, out=outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spacekit", usage="python -m spacekit.skopes.hst.svm.corrupt ./singlevisits ./synthetic -e=sub -m=stat"
    )
    parser.add_argument("srcpath", type=str, help="single visit dataset(s) directory")
    parser.add_argument(
        "outputs", type=str, help="path for saving corrupt versions of HAP files"
    )
    parser.add_argument(
        "palette",
        type=str,
        choices=["rex", "rfi", "mfi", "multi"],
        help="`rex`: randomly select subset of exposures from any filter; `rfi`: select all exposures from randomly selected filter; `mfi`: exposures of one filter, repeated for every filter in dataset. 'multi' creates sub- and all- MFI permutations",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*",
        help="glob search pattern - default is wildcard *",
    )
    parser.add_argument(
        "-e",
        "--expos",
        type=str,
        choices=["all", "sub"],
        default="all",
        help="all or subset of exposures",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["stat", "stoc"],
        default="stoc",
        help="apply consistent (static) or randomly varying (stochastic) corruptions to each exposure",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=str,
        choices=["major", "standard", "minor", "any"],
        default="any",
        help="lambda relative error threshold",
    )
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        choices=["block", "pipe"],
        default="block",
        help="block (default): run all datasets through one process before starting next (if using more than one); pipe: run the entire workflow one dataset at a time.",
    )
    parser.add_argument(
        "-c",
        "--crpt",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 (default) run corruption workflow",
    )
    parser.add_argument(
        "-r",
        "--runsvm",
        type=int,
        choices=[0, 1],
        default=0,
        help="1: run svm drizzle workflow; 0 (default): skip",
    )
    parser.add_argument(
        "-i",
        "--imagegen",
        type=int,
        choices=[0, 1],
        default=0,
        help="1: run imagegen workflow; 0 (default): skip",
    )
    # get user-defined args and/or set defaults
    args = parser.parse_args()
    outputs = args.outputs
    workflow = args.workflow
    datasets = glob.glob(f"{args.srcpath}/{args.pattern}")
    if len(datasets) < 1:
        print("No datasets found matching the search pattern.")
        sys.exit(1)
    procs = dict(crpt=args.crpt, runsvm=args.runsvm, imagegen=args.imagegen)
    cfg = dict(
        palette=args.palette, expos=args.expos, mode=args.mode, thresh=args.threshold
    )
    if workflow == "block":
        run_blocks(datasets, outputs, procs, cfg)
    else:
        run_pipes(datasets, outputs, procs, cfg)
