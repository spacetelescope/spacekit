"""
Spacekit HST Single Visit Mosaic Data/Image Preprocessing

Ex 1:
input_path = "path/to/datasets"
data_file = run_preprocessing(input_path, fname="svm_data")

Ex 2: synthetic (artificially corrupted) data:
data_file = run_preprocessing(input_path, fname=fname, json_pattern="*_total*_svm_*.json", crpt=1, draw_images=0)
"""

import argparse
import os
from spacekit.extractor.scrape import JsonScraper
from spacekit.preprocessor.scrub import ScrubSvm
from spacekit.generator.draw import DrawMosaics


def run_preprocessing(
    input_path,
    h5=None,
    fname="svm_data",
    output_path=None,
    json_pattern="*_total*_svm_*.json",
    crpt=0,
    draw_images=1,
):
    """Scrapes SVM data from raw files, preprocesses dataframe for MLP classifier and generates png images for image classifier.

    Parameters
    ----------
    input_path : str
        path to SVM dataset directory
    h5 : str, optional
        load from existing hdf5 file, by default None
    fname : str, optional
        base filename to give the output files, by default "svm_data"
    output_path : str, optional
        where to save output files. Defaults to current working directory., by default None
    json_pattern : str, optional
        glob-based search pattern, by default "*_total*_svm_*.json"
    crpt : int, optional
        set to 1 if using synthetic corruption data, by default 0
    draw_images : int, optional
        generate png images from dataset, by default 1

    Returns
    -------
    str
        path to csv file of preprocessed Pandas dataframe
    """
    if output_path is None:
        output_path = os.getcwd()
    os.makedirs(output_path, exist_ok=True)
    fname = os.path.basename(fname).split(".")[0]
    if h5 is None:
        patterns = json_pattern.split(",")
        jsc = JsonScraper(
            search_path=input_path,
            search_patterns=patterns,
            file_basename=fname,
            crpt=crpt,
            output_path=output_path,
        )
        jsc.json_harvester()
        jsc.h5store()
    else:
        jsc = JsonScraper(h5_file=h5).load_h5_file()
    if crpt == 1:
        jsc.data["label"] = 1
    scrub = ScrubSvm(jsc.data, input_path, output_path, fname)
    scrub.preprocess_data()
    fname = scrub.data_path
    if draw_images:
        img_outputs = os.path.join(output_path, "img")
        draw = DrawMosaics(
            input_path,
            output_path=img_outputs,
            fname=fname,
            gen=3,
            size=(24, 24),
            crpt=crpt,
        )
        draw.generate_total_images()
    return fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spacekit SVM",
        usage="python -m spacekit.skopes.hst.svm.prep path/to/raw_data",
    )
    parser.add_argument("input_path", type=str, help="path to SVM dataset directory")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="where to save output files. Defaults to current working directory.",
    )
    parser.add_argument(
        "--h5",
        type=str,
        default=None,
        help="load from existing hdf5 file",
    )
    parser.add_argument(
        "-f",
        "--fname",
        type=str,
        default="svm_data",
        help="output filename to create",
    )
    parser.add_argument("-j", "--json_pattern", type=str, default="*_total*_svm_*.json", help="glob-based search pattern")
    parser.add_argument(
        "-c",
        "--crpt",
        type=int,
        default=0,
        choices=[0, 1],
        help="set to 1 if using synthetic corruption data",
    )
    parser.add_argument(
        "-d", "--draw", type=int, default=1, help="generate png images from dataset"
    )
    args = parser.parse_args()
    run_preprocessing(
        args.input_path,
        h5=args.h5,
        fname=args.fname,
        json_pattern=args.json_pattern,
        crpt=args.crpt,
        draw_images=args.draw,
    )
