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
):
    """[summary]
    Scrapes SVM data from raw files, preprocesses dataframe for MLP classifier and generates png images for image classifier.
    Args:
        input_path ([type]): [description]
        h5 ([type], optional): [description]. Defaults to None.
        fname (str, optional): [description]. Defaults to "svm_data".
        output_path ([type], optional): [description]. Defaults to None.
        json_pattern (str, optional): [description]. Defaults to "*_total*_svm_*.json".
        crpt (int, optional): [description]. Defaults to 0.

    Returns:
        [Pandas dataframe]: preprocessed dataframe for SVM QA data
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
    scrub = ScrubSvm(jsc.data, input_path, output_path, fname)
    scrub.preprocess_data()
    fname = scrub.data_path
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
        usage="python prep.py path/to/raw_data -f=svm_data.csv",
    )
    parser.add_argument("input_path", type=str, help="path to SVM dataset directory")
    parser.add_argument(
        "--hdf5",
        type=str,
        default=None,
        help="hdf5 file to create (or load if already exists)",
    )
    parser.add_argument(
        "-f",
        "--fname",
        type=str,
        default="svm_data",
        help="csv output filename to create",
    )
    parser.add_argument("-j", "--json_pattern", type=str, default="*_total*_svm_*.json")
    parser.add_argument(
        "-c",
        "--crpt",
        type=int,
        default=0,
        choices=[0, 1],
        help="set to 1 if using synthetic corruption data",
    )
    args = parser.parse_args()
    input_path = args.input_path
    h5 = args.hdf5
    fname = args.fname
    json_pattern = args.json_pattern
    crpt = args.crpt
    run_preprocessing(
        input_path, h5=h5, fname=fname, json_pattern=json_pattern, crpt=crpt
    )
