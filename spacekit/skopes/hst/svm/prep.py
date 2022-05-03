"""
Spacekit HST Single Visit Mosaic Data/Image Preprocessing

Step 1: SCRAPE JSON FILES and make dataframe
Step 2: Scrape Fits Headers and SCRUB DATAFRAME
Step 3: DRAW Mosaic images

Examples:
df = run_preprocessing("home/singlevisits")

df = run_preprocessing("home/syntheticdata", fname="synth2", crpt=1, draw=0)

"""
import argparse
import os
from spacekit.extractor.scrape import JsonScraper
from spacekit.preprocessor.scrub import SvmScrubber
from spacekit.generator.draw import DrawMosaics


def run_preprocessing(
    input_path,
    h5=None,
    fname="svm_data",
    output_path=None,
    json_pattern="*_total*_svm_*.json",
    visit=None,
    crpt=0,
    draw=1,
):
    """Scrapes SVM data from raw files, preprocesses dataframe for MLP classifier and generates png images for image CNN.
    #TODO: if no JSON files found, look for results_*.csv file instead and preprocess via alternative method

    Parameters
    ----------
    input_path : str
        path to SVM dataset directory
    h5 : str, optional
        load from existing hdf5 file, by default None
    fname : str, optional
        base filename to give the output files, by default "svm_data"
    output_path : str, optional
        where to save output files. Defaults to current working directory, by default None
    json_pattern : str, optional
        glob-based search pattern, by default "*_total*_svm_*.json"
    visit: str, optional
        single visit name (e.g. "id8f34") matching subdirectory of input_path; will search and preprocess this visit only (rather than all visits contained in the input_path), by default None
    crpt : int, optional
        set to 1 if using synthetic corruption data, by default 0
    draw : int, optional
        generate png images from dataset, by default 1

    Returns
    -------
    dataframe
        preprocessed Pandas dataframe
    """
    if output_path is None:
        output_path = os.getcwd()
    os.makedirs(output_path, exist_ok=True)
    fname = os.path.basename(fname).split(".")[0]
    # 1: SCRAPE JSON FILES and make dataframe
    if h5 is None:
        search_path = os.path.join(input_path, visit) if visit else input_path
        patterns = json_pattern.split(",")
        jsc = JsonScraper(
            search_path=search_path,
            search_patterns=patterns,
            file_basename=fname,
            crpt=crpt,
            output_path=output_path,
        )
        jsc.json_harvester()
        jsc.h5store()
    else:
        jsc = JsonScraper(h5_file=h5).load_h5_file()
    # 2: Scrape Fits Files and SCRUB DATAFRAME
    scrub = SvmScrubber(
        input_path, data=jsc.data, output_path=output_path, output_file=fname, crpt=crpt
    )
    scrub.preprocess_data()
    # 3:  DRAW IMAGES
    if draw:
        img_outputs = os.path.join(output_path, "img")
        mos = DrawMosaics(
            input_path,
            output_path=img_outputs,
            fname=scrub.data_path,
            pattern="",
            gen=3,
            size=(24, 24),
            crpt=crpt,
        )
        mos.generate_total_images()
    return scrub.df, scrub.data_path


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
    parser.add_argument(
        "-j",
        "--json_pattern",
        type=str,
        default="*_total*_svm_*.json",
        help="glob-based search pattern",
    )
    parser.add_argument(
        "-c",
        "--crpt",
        type=int,
        default=0,
        choices=[0, 1],
        help="set to 1 if using synthetic corruption data",
    )
    parser.add_argument(
        "-d",
        "--draw",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 (default): generate png images from dataset, 0: turn images off",
    )
    args = parser.parse_args()
    _, _ = run_preprocessing(
        args.input_path,
        h5=args.h5,
        fname=args.fname,
        output_path=args.output_path,
        json_pattern=args.json_pattern,
        crpt=args.crpt,
        draw=args.draw,
    )
