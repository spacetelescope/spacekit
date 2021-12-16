import argparse
import os
from spacekit.extractor.load import Hdf5IO
from spacekit.preprocessor.scrub import ScrubSvm
from spacekit.generator.draw import DrawMosaics

def prep_mlp_data(
    input_path, h5=None, filename="svm_data.csv", output_path=None, json_pattern="*_total*_svm_*.json", crpt=0
    ):
    output_file = os.path.basename(filename)
    if output_path is None:
        output_path = os.getcwd()
    else:
        os.makedirs(output_path, exist_ok=True)
    if h5:
        h5io = Hdf5IO(h5_file=h5).load_h5_file()
    else:
        patterns = json_pattern.split(",")
        h5io = Hdf5IO(search_path=input_path, patterns=patterns, crpt=crpt, save_file_as=filename, outpath=output_path).make_h5_file()
    df = ScrubSvm(h5io.data, input_path, output_path, output_file).preprocess_data()
    return df


def run_preprocessing(input_path, h5=None, filename=None, output_path=None, json_pattern="*_total*_svm_*.json", crpt=0):
    if filename is None:
        filename = "svm_data.csv"
    if output_path is None:
        output_path = os.getcwd()
    df = prep_mlp_data(input_path, h5=h5, filename=filename, json_pattern=json_pattern, crpt=crpt)
    img_outputs = os.path.join(output_path, "img")
    draw = DrawMosaics(input_path, output_path=img_outputs, fname=filename, gen=3, size=(24,24), crpt=crpt)
    draw.generate_total_images()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spacekit SVM",
        usage="python prep.py path/to/raw_data -f=svm_data.csv",
    )
    parser.add_argument(
        "input_path", type=str, help="path to SVM dataset directory"
    )
    parser.add_argument(
        "--hdf5",
        type=str,
        default=None,
        help="hdf5 file to create (or load if already exists)",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="svm_data.csv",
        help="csv output filepath to create",
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
    h5 = args.h5
    filename = args.filename
    json_pattern = args.json_pattern
    crpt = args.crpt
    run_preprocessing(input_path, h5=h5, filename=filename, json_pattern=json_pattern, crpt=crpt)