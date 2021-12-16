import argparse
from spacekit.skopes.hst.svm.prep import run_preprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spacekit SVM",
        usage="python prep.py path/to/raw_data -f=svm_data.csv",
        )
    parser.add_argument(
        "input_path", type=str, help="path to SVM dataset directory"
    )
    # prep
    args = parser.parse_args()
    input_path = args.input_path
    df = run_preprocessing(input_path)
    # TODO:
    # if train
    # do train
    # if predict
    # do predict
    # if corrupt 
    # do corrupt