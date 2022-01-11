"""
Spacekit HST "Calibration in the Cloud" (calcloud) Job Resource Allocation Model Training

This script imports and preprocesses job metadata for the Hubble Space Telescope data calibration pipeline,
which is then used as inputs to build, train and evaluate 3 neural networks for estimating AWS batch compute job resource requirements.

The networks include one multi-class classifier and two linear regression estimators. The classifier predicts which of 4 possible memory bin sizes (and therefore compute instance type) is most appropriate for reprocessing a given ipppssoot (i.e. "job"). The wallclock regressor estimates the maximum execution time ("wallclock" or "kill" time) in seconds needed to complete the job.

Memory Bin Classifier
---------------------
Allocating a job to a memory bin higher than required leads to unnecessary cost increases (because AWS billing tiers are directly related to compute instance types). However, if the estimated memory allocation is too low, the job will fail and have to be re-submitted at the next highest tier, also leading to increased costs. The majority of data being reprocessed for HST falls in the <2GB range, with only a handful in the highest tier (16-64GB).

Memory Regressor
----------------
Essentially identical to the classifier, the difference being that it returns a precise estimation value for memory in GB rather than a class. This is not needed for the pipeline (because it only needs to know which bin size) but we decided to keep the model for reference and analysis purposes.

Wallclock Regressor
-------------------
Estimates maximum execution or "kill" time in seconds it will take to complete the job. AWS Batch requires a minimum threshold of 60 seconds, with a large proportion of jobs completing below the one minute mark. Some jobs can take hours or even days to complete - if a job fails in memory after running for 24 hours, it would have to be re-submitted (huge cost). Likewise, if a job is allocated appropriate memory size but fails prematurely because it passes the maximum wallclock threshold, it would have to be resubmitted at a higher time allocation (huge cost). The relationship between memory needs and wallclock time is not linear, hence why there is a need for two separate models.



Ex:
python -m spacekit.skopes.hst.cal.train data/2021-11-04-1636048291
"""

from argparse import ArgumentParser
from spacekit.datasets import hst_cal
from spacekit.preprocessor.scrub import ScrubCal
from spacekit.builder.networks import (
    MemoryClassifier,
    MemoryRegressor,
    WallclockRegressor,
)
from spacekit.analyzer.compute import ComputeMulti, ComputeRegressor

"""
## to load results from disk in a separate session (for plotting, analysis etc):
# bcom2 = ComputeMulti(res_path=f"{res_path}/mem_bin")
# bin_out = bcom2.upload()
# bcom2.load_results(bin_out)
"""


def load_prep(data_path="data/2021-11-04-1636048291/latest.csv"):
    df = hst_cal.load_data(fpath=data_path)
    data = ScrubCal(df).prep_data()
    return data


def build_fit(BuildClass, data, y_train, y_test, test_idx, model_path=None):
    builder = BuildClass(data.X_train, y_train, data.X_test, y_test, test_idx=test_idx)
    builder.build_mlp()
    builder.fit()
    if model_path:
        builder.save_model(output_path=model_path)
    return builder


def compute_cache(builder, res_path):
    if builder.algorithm == "linreg":
        com = ComputeRegressor(builder=builder, res_path=res_path)
        com.calculate_results()
    elif builder.algorithm == "multiclass":
        com = ComputeMulti(builder=builder, res_path=res_path)
        com.calculate_multi()
    outputs = com.make_outputs()
    print(outputs.keys())
    return com


def train_models(data, res_path, mem_clf=1, mem_reg=1, wall_reg=1, model_path=None):
    model_objects = {}
    if mem_clf:
        y_train, y_test, test_idx = data.y_bin_train, data.y_bin_test, data.bin_test_idx
        clf = build_fit(
            MemoryClassifier, data, y_train, y_test, test_idx, model_path=model_path
        )
        bcom = compute_cache(clf, res_path=f"{res_path}/mem_bin")
        model_objects["mem_clf"] = {"builder": clf, "results": bcom}
    if mem_reg:
        y_train, y_test, test_idx = data.y_mem_train, data.y_mem_test, data.mem_test_idx
        mem = build_fit(
            MemoryRegressor, data, y_train, y_test, test_idx, model_path=model_path
        )
        mcom = compute_cache(mem, res_path=f"{res_path}/memory")
        model_objects["mem_reg"] = {"builder": mem, "results": mcom}
    if wall_reg:
        y_train, y_test, test_idx = (
            data.y_wall_train,
            data.y_wall_test,
            data.wall_test_idx,
        )
        wall = build_fit(
            WallclockRegressor, data, y_train, y_test, test_idx, model_path=model_path
        )
        wcom = compute_cache(wall, res_path=f"{res_path}/wallclock")
        model_objects["wall_reg"] = {"builder": wall, "results": wcom}
    return model_objects


def parse_user_args(args):
    # import and preprocess data
    data = load_prep(data_path=f"{args.source_path}/data/{args.fname}")
    res_path = f"{args.source_path}/results"
    if args.save_models is True:
        model_path = args.source_path  # auto creates "models" subdir
    else:
        model_path = None  # don't save the models (for testing)
    # build, train and evaluate models
    return train_models(
        data,
        res_path,
        mem_clf=args.mem_clf,
        mem_reg=args.mem_reg,
        wall_reg=args.wall_reg,
        model_path=model_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser(prog="spacekit hst calibration model training")
    parser.add_argument(
        "source_path",
        type=str,
        default="data/2021-11-04-1636048291",
        help="source data directory base path",
    )
    parser.add_argument(
        "--fname", type=str, default="latest.csv", help="name of training data csv file"
    )
    parser.add_argument(
        "--mem_clf", type=int, default=1, help="bool: train memory bin classifier"
    )
    parser.add_argument(
        "--mem_reg", type=int, default=1, help="bool: train memory regressor"
    )
    parser.add_argument(
        "--wall_reg", type=int, default=1, help="bool: train wallclock regressor"
    )
    parser.add_argument(
        "--save_models",
        type=int,
        default=0,
        help="save trained models and weights to disk",
    )
    parse_user_args(parser.parse_args())
