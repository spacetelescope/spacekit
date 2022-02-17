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
import sys
import os
from spacekit.datasets import load
from spacekit.preprocessor.prep import CalPrep
from spacekit.builder.architect import (
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

COLUMN_ORDER = [
    "n_files",
    "total_mb",
    "drizcorr",
    "pctecorr",
    "crsplit",
    "subarray",
    "detector",
    "dtype",
    "instr",
]


# def get_paths(timestamp):
#     if timestamp == "now":
#         train_time = dt.datetime.now()
#     elif isinstance(timestamp, str):
#         if len(timestamp) <= 14:
#             train_time = dt.datetime.fromtimestamp(int(timestamp))
#         else:
#             train_time = dt.datetime.fromisoformat(timestamp)
#     elif isinstance(timestamp, int) or isinstance(timestamp, float):
#         train_time = dt.datetime.fromtimestamp(timestamp)
#     else:
#         print(
#             f"Timestamp type must be a string (datetime, isoformat) or int/float (timestamp). You passed {type(timestamp)}."
#         )
#         raise ValueError
#     t0 = train_time.timestamp()
#     data_path = f"{dt.date.fromtimestamp(t0).isoformat()}-{str(int(t0))}"
#     return data_path


def find_data_sources(source_path, fname=None, date_key=None):
    fpath = []
    for root, _, files in os.walk(source_path):
        if fname is not None:
            name = os.path.join(root, fname)
            if os.path.exists(name):
                # print(f"Found dataset: {name}")
                fpath.append(name)
        else:
            for f in files:
                if f.split(".")[-1] == "csv":
                    name = os.path.join(root, f)
                    fpath.append(name)
    if len(fpath) > 0:
        if date_key is None:
            print(f"Found datasets: \n {fpath}")
            print(f"Defaulting to most recent: {fpath[-1]}")
        else:
            for f in fpath:
                if date_key in f:
                    fpath = [f]
                    print(f"Found matching dataset: {f}")
        fpath = fpath[-1]
    else:
        print(
            "No datasets found :( \n Check the source_path exists and there's a .csv file in one of its subdirectories."
        )
        sys.exit(1)
    return fpath


def load_prep(date_key=None, fpath=None):
    df = load(name="calcloud", date_key=date_key, fpath=fpath)
    data = CalPrep(df, "mem_bin")
    data.prep_data()
    return data


# def save_preprocessed(self):
# if src == "ddb":  # dynamodb 'calcloud-hst-data'
#     ddb_data = io.ddb_download(table_name, attr)
#     io.write_to_csv(ddb_data, "batch.csv")
#     df = pd.read_csv("batch.csv", index_col="ipst")

# def preprocess(bucket_mod, prefix, src, table_name, attr):
#     # MAKE TRAINING SET - single df for ingested data

#     # update power transform
#     df, pt_transform = update_power_transform(df)
#     io.save_dataframe(df, "latest.csv")
#     io.save_json(pt_transform, "pt_transform")
#     io.s3_upload(["pt_transform", "latest.csv"], bucket_mod, f"{prefix}/data")
#     return df


def build_fit(BuildClass, data, y_train, y_test, test_idx, model_path=None):
    builder = BuildClass(data.X_train, y_train, data.X_test, y_test, test_idx=test_idx)
    builder.build()
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
        clf = build_fit(
            MemoryClassifier,
            data,
            data.y_bin_train,
            data.y_bin_test,
            data.bin_test_idx,
            model_path=model_path,
        )
        bcom = compute_cache(clf, res_path=f"{res_path}/mem_bin")
        model_objects["mem_clf"] = {"builder": clf, "results": bcom}
    if mem_reg:
        mem = build_fit(
            MemoryRegressor,
            data,
            data.y_mem_train,
            data.y_mem_test,
            data.mem_test_idx,
            model_path=model_path,
        )
        mcom = compute_cache(mem, res_path=f"{res_path}/memory")
        model_objects["mem_reg"] = {"builder": mem, "results": mcom}
    if wall_reg:
        wall = build_fit(
            WallclockRegressor,
            data,
            data.y_wall_train,
            data.y_wall_test,
            data.wall_test_idx,
            model_path=model_path,
        )
        wcom = compute_cache(wall, res_path=f"{res_path}/wallclock")
        model_objects["wall_reg"] = {"builder": wall, "results": wcom}
    return model_objects


def parse_user_args(args):
    # import and preprocess data
    fpath = None
    model_path = None
    res_path = os.path.join(os.getcwd(), "results")
    if args.source_path:
        fpath = find_data_sources(
            args.source_path, fname=args.fname, date_key=args.date_key
        )
        if args.overwrite:
            model_path = args.source_path  # auto creates "models" subdir
            res_path = os.path.join(args.source_path, "results")
    data = load_prep(date_key=args.date_key, fpath=fpath)

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
        "--archive_date",
        type=str,
        default="2021-11-04",
        help="date key of the archived dataset in the spacekit collection (defaults to most recent)",
    )  # data/2021-11-04-1636048291/data
    parser.add_argument(
        "--source_path",
        type=str,
        default=None,
        help="top level (parent) directory of source data (absolute path or relative to current working directory, e.g. `data/2021-11-04-1636048291/` or just `data`",
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
        "--overwrite",
        type=int,
        default=0,
        help="overwrite saved results and models on disk",
    )
    parse_user_args(parser.parse_args())
