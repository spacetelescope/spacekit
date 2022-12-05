"""
Spacekit HST "Calibration in the Cloud" (calcloud) Job Resource Allocation Model Training

This script imports and preprocesses job metadata for the Hubble Space Telescope data calibration pipeline,
which is then used as inputs to build, train and evaluate 3 neural networks for estimating AWS batch compute job resource requirements.

The networks include one multi-class classifier and two linear regression estimators. The classifier predicts which of 4 possible memory bin sizes (and therefore compute instance type) is most appropriate for reprocessing a given ipppssoot (i.e. "job"). The wallclock regressor estimates the maximum execution time ("wallclock" or "kill" time) in seconds needed to complete the job.

Ex:
python -m spacekit.skopes.hst.cal.train data/2021-11-04-1636048291

To load results from disk in a separate session (for plotting, analysis etc):

bcom2 = ComputeMulti(res_path=f"{res_path}/mem_bin")
bin_out = bcom2.upload()
bcom2.load_results(bin_out)
test_idx = bin_out["test_idx"]

"""

from argparse import ArgumentParser
import os
import datetime as dt
import numpy as np
import pandas as pd
from spacekit.datasets import load
from spacekit.extractor.load import find_local_dataset
from spacekit.extractor.scrape import DynamoDBScraper, S3Scraper
from spacekit.preprocessor.prep import CalPrep
from spacekit.builder.architect import (
    MemoryClassifier,
    MemoryRegressor,
    WallclockRegressor,
)
from spacekit.analyzer.compute import ComputeMulti, ComputeRegressor
from spacekit.skopes.hst.cal.validate import run_kfold

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


def make_timestamp_path(timestamp):
    if timestamp == "now":
        train_time = dt.datetime.now()
    elif isinstance(timestamp, str):
        if len(timestamp) <= 14:
            train_time = dt.datetime.fromtimestamp(int(timestamp))
        else:
            train_time = dt.datetime.fromisoformat(timestamp)
    elif isinstance(timestamp, int) or isinstance(timestamp, float):
        train_time = dt.datetime.fromtimestamp(timestamp)
    else:
        print(
            f"Timestamp type must be a string (datetime, isoformat) or int/float (timestamp). You passed {type(timestamp)}."
        )
        raise ValueError
    t0 = train_time.timestamp()
    data_path = f"{dt.date.fromtimestamp(t0).isoformat()}-{str(int(t0))}"
    return data_path


def scrape_dynamodb(table_name, timestamp="now", fname=None, attr={}):
    fname = "batch.csv" if fname is None else fname
    data_path = os.path.join(make_timestamp_path(timestamp), "data")
    os.makedirs(os.path.join(os.path.expanduser("~"), data_path), exist_ok=True)

    ddb = DynamoDBScraper(
        table_name,
        attr=attr,
        fname=fname,
        cache_dir="~",
        cache_subdir=data_path,
        format="zip",
        extract=True,
        clean=True,
    )
    ddb.ddb_download()
    ddb.write_to_csv()
    return ddb.fpath


def load_prep(date_key=None, fpath=None):
    df = load(name="calcloud", date_key=date_key, fpath=fpath)
    data = CalPrep(df, "mem_bin")
    data.prep_data()
    return data


def build_fit(BuildClass, data, y_train, y_test, test_idx, model_path=None):
    builder = BuildClass(data.X_train, y_train, data.X_test, y_test, test_idx=test_idx)
    builder.build()
    builder.fit()
    # TODO shouldn't default be to save even if model_path=None?
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


def save_csv_file(df, index_col="ipst", save_to="."):
    df[index_col] = df.index
    df.to_csv(save_to, index=False)
    # df.set_index(index_col, inplace=True)


# TODO: SVM repro model training - can use same builder objects, just set diff hyperparams after instantiating
# mosaic=None, mosaic="svm", mosaic="mvm"
# add params to builder.blueprints


def train_models(data, res_path, mem_clf=1, mem_reg=1, wall_reg=1, model_path=None):
    model_objects = {}
    if mem_clf:
        clf = build_fit(
            MemoryClassifier,
            data,
            data.y_bin_train,
            data.y_bin_test,
            data.test_idx,
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
            data.test_idx,
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
            data.test_idx,
            model_path=model_path,
        )
        wcom = compute_cache(wall, res_path=f"{res_path}/wallclock")
        model_objects["wall_reg"] = {"builder": wall, "results": wcom}
    return model_objects


def xy_pred(df, model_objects, dict_preds):
    """Generates predictions for the entire dataset. This is a fallback for cases
    where the test index from training is unknown.

    Parameters
    ----------
    df : pandas.DataFrame
        _description_
    model_objects : dict
        dictionary of spacekit.compute.Computer subclass objects
    dict_preds : dict
        key-value pairs of the model names and associated target classes

    Returns
    -------
    pandas.DataFrame
        training data with y_preds for each model type included as '{target}_pred' columns.
    """
    for m, c in dict_preds.items():
        if c == "bin_pred":
            df[c] = np.argmax(
                model_objects[m]["builder"].model.predict(df[COLUMN_ORDER]), axis=-1
            )
        else:
            df[c] = model_objects[m]["builder"].model.predict(df[COLUMN_ORDER])
    return df


def tt_pred(df, model_objects, dict_preds):
    """Generates predictions for the training set and combines these
    with test set predictions already recorded during training.

    Parameters
    ----------
    df : pandas.DataFrame
        _description_
    model_objects : dict
        dictionary of spacekit.compute.Computer subclass objects
    dict_preds : dict
        key-value pairs of the model names and associated target classes

    Returns
    -------
    pandas.DataFrame
        training data with y_preds for each model type included as '{target}_pred' columns.
    """
    test = df.loc[df["split"] == "test"]
    train = df.loc[df["split"] == "train"]
    for m, c in dict_preds.items():
        if c == "bin_pred":
            test[c] = np.argmax(model_objects[m]["results"].y_pred, axis=-1)
            train[c] = np.argmax(
                model_objects[m]["builder"].model.predict(train[COLUMN_ORDER]), axis=-1
            )
        else:
            test[c] = model_objects[m]["results"].y_pred
            train[c] = model_objects[m]["builder"].model.predict(train[COLUMN_ORDER])
    for c in dict_preds.values():
        df.loc[df[test.index], c] = test[c]
        df.loc[df[train.index], c] = train[c]
    return df


def generate_preds(data, model_objects, save_csv=None):
    """_summary_

    Parameters
    ----------
    data : spacekit.preprocessor.prep.CalPrep object
        _description_
    model_objects : dict
        dictionary of spacekit.compute.Computer subclass objects
    save_csv : string or path, optional
        local path to save dataframe with new predictions, by default None

    Returns
    -------
    _type_
        _description_
    """
    dict_preds = dict(mem_clf="bin_pred", mem_reg="mem_pred", wall_reg="wall_pred")
    if data.test_idx:
        if "split" not in data.data.columns:
            data.data["split"] = "train"
        data.data.loc[data.test_idx.index, "split"] = "test"
        data.data = tt_pred(data.data, model_objects, dict_preds)
    else:
        print("Warning: test_idx attribute not found.")
        data.data = xy_pred(data.data, model_objects, dict_preds)
    if save_csv:
        save_csv_file(data.data, index_col="ipst", save_to=save_csv)
    return data.data


def wallclock_stats(df, save_csv=None):
    cols = ["wc_mean", "wc_std", "wc_err"]
    drop_cols = [col for col in cols if col in df.columns]
    df = df.drop(drop_cols, axis=1)

    dfw = df[["wall_pred", "wallclock"]]
    wc_dict = {}
    wc_stats = {}
    wc_preds = list(df["wall_pred"].unique())
    for p in wc_preds:
        wc_dict[p] = {}
        wall = df.loc[df.wall_pred == p]["wallclock"]
        std = np.std(wall)
        wc_dict[p]["wc_mean"] = np.mean(wall)
        wc_dict[p]["wc_std"] = std
        wc_dict[p]["wc_err"] = std / np.sqrt(len(wall))
    for idx, row in dfw.iterrows():
        wc_stats[idx] = {}
        wp = row["wall_pred"]
        if wp in wc_dict:
            wc_stats[idx]["wc_mean"] = wc_dict[wp]["wc_mean"]
            wc_stats[idx]["wc_std"] = wc_dict[wp]["wc_std"]
            wc_stats[idx]["wc_err"] = wc_dict[wp]["wc_err"]
    df_stats = pd.DataFrame.from_dict(wc_stats, orient="index")
    df_new = df.join(df_stats, how="left")
    if save_csv:
        save_csv_file(df_new, index_col="ipst", save_to=save_csv)
    return df_new


def upload_results(train_path, ddb_table=None, s3_bucket=None):
    if ddb_table:
        dataset_path = os.path.join(train_path, "data", "batch.csv")
        ddb = DynamoDBScraper(table_name=ddb_table)
        ddb.batch_ddb_writer(dataset_path)
    if s3_bucket:
        model_path = os.path.join(train_path, "models")
        pfx = str(os.path.basename(train_path))  # the timestamp
        s3 = S3Scraper(s3_bucket, pfx=pfx)
        # zip everything in ~/data/timestamp/
        archive = s3.compress_files(train_path)
        s3.s3_upload([archive], s3_bucket, "archive")
        # create models.zip
        model_zip = s3.compress_files(model_path)
        s3.s3_upload([train_path, model_zip], s3_bucket, pfx)


def main(
    date_key=None,
    fpath=None,  # data/timestamp
    model_path=None,
    res_path=None,
    mem_clf=1,
    mem_reg=1,
    wall_reg=1,
    cross_val=None,
    njobs=-2,
    ddb_table=None,
    s3_bucket=None,
):
    # prep data
    data = load_prep(date_key=date_key, fpath=fpath)

    # TODO: kfold cross-validation
    mods = dict(mem_bin=mem_clf, memory=mem_reg, wallclock=wall_reg)
    modelnames = [k for k, v in mods.items() if v == 1]
    train_path = os.path.dirname(os.path.dirname(fpath))
    if cross_val == "only":
        # run_kfold, skip training
        run_kfold(
            data, s3=s3_bucket, data_path=train_path, models=modelnames, n_jobs=njobs
        )
    else:
        # build, train and evaluate models
        model_objs = train_models(
            data,
            res_path,
            mem_clf=mem_clf,
            mem_reg=mem_reg,
            wall_reg=wall_reg,
            model_path=model_path,
        )

        train_path = os.path.dirname(os.path.dirname(fpath))
        data.data = generate_preds(
            data, model_objs, save_csv=f"{train_path}/data/batch.csv"
        )
        data.data = wallclock_stats(data.data, save_csv=f"{train_path}/data/batch.csv")

        upload_results(train_path, ddb_table=ddb_table, s3_bucket=s3_bucket)

        if cross_val == "skip":
            print("Skipping KFOLD")
        else:
            run_kfold(
                data,
                s3=s3_bucket,
                data_path=train_path,
                models=modelnames,
                n_jobs=njobs,
            )


def parse_user_args(args):
    # import and preprocess data
    fpath = None
    model_path = None
    res_path = os.path.join(os.getcwd(), "results")
    date_key = args.date_key if args.src == "arch" else None
    ddb_table = args.tablename if args.save_ddb is True else None
    s3_bucket = args.bucket_name if args.save_s3 is True else None
    train_path = None

    # TODO: if args.src == "s3":

    if args.src == "ddb":
        attr = dict(
            name=args.attrname,
            method=args.attrmethod,
            type=args.attrtype,
            value=args.attrvalue,
        )
        fpath = scrape_dynamodb(args.tablename, fname=args.fname, attr=attr)
        train_path = os.path.dirname(os.path.dirname(fpath))
        model_path = train_path  # "models" subdir will be saved here automatically
        res_path = os.path.join(train_path, "results")

    elif args.src == "file":
        fpath = find_local_dataset(
            args.source_path, fname=args.fname, date_key=args.date_key
        )
        if args.overwrite:
            model_path = (
                args.source_path
            )  # "models" subdir will be saved here automatically
            res_path = os.path.join(args.source_path, "results")

    return dict(
        date_key=date_key,
        fpath=fpath,
        model_path=model_path,
        res_path=res_path,
        mem_clf=args.mem_clf,
        mem_reg=args.mem_reg,
        wall_reg=args.wall_reg,
        cross_val=args.cross_val,
        njobs=args.njobs,
        ddb_table=ddb_table,
        s3_bucket=s3_bucket,
    )


if __name__ == "__main__":
    parser = ArgumentParser(prog="spacekit hst calibration model training")
    parser.add_argument(
        "src",
        type=str,
        choices=["ddb", "s3", "arch", "file"],
        help="ddb:dynamodb, s3:aws bucket, arch:spacekit archive, file:local csv file",
    )

    # ddb (Dynamo DB)
    parser.add_argument(
        "--tablename", type=str, default=os.environ.get("DDBTABLE", "calcloud-model-sb")
    )
    parser.add_argument(
        "--attrname", type=str, default=os.environ.get("ATTRNAME", "None")
    )
    parser.add_argument(
        "--attrmethod", type=str, default=os.environ.get("ATTRMETHOD", "None")
    )
    parser.add_argument(
        "--attrtype", type=str, default=os.environ.get("ATTRTYPE", "None")
    )
    parser.add_argument(
        "--attrvalue", type=str, default=os.environ.get("ATTRVAL", "None")
    )

    # s3 (amazon s3 bucket)
    parser.add_argument(
        "--bucketname", default=os.environ.get("S3MOD", "calcloud-modeling-sb")
    )

    # arch (spacekit collection archive dataset)
    parser.add_argument(
        "--date_key",
        "-d",
        type=str,
        default="2021-11-04",
        help="YYYY-MM-DD date key if retraining archived data from the spacekit collection (src=arch).Defaults to most recent)",
    )  # data/2021-11-04-1636048291/data

    # file (local csv file)
    parser.add_argument(
        "--source_path",
        type=str,
        default=None,
        help="if src=file, top level (parent) directory of source data (absolute path or relative to current working directory, e.g. `data/2021-11-04-1636048291/` or just `data`",
    )

    # Optional args for any data source
    parser.add_argument(
        "--timestamp",
        "-t",
        type=str,
        default="now",
        help="timestamp to record for this training iteration. Default is `now` (current timestamp at runtime).",
    )
    parser.add_argument(
        "--fname", type=str, default=None, help="name of training data csv file"
    )
    parser.add_argument(
        "--mem_clf",
        type=int,
        default=1,
        choices=[0, 1],
        help="bool: train memory bin classifier",
    )
    parser.add_argument(
        "--mem_reg",
        type=int,
        default=1,
        choices=[0, 1],
        help="bool: train memory regressor",
    )
    parser.add_argument(
        "--wall_reg",
        type=int,
        default=1,
        choices=[0, 1],
        help="bool: train wallclock regressor",
    )
    parser.add_argument(
        "--cross_val",
        "-k",
        choices=["only", "skip", "None", None],
        default=os.environ.get("KFOLD", None),
    )
    parser.add_argument("--njobs", "-j", default=int(os.environ.get("NJOBS", -2)))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If using a local file source, overwrite saved results and models on disk. Use with caution.",
    )
    parser.add_argument("--save_ddb", action="store_true")
    parser.add_argument("--save_s3", action="store_true")
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2], default=os.environ.get("VERBOSE", 0)
    )

    kwargs = parse_user_args(parser.parse_args())
    main(**kwargs)
