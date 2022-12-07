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

__name__ = "HST CAL MODEL TRAINING"

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


class Train:
    def __init__(
        self,
        project="standard",
        date_key=None,
        fpath=None,
        train_path=None,
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
        self.project = project
        self.date_key = date_key
        self.fpath = fpath  # data/timestamp
        self.train_path = train_path
        self.model_path = model_path
        self.res_path = res_path
        self.mem_clf = mem_clf
        self.mem_reg = mem_reg
        self.wall_reg = wall_reg
        self.cross_val = cross_val
        self.njobs = njobs
        self.ddb_table = ddb_table
        self.s3_bucket = s3_bucket
        self.data = None
        self.model_objs = {}
        self.save_csv = None
        self.dict_preds = dict(
            mem_clf="bin_pred", mem_reg="mem_pred", wall_reg="wall_pred"
        )

    def main(self):
        # TODO error handling / log msg
        if self.fpath is None:
            return
        if self.train_path is None:
            self.train_path = os.path.dirname(os.path.dirname(self.fpath))

        self.save_csv = f"{self.train_path}/data/batch.csv"
        # prep data
        self.load_prep(date_key=self.date_key, fpath=self.fpath)

        # TODO: kfold cross-validation
        mods = dict(mem_bin=self.mem_clf, memory=self.mem_reg, wallclock=self.wall_reg)
        modelnames = [k for k, v in mods.items() if v == 1]

        if self.cross_val == "only":
            # run_kfold, skip training
            run_kfold(
                self.data,
                s3=self.s3_bucket,
                data_path=self.train_path,
                models=modelnames,
                n_jobs=self.njobs,
            )
        else:
            # build, train and evaluate models
            self.train_models()  # -> self.model_objs
            self.generate_preds(save_csv=self.save_csv)
            self.wallclock_stats(self.data.data, save_csv=self.save_csv)
            self.upload_results()

            if self.cross_val == "skip":
                print("Skipping KFOLD")
            else:
                run_kfold(
                    self.data,
                    s3=self.s3_bucket,
                    data_path=self.train_path,
                    models=modelnames,
                    n_jobs=self.njobs,
                )

    def load_prep(self):
        df = load(name="calcloud", date_key=self.date_key, fpath=self.fpath)
        self.data = CalPrep(df, "mem_bin")
        self.data.prep_data()

    # TODO: SVM repro model training - can use same builder objects, just set diff hyperparams after instantiating (add params to builder.blueprints)

    def train_models(self):
        if self.mem_clf:
            clf = self.build_fit(
                MemoryClassifier,
                self.data.y_bin_train,
                self.data.y_bin_test,
                self.data.test_idx,
            )
            bcom = self.compute_cache(clf, res_path=f"{self.res_path}/mem_bin")
            self.model_objects["mem_clf"] = {"builder": clf, "results": bcom}
        if self.mem_reg:
            mem = self.build_fit(
                MemoryRegressor,
                self.data.y_mem_train,
                self.data.y_mem_test,
                self.data.test_idx,
            )
            mcom = self.compute_cache(mem, res_path=f"{self.res_path}/memory")
            self.model_objects["mem_reg"] = {"builder": mem, "results": mcom}
        if self.wall_reg:
            wall = self.build_fit(
                WallclockRegressor,
                self.data.y_wall_train,
                self.data.y_wall_test,
                self.data.test_idx,
            )
            wcom = self.compute_cache(wall, res_path=f"{self.res_path}/wallclock")
            self.model_objects["wall_reg"] = {"builder": wall, "results": wcom}

    def build_fit(self, BuildClass, y_train, y_test, test_idx):
        builder = BuildClass(
            self.data.X_train, y_train, self.data.X_test, y_test, test_idx=test_idx
        )
        builder.build()
        builder.fit()
        if self.model_path:
            builder.save_model(output_path=self.model_path)
        return builder

    def compute_cache(self, builder, res_path=None):
        if res_path is None:
            res_path = self.res_path if self.res_path else "./results"
        if builder.algorithm == "linreg":
            com = ComputeRegressor(builder=builder, res_path=res_path)
            com.calculate_results()
        elif builder.algorithm == "multiclass":
            com = ComputeMulti(builder=builder, res_path=res_path)
            com.calculate_multi()
        outputs = com.make_outputs()
        print(outputs.keys())
        return com

    def generate_preds(self, save_csv=None):
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
        if self.data.test_idx:
            if "split" not in self.data.data.columns:
                self.data.data["split"] = "train"
            self.data.data.loc[self.data.test_idx.index, "split"] = "test"
            self.tt_pred()
        else:
            print("Warning: test_idx attribute not found.")
            self.xy_pred()
        if save_csv:
            self.save_csv_file(self.data.data, index_col="ipst", save_to=save_csv)

    def save_csv_file(self, df, index_col="ipst", save_to="."):
        df[index_col] = df.index
        df.to_csv(save_to, index=False)

    def xy_pred(self):
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
        for m, c in self.dict_preds.items():
            if c == "bin_pred":
                self.data.data[c] = np.argmax(
                    self.model_objects[m]["builder"].model.predict(
                        self.data.data[COLUMN_ORDER]
                    ),
                    axis=-1,
                )
            else:
                self.data.data[c] = self.model_objects[m]["builder"].model.predict(
                    self.data.data[COLUMN_ORDER]
                )

    def tt_pred(self):
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
        test = self.data.data.loc[self.data.data["split"] == "test"]
        train = self.data.data.loc[self.data.data["split"] == "train"]
        for m, c in self.dict_preds.items():
            if c == "bin_pred":
                test[c] = np.argmax(self.model_objects[m]["results"].y_pred, axis=-1)
                train[c] = np.argmax(
                    self.model_objects[m]["builder"].model.predict(train[COLUMN_ORDER]),
                    axis=-1,
                )
            else:
                test[c] = self.model_objects[m]["results"].y_pred
                train[c] = self.model_objects[m]["builder"].model.predict(
                    train[COLUMN_ORDER]
                )
        for c in self.dict_preds.values():
            self.data.data.loc[self.data.data[test.index], c] = test[c]
            self.data.data.loc[self.data.data[train.index], c] = train[c]

    def wallclock_stats(self, df, save_csv=None):
        """Recorded wallclock time across groups of datasets with otherwise identical inputs
        have been shown in some cases (mostly ACS/WFC) to exhibit significantly large variance.
        This limits the model's ability to make accurate predictions for these subsets.
        To overcome this limitation, this method calculates comparative statistics for wallclock predictions
        generated by the model vs. actual recorded wallclock times (ground truth).

        The dataframe returned by the function is based on a nested dictionary of key:value pairs.
        The keys are a list of each unique wallclock prediction in the dataset; we then group the data
        into subsets according to these y_pred values. The next layer of key:value pairs consists of
        the calculated mean, standard deviation, and standard error of the ground truth wallclock times
        for that subset. The dictionary is converted to a dataframe and merged with the original input
        dataset.

        When the model makes a prediction for a new dataset it hasn't seen before, we can use what we know about
        the statistics of past predictions to adjust the maximum allowed wallclock time for processing based on
        a previous margin of error.

        NOTE: Additional features derived from the raw dataset inputs might explain the differentiation
        between otherwise identical metadata, and thereby resolve the variance problem. Preliminary analysis
        of these subsets indicates the likely source of distinction relates to target field content of the image.
        For example, scenes with a large number of bright sources tend to take much longer to process than those
        with fewer bright sources. Unfortunately, identifying these cases in advance, without opening the
        fits files and looking at the images, is currently not feasible. In the mean time, this method
        has proved to be a sufficient workaround for HST production pipeline calibration reprocessing.

        Parameters
        ----------
        df : pandas.DataFrame
            input dataset with ground truth wallclock times and wallclock prediction values for each ipppssoot.
        save_csv : str or path, optional
            where to save the dataframe with added wc stat info as a csv file on local disk, by default None
        """
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
        self.data.data = df.join(df_stats, how="left")
        if save_csv:
            self.save_csv_file(self.data.data, index_col="ipst", save_to=save_csv)

    def upload_results(self):
        if self.ddb_table:
            dataset_path = os.path.join(self.train_path, "data", "batch.csv")
            ddb = DynamoDBScraper(table_name=self.ddb_table)
            ddb.batch_ddb_writer(dataset_path)
        if self.s3_bucket:
            model_path = os.path.join(self.train_path, "models")
            pfx = str(os.path.basename(self.train_path))  # the timestamp
            s3 = S3Scraper(self.s3_bucket, pfx=pfx)
            # zip everything in ~/data/timestamp/
            archive = s3.compress_files(self.train_path)
            s3.s3_upload([archive], self.s3_bucket, "archive")
            # create models.zip
            model_zip = s3.compress_files(model_path)
            s3.s3_upload([self.train_path, model_zip], self.s3_bucket, pfx)


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


def parse_user_args(args):
    # import and preprocess data
    fpath = None
    train_path = None
    model_path = None
    res_path = os.path.join(os.getcwd(), "results")
    date_key = args.date_key if args.src == "arch" else None
    ddb_table = args.tablename if args.save_ddb is True else None
    s3_bucket = args.bucket_name if args.save_s3 is True else None

    # TODO: if args.src == "s3":

    if args.src == "ddb":
        attr = dict(
            name=args.attrname,
            method=args.attrmethod,
            type=args.attrtype,
            value=args.attrvalue,
        )
        fpath = scrape_dynamodb(
            args.tablename, timestamp=args.timestamp, fname=args.fname, attr=attr
        )
        train_path = os.path.dirname(os.path.dirname(fpath))
        model_path = train_path  # "models" subdir will be saved here automatically
        res_path = os.path.join(train_path, "results")

    elif args.src == "file":
        fpath = find_local_dataset(
            args.source_path, fname=args.fname, date_key=args.date_key
        )
        if args.overwrite:  # retrain and overwrite existing local data and results
            train_path = (
                model_path
            ) = args.source_path  # "models" subdir will be saved here automatically
            res_path = os.path.join(args.source_path, "results")
        else:
            data_path = os.path.join(make_timestamp_path(args.timestamp), "data")
            os.makedirs(os.path.join(os.path.expanduser("~"), data_path), exist_ok=True)
            train_path = os.path.dirname(os.path.dirname(data_path))

    return dict(
        date_key=date_key,
        fpath=fpath,
        train_path=train_path,
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
    parser.add_argument(
        "--project",
        "-p",
        type=int,
        choices=["cal", "svm", "mvm"],
        default=os.environ.get("PROJECT", "cal"),
    )

    kwargs = parse_user_args(parser.parse_args())
    Train(**kwargs).main()
