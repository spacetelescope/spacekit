import os
import re
import json
from argparse import ArgumentParser
import datetime as dt
import pandas as pd
import numpy as np
from collections import OrderedDict
from pprint import pprint
from sklearn.model_selection import KFold
from spacekit.preprocessor.prep import JwstCalPrep
from spacekit.builder.architect import BuilderMLP
from spacekit.analyzer.compute import ComputeRegressor
from spacekit.logger.log import SPACEKIT_LOG, Logger


class JwstCalTrain:

    def __init__(self, training_data=None, out=None, exp_mode="image", norm=1, cross_val=0, early_stopping=None, threshold=100, **log_kws):
        """Training can be run in 1 of 2 modes: default or kfold cross-validation.

        Default: Single training iteration run on the dataset using a randomized 80-20 train-test split

        Cross-validation: Training is run on the dataset K times, where k is the number of randomly shuffled train-test split folds 
        (initialized by the value passed into `cross_val` kwarg). Additional metrics are accumulated and recorded for evaluating 
        overall model performance across iterations.

        Parameters
        ----------
        training_data : str or Path, optional
            path on local disk to directory where training data files are stored, by default None
        out : str or Path, optional
            path on local disk for saving outputs (leave blank to auto set using timestamp), by default None
        exp_mode : str, optional
            specifies which model to train ('image' or 'spec'), by default "image"
        norm : int, optional
            apply normalization and scaling, by default 1 (True)
        cross_val : int, optional
            Run cross-validation using k number of folds (10 recommended), by default 0
        early_stopping : str, optional
            Either 'val_loss' or 'val_rmse' ends training when this metric is no longer improving, by default None
        threshold : int, optional
            minimum value designating an observation's target value being classified as large/high, by default 100
        """
        self.training_data = training_data
        self.exp_mode = exp_mode.lower()
        self.set_outpath(out=out)
        self.prep_kwargs = dict(exp_mode=self.exp_mode, normalize=norm)
        self.cross_val = cross_val
        self.early_stopping = early_stopping
        self.threshold = threshold
        self.builder = None
        self.data = None
        self.itn = ""
        self.jp = None
        self.metrics = None
        self.dm = None
        self.scores = None
        self.metrics_file = f"{self.exp_mode}-cv-metrics.csv"
        self.__name__ = "JwstCalTrain"
        self.log = Logger(self.__name__, **log_kws).setup_logger(logger=SPACEKIT_LOG)
        self.log_kws = dict(log=self.log, **log_kws)
        self.initialize()

    def __str__(self):
        attrs = dict(
            output_path=self.output_path,
            prep_kwargs=self.prep_kwargs,
            threshold=self.threshold,
            cross_val=self.cross_val, 
        )
        string = f"JwstCalTrain attributes:\n\n"
        for k,v in attrs.items():
            string += f"\n\t{k}: {v}"
        if self.builder is not None:
            params=[self.builder.get_build_params(), self.builder.get_fit_params()]
            string += "\n\nModel Parameters:\n"
            for p in params:
                for k,v in p.items():
                    string += f"\n\t{k}: {v}"
        return string

    def set_outpath(self, out=None):
        """Sets up the output directory for model training data, models and results.
        If `out` is formatted as a {date}_{timestamp} string, it becomes the output_path attr and
        a directory is created if it doesn't exist. Otherwise, `out` is used as the top directory,
        under which a new {date}_{timestamp} folder is created using the current date/time. 

        Parameters
        ----------
        out : str, optional
            name of directory in which to save training outputs, by default None
        """
        base_out = "." if out is None else out
        if out:
            date_timestamp_re = re.compile('^[0-9]{4}\-[0-9]{2}\-[0-9]{2}\_[0-9]{10}')
            match_existing = date_timestamp_re.match(str(os.path.basename(out)))
            if match_existing:
                self.output_path = os.path.join(base_out, self.exp_mode)
                if not os.path.exists(self.output_path):
                    os.makedirs(self.output_path, exist_ok=True)
                return
        today = dt.date.today().isoformat()
        timestamp = str(dt.datetime.now().timestamp()).split('.')[0]
        self.output_path = f"{base_out}/{today}_{timestamp}/{self.exp_mode}"
        os.makedirs(self.output_path, exist_ok=True)
    
    def initialize(self):
        """Creates output path subdirectories. If running cross validation, the training iteration metrics file 
        is also loaded and instantiated into the `self.metrics` attribute, with `self.itn` initialized to a value 1 higher
        than the length of iterations in the file so far.
        """
        global DATA, MODELS, RESULTS, SUMMARY
        DATA = os.path.join(self.output_path, "data")
        MODELS = os.path.join(self.output_path, "models")
        RESULTS = os.path.join(self.output_path, "results")
        SUMMARY = os.path.join(self.output_path, "summary")
        for p in [DATA, MODELS, RESULTS, SUMMARY]:
            os.makedirs(p, exist_ok=True)
        self.metrics_file = f"{SUMMARY}/" + self.metrics_file
        self.load_metrics()
        if self.metrics is not None:
            self.itn = str(len(list(self.metrics.keys())))

    def load_data(self, tts=None):
        """Loads the training dataset from csv file on local disk.
        Expects the following file naming convention and directory location:
        self.training_data / train-{exp_mode}.csv

        If running cross validation (tts): self.output_path / data / {exp_mode}-tts_{tts}.csv

        Parameters
        ----------
        tts : str or int, optional
            train test split iteration count (integer), by default None
        """
        if tts:
            fpath = f"{DATA}/{self.exp_mode}-tts_{tts}.csv"
        else:
            fpath = os.path.join(self.training_data, f"train-{self.exp_mode}.csv")
        if not os.path.exists(fpath):
            self.log.error(f"Training data filepath not found: {fpath}")
            return
        self.data = pd.read_csv(fpath, index_col="Dataset")
        ndupes = len(self.data.loc[self.data.duplicated(subset='pname')])
        if ndupes > 0:
            self.log.info(f"Dropping {ndupes} duplicates")
            self.data = self.data.sort_values(by=['date', 'imagesize']).drop_duplicates(subset='pname', keep='last')

    def generate_kfolds(self):
        """Generates k number of train-test splits and saves them to disk to be loaded at each training iteration.
        The rationale for this approach is based on the fact that the dataset being used to train these models
        tends to evolve over time (due to changes in jwst calibration code) and it is sometimes helpful to know which
        observations were used in training vs test when evaluating a model's performance.
        """
        kfold = KFold(n_splits=self.cross_val, shuffle=True)
        itn = 0
        for _, test_idx in kfold.split(np.zeros(self.data.shape[0]), pd.concat(self.jp.get_y_train_test('imgsize_gb'), axis=0)):
            self.data['split'] = 'train'
            self.data.loc[self.data.iloc[test_idx].index, 'split'] = 'test'
            self.data['Dataset'] = self.data.index
            self.data.to_csv(f"{DATA}/{self.exp_mode}-tts_{str(itn)}.csv", index=False)
            itn += 1

    def run_cross_val(self, custom_arch=None):
        """Loops through k number of train/test splits to load data, run training, and record model performance metrics on each iteration.
        Requires attribute `cross_val` to be greater than 0 in order to run (standard practice is k=10)
        """
        for i in list(range(self.cross_val)):
            save_diagram = True if i == 0 else False
            self.data = None
            self.load_train_test(tts=str(i))
            self.run_training(save_diagram=save_diagram, custom_arch=custom_arch)
            self.compute_cache()
        self.scores = {}
        self.dm = pd.DataFrame.from_dict(self.metrics)
        for m in list(self.dm.index):
            self.scores[m] = np.average(self.dm.loc[self.dm.index == m].values[0])
        pprint(self.scores)
        with open(f"{SUMMARY}/{self.exp_mode}-cv-scores.json", "w") as j:
            json.dump(self.scores, j)
            

    def prep_train_test(self):
        """Loads and splits training dataset into train and test sets.
        """
        if self.data is None:
            self.load_data()
        self.jp = JwstCalPrep(self.data, **self.prep_kwargs)
        self.jp.prep_data()
        self.jp.prep_targets()
 
    def load_train_test(self, tts="0"):
        """Loads the training data and splits into train and test sets. 

        Parameters
        ----------
        tts : str or int, optional
            train test split iteration count (integer), by default "0"
        """
        if self.data is None:
            self.load_data(tts=tts)
        self.jp = JwstCalPrep(self.data, **self.prep_kwargs)
        self.jp.prep_data(existing_splits=True)
        self.jp.prep_targets()
        if isinstance(tts, int):
            tts = str(tts)
        if tts != self.itn:
            self.itn = tts

    @property
    def architecture(self):
        return dict(
            image="jwst_img3_reg",
            spec="jwst_spec3_reg"
        )[self.exp_mode]
    
    def build_model(self, custom_arch=None):
        """Build the functional model using standard blueprint. Optionally customize hyperparameters using `custom_arch`.
        Ex:
        custom_arch=dict(
        build_params=dict(layers=[18, 36, 72, 144, 288, 144, 72, 36, 18],
            activation="relu",
            cost_function="linear",
            lr_sched=True,
            optimizer=Adam),
        fit_params=dict(batch_size=32,
            epochs=2000,
            lr=1e-4,
            decay=[100000, 0.96],
            early_stopping=None,
            verbose=0)
        )

        Parameters
        ----------
        custom_arch : dict, optional
            nested dict with keys `build_params`, `fit_params` as a way of tuning hyperparameters, by default None
        """
        self.builder = BuilderMLP(
            X_train=self.jp.X_train,
            y_train=self.jp.y_reg_train,
            X_test=self.jp.X_test,
            y_test=self.jp.y_reg_test,
            blueprint="mlp",
        )
        draft = self.builder.get_blueprint(self.architecture)
        if custom_arch is not None:
            try:
                if "build_params" in custom_arch.keys():
                    build_params = draft.building().update(custom_arch["build_params"])
                    self.builder.set_build_params(**build_params)
                if "fit_params" in custom_arch.keys():
                    fit_params = draft.fitting().update(custom_arch["fit_params"])
                    self.builder.fit_params(**fit_params)
            except Exception as e:
                self.log.error(f"Custom architecture must be a nested dict with keys \
                               `build_params` and `fit_params` using standard key-value pairs \
                               for building and fitting a model. Logged Exception:\n {e}")
        self.builder.model = self.builder.build()

    def run_training(self, save_diagram=True, custom_arch=None):
        """Build, train and save a model

        Parameters
        ----------
        save_diagram : bool, optional
            Save a png diagram image of the model architecture, by default True
        custom_arch : dict, optional
            pass in custom build_params and fit_params as a way of tuning hyperparameters, by default None
        """
        self.build_model(custom_arch=custom_arch)
        if save_diagram is True:
            self.builder.model_diagram(output_path=MODELS, show_layer_names=True)
        self.builder.early_stopping = self.early_stopping
        self.builder.fit()
        self.builder.save_model(output_path=MODELS, parent_dir=self.itn)

    def compute_cache(self):
        """Generates and stores model training results 
        """
        self.builder.test_idx = list(self.jp.test_idx)
        it = "" if not self.itn else f"/{self.itn}"
        self.com = ComputeRegressor(
            builder=self.builder,
            algorithm="linreg",
            res_path=RESULTS+it,
            show=True,
            validation=False,
        )
        self.com.calculate_results()
        _ = self.com.make_outputs()
        self.res_fig = self.com.resid_plot(desc=f"{self.exp_mode} tts_{self.itn}")
        self.loss_fig = self.com.keras_loss_plot(desc=f"{self.exp_mode} tts_{self.itn}")
        self.record_metrics()

    def load_metrics(self):
        """Loads a csv file from local disk (if it exists) to record results of subsequent model training iterations.
        """
        if os.path.exists(self.metrics_file):
            self.dm = pd.read_csv(self.metrics_file, index_col="index")
            self.metrics = self.dm.to_dict()

    @property
    def itr_metrics(self):
        """Descriptive statistics about the training and test sets:
        tr_size/ts_size : size of the training/test set
        tr_lrg_ct/ts_lrg_ct : count of datasets over 100gb in train/test (informs target class balance)
        tr_lrg_mean/ts_lrg_mean: mean (average) of targets over 100gb in train/test
        tr_lrg_max/ts_lrg_max: largest target value dataset in train/test

        Returns
        -------
        collections.OrderedDict
            Additional metadata including descriptive statistics about the training and test sets.  
        """
        return OrderedDict(
            tr_size=0,
            ts_size=0,
            tr_lrg_ct=0,
            ts_lrg_ct=0,
            tr_lrg_mean=0,
            ts_lrg_mean=0,
            tr_lrg_max=0,
            ts_lrg_max=0
        )

    def record_metrics(self):
        """Generates a dataframe of model scores for each training iteration then saves it as a .csv file on local disk.
        """
        itr_metrics = self.itr_metrics.copy()
        dd = dict(
            train=self.jp.data.loc[self.jp.data.split == 'train'],
            test=self.jp.data.loc[self.jp.data.split == 'test']
        )
        for s, g in dict(zip(['tr', 'ts'],['train', 'test'])).items():
            itr_metrics[f'{s}_size'] = dd[g].shape[0]
            itr_metrics[f'{s}_lrg_ct'] = dd[g].loc[dd[g].imgsize_gb>self.threshold].shape[0]
            itr_metrics[f'{s}_lrg_mean'] = dd[g].loc[dd[g].imgsize_gb>self.threshold].imgsize_gb.mean()
            itr_metrics[f'{s}_lrg_max'] = dd[g].loc[dd[g].imgsize_gb>self.threshold].imgsize_gb.max()

        itr_metrics.update(self.com.loss)
        if self.metrics is None:
            self.itn = "0"
            self.metrics = {self.itn:itr_metrics}
        else:
            self.itn = str(len(list(self.metrics.keys())))
            self.metrics[self.itn] = itr_metrics
        dm = pd.DataFrame.from_dict(self.metrics)
        dm['index'] = dm.index
        dm.to_csv(self.metrics_file, index=False)
        self.dm = dm.drop('index', axis=1, inplace=True)

    def main(self):
        """Main calling function used to run the full script of preprocessing, training, 
        cross-validation (if selected), and model evaluation scoring.
        """
        self.prep_train_test()
        if self.cross_val > 0:
            self.generate_kfolds()
            self.run_cross_val()
        else:
            self.run_training(save_diagram=True)
            self.compute_cache()
        with open(f"{SUMMARY}/{self.exp_mode}-hyperparameters.txt", "w") as f:
            f.write(str(self))


if __name__ == "__main__":
    parser = ArgumentParser(prog="spacekit hst calibration model training")
    parser.add_argument(
        "-d",
        "--training_data",
        type=str,
        default=None,
        help="path on local disk to directory where training data files are stored",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="path on local disk for saving training results, saved models, and train-test splits",
    )
    parser.add_argument(
        "-e",
        "--exp_mode",
        choices=["image", "spec"],
        default="image",
        help="image: train image model, spec: train spec model"
    )
    parser.add_argument(
        "-n",
        "--norm",
        type=int,
        default=1,
        help="apply normalization and scaling (bool)",
    )
    parser.add_argument(
        "-k",
        "--cross_val",
        type=int,
        default=0,
        help="Run k-fold cross-validation on k number of training iterations (standard is 10)",
    )
    parser.add_argument(
        "-s",
        "--early_stopping",
        type=str,
        default=None,
        help="Either 'val_loss' or 'val_rmse' ends training when this metric is no longer improving"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=100,
        help="minimum value designating an observation's target value being classified as large/high"
    )
    parser.add_argument(
        "--console_log_level",
        type=str,
        choices=["info", "debug", "error", "warning"],
        default="info",
    )
    parser.add_argument(
        "--logfile_log_level",
        type=str,
        choices=["info", "debug", "error", "warning"],
        default="debug",
    )
    parser.add_argument(
        "--logfile",
        type=bool,
        default=True,
    )
    parser.add_argument("--logdir", type=str, default=".")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()
    JwstCalTrain(**vars(args)).main()