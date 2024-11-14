import os
import re
from argparse import ArgumentParser
import datetime as dt
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import KFold
from spacekit.preprocessor.prep import JwstCalPrep
from spacekit.builder.architect import BuilderMLP
from spacekit.analyzer.compute import ComputeRegressor
from spacekit.logger.log import SPACEKIT_LOG, Logger

"""
Training can be run in 1 of 2 modes: standard, or cross-validation.

Standard:

Single training iteration run on the dataset using a randomized 80-20 train-test split


Cross-validation mode:

Training is run on the dataset K times, where k is the number of randomly shuffled train-test split folds.
Additional metrics are accumulated and recorded for evaluating overall model performance across iterations.


"""

class JwstCalTrain:

    def __init__(self, training_data=None, out=None, exp_mode="image", norm=1, cross_val=0, early_stopping=None, **log_kws):
        """_summary_

        Parameters
        ----------
        training_data : str or Path, optional
            path on local disk to directory where training data files are stored, by default None
        out : str or Path, optional
            path on local disk for saving training results, saved models, and train-test splits, by default None
        exp_mode : str, optional
            image: specifies which model to train (image or spec), by default "image"
        norm : int, optional
            apply normalization and scaling, by default 1
        cross_val : int, optional
            Run cross-validation using k number of folds (10 recommended), by default 0
        early_stopping : str, optional
            Either 'val_loss' or 'val_rmse' ends training when this metric is no longer improving, by default None
        """
        self.training_data = training_data
        self.set_outpath(out=out)
        self.exp_mode = exp_mode.lower()
        self.prep_kwargs = dict(exp_mode=self.exp_mode, normalize=norm)
        self.cross_val = cross_val
        self.early_stopping = early_stopping
        self.data = None
        self.itn = ""
        self.jp = None
        self.metrics = None
        self.dm = None
        self.__name__ = "JwstCalTrain"
        self.log = Logger(self.__name__, **log_kws).setup_logger(logger=SPACEKIT_LOG)
        self.log_kws = dict(log=self.log, **log_kws)
        self.initialize()

    def set_outpath(self, out=None):
        """Sets up the output directory for model training data, models and results.
        If `out` is formatted as a {date}_{timestamp} string this becomes the output_path attr and
        the directory for it is created if it does not exist already. Otherwise, `out` is used as the top directory,
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
                self.output_path = base_out
                if not os.path.exists(self.output_path):
                    os.makedirs(self.outpath, exist_ok=True)
                return
        today = dt.date.today().isoformat()
        timestamp = str(dt.datetime.now().timestamp()).split('.')[0]
        self.output_path = f"{base_out}/{today}_{timestamp}"
        os.makedirs(self.output_path, exist_ok=True)
    
    def initialize(self):
        global DATA, MODELS, RESULTS
        DATA = os.path.join(self.output_path, "data")
        MODELS = os.path.join(self.output_path, "models")
        RESULTS = os.path.join(self.output_path, "results")
        for p in [DATA, MODELS, RESULTS]:
            os.makedirs(p, exist_ok=True)
        self.load_metrics()
        if self.metrics is not None: # 1 higher than 0-valued list of iterations
            self.itn = str(len(list(self.metrics.keys())))

    def load_data(self, tts=None):
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
        kfold = KFold(n_splits=self.cross_val, shuffle=True)
        itn = 0
        for _, test_idx in kfold.split(np.zeros(self.data.shape[0]), pd.concat(self.jp.get_y_train_test('imgsize_gb'), axis=0)):
            self.data['split'] = 'train'
            self.data.loc[self.data.iloc[test_idx].index, 'split'] = 'test'
            self.data.to_csv(f"{DATA}/{self.exp_mode}-tts_{str(itn)}.csv", index=False)
            itn += 1

    def run_cross_val(self):
        for i in list(range(self.cross_val)):
            save_diagram = True if i == 0 else False
            self.data = None
            self.load_train_test(tts=str(i))
            self.run_training(save_diagram=save_diagram)
            self.compute_cache()

    def prep_train_test(self):
        if self.data is None:
            self.load_data()
        self.jp = JwstCalPrep(self.data, **self.prep_kwargs)
        self.jp.prep_data()
        self.jp.prep_targets()
 
    def load_train_test(self, tts="0"):
        if self.data is None:
            self.load_data(tts=tts)
        self.jp = JwstCalPrep(self.data, **self.prep_kwargs)
        self.jp.prep_data(existing_splits=True)
        self.jp.prep_targets()
        if tts != self.itn:
            self.itn = tts

    @property
    def architecture(self):
        return dict(
            image="jwst_img3_reg",
            spec="jwst_spec3_reg"
        )[self.exp_mode]

    def run_training(self, save_diagram=True):
        """Build, train and save a model

        Parameters
        ----------
        save_diagram : bool, optional
            Save a png diagram image of the model architecture, by default True
        """
        self.builder = BuilderMLP(
            X_train=self.jp.X_train,
            y_train=self.jp.y_reg_train,
            X_test=self.jp.X_test,
            y_test=self.jp.y_reg_test,
            blueprint="mlp",
        )
        self.builder.get_blueprint(self.architecture)
        self.builder.model = self.builder.build()
        if save_diagram is True:
            self.builder.model_diagram(output_path=MODELS, show_layer_names=True)
        self.builder.early_stopping = self.early_stopping
        self.builder.fit()
        self.builder.save_model(output_path=MODELS, parent_dir=self.itn)

    def compute_cache(self):
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
        self.loss_fig = self.com.keras_loss_plot()
        self.record_metrics()

    def load_metrics(self):
        metrics_file = f"{DATA}/training_metrics-{self.exp_mode}.csv"
        if os.path.exists(metrics_file):
            self.dm = pd.read_csv(metrics_file, index_col="index")
            self.metrics = self.dm.to_dict()

    @property
    def itr_metrics(self):
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
        itr_metrics = self.itr_metrics.copy()
        dd = dict(
            train=self.jp.data.loc[self.jp.data.split == 'train'],
            test=self.jp.data.loc[self.jp.data.split == 'test']
        )
        for s, g in dict(zip(['tr', 'ts'],['train', 'test'])):
            itr_metrics[f'{s}_size'] = dd[g].shape[0]
            itr_metrics[f'{s}_lrg_ct'] = dd[g].loc[dd[g].imgsize_gb>100].shape[0]
            itr_metrics[f'{s}_lrg_mean'] = dd[g].loc[dd[g].imgsize_gb>100].imgsize_gb.mean()
            itr_metrics[f'{s}_lrg_max'] = dd[g].loc[dd[g].imgsize_gb>100].imgsize_gb.max()

        itr_metrics.update(self.com.loss)
        if self.metrics is None:
            self.itn = "0"
            self.metrics = {self.itn:itr_metrics}
        else:
            self.itn = str(len(list(self.metrics.keys())))
            self.metrics[self.itn] = itr_metrics
        dm = pd.DataFrame.from_dict(self.metrics)
        dm['index'] = dm.index
        dm.to_csv(f"{DATA}/training_metrics-{self.exp_mode}.csv", index=False)
        self.dm = dm.drop('index', axis=1, inplace=True)

    def main(self):
        self.prep_train_test()
        if self.cross_val > 0:
            self.generate_kfolds()
            self.run_cross_val()
        else:
            self.run_training(save_diagram=True)
            self.compute_cache()


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
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--early_stopping",
        type=str,
        default=None,
        help="Either 'val_loss' or 'val_rmse' ends training when this metric is no longer improving"
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