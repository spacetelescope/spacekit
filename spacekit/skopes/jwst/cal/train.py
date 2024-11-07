import os
import re
from argparse import ArgumentParser
import datetime as dt
import pandas as pd
from sklearn.model_selection import KFold
from spacekit.preprocessor.prep import JwstCalPrep
from spacekit.builder.architect import BuilderMLP
from spacekit.analyzer.compute import ComputeRegressor
from spacekit.logger.log import SPACEKIT_LOG, Logger


class JwstCalTrain:

    def __init__(self, training_data=None, out=None, expmode="image", norm=1, cross_val=False, nfolds=10, **log_kws):
        self.training_data = training_data
        self.set_outpath(out=out)
        self.expmode = expmode.lower()
        self.norm = norm
        self.cross_val = cross_val
        self.nfolds = nfolds
        self.data = None
        self.iteration = ""
        self.jp = None
        self.metrics = None
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
        self.metrics = self.load_metrics()
        if self.metrics is not None: # 1 higher than 0-valued list of iterations
            self.iteration = str(len(list(self.metrics.keys())))

    def load_data(self, tts=None):
        if tts:
            fpath = f"{DATA}/{self.expmode}-tts_{tts}.csv"
        else:
            fpath = os.path.join(self.training_data, f"train-{self.expmode}.csv")
        if not os.path.exists(fpath):
            self.log.error(f"Training data filepath not found: {fpath}")
            return
        self.data = pd.read_csv(fpath, index_col="Dataset")

    def generate_kfolds(self):
        # TODO
        kfold = KFold(n_splits=self.nfolds, shuffle=True)
        self.jp.prep_data(existing_splits=True)
        # for train, test in kfold.split(X, y)

    def prep_train_test(self, **prep_kwargs):
        if self.data is None:
            self.load_data()
        ndupes = len(self.data.loc[self.data.duplicated(subset='pname')])
        if ndupes > 0:
            self.log.info(f"Dropping {ndupes} duplicates")
            self.data = self.data.sort_values(by=['date', 'imagesize']).drop_duplicates(subset='pname', keep='last')
        self.jp = JwstCalPrep(self.data, **prep_kwargs)
        self.jp.prep_data()
        self.jp.prep_targets()
        self.data['Dataset'] = self.data.index
        it = "0" if not self.iteration else self.iteration
        self.data.to_csv(f"{DATA}/{self.expmode}-tts_{it}.csv", index=False)
        #TODO
        # if self.cross_val is True:
        #     self.generate_kfolds()

    def load_train_test(self, tts="0", **prep_kwargs):
        if self.data is None:
            self.load_data(tts=tts)
        self.jp = JwstCalPrep(self.data, **prep_kwargs)
        self.jp.prep_data(existing_splits=True)
        self.jp.prep_targets()
        if tts != self.iteration:
            self.iteration = tts

    def architectures(self):
        return dict(
            image="jwst_img3_reg",
            spec="jwst_spec3_reg"
        )[self.expmode]

    def train_models(self, save_diagram=True):
        self.builder = BuilderMLP(
            X_train=self.jp.X_train,
            y_train=self.jp.y_reg_train,
            X_test=self.jp.X_test,
            y_test=self.jp.y_reg_test,
            blueprint="mlp",
        )
        self.builder.get_blueprint(self.architectures())
        self.builder.model = self.builder.build()
        if save_diagram is True:
            self.builder.model_diagram(output_path=MODELS, show_layer_names=True)
        self.builder.fit()
        self.builder.save_model(output_path=MODELS, parent_dir=self.iteration)

    def compute_cache(self):
        self.builder.test_idx = list(self.jp.test_idx)
        self.com = ComputeRegressor(
            builder=self.builder,
            algorithm="linreg",
            res_path=RESULTS+f"/{self.iteration}",
            show=True,
            validation=False,
        )
        self.com.calculate_results()
        outputs = self.com.make_outputs()
        print(outputs.keys())
        self.record_metrics()

    def load_metrics(self):
        metrics_file = f"{DATA}/training_metrics-{self.expmode}.csv"
        if os.path.exists(metrics_file):
            dm = pd.read_csv(metrics_file, index_col="index")
            metrics = dm.to_dict()
            return metrics
        return None

    def record_metrics(self):
        itr_metrics = dict(
            train_size=self.jp.data.loc[self.jp.data.split == 'train'].size,
            test_size=self.jp.data.loc[self.jp.data.split == 'test'].size,
            tr_lrg_ct=self.jp.data.loc[(self.jp.data.split == 'train') & (self.jp.data.imgsize_gb>100)].size,
            ts_lrg_ct=self.jp.data.loc[(self.jp.data.split == 'test') & (self.jp.data.imgsize_gb>100)].size,
            tr_lrg_mean=self.jp.data.loc[(self.jp.data.split == 'train') & (self.jp.data.imgsize_gb>100)].imgsize_gb.mean(),
            ts_lrg_mean=self.jp.data.loc[(self.jp.data.split == 'test') & (self.jp.data.imgsize_gb>100)].imgsize_gb.mean(),
            tr_lrg_max=self.jp.data.loc[(self.jp.data.split == 'train') & (self.jp.data.imgsize_gb>100)].imgsize_gb.max(),
            ts_lrg_max=self.jp.data.loc[(self.jp.data.split == 'test') & (self.jp.data.imgsize_gb>100)].imgsize_gb.max(),
        )
        itr_metrics.update(self.com.loss)

        if self.metrics is None:
            self.iteration = "0"
            self.metrics = {self.iteration:itr_metrics}
        else:
            self.iteration = str(len(list(self.metrics.keys())) - 1)
            self.metrics[self.iteration] = itr_metrics
        dm = pd.DataFrame.from_dict(self.metrics)
        dm['index'] = dm.index
        dm.to_csv(f"{DATA}/training_metrics-{self.expmode}.csv", index=False)
        self.iteration = str(int(self.iteration) + 1)

    def main(self):
        self.initialize()
        self.prep_train_test(expmode=self.expmode)
        self.train_models()
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
        "--output_path",
        type=str,
        default=None,
        help="path on local disk for saving training results, saved models, and train-test splits",
    )
    parser.add_argument(
        "-e",
        "--expmode",
        choices=["IMAGE", "SPEC"],
        default="IMAGE",
        help="IMAGE: train image model, SPEC: train spec model"
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
        "-n",
        "--nfolds",
        type=int,
        default=10,
        help="Number of folds in Kfold cross validation. Requires `--cross_val`"
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
    args.norm_cols = [str(i) for i in args.norm_cols.split(",")]
    args.expmodes = sorted([str(i).upper() for i in args.expmodes.split(",")])
    JwstCalTrain(**vars(args)).main()