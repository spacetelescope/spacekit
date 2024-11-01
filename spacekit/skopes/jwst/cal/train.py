import os
from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import KFold
from spacekit.preprocessor.prep import JwstCalPrep
from spacekit.builder.architect import BuilderMLP
from spacekit.analyzer.compute import ComputeRegressor
from spacekit.logger.log import SPACEKIT_LOG, Logger


class JwstCalTrain:

    def __init__(self, training_data=None, output_path=None, expmode="IMAGE", norm=1, cross_val=False, nfolds=10, **log_kws):
        self.training_data = training_data
        self.output_path = "." if output_path is None else output_path
        self.expmode = expmode.lower()
        self.norm = norm
        self.cross_val = cross_val
        self.nfolds
        self.data = None
        self.splits = dict()
        self.jp = None
        self.__name__ = "JwstCalTrain"
        self.log = Logger(self.__name__, **log_kws).setup_logger(logger=SPACEKIT_LOG)
        self.log_kws = dict(log=self.log, **log_kws)
        self.initialize()
    
    def initialize(self):
        global DATA, MODELS, RESULTS
        DATA = os.path.join(self.output_path, "data")
        MODELS = os.path.join(self.output_path, "models")
        RESULTS = os.path.join(self.output_path, "results")
        for p in [DATA, MODELS, RESULTS]:
            os.makedirs(p, exist_ok=True)

    def load_data(self):
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
        if len(self.data.loc[self.data.duplicated(subset='pname')]) > 0:
            self.log.info("Dropping duplicates")
            self.data = self.data.sort_values(by=['date', 'imagesize']).drop_duplicates(subset='pname', keep='last')
        self.jp = JwstCalPrep(data=self.data, exp_mode=self.expmode, **prep_kwargs)
        self.jp.prep_data()
        self.jp.prep_targets()
        #TODO
        # if self.cross_val is True:
        #     self.generate_kfolds()

    def architectures(self):
        return dict(
            image="jwst_img3_reg",
            spec="jwst_spec3_reg"
        )[self.expmode]

    def train_models(self):
        self.builder = BuilderMLP(
            X_train=self.jp.X_train,
            y_train=self.jp.y_reg_train,
            X_test=self.jp.X_test,
            y_test=self.jp.y_reg_test,
            blueprint="mlp",
        )
        self.builder.get_blueprint(self.architectures())
        self.builder.model = self.builder.build()
        self.builder.model_diagram(output_path=MODELS, show_layer_names=True)
        self.builder.fit()
        self.builder.save_model()

    def compute_cache(self):
        self.builder.test_idx = list(self.jp.test_idx)
        self.com = ComputeRegressor(
            builder=self.builder,
            algorithm="linreg",
            res_path=RESULTS,
            show=True,
            validation=False,
        )
        self.com.calculate_results()
        outputs = self.com.make_outputs()
        print(outputs.keys())
       

    def main(self):
        self.initialize()
        self.prep_train_test()
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