"""
Program (PID) of exposures comes into pipeline
1. create list of L1B exposures
2. scrape pix offset vals from scihdrs + any additional metadata
3. determine potential L3 products based on obs, filters, detectors, etc
4. calculate sky separation / reference pixel offset statistics
5. preprocessing: create dataframe of all input values, encode categoricals
6. load model
7. run inference
"""
import os
import argparse
import numpy as np
from spacekit.logger.log import SPACEKIT_LOG, Logger
from spacekit.skopes.jwst.cal.config import KEYPAIR_DATA, COLUMN_ORDER, NORM_COLS
from spacekit.preprocessor.scrub import JwstCalScrubber
from spacekit.preprocessor.transform import array_to_tensor, PowerX
from spacekit.builder.architect import Builder


# build from local filepath
# MODEL_PATH = os.environ.get("MODEL_PATH", "./models/jwst_cal")
# TX_FILE = os.path.join(MODEL_PATH, "tx_data.json")


def load_pretrained_model(**builder_kwargs):
    builder = Builder(**builder_kwargs)
    builder.load_saved_model(arch="jwst_cal")
    return builder


class JwstCalPredict:
    def __init__(
        self,
        input_path=None,
        pid=None,
        model_path=None,
        models={},
        tx_file=None,
        norm=1,
        norm_cols=[
            "offset",
            "max_offset",
            "mean_offset",
            "sigma_offset",
            "err_offset",
            "sigma1_mean"
        ],
        **log_kws,
    ):
        self.input_path = input_path
        self.pid = pid
        self.model_path = model_path
        self.models = models
        self.tx_file = tx_file
        self.norm = norm
        self.norm_cols = norm_cols
        self.input_data = None # dict of dataframes
        self.inputs = None # dict of (normalized) arrays
        self.tx_data = None
        self.X = None
        self.img3_reg = None
        # self.img3_clf = None
        # self.spec3_reg = None
        self.__name__ = "JwstCalPredict"
        self.log = Logger(self.__name__, **log_kws).setup_logger(logger=SPACEKIT_LOG)
        self.log_kws = dict(log=self.log, **log_kws)
        self.predictions = dict()
        self.probabilities = dict()
        self.initialize_models()

    def initialize_models(self):
        self.log.info("Initializing prediction models...")
        if self.models is None and not os.path.exists(self.model_path):
            self.log.warning(
                f"models path not found: {self.model_path} - defaulting to latest in spacekit-collection"
            )
            self.model_path = None
        self.load_models(models=self.models)
        self.log.info("Initialized.")

    def normalize_inputs(self, inputs, order="IMAGE"):
        xcols = COLUMN_ORDER.get(order, list(inputs.columns))
        norm_cols = NORM_COLS.get(order, self.norm_cols)
        if self.norm:
            self.log.info(f"Applying normalization [{order}]...")
            Px = PowerX(inputs, cols=norm_cols, tx_file=self.tx_file, rename=None, join_data=1)
            X = Px.Xt
            self.tx_data = Px.tx_data
            self.log.debug(f"tx_data: {self.tx_data}")
            X = X[xcols]
            X = np.asarray(X)
        else:
            X = inputs[xcols]
            X = np.asarray(X)
        return X

    def preprocess(self):
        self.input_data = dict(
            IMAGE=None,
            SPEC=None,
            FGS=None,
            TAC=None,
        )
        self.inputs = dict(
            IMAGE=None,
            SPEC=None,
            FGS=None,
            TAC=None,
        )
        self.log.info("Preprocessing inputs...")
        if self.pid is not None:
            program_id = str(self.pid)
            program_id = f"jw0{program_id}" if len(program_id) == 4 else f"jw{program_id}"
            self.pid = program_id
        else:
            self.pid = ""
        scrubber = JwstCalScrubber(
            self.input_path, pfx=self.pid, sfx="_uncal.fits", encoding_pairs=KEYPAIR_DATA, **self.log_kws
        )
        for exp_type in ["IMAGE"]: #["IMAGE", "SPEC", "TAC", "FGS"]:
            inputs = scrubber.scrub_inputs(exp_type=exp_type)
            if inputs is not None:
                self.input_data[exp_type] = inputs
                self.inputs[exp_type] = self.normalize_inputs(inputs, order=exp_type)

    def load_models(self, models={}):
        # self.img3_clf = models.get(
        #     "img3_clf",
        #     load_pretrained_model(
        #         model_path=self.model_path, name="img3_clf", **self.log_kws
        #     ),
        # )
        self.img3_reg = models.get(
            "img3_reg",
            load_pretrained_model(
                model_path=self.model_path, name="img3_reg", **self.log_kws
            ),
        )
        # self.spec3_reg = models.get(
        #     "spec3_reg",
        #     load_pretrained_model(
        #         model_path=self.model_path, name="spec3_reg", **self.log_kws
        #     ),
        # )
        if self.model_path is None:
            self.model_path = os.path.dirname(self.img3_reg.model_path)
        if self.tx_file is None or not os.path.exists(self.tx_file):
            self.img3_reg.find_tx_file()
            self.tx_file = self.img3_reg.tx_file

    def classifier(self, model, data):
        """Returns class prediction"""
        reshape = True if len(data.shape) == 1 else False
        shape = (1,-1) if reshape is True else data.shape
        X = array_to_tensor(data, reshape=reshape, shape=shape)
        pred_proba = model.predict(X)
        pred = int(np.argmax(pred_proba, axis=-1))
        return pred, pred_proba

    def regressor(self, model, data):
        """Returns Regression model prediction"""
        reshape = True if len(data.shape) == 1 else False
        shape = (1,-1) if reshape is True else data.shape
        X = array_to_tensor(data, reshape=reshape, shape=shape)
        pred = model.predict(X)
        return pred

    def run_image_inference(self):
        input_data = self.input_data.get("IMAGE", None)
        X = self.inputs.get("IMAGE", None)
        if X is None or input_data is None:
            return
        product_index = list(input_data.index)
        imgsize = self.regressor(self.img3_reg.model, X)
        # imgbin, pred_proba = self.classifier(self.img3_clf.model, X)
        for i, _ in enumerate(X):
            rpred = np.round(float(imgsize[i]), 2)
            self.predictions[product_index[i]] = {"gbSize": rpred} # "imgBin": imgbin[0]
            # self.probabilities[product_index[i]] = {"probabilities": pred_proba[0]}

    def run_spec_inference(self):
        input_data = self.input_data.get("SPEC", None)
        X = self.inputs.get("SPEC", None)
        if X is None or input_data is None:
            return
        product_index = list(input_data.index)
        imgsize = self.regressor(self.spec3_reg.model, X)
        # imgbin, pred_proba = self.classifier(self.img3_clf.model, X)
        for i, _ in enumerate(X):
            rpred = np.round(float(imgsize[i]), 2)
            self.predictions[product_index[i]] = {"gbSize": rpred} # "imgBin": imgbin[0]
            # self.probabilities[product_index[i]] = {"probabilities": pred_proba[0]}

    def run_inference(self):
        if not self.inputs:
            self.preprocess()
        self.log.info("Estimating Level 3 output image sizes...")
        self.run_image_inference()
        # self.run_spec_inference()
        self.log.info(f"predictions: {self.predictions}")
        # self.log.info(f"probabilities: {self.probabilities}")


def predict_handler(input_path, **kwargs):
    """handles local invocations"""
    pred = JwstCalPredict(input_path, **kwargs)
    pred.run_inference()
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spacekit", usage="spacekit.skopes.jwst.cal.predict input_path"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="path to input exposure fits files",
    )
    parser.add_argument(
        "-p",
        "--pid",
        type=int,
        default=None,
        help="restrict to input files matching a specific program ID e.g. 1018"
    )
    parser.add_argument(
        "-n",
        "--norm",
        type=int,
        default=0,
        help="apply normalization and scaling (bool)",
    )
    parser.add_argument(
        "-c",
        "--norm_cols",
        type=str,
        default="offset,max_offset,mean_offset,sigma_offset,err_offset,sigma1_mean",
        help="comma-separated index of input columns to apply normalization",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=os.environ.get("MODEL_PATH", None),
        help="path to saved model directory",
    )
    parser.add_argument(
        "-t",
        "--tx_file",
        type=str,
        default=None,
        help="path to transformer metadata json file",
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
        "--verbose",
        "-v",
        action="store_true",
    )
    args = parser.parse_args()
    args.norm_cols = [int(i) for i in args.norm_cols.split(",")]
    predict_handler(**vars(args))
