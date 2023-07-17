"""
Program (PID) of exposures comes into pipeline
1. create list of L1B exposures
2. scrape refpix vals from scihdrs + any additional metadata
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
from spacekit.skopes.jwst.cal.config import KEYPAIR_DATA
from spacekit.preprocessor.scrub import JwstCalScrubber
from spacekit.preprocessor.transform import array_to_tensor, PowerX
from spacekit.builder.architect import Builder


# build from local filepath
# MODEL_PATH = os.environ.get("MODEL_PATH", "./models") #"data/2023-07-14-1644848448/models"
# TX_FILE = os.path.join(MODEL_PATH, "tx_data.json")


def load_pretrained_model(**builder_kwargs):
    builder = Builder(**builder_kwargs)
    builder.load_saved_model(arch="jwstcal")
    return builder


class JwstCalPredict:
    def __init__(
        self,
        input_path=None,
        model_path=None,
        models={},
        tx_file=None,
        norm=0,
        norm_cols=[],
        **log_kws,
    ):
        self.input_path = input_path
        self.model_path = model_path
        self.models = models
        self.tx_file = tx_file
        self.norm = norm
        self.norm_cols = norm_cols
        self.input_data = None
        self.inputs = None
        self.tx_data = None
        self.X = None
        self.mem_clf = None
        self.mem_reg = None
        self.__name__ = "jwstpredict"
        self.log = Logger(self.__name__, **log_kws).setup_logger(logger=SPACEKIT_LOG)
        self.log_kws = dict(log=self.log, **log_kws)
        self.predictions = None
        self.probabilities = None
        self.initialize_models()

    def initialize_models(self):
        self.log.info("Initializing prediction models")
        if self.models is None and not os.path.exists(self.model_path):
            self.log.warning(
                f"models path not found: {self.model_path} - defaulting to latest in spacekit-collection"
            )
            self.model_path = None
        self.load_models(models=self.models)

    def normalize_inputs(self):
        if self.norm:
            self.log.info("Applying normalization")
            Px = PowerX(self.inputs, cols=self.norm_cols, tx_file=self.tx_file)
            self.X = Px.Xt
            self.tx_data = Px.tx_data
            self.log.debug(f"tx_data: {self.tx_data}")
            self.log.info(f"dataset: {self.dataset} normalized inputs (X): {self.X}")
        else:
            self.X = self.inputs

    def preprocess(self):
        self.log.info("Preprocessing input data")
        self.input_data = JwstCalScrubber(
            self.input_path, keypair=KEYPAIR_DATA, **self.log_kws
        ).scrub_inputs()
        for product, inputs in self.input_data.iterrows():
            self.log.info(f"product: {product} features: {inputs}")
        self.normalize_inputs()

    def load_models(self, models={}):
        self.mem_clf = models.get(
            "mem_clf",
            load_pretrained_model(
                model_path=self.model_path, name="mem_clf", **self.log_kws
            ),
        )
        self.wall_reg = models.get(
            "wall_reg",
            load_pretrained_model(
                model_path=self.model_path, name="wall_reg", **self.log_kws
            ),
        )
        self.mem_reg = models.get(
            "mem_reg",
            load_pretrained_model(
                model_path=self.model_path, name="mem_reg", **self.log_kws
            ),
        )
        if self.model_path is None:
            self.model_path = os.path.dirname(self.mem_clf.model_path)
        if self.tx_file is None or not os.path.exists(self.tx_file):
            self.mem_clf.find_tx_file()
            self.tx_file = self.mem_clf.tx_file

    def classifier(self, model, data):
        """Returns class prediction"""
        X = array_to_tensor(data)
        pred_proba = model.predict(X)
        pred = int(np.argmax(pred_proba, axis=-1))
        return pred, pred_proba

    def regressor(self, model, data):
        """Returns Regression model prediction"""
        X = array_to_tensor(data)
        pred = model.predict(X)
        return pred

    def run_inference(self, models={}):
        if self.X is None:
            self.preprocess()
        membin, pred_proba = self.classifier(self.mem_clf.model, self.X)
        memval = np.round(float(self.regressor(self.mem_reg.model, self.X)), 2)
        self.predictions = {"memBin": membin, "memVal": memval}
        self.log.info(f"dataset: {self.dataset} predictions: {self.predictions}")
        self.probabilities = {"dataset": self.dataset, "probabilities": pred_proba}
        self.log.info(self.probabilities)


def predict_handler(input_path, **kwargs):
    """handles non-lambda invocations"""
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
        "-n",
        "--norm",
        type=int,
        default=1,
        help="apply normalization and scaling (bool)",
    )
    parser.add_argument(
        "-c",
        "--norm_cols",
        type=str,
        default="0,1",
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
