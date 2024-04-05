"""
Spacekit HST Calibration Dataset Reprocessing Resource Prediction

Step 1: SCRAPE inputs from s3 text file (import data)
Step 2: SCRUB inputs (preprocessing)
Step 3: PREDICT resource requirements (inference)


Examples:
df = run_preprocessing("home/singlevisits")

df = run_preprocessing("home/syntheticdata", fname="synth2", crpt=1, draw=0)

This module loads a pre-trained ANN to predict job resource requirements for HST.
# 1 - load job metadata inputs from text file in s3
# 2 - encode strings as int/float values in numpy array
# 3 - load models and generate predictions
# 4 - return preds as json to parent lambda function

MEMORY BIN: classifier predicts which of 4 memory bins is most likely to be needed to process an HST dataset (ipppssoot) 
successfully. The probabilities of each bin are output to Cloudwatch logs and the highest bin probability is returned to the 
Calcloud job submit lambda invoking this one. Bin sizes are as follows:

Memory Bins:
0: < 2GB
1: 2-8GB
2: 8-16GB
3: >16GB

WALLCLOCK REGRESSION: regression generates estimate for specific number of seconds needed to process the dataset using the same 
input data. This number is then tripled in Calcloud for the sake of creating an extra buffer of overhead in order to prevent 
larger jobs from being killed unnecessarily.

MEMORY REGRESSION: A third regression model is used to estimate the actual value of memory needed for the job. This is mainly for 
the purpose of logging/future analysis and is not currently being used for allocating memory in calcloud jobs.
"""
import os
import argparse
import numpy as np
from spacekit.extractor.scrape import S3Scraper
from spacekit.preprocessor.scrub import HstCalScrubber
from spacekit.preprocessor.transform import PowerX, array_to_tensor
from spacekit.builder.architect import Builder
from spacekit.logger.log import SPACEKIT_LOG, Logger


# build from local filepath
# MODEL_PATH = os.environ.get("MODEL_PATH", "./models") #"data/2022-02-14-1644848448/models"
# TX_FILE = os.path.join(MODEL_PATH, "tx_data.json")


def load_pretrained_model(**builder_kwargs):
    builder = Builder(**builder_kwargs)
    builder.load_saved_model(arch="hst_cal", keras_archive=True)
    return builder


class Predict:
    def __init__(
        self,
        dataset,
        bucket_name=None,
        key=None,
        model_path=None,
        models={},
        tx_file=None,
        norm=1,
        norm_cols=[0, 1],
        **log_kws,
    ):
        self.dataset = dataset
        self.bucket_name = bucket_name
        self.key = "control" if key is None else key
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
        self.wall_reg = None
        self.mem_reg = None
        self.__name__ = "predict"
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

    def scrape_s3_inputs(self):
        if self.key == "control":
            self.key = f"control/{self.dataset}/{self.dataset}_MemModelFeatures.txt"
        self.input_data = S3Scraper(
            bucket=self.bucket_name, pfx=self.key, **self.log_kws
        ).import_dataset()

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
        self.log.info("Acquiring input data")
        if self.input_data is None:
            self.scrape_s3_inputs()
        self.log.info(f"dataset: {self.dataset} keys: {self.input_data}")
        self.log.info("Preprocessing features")
        self.inputs = HstCalScrubber(
            data={self.dataset: self.input_data}, **self.log_kws
        ).scrub_inputs()
        self.log.info(f"dataset: {self.dataset} features: {self.inputs}")
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
        clocktime = int(self.regressor(self.wall_reg.model, self.X))
        self.predictions = {"memBin": membin, "memVal": memval, "clockTime": clocktime}
        self.log.info(f"dataset: {self.dataset} predictions: {self.predictions}")
        self.probabilities = {"dataset": self.dataset, "probabilities": pred_proba}
        self.log.info(self.probabilities)


def local_handler(dataset, **kwargs):
    """handles non-lambda invocations"""
    pred = Predict(dataset, **kwargs)
    pred.run_inference()
    return pred


def lambda_handler(event, context):
    """Predict Resource Allocation requirements for memory (GB) and max execution `kill time` / `wallclock` (seconds) using three
    pre-trained neural networks. This lambda is invoked from the Job Submit lambda which json.dumps the s3 bucket and key to the
    file containing job input parameters. The path to the text file in s3 assumes the following format: `control/ipppssoot/
    ipppssoot_MemModelFeatures.txt`."""
    MODEL_PATH = os.environ.get(
        "MODEL_PATH", "./models"
    )  # "data/2022-02-14-1644848448/models"
    TX_FILE = os.path.join(MODEL_PATH, "tx_data.json")
    # load models here for warm starts in aws lambda
    mem_clf = load_pretrained_model(model_path=MODEL_PATH, name="mem_clf")
    wall_reg = load_pretrained_model(model_path=MODEL_PATH, name="wall_reg")
    mem_reg = load_pretrained_model(model_path=MODEL_PATH, name="mem_reg")
    models = dict(mem_clf=mem_clf, wall_reg=wall_reg, mem_reg=mem_reg)
    # import and prep data: control/dataset/dataset_MemModelFeatures.txt
    pred = Predict(
        event["Dataset"],
        bucket_name=event["Bucket"],
        key=event["Key"],
        models=models,
        tx_file=TX_FILE,
    )
    pred.run_inference()
    return pred.predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spacekit", usage="spacekit.skopes.hst.cal.predict dataset"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="ipppssooot or visit name",
    )
    parser.add_argument(
        "-b",
        "--bucket_name",
        type=str,
        default=os.environ.get("BUCKET", None),
        help="name of s3 bucket containing input metadata",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        default=None,
        help="s3 object key for input metadata text file",
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
    local_handler(**vars(args))
