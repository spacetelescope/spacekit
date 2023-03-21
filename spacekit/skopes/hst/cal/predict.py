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

MEMORY BIN: classifier predicts which of 4 memory bins is most likely to be needed to process an HST dataset (ipppssoot) successfully. The probabilities of each bin are output to Cloudwatch logs and the highest bin probability is returned to the Calcloud job submit lambda invoking this one. Bin sizes are as follows:

Memory Bins:
0: < 2GB
1: 2-8GB
2: 8-16GB
3: >16GB

WALLCLOCK REGRESSION: regression generates estimate for specific number of seconds needed to process the dataset using the same input data. This number is then tripled in Calcloud for the sake of creating an extra buffer of overhead in order to prevent larger jobs from being killed unnecessarily.

MEMORY REGRESSION: A third regression model is used to estimate the actual value of memory needed for the job. This is mainly for the purpose of logging/future analysis and is not currently being used for allocating memory in calcloud jobs.
"""
import os
import argparse
import numpy as np
from spacekit.extractor.scrape import S3Scraper
from spacekit.preprocessor.scrub import CalScrubber
from spacekit.preprocessor.transform import PowerX
from spacekit.builder.architect import Builder
from spacekit.logger.log import Logger


# build from local filepath
MODEL_PATH = os.environ.get("MODEL_PATH", "./models") #"data/2022-02-14-1644848448/models"
TX_FILE = os.path.join(MODEL_PATH, "tx_data.json")


def load_pretrained_model(mpath=None, name="wall_reg"):
    # build from spacekit package trained_networks
    if mpath is None:
        builder = Builder(blueprint=name)
        builder.load_saved_model(arch="calmodels")
    else:
        builder = Builder(blueprint=name, model_path=os.path.join(mpath, name))
        builder.load_saved_model()
    return builder

class Predict:

    def __init__(
        self, dataset, bucket_name=None, key=None, model_path=None, tx_file=None, norm=1, norm_cols=[0,1],
        ):
        self.dataset = dataset
        self.bucket_name = bucket_name
        self.key = "control" if key is None else key
        self.model_path = MODEL_PATH if model_path is None else model_path
        self.tx_file = TX_FILE if tx_file is None else tx_file
        self.norm = norm
        self.norm_cols = norm_cols
        self.input_data = None
        self.inputs = None
        self.X = None
        self.mem_clf = None
        self.wall_reg = None
        self.mem_reg = None
        self.__name__ = "Predict"
        self.loglevel = "info"
        self.log = Logger(self.__name__, console_log_level=self.loglevel).setup_logger()
        self.predictions = None
        self.probabilities = None

    def scrape_s3_inputs(self):
        if self.key == "control":
            self.key = f"control/{self.dataset}/{self.dataset}_MemModelFeatures.txt"
        self.input_data = S3Scraper(bucket=self.bucket_name, pfx=self.key).import_dataset()

    def preprocess(self):
        if self.input_data is None:
            try:
                self.scrape_s3_inputs()
            except Exception as e:
                self.log.error(e)
        self.inputs = CalScrubber(data={self.dataset:self.input_data}).scrub_inputs()
        Px = PowerX(self.inputs, cols=self.norm_cols, tx_file=self.tx_file)
        self.X = Px.Xt
        self.log.info(f"tx_data: {Px.tx_data}")
        self.log.info(f"dataset: {self.dataset} keys: {self.input_data}")
        self.log.info(f"dataset: {self.dataset} features: {self.inputs}")
        self.log.info(f"dataset: {self.dataset} normalized inputs (X): {self.X}")

    def load_models(self, models={}):
        self.mem_clf = models.get('mem_clf', load_pretrained_model(mpath=self.model_path, name="mem_clf"))
        self.wall_reg = models.get('wall_reg', load_pretrained_model(mpath=self.model_path, name="wall_reg"))
        self.mem_reg = models.get('mem_reg', load_pretrained_model(mpath=self.model_path, name="mem_reg"))

    def classifier(self, model, data):
        """Returns class prediction"""
        pred_proba = model.predict(data)
        pred = int(np.argmax(pred_proba, axis=-1))
        return pred, pred_proba

    def regressor(self, model, data):
        """Returns Regression model prediction"""
        pred = model.predict(data)
        return pred

    def run_inference(self, models={}):
        self.load_models(models=models)
        if not os.path.exists(self.tx_file):
            self.tx_file = self.mem_clf.find_tx_file()
        if self.X is None:
            self.preprocess()
        membin, pred_proba = self.classifier(self.mem_clf, self.X)
        memval = np.round(float(self.regressor(self.mem_reg, self.X)), 2)
        clocktime = int(self.regressor(self.wall_reg, self.X))
        self.predictions = {"memBin": membin, "memVal": memval, "clockTime": clocktime}
        self.log.info(dict(dataset=self.dataset).update(self.predictions))
        self.probabilities = {"dataset": self.dataset, "probabilities": pred_proba}
        self.log.info(self.probabilities)


def local_handler(dataset, **kwargs):
    """handles non-lambda invocations"""
    pred = Predict(dataset, **kwargs)
    pred.preprocess()
    pred.run_inference()
    # TODO: write to ddb (ingest) or save to csv
    return pred


def lambda_handler(event, context):
    """Predict Resource Allocation requirements for memory (GB) and max execution `kill time` / `wallclock` (seconds) using three pre-trained neural networks. This lambda is invoked from the Job Submit lambda which json.dumps the s3 bucket and key to the file containing job input parameters. The path to the text file in s3 assumes the following format: `control/ipppssoot/ipppssoot_MemModelFeatures.txt`.
    """
    # load models here for warm starts in aws lambda
    mem_clf = load_pretrained_model(model_path=MODEL_PATH, model_name="mem_clf/")
    wall_reg = load_pretrained_model(model_path=MODEL_PATH, model_name="wall_reg/")
    mem_reg = load_pretrained_model(model_path=MODEL_PATH, model_name="mem_reg/")
    bucket_name = event["Bucket"]
    key = event["Key"]
    dataset = event["Dataset"]
    if not os.path.exists(TX_FILE):
        tx_file = mem_clf.find_tx_file()
    # import and prep data: control/dataset/dataset_MemModelFeatures.txt
    pred = Predict(dataset, bucket_name=bucket_name, key=key, tx_file=TX_FILE)
    pred.preprocess()
    pred.run_inference(models=dict(mem_clf=mem_clf, wall_reg=wall_reg, mem_reg=mem_reg))
    return pred.predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spacekit",
        usage="spacekit.skopes.hst.cal.predict dataset"
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
        default=None,
        help="s3 bucket name",
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
        help="apply normalization and scaling",
    )
    parser.add_argument(
        "-c",
        "--norm_cols",
        type=str,
        default="0,1",
        help=""
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=None,
        help="path to saved model directory"
    )
    parser.add_argument(
        "-t",
        "--tx_file",
        type=str,
        default="",
        help=""
    )
    args = parser.parse_args()
    args.norm_cols = [int(i) for i in args.norm_cols.split(",")]
    local_handler(args.dataset, **args)
