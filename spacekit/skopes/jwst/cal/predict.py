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
# TX_FILE = MODEL_PATH + "/tx_data-{}.json"

global JWST_CAL_MODELS
JWST_CAL_MODELS = {}

def load_pretrained_model(**builder_kwargs):
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    builder = Builder(**builder_kwargs)
    builder.load_saved_model(arch="jwst_cal")
    model_name = builder_kwargs.get("name")
    JWST_CAL_MODELS[model_name] = builder
    return builder


class JwstCalPredict:
    """Generate predicted memory footprint of Level 3 products using metadata from uncalibrated (Level 1) exposures)."""

    def __init__(
        self,
        input_path=None,
        pid=None,
        obs=None,
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
            "sigma1_mean",
        ],
        sfx="uncal.fits",
        expmodes=["IMAGE", "SPEC"],
        **log_kws,
    ):
        """Initializes JwstCalPredict class object. This class can be used to estimate the memory footprint of Level 3 products based on metadata scraped from uncalibrated (Level 1) exposures.

        Parameters
        ----------
        input_path : str or Path, optional
            path to input exposure fits files, by default None
        pid : int, optional
            restrict to exposures matching a specific program ID e.g. 1018, by default None
        obs : str or int, optional
            restrict to exposures matching a specific observation number (requires `pid`), by default None
        model_path : str or Path, optional
            path to saved model directory, by default None
        models : dict, optional
            dictionary of spacekit.builder.architect.Builder type objects, by default {}
        tx_file : str or Path, optional
            path to transformer metadata json file, by default None
        norm : int, optional
            apply normalization and scaling (bool), by default 1
        norm_cols : list, optional
            index of input columns on which to apply normalization, by default [ "offset", "max_offset", "mean_offset", "sigma_offset", "err_offset", "sigma1_mean"]
        sfx : str, optional
            restrict to exposures matching a specifc filename suffix, by default "uncal.fits"
        expmodes : list, optional
            specifies which exposure modes to turn on for inference, by default ["IMAGE", "SPEC"]
        """
        self.input_path = input_path
        self.pid = pid
        self.obs = obs
        self.model_path = model_path
        self.models = models
        self.tx_file = tx_file
        self.norm = norm
        self.norm_cols = norm_cols
        self.sfx = sfx
        self.expmodes = expmodes
        self.input_data = None  # dict of dataframes
        self.inputs = None  # dict of (normalized) arrays
        self.tx_data = None
        self.X = None
        self.img3_reg = None
        self.spec3_reg = None
        # self.tac3_reg = None
        # self.jmem_clf = None
        self.__name__ = "JwstCalPredict"
        self.log = Logger(self.__name__, **log_kws).setup_logger(logger=SPACEKIT_LOG)
        self.log_kws = dict(log=self.log, **log_kws)
        self.predictions = dict()
        self.probabilities = dict()
        self.initialize_models()

    def initialize_models(self):
        """Initializes pre-trained models used for inference. Once loaded initially,
        the models are stored in the global variable `JWST_CAL_MODELS` to avoid unnecessary reloads over
        multiple iterations of object instantiation.
        """
        self.log.info("Initializing prediction models...")
        if self.models is None and not os.path.exists(self.model_path):
            self.log.warning(
                f"models path not found: {self.model_path} - defaulting to latest in spacekit-collection"
            )
            self.model_path = None
        if JWST_CAL_MODELS:
            self.models = JWST_CAL_MODELS
        self.load_models(models=self.models)
        self.log.info("Initialized.")

    def normalize_inputs(self, inputs, order="IMAGE"):
        """Applies normalization and scaling to continuous data type feature inputs.

        Parameters
        ----------
        inputs : pandas.DataFrame
            _description_
        order : str, optional
            inference exposure mode matching L3 data group, by default "IMAGE"

        Returns
        -------
        np.array
            array of input features preprocessed and normalized for ML inference
        """
        xcols = COLUMN_ORDER.get(order, list(inputs.columns))
        norm_cols = NORM_COLS.get(order, self.norm_cols)
        if self.norm:
            self.log.info(f"Applying normalization [{order}]...")
            tx_file = self.tx_file.format(order.lower())
            Px = PowerX(
                inputs, cols=norm_cols, tx_file=tx_file, rename=None, join_data=1
            )
            X = Px.Xt
            self.tx_data = Px.tx_data
            self.log.debug(f"tx_data: {self.tx_data}")
            X = X[xcols]
            X = np.asarray(X)
        else:
            X = inputs[xcols]
            X = np.asarray(X)
        return X

    def verify_input_path(self):
        """Verifies input path exists and checks if file or directory.
        If input_path is a directory, check/set self.pid value
        If self.obs is not None, validate format (1-3 digits) and append to self.pid
        If input_path is a file, any files matching first 9 chars and suffix 
        (typically detector, e.g. "nrcb4_uncal.fits")
        found in the same directory will be included automatically (assumes
        standard naming convention of JWST input exposures).
        - self.input_path is reset to top/parent directory
        - self.pid is set to the first 9 characters
        NB these variables are passed through to the Scrubber and Scraper classes
        to handle the actual searching on local disk for input files.
        """
        if not os.path.exists(self.input_path):
            self.log.error(f"No files/directories found at the specified path: {self.input_path}")
            raise FileNotFoundError
        elif os.path.isfile(self.input_path):
            fname = str(os.path.basename(self.input_path))
            self.log.debug("Acquiring data from single input file")
            # reset input path to parent directory
            self.input_path = os.path.dirname(self.input_path)
            prefix = fname.split("_")[0][:10]
            self.pid = prefix
            return
        if self.pid is not None:
            self.pid = 'jw{:0>5}'.format(str(self.pid).lstrip('jw'))
            if self.obs:
                try:
                    self.obs = '{:0>3}'.format(int(self.obs))
                except ValueError:
                    self.obs = ''
                self.pid += self.obs
        else:
            self.pid = ""
            return

    def preprocess(self):
        """Runs necessary preprocessing steps on input exposure data prior to inference.
        """
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
        self.verify_input_path()
        scrubber = JwstCalScrubber(
            self.input_path,
            pfx=self.pid,
            sfx=self.sfx,
            encoding_pairs=KEYPAIR_DATA,
            **self.log_kws,
        )
        for exp_type in self.expmodes:
            inputs = scrubber.scrub_inputs(exp_type=exp_type)
            if inputs is not None:
                self.input_data[exp_type] = inputs
                self.inputs[exp_type] = self.normalize_inputs(inputs, order=exp_type)

    def load_models(self, models={}):
        """_summary_

        Parameters
        ----------
        models : dict, optional
            Builder objects with pre-loaded functional models, by default {}
        """
        # self.jmem_clf = models.get(
        #     "jmem_clf",
        #     load_pretrained_model(
        #         model_path=self.model_path, name="jmem_clf", **self.log_kws
        #     ),
        # )
        self.img3_reg = models.get(
            "img3_reg",
            load_pretrained_model(
                model_path=self.model_path, name="img3_reg", **self.log_kws
            ),
        )
        self.spec3_reg = models.get(
            "spec3_reg",
            load_pretrained_model(
                model_path=self.model_path, name="spec3_reg", **self.log_kws
            ),
        )
        if self.model_path is None:
            self.model_path = os.path.dirname(self.img3_reg.model_path)
        if self.tx_file is None or not os.path.exists(self.tx_file):
            self.tx_file = self.model_path + "/tx_data-{}.json"

    def classifier(self, model, data):
        """Returns class prediction"""
        reshape = True if len(data.shape) == 1 else False
        shape = (1, -1) if reshape is True else data.shape
        X = array_to_tensor(data, reshape=reshape, shape=shape)
        pred_proba = model.predict(X)
        pred = int(np.argmax(pred_proba, axis=-1))
        return pred, pred_proba

    # def run_classifier(self, expmode):
    #     input_data = self.input_data.get(expmode, None)
    #     X = self.inputs.get(expmode, None)
    #     if X is None or input_data is None:
    #         return
    #     self.log.info(f"Estimating memory bin : L3 {expmode}")
    #     product_index = list(input_data.index)
    #     if expmode == "IMAGE":
    #         imgbin, pred_proba = self.classifier(self.jmem_clf.model, X)
    #     for i, _ in enumerate(X):
    #         self.predictions[product_index[i]] = {
    #             "imgBin": imgbin[0]
    #         } 
    #         self.probabilities[product_index[i]] = {"probabilities": pred_proba[0]}
    #     # self.log.info(f"probabilities: {self.probabilities}")

    def regressor(self, model, data):
        """_summary_

        Parameters
        ----------
        model : tf.keras.model
            keras functional model
        data : numpy.array or tf.tensors
            input data on which to run inference

        Returns
        -------
        numpy.array
            Returns Regression model prediction
        """
        reshape = True if len(data.shape) == 1 else False
        shape = (1, -1) if reshape is True else data.shape
        X = array_to_tensor(data, reshape=reshape, shape=shape)
        pred = model.predict(X)
        return pred

    def run_image_inference(self):
        """Run inference for L3 Image exposure datasets
        """
        input_data = self.input_data.get("IMAGE", None)
        X = self.inputs.get("IMAGE", None)
        if X is None or input_data is None:
            return
        self.log.info("Estimating memory footprints : L3 IMAGE")
        product_index = list(input_data.index)
        imgsize = self.regressor(self.img3_reg.model, X)
        for i, _ in enumerate(X):
            rpred = np.round(float(np.squeeze(imgsize[i])), 2)
            self.predictions[product_index[i]] = {
                "gbSize": rpred
            }

    def run_spec_inference(self):
        """Run inference for L3 Spectroscopy exposure datasets
        """
        input_data = self.input_data.get("SPEC", None)
        X = self.inputs.get("SPEC", None)
        if X is None or input_data is None:
            return
        self.log.info("Estimating memory footprints : L3 SPEC")
        product_index = list(input_data.index)
        imgsize = self.regressor(self.spec3_reg.model, X)
        
        for i, _ in enumerate(X):
            rpred = np.round(float(np.squeeze(imgsize[i])), 2)
            self.predictions[product_index[i]] = {
                "gbSize": rpred
            }

    def run_inference(self):
        """Main calling function to preprocess input exposures and generate estimated memory footprints.
        """
        if not self.inputs:
            self.preprocess()
        if self.img3_reg:
            self.run_image_inference()
        if self.spec3_reg:
            self.run_spec_inference()
        self.log.info(f"predictions: {self.predictions}")


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
        help="restrict to exposures matching a specific program ID e.g. 1018",
    )
    parser.add_argument(
        "-o",
        "--obs",
        type=int,
        default=None,
        help="restrict to exposures matching a specific observation number (requires --pid)",
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
        default="offset,max_offset,mean_offset,sigma_offset,err_offset,sigma1_mean",
        help="comma-separated index of input columns to apply normalization",
    )
    parser.add_argument(
        "-s",
        "--sfx",
        type=str,
        default="_uncal.fits",
        help="restrict to exposures matching a specific filename suffix",
    )
    parser.add_argument(
        "-e",
        "--expmodes",
        type=str,
        default="IMAGE,SPEC",
        help="comma-separated string of exposure modes to turn on for inference",
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
    args.norm_cols = [str(i) for i in args.norm_cols.split(",")]
    args.expmodes = sorted([str(i).upper() for i in args.expmodes.split(",")])
    predict_handler(**vars(args))
