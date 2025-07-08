from spacekit.preprocessor.encode import encode_target_data
from spacekit.preprocessor.transform import arrays_to_tensors, y_tensors, PowerX
from sklearn.model_selection import train_test_split
from spacekit.logger.log import Logger


class Prep:
    """Base class for preprocessing data sets prior to training a machine learning model. This class can be used directly or subclassed for additional custom preprocessing. Existing subclasses for HST and JWST skopes are also available.

    Parameters
    ----------
    data : pandas.DataFrame
        training dataset to be preprocessed
    y_target : str, optional
        target column name (dependent variable), by default "imgsize_gb"
    X_cols : list, optional
        feature column names (independent variables), by default []
    tensors : bool, optional
        convert model inputs into tensors, by default True
    normalize : bool, optional
        apply normalization, by default True
    random : int, optional
        random seed for train-test splits, by default None
    tsize : float, optional
        test size ratio, by default 0.2
    encode_targets : bool, optional
        encode target values (categorical classifiers), by default False
    norm_params : dict, optional
        normalization parameters (see apply_normalization for acceptable key-val pairs), by default None
    """
    def __init__(
        self,
        data,
        y_target,
        X_cols=[],
        tensors=True,
        normalize=True,
        random=None,
        tsize=0.2,
        encode_targets=True,
        norm_params=None,
    ):
        self.data = data
        self.y_target = y_target
        self.X_cols = self.check_input_cols(X_cols)
        self.tensors = tensors
        self.normalize = normalize
        self.norm_params = norm_params
        self.random = random
        self.tsize = tsize
        self.encode_targets = encode_targets
        self.X = self.data[self.X_cols]
        self.train_idx = None
        self.test_idx = None
        self.Tx = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def check_input_cols(self, X_cols):
        if len(X_cols) == 0:
            X_cols = list(self.data.columns)
            X_cols.remove(self.y_target)
        return X_cols

    def stratify_split(self, y_target=None, stratify=True):
        if y_target is None:
            y_target = self.y_target
        y = self.data[y_target]

        strat = y if stratify is True else None
        train, test = train_test_split(
            self.X, test_size=self.tsize, stratify=strat, random_state=self.random
        )
        self.train_idx, self.test_idx = train.index, test.index
        self.data["split"] = "train"
        self.data.loc[self.test_idx, "split"] = "test"

    def get_X_y(self, group, y_target):
        if group == "train":
            X_train = self.data.loc[self.train_idx, self.X_cols]
            y_train = self.data.loc[self.train_idx, y_target]
            return X_train, y_train
        elif group == "test":
            X_test = self.data.loc[self.test_idx, self.X_cols]
            y_test = self.data.loc[self.test_idx, y_target]
            return X_test, y_test
        else:
            raise ValueError("group must be train or test.")

    def get_X_train_test(self):
        X_train = self.data.loc[self.train_idx, self.X_cols]
        X_test = self.data.loc[self.test_idx, self.X_cols]
        return X_train, X_test

    def get_y_train_test(self, y_target):
        y_train = self.data.loc[self.train_idx, y_target]
        y_test = self.data.loc[self.test_idx, y_target]
        return y_train, y_test

    def get_test_index(self, target_col):
        return self.data.loc[self.test_idx, target_col]

    def set_normalization_params(self):
        if self.norm_params is None:
            self.norm_params = dict(
                T=PowerX, cols=[], ncols=[], rename=None, join=1, save_tx=True
            )

    def _prep_data(self, y_target, stratify=True):
        """main calling function"""
        if self.train_idx is None:
            self.stratify_split(y_target=y_target, stratify=stratify)
        self.X_train, self.y_train = self.get_X_y("train", y_target)
        self.X_test, self.y_test = self.get_X_y("test", y_target)
        # y_train encode, reshape
        if self.encode_targets is True:
            self.y_train, self.y_test = self.encode_y(self.y_train, self.y_test)
        if self.normalize:
            self.set_normalization_params()
            self.apply_normalization(**self.norm_params)
        if self.tensors is True:
            train_test_data = [self.X_train, self.y_train, self.X_test, self.y_test]
            self.X_train, self.y_train, self.X_test, self.y_test = arrays_to_tensors(
                *train_test_data
            )

    def encode_y(self, y_train, y_test):
        return encode_target_data(y_train, y_test)

    def apply_normalization(
        self, T=PowerX, cols=[], ncols=[], rename=None, join=1, save_tx=True, save_as="tx_data.json",
    ):
        if len(cols) == 0:
            cols = self.X_cols
        if len(ncols) == 0:
            ncols = [i for i, c in enumerate(self.X_cols) if c in cols]
        self.Tx = T(
            self.X, cols, ncols=ncols, save_tx=save_tx, rename=rename, join_data=join, save_as=save_as,
        )
        self.X_train = T(
            self.X_train,
            cols,
            ncols=ncols,
            tx_data=self.Tx.tx_data,
            rename=rename,
            join_data=join,
        ).Xt
        if self.X_test is not None:
            self.X_test = T(
                self.X_test,
                cols,
                ncols=ncols,
                tx_data=self.Tx.tx_data,
                rename=rename,
                join_data=join,
            ).Xt


class HstCalPrep(Prep):
    def __init__(
        self,
        data,
        y_target,
        X_cols=[],
        norm_cols=["n_files", "total_mb"],
        rename_cols=["x_files", "x_size"],
        tensors=True,
        normalize=True,
        random=None,
        tsize=0.2,
        encode_targets=True,
    ):
        self.set_X_cols(X_cols)
        super().__init__(
            data,
            y_target,
            X_cols=self.X_cols,
            tensors=tensors,
            normalize=normalize,
            random=random,
            tsize=tsize,
            encode_targets=encode_targets,
        )
        self.norm_cols = norm_cols
        self.rename_cols = rename_cols
        self.mem_bin = data["mem_bin"]
        self.memory = data["memory"]
        self.wallclock = data["wallclock"]
        self.y_bin_train = None
        self.y_bin_test = None
        self.y_mem_train = None
        self.y_mem_test = None
        self.y_wall_train = None
        self.y_wall_test = None

    def set_X_cols(self, X_cols):
        if len(X_cols) == 0:
            self.X_cols = [
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
        else:
            self.X_cols = X_cols

    def prep_data(self):
        super().stratify_split(y_target="mem_bin", stratify=True)
        self.X_train, self.X_test = super().get_X_train_test()
        super().apply_normalization(
            T=PowerX, cols=self.norm_cols, rename=self.rename_cols, join=2
        )
        self.prep_mem_bin()
        self.prep_mem_reg()
        self.prep_wall_reg()

    def prep_mem_bin(self):
        """main calling function"""
        y_train, y_test = super().get_y_train_test("mem_bin")
        y_train, y_test = self.encode_y(y_train, y_test)
        self.y_bin_train, self.y_bin_test = y_tensors(y_train, y_test, reshape=True)

    def prep_mem_reg(self):
        y_train, y_test = super().get_y_train_test("memory")
        self.y_mem_train, self.y_mem_test = y_tensors(
            y_train.values, y_test.values, reshape=True
        )

    def prep_wall_reg(self):
        y_train, y_test = super().get_y_train_test("wallclock")
        self.y_wall_train, self.y_wall_test = y_tensors(
            y_train.values, y_test.values, reshape=True
        )


class JwstCalPrep(Prep):
    """Class for preprocessing JWST calibration pipeline metadata prior to training neural networks for estimating memory footprint.

    Parameters
    ----------
    data : pandas.DataFrame
        training dataset to be preprocessed
    y_target : str, optional
        target column name (dependent variable), by default "imgsize_gb"
    X_cols : list, optional
        feature column names (independent variables), by default []
    norm_cols : list, optional
        columns on which to apply normalization, by default []
    exp_mode : str, optional
        model training set (image, spec, tac, fgs), by default "image"
    tensors : bool, optional
        convert model inputs into tensors, by default True
    normalize : bool, optional
        apply normalization, by default True
    random : int, optional
        random seed for train-test splits, by default None
    tsize : float, optional
        test size ratio, by default 0.2
    encode_targets : bool, optional
        encode target values (categorical classifiers), by default False
    """
    def __init__(
        self,
        data,
        y_target="imgsize_gb",
        X_cols=[],
        norm_cols=[],
        exp_mode="image",
        tensors=True,
        normalize=True,
        random=None,
        tsize=0.2,
        encode_targets=False,
        **log_kws,
    ):
        self.exp_mode = exp_mode
        self.set_X_cols(X_cols)
        self.set_norm_cols(norm_cols=norm_cols)
        self.__name__ = "JwstCalPrep"
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        super().__init__(
            data,
            y_target,
            X_cols=self.X_cols,
            tensors=tensors,
            normalize=normalize,
            random=random,
            tsize=tsize,
            encode_targets=encode_targets,
        )
        self.target_data = data[self.y_target]
        self.y_reg_train = None
        self.y_reg_test = None
        self.y_bin_train = None
        self.y_bin_test = None


    def set_X_cols(self, X_cols):
        if len(X_cols) == 0:
            self.X_cols = dict(
                image=[
                    "instr",
                    "detector",
                    "visitype",
                    "filter",
                    "pupil",
                    "channel",
                    "subarray",
                    "bkgdtarg",
                    "nexposur",
                    "numdthpt",
                    "offset",
                    "max_offset",
                    "mean_offset",
                    "sigma_offset",
                    "err_offset",
                    "sigma1_mean",
                    "frac",
                    "targ_frac",
                ],
                spec=[
                    "instr",
                    "detector",
                    "visitype",
                    "filter",
                    "pupil",
                    "grating",
                    "subarray",
                    "band",
                    "nexposur",
                    "numdthpt",
                    "targ_max_offset",
                    "offset",
                    "max_offset",
                    "mean_offset",
                    "sigma_offset",
                    "err_offset",
                    "sigma1_mean",
                    "frac",
                ],
                fgs=[
                    "instr",
                    "detector",
                    "visitype",
                    "subarray",
                    "nexposur",
                    "numdthpt",
                    "crowdfld",
                    "gs_mag",
                ],
                tac=[
                    "instr",
                    "detector",
                    "visitype",
                    "exp_type",
                    "tsovisit",
                    "filter",
                    "grating",
                    "subarray",
                    "nexposur",
                    "numdthpt",
                    "targ_max_offset",
                    "offset",
                    "max_offset",
                    "mean_offset",
                    "sigma_offset",
                    "err_offset",
                    "sigma1_mean",
                    "frac",
                ],
            )[self.exp_mode]
        else:
            self.X_cols = X_cols

    def set_norm_cols(self, norm_cols=[]):
        if len(norm_cols) == 0:
            norm_cols = dict(
                image=[
                    "offset",
                    "max_offset",
                    "mean_offset",
                    "sigma_offset",
                    "err_offset",
                    "sigma1_mean",
                ],
                spec=[
                    "targ_max_offset",
                    "offset",
                    "max_offset",
                    "mean_offset",
                    "sigma_offset",
                    "err_offset",
                    "sigma1_mean",
                ]
            )[self.exp_mode]
        self.norm_cols = [c for c in norm_cols if c in self.X_cols]

    @property
    def memory_classes(self):
        return {
            0: [0,12],
            1: [12, 225],
            2: [225, 950],
            3: [950, 2000]
        }

    def classify_targets(self):
        """Creates temporary target class 'mem_bin' based on max RAM levels specified by
        `memory_classes` property.
        """
        y = self.y_target
        for c, rng in self.memory_classes.items():
            self.data.loc[(self.data[y] >= rng[0]) & (self.data[y] < rng[1]), 'mem_bin'] = c

    def prep_data(self, existing_splits=False, stratify=False):
        """Splits data into training (X_train) and test (X_test) sets and applies a PowerTransform
        normalization to each.

        Parameters
        ----------
        existing_splits : bool, optional
            Split the data using values in 'split' column, by default False
        stratify : bool, optional
            Stratify splits according to target class distribution (mem_bin), by default False
        """
        if existing_splits is True:
            if "split" not in self.data.columns:
                self.log.warning("'split' not found in data columns")
                return
            self.test_idx = self.data.loc[self.data.split == "test"].index
            self.train_idx = self.data.loc[self.data.split == "train"].index
        else:
            y_target = self.y_target
            if stratify is True:
                y_target = 'mem_bin'
                self.classify_targets()
            super().stratify_split(y_target=y_target, stratify=stratify)

        self.X_train, self.X_test = super().get_X_train_test()
        fname = f"tx_data-{self.exp_mode}.json"
        super().apply_normalization(T=PowerX, cols=self.norm_cols, rename=None, join=1, save_as=fname)
        self.X_train = self.X_train[self.X_cols]
        self.X_test = self.X_test[self.X_cols]

    def prep_targets(self):
        """main calling function"""
        y_train, y_test = super().get_y_train_test(self.y_target)
        self.y_reg_train, self.y_reg_test = y_tensors(
            y_train.values, y_test.values, reshape=True
        )


# TODO
class SvmPrep(Prep):
    def __init__(
        self,
        data,
        y_target="label",
        X_cols=[],
        tensors=True,
        normalize=False,
        random=None,
        tsize=0.2,
        encode_targets=False,
        norm_params=None,
    ):
        self.set_X_cols(X_cols)

        super().__init(
            data,
            y_target,
            X_cols=self.X_cols,
            tensors=tensors,
            normalize=normalize,
            random=random,
            tsize=tsize,
            encode_targets=encode_targets,
            norm_params=norm_params,
        )
        self.norm_cols = ["", ""]
        self.label = data["label"]
        self.y_train_labels = None
        self.y_test_labels = None

    def set_X_cols(self, X_cols):
        if len(X_cols) == 0:
            self.X_cols = [
                "numexp",
                "rms_ra",
                "rms_dec",
                "nmatches",
                "point",
                "segment",
                "gaia",
                "det",
                "wcs",
                "cat",
            ]
        else:
            self.X_cols = X_cols

    def prep_data(self):
        super()._prep_data(self.y_target, stratify=True)
