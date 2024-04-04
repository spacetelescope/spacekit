import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np

from spacekit.logger.log import Logger


def boolean_encoder(x):
    if x in [True, "True", "T", "t"]:
        return 1
    else:
        return 0


def nan_encoder(x, truevals):
    if x in truevals:
        return 1
    else:
        return 0


def encode_booleans(df, cols, special=False, replace=False, rename="", verbose=False):
    cols = [c for c in cols if c in df.columns]
    if verbose:
        print(f"\nNaNs to be NaNdled:\n{df[cols].isna().sum()}\n")
    df_bool = df[cols].copy()
    encoded_cols = []
    sfx = "_enc" if not rename else rename
    for col in cols:
        enc_col = f"{col}{sfx}"
        encoded_cols.append(enc_col)
        if special is True:
            truevals = list(df_bool[col].value_counts().index)
            df_bool[enc_col] = df_bool[col].apply(lambda x: nan_encoder(x, truevals))
        else:
            df_bool[enc_col] = df_bool[col].apply(lambda x: boolean_encoder(x))
    # merge back into original dataframe
    df_bool.drop(cols, axis=1, inplace=True)
    df = pd.concat([df, df_bool], axis=1)
    if replace is True:
        df.drop(cols, axis=1, inplace=True)
        if not rename:  # make encoded colnames same as the originals
            df.rename(dict(zip(encoded_cols, cols)), axis=1, inplace=True)
            encoded_cols = cols
    return df


def encode_target_data(y_train, y_test):
    """Label encodes target class training and test data for multi-classification models.

    Parameters
    ----------
    y_train : dataframe or ndarray
        training target data
    y_test : dataframe or ndarray
        test target data

    Returns
    -------
    ndarrays
        y_train, y_test
    """
    # label encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train_enc = encoder.transform(y_train)
    y_train = to_categorical(y_train_enc)
    # test set
    encoder.fit(y_test)
    y_test_enc = encoder.transform(y_test)
    y_test = to_categorical(y_test_enc)
    # ensure train/test targets have correct shape (4 bins)
    print(y_train.shape, y_test.shape)
    return y_train, y_test


class PairEncoder:
    def __init__(self, name="PairEncoder", **log_kws):
        self.__name__ = name
        self.arr = None
        self.transformed = None
        self.invpairs = None
        self.inversed = None
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()

    def lambda_func(self, inverse=False):
        if inverse is False:

            def L(x):
                return self.keypairs[x]

            return [L(a) for a in self.arr]
        else:
            self.inverse_pairs()

            def inv(i):
                return self.invpairs[i]

            return [inv(b) for b in self.transformed]

    def inverse_pairs(self):
        self.invpairs = {}
        for key, value in self.keypairs.items():
            self.invpairs[value] = key
        return self.invpairs

    def handle_unknowns(self, unknowns):
        uvals = np.unique(self.arr[unknowns])
        self.log.warning(f"Found unknown values:\n {uvals}")
        try:
            for u in uvals:
                add_encoding = max(list(self.keypairs.values())) + 1
                self.keypairs[u] = add_encoding
                self.classes_ = list(self.keypairs.keys())
        except Exception as e:
            self.log.error("Unable to add encoding for unknown value(s)", e)

    def fit(self, data, keypairs, axiscol=None, handle_unknowns=True):
        if isinstance(data, pd.DataFrame):
            if axiscol is None:
                self.log.error(
                    "Must indicate which column to fit if `data` is a `dataframe`."
                )
                return
            try:
                self.arr = np.asarray(data[axiscol], dtype=object)
            except Exception as e:
                self.log.error(e)
        elif isinstance(data, np.ndarray):
            if len(data.shape) > 1:
                if data.shape[-1] > 1:
                    if axiscol is None:
                        self.log.error("Must specify index using `axiscol`")
                        return
                    else:
                        self.arr = np.asarray(data[:, axiscol], dtype=object)
            else:
                self.arr = np.asarray(data, dtype=object)
        else:
            self.log.error("Invalid Type: `data` must be of an array or dataframe.")
            return
        self.keypairs = keypairs
        self.classes_ = list(self.keypairs.keys())
        unknowns = np.where([a not in self.classes_ for a in self.arr])[0]
        if unknowns.shape[0] > 0:
            if handle_unknowns is True:
                self.handle_unknowns(unknowns)
            else:
                self.log.error(
                    f"Found unknown values in {axiscol}:\n {self.arr[unknowns]}"
                )
                return
        try:
            self.unique = np.unique(self.arr)
        except Exception as e:
            self.log.error(e)
        return self

    def transform(self):
        if self.arr is None:
            self.log.error("Must fit the data first.")
            return
        self.transformed = self.lambda_func()
        return self.transformed

    def inverse_transform(self):
        self.inversed = self.lambda_func(inverse=True)
        return self.inversed

    def fit_transform(self, data, keypairs, axiscol=None):
        self.fit(data, keypairs, axiscol=axiscol)
        self.transform()


class CategoricalEncoder:
    def __init__(
        self,
        data,
        fkeys=[],
        names=[],
        drop=False,
        rename=False,
        keypair_file=None,
        encoding_pairs=None,
        verbose=0,
        name="CategoricalEncoder",
        **log_kws,
    ):
        self.data = data
        self.fkeys = fkeys
        self.names = names
        self.drop = drop
        self.rename = rename
        self.keypair_file = keypair_file
        self.encoding_pairs = encoding_pairs
        self.verbose = verbose
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        self.encodings = dict(zip(self.fkeys, self.names))
        self.df = self.categorical_data()

    def categorical_data(self):
        """Makes a copy of input dataframe and extracts only the categorical features based on the column names in `fkeys`.

        Returns
        -------
        df: dataframe
            dataframe with only the categorical feature columns
        """
        return self.data.copy()[self.fkeys]

    def rejoin_original(self):
        encoded = list(self.encodings.values())
        originals = list(self.encodings.keys())
        self.df.drop(originals, axis=1, inplace=True)
        self.df = self.data.join(self.df, how="left")
        if self.verbose:
            self.display_encoding()
        if self.drop is True:
            self.df.drop(originals, axis=1, inplace=True)
        if self.rename is True:
            self.df.rename(dict(zip(encoded, originals)), axis=1, inplace=True)

    def _encode_features(self):
        """Encodes input features matching column names assigned to the object's ``encodings`` keys.

        Returns
        -------
        dataframe
            original dataframe with all categorical type features label-encoded.
        """
        if self.encoding_pairs is None:
            self.log.error(
                "encoding_pairs attr must be instantiated with key-value pairs"
            )
            return
        self.log.debug("Encoding categorical features...")
        for col, name in self.encodings.items():
            keypairs = self.encoding_pairs[col]
            enc = PairEncoder()
            enc.fit_transform(self.df, keypairs, axiscol=col)
            self.df[name] = enc.transformed
            if self.verbose:
                self.log.debug(f"*** {col} --> {name} ***")
                self.log.debug(
                    f"\n\nORIGINAL:\n{self.df[col].value_counts()}\n\nENCODED:\n{self.df[name].value_counts()}\n"
                )
        self.rejoin_original()
        return self.df

    def display_encoding(self):
        self.log.info("---" * 7)
        for k, v in self.encodings.items():
            res = list(
                zip(
                    self.df[v].value_counts(),
                    self.df[v].unique(),
                    self.df[k].value_counts(),
                    self.df[k].unique(),
                )
            )
            self.log.info(f"{k}<--->{v}")
            self.log.info("#VAL\t\tENC\t\t#VAL\t\tORDINAL")
            for r in res:
                string = "\t\t".join(str(i) for i in r)
                self.log.info(string)
        self.log.info("---" * 7)

    def load_keypair_file(self):
        if os.path.exists(self.keypair_file):
            with open(self.keypair_file, "r") as j:
                self.encoding_pairs = json.load(j)

    def save_keypair_file(self, fpath):
        with open(fpath, "w") as f:
            json.dump(self.encoding_pairs, f)


class HstSvmEncoder(CategoricalEncoder):
    """Categorical encoding class for HST Single Visit Mosiac regression test data inputs."""

    def __init__(
        self,
        data,
        fkeys=["category", "detector", "wcstype"],
        names=["cat", "det", "wcs"],
        drop=False,
        rename=False,
        keypair_file=None,
        encoding_pairs=None,
        **log_kws,
    ):
        """Instantiates an HstSvmEncoder class object.

        Parameters
        ----------
        data : dataframe
            input data containing features (columns) to be encoded

        fkeys: list
            categorical-type column names (str) to be encoded

        names: list
            new names to assign columns of the encoded versions of categorical data

        """
        super().__init__(
            data,
            fkeys=fkeys,
            names=names,
            drop=drop,
            rename=rename,
            keypair_file=keypair_file,
            encoding_pairs=encoding_pairs,
            name="HstSvmEncoder",
            **log_kws,
        )
        self.make_keypairs()
        self.encode_categories()

    def __repr__(self):
        return (
            "encodings: %s \n category_keys: %s \n detector_keys: %s \n wcs_keys: %s"
            % (self.encodings, self.category_keys, self.detector_keys, self.wcs_keys)
        )

    def encode_features(self):
        return super()._encode_features()

    def make_keypairs(self):
        """Instantiates key-pair dictionaries for each of the categorical features listed in `fkeys`. Except for the target
        classification "category" feature, each string value is assigned an integer in alphabetical and increasing order,
        respectively. For the image target category feature, an integer is assigned to each abbreviated version of strings
        collected from the MAST archive). The extra abbreviation step is done to allow for debugging and analysis purposes
        (value-count of abbreviated versions are printed to stdout before the final encoding).

        Returns
        -------
        dict
            key-pair values for image target category classification (category), detectors and wcstype.
        """
        self.category_keys = {
            "C": 0,
            "SS": 1,
            "I": 2,
            "U": 3,
            "SC": 4,
            "S": 5,
            "GC": 6,
            "G": 7,
        }
        self.detector_keys = {"hrc": 0, "ir": 1, "sbc": 2, "uvis": 3, "wfc": 4}
        self.wcs_keys = {
            "a posteriori": 0,
            "a priori": 1,
            "default a": 2,
            "not aligned": 3,
        }
        self.encoding_pairs = {
            "category": self.category_keys,
            "detector": self.detector_keys,
            "wcstype": self.wcs_keys,
        }

    def init_categories(self):
        """Assigns abbreviated character code as key-pair value for each type of target category classification (as determined by
        data on MAST archive).

        Returns
        -------
        dict
            key-pair values for image target category classification.
        """
        return {
            "CALIBRATION": "C",
            "SOLAR SYSTEM": "SS",
            "ISM": "I",
            "EXT-MEDIUM": "I",
            "STAR": "S",
            "EXT-STAR": "S",
            "UNIDENTIFIED": "U",
            "STELLAR CLUSTER": "SC",
            "EXT-CLUSTER": "SC",
            "CLUSTER OF GALAXIES": "GC",
            "GALAXY": "G",
            "None": "U",
        }

    def encode_categories(self, cname="category", sep=";"):
        """Transforms the raw string inputs from MAST target category naming conventions into an abbreviated form. For example,
        `CLUSTER OF GALAXIES;GRAVITATIONA` becomes `GC` for galaxy cluster; and `STELLAR CLUSTER;GLOBULAR CLUSTER` becomes `SC`
        for stellar cluster. This serves to group similar but differently named objects into a discrete set of 8 possible
        categorizations. The 8 categories will then be encoded into integer values in the final encoding step (machine learning
        inputs must be numeric).

        Returns
        -------
        dataframe
            original dataframe with category input feature values encoded.
        """
        CAT = {}
        ckeys = self.init_categories()
        for idx, cat in self.df[cname].items():
            c = cat.split(sep)[0]
            if c in ckeys:
                CAT[idx] = ckeys[c]
        df_cat = pd.DataFrame.from_dict(CAT, orient="index", columns=["category"])
        self.df.drop("category", axis=1, inplace=True)
        self.df = self.df.join(df_cat, how="left")
        return self.df


class HstCalEncoder(CategoricalEncoder):

    """Categorical encoding class for HST Calibration in the Cloud Reprocessing inputs."""

    def __init__(
        self,
        data,
        fkeys=["DETECTOR", "SUBARRAY", "DRIZCORR", "PCTECORR"],
        names=["detector", "subarray", "drizcorr", "pctecorr"],
        keypair_file=None,
        encoding_pairs=None,
        **log_kws,
    ):
        """Instantiates a CalEncoder class object.

        Parameters
        ----------
        data : dataframe
            input data containing features (columns) to be encoded

        fkeys: list
            categorical-type column names (str) to be encoded

        names: list
            new names to assign columns of the encoded versions of categorical data

        """
        self.fkeys = fkeys
        self.names = names
        super().__init__(
            data,
            fkeys=fkeys,
            names=names,
            keypair_file=keypair_file,
            encoding_pairs=encoding_pairs,
            name="HstCalEncoder",
            **log_kws,
        )
        self.make_keypairs()

    def __repr__(self):
        return "encodings: %s \n keypairs: %s \n" % (
            self.encodings,
            self.encoding_pairs,
        )

    def set_calibration_keys(self):
        return {
            "PERFORM": 1,
            "OTHER": 0,
        }

    def set_detector_keys(self):
        return {"UVIS": 1, "WFC": 1, "OTHER": 0}

    def set_subarray_keys(self):
        return {"True": 1, "False": 0}

    def set_crsplit_keys(self):
        return {"NaN": 0, "1.0": 1, "OTHER": 2}

    def set_dtype_keys(self, i):
        return {"0": 1, "OTHER": 0}

    def set_instr_keys(self, i):
        return dict(j=0, l=1, o=2, i=3)

    def make_keypairs(self):
        self.encoding_pairs = dict(
            drizcorr=self.set_calibration_keys(),
            pctecorr=self.set_calibration_keys(),
            detector=self.set_detector_keys(),
            subarray=self.set_subarray_keys(),
            crsplit=self.set_crsplit_keys(),
            dtype=self.set_dtype_keys(),
            instr=self.set_instr_keys(),
        )

    def encode_features(self):
        super()._encode_features()


class JwstEncoder(CategoricalEncoder):
    def __init__(
        self,
        data,
        fkeys=[],
        names=[],
        drop=True,
        rename=True,
        encoding_pairs=None,
        keypair_file=None,
        **log_kws,
    ):
        if not names:
            names = [c + "_enc" for c in fkeys]
        super().__init__(
            data,
            fkeys=fkeys,
            names=names,
            drop=drop,
            rename=rename,
            keypair_file=keypair_file,
            encoding_pairs=encoding_pairs,
            name="JwstEncoder",
            **log_kws,
        )
        # self.make_keypairs() # for training only

    def make_keypairs(self):
        """Instantiates key-pair dictionaries for each of the categorical features."""
        self.abbreviate_strings(self, cname="subarray", ckeys=["MASK", "SUB", "WFSS"])
        keymaker = CategoricalKeymaker(
            self.df, list(self.df.columns), recast=["channel"]
        )
        self.encoding_pairs = keymaker.encode_categories()

    def abbreviate_strings(self, cname="subarray", ckeys=["MASK", "SUB", "WFSS"]):
        """Abbreviates the original values based on the starting values of the string.
        For example, if "MASK" is passed as a value in the `ckeys` keyword arg,
        any value starting with "MASK" within the `cname` column is shortened to "MASK".
        For the "subarray" column in JWST, this reduces the number of possible encodings to 7 values.
        The 7 subarray values will then be encoded into integers in the final encoding step.

        Returns
        -------
        dataframe
            original dataframe with subarray input feature values encoded.
        """
        for key in ckeys:
            self.df.loc[self.df[cname].str.startswith(key), cname] = key

    def encode_features(self):
        super()._encode_features()


class CategoricalKeymaker:
    def __init__(
        self,
        df,
        cols,
        keypair_file=None,
        recast=[],
        codify=[],
        forced_zeros={},
        name="CategoricalKeymaker",
        **log_kws,
    ):
        self.df = df
        self.cols = [c for c in cols if c in self.df.columns]
        self.keypair_file = keypair_file
        self.recast = recast
        self.codify = codify
        self.forced_zeros = forced_zeros
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        self.non_defaults()
        self.set_default_kwargs()
        self.set_recast_kwargs()
        self.set_codify_kwargs()
        self.set_encoding_kwargs()

    def load_keypair_data(self, keypair_file):
        if os.path.exists(keypair_file):
            with open(keypair_file, "r") as j:
                self.encoding_pairs = json.load(j)

    def save_keypair_data(self, fpath):
        with open(fpath, "w") as f:
            json.dump(self.encoding_pairs, f)

    def encode_categories(self, inverse=True):
        self.encoding_pairs = {}
        for col in self.cols:
            try:
                enc_kwargs = self.encoding_kwargs.get(col, self.default_kwargs)
                encoding_key = self.make_encoding_key(col, **enc_kwargs)
                self.encoding_pairs[col] = encoding_key
            except KeyError:
                self.log.error(f"Key Error occurred while encoding {col}")
        if inverse is True:
            enc_pairs = {}
            for col, pairs in self.encoding_pairs.items():
                enc_pairs[col] = {}
                for k, v in pairs.items():
                    enc_pairs[col][v] = k
            self.encoding_pairs = enc_pairs
        return self.encoding_pairs

    def make_encoding_key(self, col, forced_zero="NONE", recast=None, codify=None):
        # convert values / apply datatype recasting
        if recast:
            self.recast_data(col, **recast)

        keypairs = None
        if codify:
            # convert long string to abbreviated string prior to numeric encoding
            coded, keypairs = self.codify_keypairs(
                col=col, forced_zero=forced_zero, **codify
            )
            self.df[col + "_c"] = self.df[col].apply(
                lambda x: self.abbreviator(x, keypairs)
            )
            col += "_c"
        else:
            coded = self.make_default_keypairs(col, zero_val=forced_zero)

        encoding_key = dict(zip(coded.values(), coded.keys()))

        if keypairs:
            for i, j in encoding_key.items():
                for k, v in keypairs.items():
                    if j == v:
                        encoding_key[i] = {j: k}
        return encoding_key

    def get_inversed_keypairs(self):
        self.inversed_pairs = dict()
        for column, keypairs in self.encoding_pairs.items():
            inverse_pairs = self.inverse_keypairs(keypairs)
            self.inversed_pairs[column] = inverse_pairs

    def inverse_keypairs(self, keypairs):
        inverse_pairs = {}
        for k, v in keypairs.items():
            if isinstance(v, dict):
                for i, j in v.items():
                    inverse_pairs[j] = k
            else:
                inverse_pairs[v] = k
        return inverse_pairs

    def keypair_encoder(self, x, keypairs, col):
        if x not in list(keypairs.values()):
            self.log.warning(f"New value not in keypairs - adding {x}...")
            keys = sorted(int(k) for k in list(keypairs.keys()))
            new_key = keys[-1] + 1
            keypairs[str(new_key)] = x
            self.encoding_pairs[col] = keypairs
            self.inversed_pairs[col] = self.inverse_keypairs(keypairs)
            return new_key
        else:
            return self.inversed_pairs[col].get(x)

    def encode_from_keypairs(self):
        self.get_inversed_keypairs()
        for col in self.cols:
            keypairs = self.encoding_pairs.get(col, None)
            if keypairs:
                self.df[col] = self.df[col].apply(
                    lambda x: self.keypair_encoder(x, keypairs, col)
                )
        return self.df

    def non_defaults(self):
        non_default_cols = self.recast + self.codify + list(self.forced_zeros.keys())
        self.non_default_cols = list(set(non_default_cols))

    def set_encoding_kwargs(self):
        self.encoding_kwargs = dict()
        for col in self.df.columns:
            if col in self.non_default_cols:
                recast_kwargs = self.recast_kwargs if col in self.recast else None
                codify_kwargs = self.codify_kwargs if col in self.codify else None
                forced_zero = self.forced_zeros.get(col, "NONE")
                self.encoding_kwargs.update(
                    {
                        col: dict(
                            forced_zero=forced_zero,
                            recast=recast_kwargs,
                            codify=codify_kwargs,
                        )
                    }
                )

    def set_default_kwargs(self, forced_zero="NONE", recast=None, codify=None):
        self.default_kwargs = dict(
            forced_zero=forced_zero, recast=recast, codify=codify
        )

    def set_recast_kwargs(
        self, stringify=True, splitify=True, make_upper=True, splitter=".", i=0
    ):
        self.recast_kwargs = dict(
            stringify=stringify,
            splitify=splitify,
            make_upper=make_upper,
            splitter=splitter,
            i=i,
        )

    def set_codify_kwargs(self, abbr=True, keep_orig=False, inverse=True):
        self.codify_kwargs = dict(
            abbr=abbr,
            keep_orig=keep_orig,
            inverse=inverse,
        )

    def find_unique_values(self, col="visitype"):
        val_types = []
        for val in list(self.df[col].value_counts().index):
            if isinstance(val, list):
                for t in val:
                    val_types.append(t)
            else:
                val_types.append(val)
        val_types = sorted(list(set(val_types)))
        vtypes = []
        for v in val_types:
            if not isinstance(v, str):
                v = str(int(v))
            vtypes.append(v)
        return list(set(vtypes)), list(set(val_types))

    def abbreviate_names(self, vtypes, strips=".+"):
        vtypes_new = []
        for v in vtypes:
            if strips:
                v = v.strip(strips)
            words = v.split("_")
            name = ""
            for w in words:
                name += w[0]
            vtypes_new.append(name)
        return vtypes_new

    def match_keypairs(self, vtypes1, vtypes2):
        keypairs = {}
        for v2 in vtypes2:
            keypairs[v2] = []
        for v1 in vtypes1:
            if not isinstance(v1, str):
                v = str(int(v1))
            else:
                v = v1
            keypairs[v].append(v1)
        return keypairs

    def create_keypair_dict(self, col, abbr=True, keep_orig=False, inverse=True):
        keypairs = {}
        if keep_orig is True:
            vtypes2, vtypes = self.find_unique_values(col=col)
        else:
            vtypes, _ = self.find_unique_values(col=col)
            vtypes2 = None

        if vtypes2 is not None:
            keypairs = self.match_keypairs(vtypes, vtypes2)

        vtypes_new = self.abbreviate_names(vtypes) if abbr is True else None
        if vtypes_new is not None:
            for a, b in list(zip(vtypes, vtypes_new)):
                keypairs[b] = [a]
        if inverse is True:
            keypairs_inv = {}
            for k, v in keypairs.items():
                keypairs_inv[v[0]] = k
            keypairs = keypairs_inv
        return keypairs

    def make_default_keypairs(self, col, zero_val="NONE"):
        keys = sorted(list(self.df[col].unique()))
        if zero_val in keys and keys[0] != zero_val:
            try:
                idx = np.where([np.asarray(keys) == zero_val])[1][0]
                self.log.info(f"Moving {zero_val} index from {idx} to 0")
                keys.pop(idx)
                keys.insert(0, zero_val)
            except Exception as e:
                self.log.error(
                    "Unable to locate zero_val index while making default keypairs",
                    str(e),
                )
        vals = list(range(len(keys)))
        keypairs = dict(zip(keys, vals))
        return keypairs

    def codify_keypairs(self, col, forced_zero="NONE", **kwargs):
        # convert long string to abbreviated string prior to numeric encoding
        # e.g. 'PRIME_UNTARGETED': 'PU'
        keypairs = self.create_keypair_dict(col, **kwargs)
        forced_key = keypairs[forced_zero]
        keylist = sorted(list(keypairs.values()))
        if keylist[0] != forced_zero:
            del keypairs[forced_zero]
            keypairs_zero = {forced_zero: forced_key}
            keypairs_zero.update(keypairs)
            keylist = sorted(list(keypairs_zero.values()))
            keypairs = keypairs_zero
        keys = list(dict(enumerate(keylist)).values())
        vals = list(dict(enumerate(keylist)).keys())
        coded_keys = dict(zip(keys, vals))
        return coded_keys, keypairs

    def abbreviator(self, x, keypairs):
        return keypairs.get(x, x)

    def string_encoder(self, x, coded):
        return coded.get(x, x)

    def split_caster(self, x, splitter=".", i=0):
        # split string on splitter, return index i of split string
        # e.g. "13.0" becomes "13"
        return x.split(splitter)[i]

    def string_caster(self, x):
        # convert to string
        if not isinstance(x, str):
            return str(x)
        else:
            return x

    def recast_data(
        self, col, stringify=True, splitify=True, make_upper=True, **kwargs
    ):
        if stringify is True:
            self.df[col] = self.df[col].apply(lambda x: self.string_caster(x))
        if splitify is True:
            self.df[col] = self.df[col].apply(lambda x: self.split_caster(x, **kwargs))
        if make_upper is True:
            self.df[col] = self.df[col].apply(lambda x: x.upper())
