import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np


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
    def __init__(self):
        self.arr = None
        self.transformed = None
        self.invpairs = None
        self.inversed = None

    def lambda_func(self, inverse=False):
        if inverse is False:
            L = lambda x: self.keypairs[x]
            return [L(a) for a in self.arr]
        else:
            self.inverse_pairs()
            I = lambda i: self.invpairs[i]
            return [I(b) for b in self.transformed]

    def inverse_pairs(self):
        self.invpairs = {}
        for key, value in self.keypairs.items():
            self.invpairs[value] = key
        return self.invpairs

    def warn_unknowns(self):
        unknowns = np.where([a not in self.classes_ for a in self.arr])
        print(f"WARNING: Found unknown values:\n {self.arr[unknowns]}")

    def handle_unknowns(self):
        unknowns = np.where([a not in self.classes_ for a in self.arr])
        add_encoding = max(list(self.keypairs.values())) + 1
        try:
            # TODO handle multiple different unknowns
            self.keypairs[self.arr[unknowns][0]] = add_encoding
            self.classes_ = list(self.keypairs.keys())
            print("Successfully added encoding for unknown values.")
        except Exception as e:
            print("Error: unable to add encoding for unknown value(s)")
            print(e)

    def fit(self, data, keypairs, axiscol=None, handle_unknowns=True):
        if isinstance(data, pd.DataFrame):
            if axiscol is None:
                print(
                    "Error: Must indicate which column to fit if `data` is a `dataframe`."
                )
                return
            try:
                self.arr = np.asarray(data[axiscol], dtype=object)
            except Exception as e:
                print(e)
        elif isinstance(data, np.ndarray):
            if len(data.shape) > 1:
                if data.shape[-1] > 1:
                    if axiscol is None:
                        print("Error - must specify index using `axiscol`")
                        return
                    else:
                        self.arr = np.asarray(data[:, axiscol], dtype=object)
            else:
                self.arr = np.asarray(data, dtype=object)
        else:
            print("Invalid Type: `data` must be of an array or dataframe.")
            return
        self.keypairs = keypairs
        self.classes_ = list(self.keypairs.keys())
        if self.arr.any() not in self.classes_:
            # if self.arr.any() not in self.classes_:
            self.warn_unknowns()
            if handle_unknowns is True:
                self.handle_unknowns()
            else:
                return
        try:
            self.unique = np.unique(self.arr)
        except Exception as e:
            print(e)
        return self

    def transform(self):
        if self.arr is None:
            print("Error - Must fit the data first.")
            return
        self.transformed = self.lambda_func()
        return self.transformed

    def inverse_transform(self):
        inverse_pairs = {}
        for key, value in self.keypairs.items():
            inverse_pairs[value] = key
        # TODO handle unknowns/nans inversely
        self.inversed = self.lambda_func(inverse=True)
        return self.inversed

    def fit_transform(self, data, keypairs, axiscol=None):
        self.fit(data, keypairs, axiscol=axiscol)
        self.transform()


class SvmEncoder:
    """Categorical encoding class for HST Single Visit Mosiac regression test data inputs."""

    def __init__(
        self,
        data,
        fkeys=["category", "detector", "wcstype"],
        names=["cat", "det", "wcs"],
    ):
        """Instantiates an SvmEncoder class object.

        Parameters
        ----------
        data : dataframe
            input data containing features (columns) to be encoded

        fkeys: list
            categorical-type column names (str) to be encoded

        names: list
            new names to assign columns of the encoded versions of categorical data

        """
        self.data = data
        self.fkeys = fkeys
        self.names = names
        self.df = self.categorical_data()
        self.make_keypairs()

    def __repr__(self):
        return (
            "encodings: %s \n category_keys: %s \n detector_keys: %s \n wcs_keys: %s"
            % (self.encodings, self.category_keys, self.detector_keys, self.wcs_keys)
        )

    def categorical_data(self):
        """Makes a copy of input dataframe and extracts only the categorical features based on the column names in `fkeys`.

        Returns
        -------
        df: dataframe
            dataframe with only the categorical feature columns
        """
        return self.data.copy()[self.fkeys]

    def make_keypairs(self):
        """Instantiates key-pair dictionaries for each of the categorical features."""
        self.encodings = dict(zip(self.fkeys, self.names))
        self.category_keys = self.set_category_keys()
        self.detector_keys = self.set_detector_keys()
        self.wcs_keys = self.set_wcs_keys()

    def init_categories(self):
        """Assigns abbreviated character code as key-pair value for each type of target category classification (as determined by data on MAST archive).

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
            "CALIBRATION": "C",
            "UNIDENTIFIED": "U",
            "STELLAR CLUSTER": "SC",
            "EXT-CLUSTER": "SC",
            "STAR": "S",
            "EXT-STAR": "S",
            "CLUSTER OF GALAXIES": "GC",
            "GALAXY": "G",
            "None": "U",
        }

    def set_category_keys(self):
        """Assigns an integer for each abbreviated character for target category classifications (as determined by data on MAST archive). Note - this could have been directly on the raw inputs from MAST, but the extra step is done to allow for debugging and analysis purposes (value-count of abbreviated versions are printed to stdout before the final encoding).

        Returns
        -------
        dict
            key-pair values for image target category classification.
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
        return self.category_keys

    def set_detector_keys(self):
        """Assigns a hardcoded integer to each 'detector' key in alphabetical and increasing value.

        Returns
        -------
        dict
            detector names and their associated integer encoding
        """
        self.detector_keys = {"hrc": 0, "ir": 1, "sbc": 2, "uvis": 3, "wfc": 4}
        return self.detector_keys

    def set_wcs_keys(self):
        """Assigns a hardcoded integer to each 'wcs' key in alphabetical and increasing value.

        Returns
        -------
        _type_
            _description_
        """
        self.wcs_keys = {
            "a posteriori": 0,
            "a priori": 1,
            "default a": 2,
            "not aligned": 3,
        }
        return self.wcs_keys

    def svm_keypairs(self, column):
        keypairs = {
            "category": self.category_keys,
            "detector": self.detector_keys,
            "wcstype": self.wcs_keys,
        }
        return keypairs[column]

    def encode_categories(self, cname="category", sep=";"):
        """Transforms the raw string inputs from MAST target category naming conventions into an abbreviated form. For example, `CLUSTER OF GALAXIES;GRAVITATIONA` becomes `GC` for galaxy cluster; and `STELLAR CLUSTER;GLOBULAR CLUSTER` becomes `SC` for stellar cluster. This serves to group similar but differently named objects into a discrete set of 8 possible categorizations. The 8 categories will then be encoded into integer values in the final encoding step (machine learning inputs must be numeric).

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
        df_cat = pd.DataFrame.from_dict(CAT, orient="index", columns={"category"})
        self.df.drop("category", axis=1, inplace=True)
        self.df = self.df.join(df_cat, how="left")
        return self.df

    def rejoin_original(self):
        originals = list(self.encodings.keys())
        self.df.drop(originals, axis=1, inplace=True)
        self.df = self.data.join(self.df, how="left")

    def encode_features(self):
        """Encodes input features matching column names assigned to the object's ``encodings`` keys.

        Returns
        -------
        dataframe
            original dataframe with all categorical type features label-encoded.
        """
        self.encode_categories()
        print("\n\nENCODING CATEGORICAL FEATURES")
        for col, name in self.encodings.items():
            keypairs = self.svm_keypairs(col)
            enc = PairEncoder()
            enc.fit_transform(self.df, keypairs, axiscol=col)
            self.df[name] = enc.transformed
            print(f"\n*** {col} --> {name} ***")
            print(
                f"ORIGINAL:\n{self.df[col].value_counts()}\n\nENCODED:\n{self.df[name].value_counts()}\n"
            )
        self.rejoin_original()
        return self.df

    def display_encoding(self):
        print("---" * 7)
        for k, v in self.encodings.items():
            res = list(
                zip(
                    self.df[v].value_counts(),
                    self.df[v].unique(),
                    self.df[k].value_counts(),
                    self.df[k].unique(),
                )
            )
            print(f"\n{k}<--->{v}\n")
            print("#VAL\t\tENC\t\t#VAL\t\tORDINAL")
            for r in res:
                string = "\t\t".join(str(i) for i in r)
                print(string)
            print("\n")
        print("---" * 7)
