import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from tensorflow.keras.utils import to_categorical


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


# class Encoder:
#     def __init__(self, df):
#         self.df = df


class SvmEncoder:
    """Categorical encoding class for HST Single Visit Mosiac regression test data inputs."""

    def __init__(self, data):
        """Instantiates an SvmEncoder class object.

        Parameters
        ----------
        data : dataframe
            input data containing features (columns) to be encoded
        """
        self.data = data
        self.sep = ";"
        self.category_keys = self.set_category_keys()
        self.categories = ['GC', 'G', 'I', 'S', 'C', 'U', 'SC', 'SS']
        self.detectors = ['hrc', 'ir', 'sbc', 'uvis', 'wfc']
        self.wcstypes = ['a posteriori', 'a priori', 'default a', 'not aligned']
        self.encodings = {"ctg": "cat", "detector": "det", "wcstype": "wcs"}
        self.groups = self.set_groups()
        self.df_cat = self.encode_categories()
        self.df = self.data.join(self.df_cat, how="left")
        self.ordinal = self.df[list(self.encodings.keys())] # 'ctg','detector','wcstype'
        self.c = [v for v in self.groups.values()]
        self.encoder = OrdinalEncoder(categories=self.c, dtype='int')
        self.encoded = None
    
    def set_groups(self):
        names = list(self.encodings.keys())
        grps = [self.categories, self.detectors, self.wcstypes]
        self.groups = dict(zip(names, grps))
        return self.groups

    def set_category_keys(self):
        """Assigns abbreviated character code as key-pair value for each type of target category classification (as determined by data on MAST archive).

        Returns
        -------
        dict
            key-pair values for image target category classification.
        """
        self.category_keys = {
            "CLUSTER OF GALAXIES": "GC",
            "GALAXY": "G",
            "ISM": "I",
            "EXT-MEDIUM": "I",
            "STAR": "S",
            "EXT-STAR": "S",
            "CALIBRATION": "C",
            "UNIDENTIFIED": "U",
            "STELLAR CLUSTER": "SC",
            "EXT-CLUSTER": "SC",
            "SOLAR SYSTEM": "SS",
        }
        return self.category_keys

    def encode_categories(self):
        """Encodes the target categories of a dataframe as integer (numeric) datatype, which is required for machine learning inputs.

        Returns
        -------
        dataframe
            original dataframe with category input feature values encoded.
        """
        print("\n*** Abbreviating HAP Categories ***")
        CAT = {}
        for idx, cat in self.data.category.items():
            c = cat.split(self.sep)[0]
            if c in self.category_keys:
                CAT[idx] = self.category_keys[c]
            else:
                CAT[idx] = "U"
        self.df_cat = pd.DataFrame.from_dict(CAT, orient="index", columns={"ctg"})
        print("\nHAP Category abbreviation complete.")
        print(self.df_cat["ctg"].value_counts())
        return self.df_cat

    def encode_features(self):
        """Encodes input features matching column names assigned to the object's ``encodings`` keys.

        Returns
        -------
        dataframe
            original dataframe with all categorical type features label-encoded.
        """
        self.encoder.fit(self.ordinal)
        self.encoded = self.encoder.transform(self.ordinal)
        self.encoded = pd.DataFrame(self.encoded, index=self.ordinal.index, columns=list(self.encodings.values()))
        self.df = self.df.join(self.encoded, how="left")
        return self.df
    
    def display_encoding(self):
        print("---"*7)
        for k, v in self.encodings.items():
            res = list(zip(self.df[v].value_counts(), self.df[v].unique(), self.df[k].value_counts(), self.df[k].unique()))
            print(f"\n{k}<--->{v}\n")
            print(f"#VAL\t\tENC\t\t#VAL\t\tORDINAL")
            for r in res:
                string = f'\t\t'.join(str(i) for i in r)
                print(string)
            print("\n")
        print("---"*7)
