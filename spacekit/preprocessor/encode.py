import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def encode_target_data(y_train, y_test):
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


class Encoder:
    def __init__(self, df):
        self.df = df


class SvmEncoder(Encoder):
    def __init__(self, data):
        self.data = data
        self.df = data.copy()
        self.sep = ";"
        self.encodings = {"wcstype": "wcs", "cat": "cat", "detector": "det"}
        self.category_keys = self.set_category_keys()
        self.df_cat = self.encode_categories()

    def set_category_keys(self):
        self.category_keys = {
            "CALIBRATION": "C",
            "SOLAR SYSTEM": "SS",
            "ISM": "I",
            "EXT-MEDIUM": "I",
            "UNIDENTIFIED": "U",
            "STELLAR CLUSTER": "SC",
            "EXT-CLUSTER": "SC",
            "STAR": "S",
            "EXT-STAR": "S",
            "CLUSTER OF GALAXIES": "GC",
            "GALAXY": "G",
        }
        return self.category_keys

    def encode_categories(self):
        print("\n*** Encoding Category Names ***")
        CAT = {}
        for idx, cat in self.df.category.items():
            c = cat.split(self.sep)[0]
            if c in self.category_keys:
                CAT[idx] = self.category_keys[c]
        self.df_cat = pd.DataFrame.from_dict(CAT, orient="index", columns={"cat"})
        print("\nCategory encoding complete.")
        print(self.df_cat["cat"].value_counts())
        self.df = self.df.join(self.df_cat, how="left")
        return self.df

    def encode_features(self):
        for col, name in self.encodings.items():
            encoder = LabelEncoder().fit(self.df[col])
            self.df[name] = encoder.transform(self.df[col])
        return self.df
