"""
Classes and methods primarily used by spacekit.dashboard but can easily be repurposed.
"""
import os
import pandas as pd
import glob
from spacekit.analyzer.compute import ComputeClassifier, ComputeRegressor

def decode_categorical(df, decoder_key):
    """Returns dataframe with added decoded column (using "{column}_key" suffix)"""
    # instrument_key = {"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}}
    # detector_key = {"det": {0: "hrc", 1: "ir", 2: "sbc", 3: "uvis", 4: "wfc"}}
    for key, pairs in decoder_key.items():
        for i, name in pairs.items():
            df.loc[df[key] == i, f"{key}_key"] = name
    return df


def import_dataset(filename=None, kwargs=dict(index_col="ipst"), decoder_key=None):
    """Imports and loads dataset from csv file via local, https, s3, or dynamodb.
    Returns Pandas dataframe.
    *args*
    src: data source ("file", "s3", or "ddb")
    uri: local file path
    kwargs: dict of keyword args to pass into pandas read_csv method e.g. set index_col: kwargs=dict(index_col="ipst")
    decoder_key: nested dict of column and key value pairs for decoding a categorical feature into strings
    Ex: {"instr": {{0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}}}
    """
    if not os.path.exists(filename):
        print("File could not be found")
    # load dataset
    df = pd.read_csv(filename, **kwargs)
    if decoder_key:
        df = decode_categorical(df, decoder_key)  # adds instrument label (string)
    return df

class MegaScanner:
    def __init__(self, prefix=f"data/20??-*-*-*"):
        self.prefix = prefix
        self.datasets = sorted(list(glob.glob(prefix)))
        self.timestamps = [int(t.split('-')[-1]) for t in self.datasets] # [1636048291, 1635457222, 1629663047]
        self.dates = [v[5:15] for v in self.datasets] # ["2021-11-04", "2021-10-28", "2021-08-22"]
        self.selection = None
        self.versions = None
        self.res_keys = None # {"mem_bin": {}, "memory": {}, "wallclock": {}}
        self.mega = None

    def select_dataset(self):
        if self.selection is None:
            self.selection = f"{self.dates[-1]}-{self.timestamps[-1]}"
        self.dataset = glob.glob(f"data/{self.selection}/*.csv")[0]

    def make_mega(self):
        self.mega = {}
        versions = []
        for i, (d, t) in enumerate(zip(self.dates, self.timestamps)):
            if self.versions is None:
                v = f"v{str(i)}"
                versions.append(v)
            else:
                v = self.versions[i]
            self.mega[v] = {"date": d, "time": t, "res": self.res_keys}
        if len(versions) > 0:
            self.versions = versions
        return self.mega

class CalRes(MegaScanner):
    def __init__(self, prefix=f"data/20??-*-*-*"):
        super().init(prefix)
        self.classes = [0,1,2,3]
        self.res_keys = {"mem_bin": {}, "memory": {}, "wallclock": {}}
    
    def scan_results(self):
        self.mega = self.make_mega()
        for i, d in enumerate(self.datasets):
            v = self.versions[i]
            bCom = ComputeClassifier(computation="clf", classes=[0,1,2,3], res_path=f"{d}/results/mem_bin")
            bCom.upload()
            self.mega[v]["res"]["mem_bin"] = bCom
            mCom = ComputeRegressor(computation="reg", res_path=f"{d}/results/memory")
            mCom.upload()
            self.mega[v]["res"]["memory"] = mCom
            wCom = ComputeRegressor(computation="reg", res_path=f"{d}/results/wallclock")
            wCom.upload()
            self.mega[v]["res"]["wallclock"] = wCom
        return self.mega

    # TODO: update results files and get rid of this
    def get_scores(self):
        df_list = []
        for v in self.versions:
            score_dict = self.mega[v]["res"]["mem_bin"]["scores"]
            df = pd.DataFrame.from_dict(score_dict, orient="index", columns=[v])
            df_list.append(df)
        df_scores = pd.concat([d for d in df_list], axis=1)
        return df_scores

class SvmRes(MegaScanner):
    def __init__(self, prefix=f"data/20??-*-*-*"):
        super().__init__(prefix)
        self.datasets = sorted(list(glob.glob(prefix)))
        self.timestamps = [int(t.split('-')[-1]) for t in self.datasets]
        self.dates = [v[5:15] for v in self.datasets]
        self.selection = f"{self.dates[-1]}-{self.timestamps[-1]}"
        self.dataset = glob.glob(f"data/{self.selection}/*.csv")[0]
        self.classes = ["aligned", "misaligned"]
        self.res_keys = {"test": {}, "val": {}}

    def scan_results(self):
        self.mega = self.make_mega()
        for i, d in enumerate(self.datasets):
            v = self.versions[i]
            tCom = ComputeClassifier(algorithm="test", classes=self.classes, res_path=f"{d}/results/test")
            tCom.upload()
            self.mega[v]["res"]["test"] = tCom

            vCom = ComputeClassifier(algorithm="val", classes=self.classes, res_path=f"{d}/results/val")
            vCom.upload()
            self.mega[v]["res"]["val"] = vCom
