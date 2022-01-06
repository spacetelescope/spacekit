import os
import pandas as pd
import numpy as np
from astropy.io import fits
from sklearn.model_selection import train_test_split
from spacekit.extractor.scrape import MastScraper
from spacekit.preprocessor.encode import SvmEncoder
from spacekit.preprocessor.encode import encode_target_data
from spacekit.preprocessor.transform import array_to_tensor


class ScrubCols:
    def __init__(self, data, dropnans=True):
        self.data = data
        self.dropnans = dropnans
        self.df = self.data.copy()  # self.scrub_columns()
        self.new_cols = self.set_new_cols()
        self.col_pfx = self.set_prefix_cols()
        self.raw_cols = self.get_raw_cols()

    def scrub_columns(self):
        """Main calling function"""
        self.df = self.rename_cols()
        extract = [c for c in self.new_cols if c in self.df.columns]
        self.df = self.df[extract]
        print("New column names: ", self.df.columns)
        if self.dropnans:
            self.df.dropna(axis=0, inplace=True)
        index_data = self.split_index()
        self.df = index_data.join(self.df, how="left")
        self.df.rename({"number_of_gaia_sources": "gaia"}, axis=1, inplace=True)
        return self.df

    def set_new_cols(self):
        self.new_cols = [
            "targname",
            "ra_targ",
            "dec_targ",
            "numexp",
            "imgname",
            "point",
            "segment",
            "number_of_gaia_sources",
        ]
        return self.new_cols

    def set_prefix_cols(self):
        self.col_pfx = [
            "header",
            "gen_info",
            "number_of_sources",
            "Number_of_GAIA_sources.",  # incl. trailing period
        ]
        return self.col_pfx

    def get_raw_cols(self):
        print("\n*** Extracting FITS header prefix columns ***")
        self.raw_cols = []
        for c in self.col_pfx:
            self.raw_cols += [col for col in self.df if c in col]
        return self.raw_cols

    def col_splitter(self, splitter="."):
        cols = [col.split(splitter)[-1].lower() for col in self.df.columns]
        return cols

    def rename_cols(self):
        print("\nRenaming columns")
        cols = self.col_splitter()
        hc = dict(zip(self.df.columns, cols))
        self.df.rename(hc, axis="columns", inplace=True)
        return self.df

    def split_index(self):
        idx_dct = {}
        for idx, _ in self.df.iterrows():
            n = str(idx)
            items = n.split("_")
            idx_dct[n] = {}
            idx_dct[n]["detector"] = items[4]
            if len(items) > 7:
                idx_dct[n]["dataset"] = "_".join(items[6:])
                # idx_dct[n]['dataset'] = items[6][:6] + '_' + items[7]
            else:
                idx_dct[n]["dataset"] = items[6]
                # idx_dct[n]['dataset'] = items[6][:6]
        index_df = pd.DataFrame.from_dict(idx_dct, orient="index")
        return index_df


class ScrubFits:
    def __init__(self, data, input_path):
        self.data = data
        self.df = self.data.copy()
        self.input_path = input_path
        self.fits_keys = ["rms_ra", "rms_dec", "nmatches", "wcstype"]
        self.drz_paths = self.find_drz_paths()
        # self.data = self.extract_fits_data()

    def find_drz_paths(self):
        self.drz_paths = {}
        for idx, row in self.df.iterrows():
            self.drz_paths[idx] = ""
            dname = row["dataset"]
            drz = row["imgname"]
            path = os.path.join(self.input_path, dname, drz)
            self.drz_paths[idx] = path
        return self.drz_paths

    def scrub_fits(self):
        print("\n*** Extracting fits data ***")
        fits_dct = {}
        for key, path in self.drz_paths.items():
            fits_dct[key] = {}
            scihdr = fits.getheader(path, ext=1)
            for k in self.fits_keys:
                if k in scihdr:
                    if k == "wcstype":
                        wcs = " ".join(scihdr[k].split(" ")[1:3])
                        fits_dct[key][k] = wcs
                    else:
                        fits_dct[key][k] = scihdr[k]
                else:
                    fits_dct[key][k] = 0
        fits_data = pd.DataFrame.from_dict(fits_dct, orient="index")
        self.df = self.df.join(fits_data, how="left")
        return self.df


class ScrubSvm:
    def __init__(
        self,
        df,
        input_path,
        output_path=None,
        output_file="svm_data",
        save_raw=True,
        make_pos_list=True,
        crpt=False,
        make_subsamples=False,
    ):
        self.df = df
        self.input_path = input_path
        self.output_path = output_path
        self.output_file = output_file
        self.save_raw = save_raw
        self.data_path = None
        self.make_pos_list = make_pos_list
        self.crpt = crpt
        self.make_subsamples = make_subsamples

    def preprocess_data(self):
        """Main calling function"""
        self.df = ScrubCols(self.df).scrub_columns()
        self.df = ScrubFits(self.df, self.input_path).scrub_fits()
        self.df = MastScraper(self.df).scrape_mast()
        if self.save_raw is True:
            self.save_csv_file(raw=True)
        self.df = SvmEncoder(self.df).encode_features()
        self.df = self.drop_and_set_cols()
        self.make_pos_label_list()
        if self.crpt:
            self.df = self.add_crpt_labels()
        if self.make_subsamples:
            self.find_subsamples()
        self.save_csv_file()

    def save_csv_file(self, raw=False):
        if raw is True:
            self.data_path = f"{self.output_path}/raw_{self.output_file}.csv"
        else:
            self.data_path = f"{self.output_path}/{self.output_file}.csv"
        self.df["index"] = self.df.index
        self.df.to_csv(self.data_path, index=False)
        print("Data saved to: ", self.data_path)
        self.df.drop("index", axis=1, inplace=True)
        return self.data_path

    def drop_and_set_cols(self):
        column_order = [
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
        if "label" in self.df.columns:
            column_order.append("label")
        drops = [col for col in self.df.columns if col not in column_order]
        self.df = self.df.drop(drops, axis=1)
        self.df = self.df[column_order]
        return self.df

    def make_pos_label_list(self):
        if self.make_pos_list is True:
            if "label" in self.df.columns:
                pos = list(self.df.loc[self.df["label"] == 1].index.values)
                if len(pos) > 0:
                    with open(f"{self.output_path}/pos.txt", "w") as f:
                        for i in pos:
                            f.writelines(f"{i}\n")
            else:
                print("No labels found - skipping")

    def add_crpt_labels(self):
        labels = []
        for _ in range(len(self.df)):
            labels.append(1)
        self.df["label"] = pd.Series(labels).values
        return self.df

    def find_subsamples(self):
        if "label" not in self.df.columns:
            return
        self.df = self.df.loc[self.df["label"] == 0]
        subsamples = []
        categories = list(self.df["cat"].unique())
        detectors = list(self.df["det"].unique())
        for d in detectors:
            det = self.df.loc[self.df["det"] == d]
            for c in categories:
                cat = det.loc[det["cat"] == c]
                if len(cat) > 0:
                    idx = np.random.randint(0, len(cat))
                    samp = cat.index[idx]
                    subsamples.append(samp.split("_")[-1])
        output_path = os.path.dirname(self.output_file)
        with open(f"{output_path}/subset.txt", "w") as f:
            for s in subsamples:
                f.writelines(f"{s}\n")


class ScrubCal:
    def __init__(self, data, tensors=True):
        self.data = data
        self.tensors = tensors
        self.mem_bin = data["mem_bin"]
        self.memory = data["memory"]
        self.wallclock = data["wallclock"]
        self.input_cols = [
            "x_files",
            "x_size",
            "drizcorr",
            "pctecorr",
            "crsplit",
            "subarray",
            "detector",
            "dtype",
            "instr",
        ]
        self.X = self.data[self.input_cols]
        self.X_train = None
        self.X_test = None
        self.test_idx = None
        self.train_idx = None
        self.y_bin_train = None
        self.y_bin_test = None
        self.y_mem_train = None
        self.y_mem_test = None
        self.y_wall_train = None
        self.y_wall_test = None
        self.bin_test_idx = None
        self.mem_test_idx = None
        self.wall_test_idx = None

    def prep_data(self):
        """main calling function"""
        self.X_train, self.X_test, y_train, y_test = self.stratify_split(self.mem_bin)
        self.test_idx = self.make_test_index(y_test)
        self.train_idx = self.make_test_index(y_train)
        self.bin_test_idx = self.test_idx
        self.y_bin_train, self.y_bin_test = self.encode_y(y_train, y_test)
        self.y_mem_train, self.y_mem_test, self.mem_test_idx = self.prep_reg(
            target_col="memory"
        )
        self.y_wall_train, self.y_wall_test, self.wall_test_idx = self.prep_reg(
            target_col="wallclock"
        )
        if self.tensors is True:
            self.X_train = array_to_tensor(self.X_train)
            self.X_test = array_to_tensor(self.X_test)
        return self

    def stratify_split(self, y):
        return train_test_split(self.X, y, test_size=0.2, stratify=y)

    def make_test_index(self, y, target_col="mem_bin"):
        test_idx = pd.DataFrame(y, index=y.index, columns={target_col})
        return test_idx

    def encode_y(self, y_train, y_test):
        y_train, y_test = encode_target_data(y_train, y_test)
        if self.tensors is True:
            y_train = array_to_tensor(y_train)
            y_test = array_to_tensor(y_test)
        return y_train, y_test

    def prep_reg(self, target_col="memory"):
        y_test = self.data.loc[self.test_idx.index][target_col]
        y_train = self.data.loc[self.train_idx.index][target_col]
        test_idx = self.make_test_index(y_test, target_col=target_col)
        y_train, y_test = y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)
        if self.tensors is True:
            y_train, y_test = array_to_tensor(y_train), array_to_tensor(y_test)
        return y_train, y_test, test_idx
