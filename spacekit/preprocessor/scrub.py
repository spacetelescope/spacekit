import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from spacekit.extractor.scrape import MastScraper, FitsScraper, scrape_catalogs
from spacekit.preprocessor.encode import SvmEncoder
from spacekit.preprocessor.encode import encode_target_data
from spacekit.preprocessor.transform import array_to_tensor

# TODO: test alt scrub process (no json files)


class Scrubber:
    """Base parent class for preprocessing data. Includes some basic column scrubbing methods for pandas dataframes. The heavy lifting is done via subclasses below."""

    def __init__(self, data=None):
        self.df = self.cache_data(cache=data)

    def cache_data(self, cache=None):
        return cache.copy() if cache is not None else cache

    def extract_matching_columns(self, cols):
        # print("\n*** Extracting FITS header prefix columns ***")
        # extracted = []
        # for c in cols:
        #     extracted += [col for col in self.df if c in col]
        # return extracted
        extract = [c for c in cols if c in self.df.columns]
        print("Extract matching columns: ", extract)
        return self.df[extract]

    def col_splitter(self, splitter=".", keep=-1, make_lowercase=True):
        if make_lowercase is True:
            cols = [col.split(splitter)[keep].lower() for col in self.df.columns]
        else:
            cols = [col.split(splitter)[keep] for col in self.df.columns]
        print("Split columns: ", cols)
        return cols

    def rename_cols(self, cols=None):
        print("\nRenaming columns")
        if cols is None:
            cols = self.col_splitter()
        hc = dict(zip(self.df.columns, cols))
        self.df.rename(hc, axis="columns", inplace=True)
        print("New column names: ", self.df.columns)
        return self.df


class SvmScrubber(Scrubber):
    """Class for invocating standard preprocessing steps of Single Visit Mosaic regression test data. This class quietly relies on other classes in the module to instantiate other scrubbing objects, although they are distinct and non-hierarchical (no inheritance between them)."""

    def __init__(
        self,
        input_path,
        data=None,
        output_path=None,
        output_file="svm_data",
        dropnans=True,
        save_raw=True,
        make_pos_list=True,
        crpt=0,
        make_subsamples=False,
    ):
        super().__init__(data=data)
        self.input_path = input_path
        self.output_path = output_path
        self.output_file = output_file
        self.dropnans = dropnans
        self.save_raw = save_raw
        self.make_pos_list = make_pos_list
        self.crpt = crpt
        self.make_subsamples = make_subsamples
        self.data_path = None
        self.set_new_cols()
        self.set_prefix_cols()

    def preprocess_data(self):
        """Main calling function to run each preprocessing step for SVM regression data."""
        # STAGE 1 scrubbing
        self.df = self.scrub_columns()
        # STAGE 2 initial encoding
        self.df = FitsScraper(self.df, self.input_path).scrape_fits()
        self.df = MastScraper(self.df).scrape_mast()
        if self.save_raw is True:
            self.save_csv_file(pfx="raw_")
        # STAGE 3 final encoding
        self.df = SvmEncoder(self.df).encode_features()
        self.df = self.drop_and_set_cols()
        # STAGE 4 target labels
        self.make_pos_label_list()
        self.add_crpt_labels()
        self.find_subsamples()
        self.save_csv_file()
        return self

    def scrub(self):
        if self.df is None:
            self.df = self.scrub_qa_summary()
        else:
            self.df = self.scrub_qa_data()
        if self.save_raw is True:
            self.save_csv_file(pfx="raw_")
        # STAGE 3 final encoding
        self.df = SvmEncoder(self.df).encode_features()
        self.df = self.drop_and_set_cols()
        # STAGE 4 target labels
        self.make_pos_label_list()
        self.add_crpt_labels()
        self.find_subsamples()
        self.save_csv_file()

    def scrub_qa_data(self):
        self.df = self.scrub_columns()
        # STAGE 2 initial encoding
        self.df = FitsScraper(self.df, self.input_path).scrape_fits()
        self.df = MastScraper(self.df).scrape_mast()
        return self.df

    def scrub_qa_summary(self, fname="single_visit_mosaics*.csv", idx=0):
        """Alternative if no .json files available (QA step not run during processing)"""
        # fname = 'single_visit_mosaics_2021-10-06.csv'
        fpath = glob.glob(os.path.join(self.input_path, fname))
        if len(fpath) == 0:
            return
        data = pd.read_csv(fpath[0], index_col=idx)
        index = {}
        for i, row in data.iterrows():
            index[i] = dict(index=None, detector=None)
            prop = row["proposal"]
            visit = row["visit"]
            num = visit[-2:]
            instr, det = row["config"].split("/")
            name = f"hst_{prop}_{num}_{instr}_{det}_total_{visit}".lower()
            index[i]["index"] = name
            index[i]["detector"] = det.lower()
            index[i]["point"] = scrape_catalogs(self.input_path, name, sfx="point")
            index[i]["segment"] = scrape_catalogs(self.input_path, name, sfx="segment")
            index[i]["gaia"] = scrape_catalogs(self.input_path, name, sfx="ref")
        add_cols = {"index", "detector", "point", "segment", "gaia"}
        df_idx = pd.DataFrame.from_dict(index, orient="index", columns=add_cols)
        df = data.join(df_idx, how="left")
        df.set_index("index", inplace=True)
        drops = [
            "dateobs",
            "config",
            "filter",
            "aec",
            "status",
            "wcsname",
            "creation_date",
        ]
        df.drop([d for d in drops if d in df.columns], axis=1, inplace=True)
        df.rename(
            {"visit": "dataset", "n_exposures": "numexp", "target": "targname"},
            axis=1,
            inplace=True,
        )
        return df

    def scrub_columns(self):
        """Initial dataframe scrubbing to extract and rename columns, drop NaNs, and set the index."""
        split_cols = super().col_splitter()
        self.df = super().rename_cols(cols=split_cols)
        self.df = super().extract_matching_columns(self.new_cols)
        if self.dropnans:
            print("Searching for NaNs...")
            print(self.df.isna().sum())
            print("Dropping NaNs")
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

    # TODO: This is pretty general and should be called from extractor.load
    def save_csv_file(self, pfx=""):
        """Saves dataframe to csv file on local disk.

        Parameters
        ----------
        pfx : str, optional
            Insert a prefix at start of filename, by default ""

        Returns
        -------
        str
            self.data_path where file is saved on disk.
        """
        self.data_path = f"{self.output_path}/{pfx}{self.output_file}.csv"
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
        """Looks for target class labels in dataframe and saves a text file listing index names of positive class. Originally this was to automate moving images into class labeled directories."""
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
        """For new synthetic datasets, adds "label" target column and assigns value of 1 to all rows.

        Returns
        -------
        dataframe
            self.df updated with label column (all values set = 1)
        """
        if self.crpt:
            labels = []
            for _ in range(len(self.df)):
                labels.append(1)
            self.df["label"] = pd.Series(labels).values
            return self.df

    def find_subsamples(self):
        """Gets a varied sampling of dataframe observations and saves to local text file. This is one way of identifying a small subset for synthetic data generation."""
        if self.make_subsamples:
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
