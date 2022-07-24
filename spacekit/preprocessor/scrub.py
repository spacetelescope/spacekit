import os
import glob
import pandas as pd
import numpy as np
from spacekit.extractor.scrape import MastScraper, FitsScraper, scrape_catalogs
from spacekit.preprocessor.encode import SvmEncoder


class Scrubber:
    """Base parent class for preprocessing data. Includes some basic column scrubbing methods for pandas dataframes. The heavy lifting is done via subclasses below."""

    def __init__(
        self,
        data=None,
        col_order=None,
        output_path=None,
        output_file=None,
        dropnans=True,
        save_raw=True,
    ):
        self.df = self.cache_data(cache=data)
        self.col_order = col_order
        self.output_path = output_path
        self.output_file = output_file
        self.dropnans = dropnans
        self.save_raw = save_raw
        self.data_path = None

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
        self.df = self.df[extract]

    def col_splitter(self, splitter=".", keep=-1, make_lowercase=True):
        if make_lowercase is True:
            cols = [col.split(splitter)[keep].lower() for col in self.df.columns]
        else:
            cols = [col.split(splitter)[keep] for col in self.df.columns]
        print("Split columns: ", cols)
        return cols

    def rename_cols(self, old=None, new=None):
        print("\nRenaming columns")
        if old is None:
            old = self.df.columns
        if new is None:
            new = self.col_splitter()
        hc = dict(zip(old, new))
        self.df.rename(hc, axis="columns", inplace=True)
        print("New column names: ", self.df.columns)

    def drop_nans(self, save_backup=True):
        if self.dropnans is True:
            print("Searching for NaNs...")
            print(self.df.isna().sum())
            if self.df.isna().sum().values.any() > 0:
                print("Dropping NaNs")
            self.df.dropna(axis=0, inplace=True)

    def drop_and_set_cols(self, label_cols=["label"]):
        if self.col_order is None:
            print("col_order attribute not set - skipping.")
            return self.df
        if label_cols:
            for lab in label_cols:
                if lab in self.df.columns and lab not in self.col_order:
                    self.col_order.append(lab)
        drops = [col for col in self.df.columns if col not in self.col_order]
        self.df = self.df.drop(drops, axis=1)
        self.df = self.df[self.col_order]

    def save_csv_file(self, df=None, pfx="", index_col="index"):
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
        if df is None:
            df = self.df
        self.data_path = f"{self.output_path}/{pfx}{self.output_file}.csv"
        df[index_col] = df.index
        df.to_csv(self.data_path, index=False)
        print("Data saved to: ", self.data_path)
        df.drop(index_col, axis=1, inplace=True)


class SvmScrubber(Scrubber):
    """Class for invocating standard preprocessing steps of Single Visit Mosaic regression test data.

    Parameters
    ----------
    input_path : str or Path
        path to directory containing data input files
    data : dataframe, optional
        dataframe containing raw inputs scraped from json (QA) files, by default None
    output_path : str or Path, optional
        location to save preprocessed output files, by default None
    output_file : str, optional
        file basename to assign preprocessed dataset, by default "svm_data"
    dropnans : bool, optional
        find and remove any NaNs, by default True
    save_raw : bool, optional
        save data as csv before any encoding is performed, by default True
    make_pos_list : bool, optional
        create a text file listing misaligned (label=1) datasets, by default True
    crpt : int, optional
        dataset contains synthetically corrupted data, by default 0
    make_subsamples : bool, optional
        save a random selection of aligned (label=0) datasets to text file, by default False
    """

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
        self.col_order = self.set_col_order()
        super().__init__(
            data=data,
            col_order=self.col_order,
            output_path=output_path,
            output_file=output_file,
            dropnans=dropnans,
            save_raw=save_raw,
        )
        self.input_path = input_path
        self.make_pos_list = make_pos_list
        self.crpt = crpt
        self.make_subsamples = make_subsamples
        self.set_new_cols()
        self.set_prefix_cols()

    def preprocess_data(self):
        """Main calling function to run each preprocessing step for SVM regression data."""
        # STAGE 1 scrubbing
        self.scrub_columns()
        # STAGE 2 initial encoding
        self.df = FitsScraper(self.df, self.input_path).scrape_fits()
        self.df = MastScraper(self.df).scrape_mast()
        # STAGE 3 final encoding
        enc = SvmEncoder(self.df)
        self.df = enc.encode_features()
        enc.display_encoding()
        if self.save_raw is True:
            super().save_csv_file(pfx="raw_")
        super().drop_and_set_cols()
        # STAGE 4 target labels
        self.make_pos_label_list()
        self.add_crpt_labels()
        self.find_subsamples()
        super().save_csv_file()
        return self

    def scrub2(self):
        if self.df is None:
            self.scrub_qa_summary()
        else:
            self.scrub_qa_data()
        if self.save_raw is True:
            super().save_csv_file(pfx="raw_")
        # STAGE 3 final encoding
        self.df = SvmEncoder(self.df).encode_features()
        super().drop_and_set_cols()
        # STAGE 4 target labels
        self.make_pos_label_list()
        self.add_crpt_labels()
        self.find_subsamples()
        super().save_csv_file()

    def scrub_qa_data(self):
        self.scrub_columns()
        # STAGE 2 initial encoding
        self.df = FitsScraper(self.df, self.input_path).scrape_fits()
        self.df = MastScraper(self.df).scrape_mast()

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
        self.df = data.join(df_idx, how="left")
        self.df.set_index("index", inplace=True)
        super().rename_cols(
            old=["visit", "n_exposures", "target"],
            new=["dataset", "numexp", "targname"],
        )

    def scrub_columns(self):
        """Initial dataframe scrubbing to extract and rename columns, drop NaNs, and set the index."""
        split_cols = super().col_splitter()
        super().rename_cols(new=split_cols)
        super().extract_matching_columns(self.new_cols)
        super().drop_nans()
        index_data = self.split_index()
        self.df = index_data.join(self.df, how="left")
        super().rename_cols(old=["number_of_gaia_sources"], new=["gaia"])

    def set_col_order(self):
        return [
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
            else:
                idx_dct[n]["dataset"] = items[6]
        index_df = pd.DataFrame.from_dict(idx_dct, orient="index")
        return index_df

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
