import os
import pandas as pd
import numpy as np
from astropy.io import fits
from spacekit.extractor.scrape import MastScraper
from spacekit.preprocessor.encode import SvmEncoder

class Scrub:
    def __init__(self, df):
        self.df = df


class ScrubCols(Scrub):
    def __init__(self, df, dropnans=True):
        super().__init__(df)
        self.dropnans = dropnans
        self.new_cols = self.set_new_cols()
        self.col_pfx = self.set_prefix_cols()
        self.raw_cols = self.get_raw_cols()
        self.df = self.scrub_columns()
    
    def scrub_columns(self):
        self.df = self.rename_cols()
        extract = [c for c in self.new_cols if c in self.df.columns]
        self.df = self.df[extract]
        print("New column names: ", self.df.columns)
        if self.dropnans:
            self.df.dropna(axis=0, inplace=True)
        index_df = self.split_index()
        self.df = index_df.join(self.df, how="left")
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
            "Number_of_GAIA_sources." # incl. trailing period
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


class ScrubFits(Scrub):
    def __init__(self, df, raw_data, join_data=True):
        super().__init__(df)
        self.raw_data = raw_data
        self.join_data = join_data
        self.fits_keys = ["rms_ra", "rms_dec", "nmatches", "wcstype"]
        self.drz_paths = self.find_drz_paths()
        self.fits_data = self.extract_fits_data()
        self.df = self.join_fits_data()

    def find_drz_paths(self):
        self.drz_paths = {}
        for idx, row in self.df.iterrows():
            self.drz_paths[idx] = ""
            dname = row["dataset"]
            drz = row["imgname"]
            path = os.path.join(self.raw_data, dname, drz)
            self.drz_paths[idx] = path
        return self.drz_paths
    
    def extract_fits_data(self):
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
        self.fits_data = pd.DataFrame.from_dict(fits_dct, orient="index")
        return self.fits_data
    
    def join_fits_data(self):
        if self.join_data is True:
            self.df.join(self.align_data, how="left")
        return self.df


class ScrubSvm(Scrub):
    def __init__(self, df, raw_data, outpath, output_file):
        super().__init__(df)
        self.raw_data = raw_data
        self.outpath = outpath
        self.output_file = output_file
        self.save_raw = True
        self.save_csv = True
        self.make_pos_list = True
        self.crpt = False
        self.make_subsamples = True
    
    def preprocess_data(self):
        self.df = ScrubCols(self.df)
        self.df = ScrubFits(self.df, self.raw_data)
        self.df = MastScraper(self.df).scrape_mast()
        if self.save_raw is True:
            self.save_csv_file(raw=True)
        self.df = SvmEncoder(self.df).encode_features()
        self.df = self.drop_and_set_columns()
        self.make_pos_label_list()
        if self.crpt:
            self.df = self.add_crpt_labels()
        if self.make_subsamples:
            self.find_subsamples()
        self.save_csv_file()

    def save_csv_file(self, raw=False):
        if raw is True:
            data_path = f"{self.outpath}/raw_{self.output_file}"
        else:
            data_path = f"{self.outpath}/{self.output_file}"
        df_copy = self.df.copy()
        df_copy.reset_index()
        df_copy.to_csv(data_path, index=False)
        print("Data saved to: ", data_path)
        del df_copy

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
            pos = list(self.df.loc[self.df["label"] == 1].index.values)
            if len(pos) > 0:
                with open(f"{self.outpath}/pos.txt", "w") as f:
                    for i in pos:
                        f.writelines(f"{i}\n")

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
