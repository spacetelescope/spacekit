import os
import glob
import time
import pandas as pd
import numpy as np
from spacekit.extractor.scrape import (
    SvmFitsScraper,
    JwstFitsScraper,
    scrape_catalogs,
)
from spacekit.preprocessor import (
    TRUEVALS,
    NANVALS,
    FALSEVALS,
    SUBNAN
)
from spacekit.preprocessor.transform import SkyTransformer
from spacekit.preprocessor.encode import HstSvmEncoder, JwstEncoder, encode_booleans
from spacekit.logger.log import Logger


class Scrubber:
    """
    Base parent class for preprocessing data. Includes some basic column scrubbing methods for pandas dataframes. The heavy lifting is done via subclasses.

    Parameters
    ----------
    data : pandas.DataFrame or dict, optional
        dataset to be scrubbed, by default None
    col_order : list, optional
        order input feature columns, by default None
    output_path : str or Path, optional
        path on local disk to save scrubbed dataset, by default None
    output_file : str, optional
        name to give scrubbed dataset file, by default None
    dropnans : bool, optional
        find and remove any NaNs, by default True
    save_raw : bool, optional
        save data as csv on local disk before any encoding is performed, by default True
    name : str, optional
        logger name (mutable for subclasses), by default "Scrubber"
    """
    def __init__(
        self,
        data=None,
        col_order=None,
        output_path=None,
        output_file=None,
        dropnans=True,
        save_raw=True,
        name="Scrubber",
        **log_kws,
    ):

        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        self.df = self.cache_data(cache=data)
        self.col_order = col_order
        self.output_path = output_path
        self.output_file = output_file
        self.dropnans = dropnans
        self.save_raw = save_raw
        self.data_path = None

    def cache_data(self, cache=None):
        return cache.copy() if cache is not None else cache

    def convert_to_dataframe(self, data=None, cache_df=False):
        if data is None or isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            data = pd.DataFrame.from_dict(data, orient="index")
            if cache_df is True:
                self.cache_data(cache=data)
            return data
        else:
            self.log.error("data must be dict, dataframe or None")

    def extract_matching_columns(self, cols):
        extract = [c for c in cols if c in self.df.columns]
        self.log.info(f"Extract matching columns: {extract}")
        self.df = self.df[extract]

    def col_splitter(self, splitter=".", keep=-1, make_lowercase=True):
        if make_lowercase is True:
            cols = [col.split(splitter)[keep].lower() for col in self.df.columns]
        else:
            cols = [col.split(splitter)[keep] for col in self.df.columns]
        self.log.info(f"Split columns: {cols}")
        return cols

    def rename_cols(self, old=None, new=None):
        if old is None:
            old = self.df.columns
        if new is None:
            new = self.col_splitter()
        hc = dict(zip(old, new))
        self.df.rename(hc, axis="columns", inplace=True)

    def drop_nans(self, save_backup=True):
        if self.dropnans is True:
            self.log.info("Searching for NaNs...")
            self.log.info(f"{self.df.isna().sum()}")
            if self.df.isna().sum().values.any() > 0:
                self.log.info("Dropping NaNs")
            self.df.dropna(axis=0, inplace=True)

    def drop_and_set_cols(self, label_cols=["label"]):
        if self.col_order is None:
            self.log.info("col_order attribute not set - skipping.")
            return self.df
        if label_cols:
            for lab in label_cols:
                if lab in self.df.columns and lab not in self.col_order:
                    self.col_order.append(lab)
        drops = [col for col in self.df.columns if col not in self.col_order]
        self.df = self.df.drop(drops, axis=1)
        self.df = self.df[self.col_order]

    def uppercase_vals(
        self, column_list, exceptions=["channel", "asn_rule", "exptype"]
    ):
        # set consistent case for categorical column values with a few exceptions
        for col in column_list:
            if col in self.df.columns and col not in exceptions:
                try:
                    self.df[col] = self.df[col].apply(lambda x: x.upper())
                except Exception as e:
                    print(col)
                    print(e)
        return self.df

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
        self.log.info(f"Data saved to: {self.data_path}")
        df.drop(index_col, axis=1, inplace=True)


class HstSvmScrubber(Scrubber):
    """Class for applying standard preprocessing steps of Single Visit Mosaic regression test data.

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
        **log_kws,
    ):
        self.col_order = self.set_col_order()
        super().__init__(
            data=data,
            col_order=self.col_order,
            output_path=output_path,
            output_file=output_file,
            dropnans=dropnans,
            save_raw=save_raw,
            name="HstSvmScrubber",
            **log_kws,
        )
        self.input_path = input_path
        self.make_pos_list = make_pos_list
        self.crpt = crpt
        self.make_subsamples = make_subsamples
        self.set_new_cols()
        self.set_prefix_cols()
        self.initialize_radio()

    def initialize_radio(self):
        from spacekit.extractor.radio import HstSvmRadio
        self.radio = HstSvmRadio

    def preprocess_data(self):
        """Main calling function to run each preprocessing step for SVM regression data."""
        # STAGE 1 scrubbing
        self.scrub_columns()
        # STAGE 2 initial encoding
        self.df = SvmFitsScraper(self.df, self.input_path).scrape_fits()
        n_retries = 3
        while n_retries > 0:
            try:
                self.df = self.radio(self.df).scrape_mast()
                n_retries = 0
            except Exception as e:
                self.log.warning(e)
                time.sleep(5)
                n_retries -= 1
        # STAGE 3 final encoding
        enc = HstSvmEncoder(self.df)
        self.df = enc.encode_features()
        if self.save_raw is True:
            super().save_csv_file(pfx="raw_")
        self.drop_and_set_cols()
        # STAGE 4 target labels
        self.make_pos_label_list()
        self.add_crpt_labels()
        self.find_subsamples()
        self.save_csv_file()
        return self

    # TODO
    def scrub2(self, summary=None, total_obj_file=None):
        if self.df is None:
            if summary:
                self.scrub_qa_summary(fname=summary)
            elif total_obj_file:
                self.run_qa(pickle_file=total_obj_file)
                self.scrub_qa_data()
            else:
                return

        if self.save_raw is True:
            super().save_csv_file(pfx="raw_")
        # STAGE 3 final encoding
        self.df = HstSvmEncoder(self.df).encode_features()
        super().drop_and_set_cols()
        # STAGE 4 target labels
        self.make_pos_label_list()
        self.add_crpt_labels()
        self.find_subsamples()
        super().save_csv_file()

    # TODO
    def run_qa(self, total_obj_file="total_obj_list_full.pickle"):
        pass
        # import pickle
        # try:
        #     from drizzlepac.hap_utils.svm_quality_analysis import run_quality_analysis
        # except ImportError:
        #     print("Running SVM QA requires drizzlepac to be installed via pip.")

        # with open(total_obj_file, 'rb') as pck:
        #     total_obj_list = pickle.load(pck)

        # run_quality_analysis(total_obj_list, run_compare_num_sources=True, run_find_gaia_sources=True,
        #                  run_compare_hla_sourcelists=True, run_compare_ra_dec_crossmatches=True,
        #                  run_characterize_gaia_distribution=True, run_compare_photometry=True,
        #                  run_compare_interfilter_crossmatches=True, run_report_wcs=True)

    def scrub_qa_data(self):
        self.scrub_columns()
        # STAGE 2 initial encoding
        self.df = SvmFitsScraper(self.df, self.input_path).scrape_fits()
        self.df = self.radio(self.df).scrape_mast()

    def scrub_qa_summary(self, csvfile="single_visit_mosaics*.csv", idx=0):
        """Alternative if no .json files available (QA step not run during processing)"""
        # fname = 'single_visit_mosaics_2021-10-06.csv'
        fpath = glob.glob(os.path.join(self.input_path, csvfile))
        if len(fpath) == 0:
            return None
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
        check_columns = [
            "targname",
            "ra_targ",
            "dec_targ",
            "numexp",
            "imgname",
            "number_of_gaia_sources",
            "detector",
            "point",
            "segment",
        ]
        missing = [c for c in check_columns if c not in split_cols]
        if missing:
            if "number_of_gaia_sources" in missing:
                self.log.warning("Inserting zero value for gaia sources")
                shape = self.df.index.values.shape
                self.df["number_of_gaia_sources.number_of_gaia_sources"] = np.zeros(
                    shape=shape, dtype=int
                )
                split_cols.append("number_of_gaia_sources")
            else:
                raise Exception(f"Dataframe is missing some data:\n {missing}")

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
        """Looks for target class labels in dataframe and saves a text file listing index names of positive class. Originally
        this was to automate moving images into class labeled directories."""
        if self.make_pos_list is True:
            if "label" in self.df.columns:
                pos = list(self.df.loc[self.df["label"] == 1].index.values)
                if len(pos) > 0:
                    with open(f"{self.output_path}/pos.txt", "w") as f:
                        for i in pos:
                            f.writelines(f"{i}\n")
            else:
                self.log.info("No labels found - skipping")

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
        """Gets a varied sampling of dataframe observations and saves to local text file. This is one way of identifying a small
        subset for synthetic data generation."""
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


class HstCalScrubber(Scrubber):
    """Class for invoking initial preprocessing on HST Pipeline calibration metadata for training compute resource estimation models.

    Parameters
    ----------
    data : pandas.DataFrame or dict, optional
        dataset to be scrubbed, by default None
    output_path : str or Path, optional
        location to save preprocessed output files, by default None
    output_file : str, optional
        file basename to assign preprocessed dataset, by default "batch.csv"
    dropnans : bool, optional
        find and remove any NaNs, by default True
    save_raw : bool, optional
        save data as csv on local disk before any encoding is performed, by default True
    """

    def __init__(
        self,
        data=None,
        output_path=None,
        output_file="batch.csv",
        dropnans=True,
        save_raw=True,
        **log_kws,
    ):
        super().__init__(
            data=data,
            col_order=self.set_col_order(),
            output_path=output_path,
            output_file=output_file,
            dropnans=dropnans,
            save_raw=save_raw,
            name="HstCalScrubber",
            **log_kws,
        )
        self.data = super().convert_to_dataframe(data=data)

    def set_col_order(self):
        return [
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

    def set_new_cols(self):
        self.new_cols = ["x_files", "x_size"]
        return self.new_cols.extend(self.col_order)

    def scrub_inputs(self):
        self.log.info(f"Scrubbing inputs for {self.data.index[0]}")
        n_files = int(self.data["n_files"][0])
        total_mb = int(np.round(float(self.data["total_mb"]), 0))
        detector = 1 if self.data["DETECTOR"][0].upper() in ["UVIS", "WFC"] else 0
        subarray = 1 if self.data["SUBARRAY"][0].title() == "True" else 0
        drizcorr = 1 if self.data["DRIZCORR"][0].upper() == "PERFORM" else 0
        pctecorr = 1 if self.data["PCTECORR"][0].upper() == "PERFORM" else 0
        cr = self.data["CRSPLIT"][0]
        if cr == "NaN":
            crsplit = 0
        elif cr in ["1", "1.0"]:
            crsplit = 1
        else:
            crsplit = 2

        i = self.data.index[0]
        # dtype (asn or singleton)
        dtype = 1 if i[-1] == "0" else 0
        # instr encoding cols
        instr_key = dict(j=0, l=1, o=2, i=3)
        for k, v in instr_key.items():
            if i[0] == k:
                instr = v
        return np.array(
            [
                n_files,
                total_mb,
                drizcorr,
                pctecorr,
                crsplit,
                subarray,
                detector,
                dtype,
                instr,
            ]
        )

    def image_pixel_scales(self):
        # calculate sky separation / reference pixel offset statistics
        return dict(
            ACS=dict(WFC=0.05),
            WFC3=dict(UVIS=0.04, IR=0.13),
        )


class JwstCalScrubber(Scrubber):
    """Class for invoking initial preprocessing of JWST calibration input data.

    Parameters
    ----------
    input_path : str or path
        path on local disk where L1 input exposures are located
    data : pd.DataFrame, optional
        dataframe of exposures to be preprocessed, by default None
    pfx : str, optional
        limit scrape search to files starting with a given prefix such as 'jw01018', by default ""
    sfx : str, optional
        limit scrape search to files ending with a given suffix, by default "_uncal.fits"
    dropnans : bool, optional
        drop null value columns, by default False
    save_raw : bool, optional
        save a copy of the dataframe before encoding, by default True
    encoding_pairs : dict, optional
        preset key-value pairs for encoding categorical data, by default None
    mode : str, optional
        determines how data is scraped and handled ('fits' for files or 'df' for dataframe), by default 'fits'
    miri_ifu_opts : dict, optional
        Optionally ignore channel and/or subchannel for MIRI IFU exposures. Setting both to False will consider exposures from all channels and subchannels of a given observation to be inputs for a single L3 product.
    """
    def __init__(
        self,
        input_path,
        data=None,
        pfx="",
        sfx="_uncal.fits",
        dropnans=False,
        save_raw=True,
        encoding_pairs=None,
        mode='fits',
        **log_kws,
    ):
        self.input_path = input_path
        self.exp_headers = None
        self.products = dict()
        self.img_products = dict()
        self.spec_products = dict()
        self.tac_products = dict()
        self.fgs_products = dict()
        self.imgpix = None
        self.specpix = None
        self.tacpix = None
        self.fgspix = None
        self.pfx = pfx
        self.sfx = sfx
        super().__init__(
            data=data,
            col_order=self.xcol_order,
            dropnans=dropnans,
            save_raw=save_raw,
            name="JwstCalScrubber",
            **log_kws,
        )
        self.xcols = self.xcol_order
        self.encoding_pairs = encoding_pairs
        self.mode = mode
        self.scrape_inputs()
        self.get_level3_products()
        self.pixel_offsets()

    @property
    def xcol_order(self):
        return self._xcol_order()

    def _xcol_order(self):
        """Used for resetting the order of columns in the final preprocessed dataframe.
        Returns
        -------
        list
            complete list of columns to include for JWST modeling
        """
        return [
            "instr",
            "detector",
            "exp_type",
            "visitype",
            "filter",
            "pupil",
            "grating",
            "fxd_slit",
            "channel",
            "subarray",
            "bkgdtarg",
            "is_imprt",
            "tsovisit",
            "nexposur",
            "numdthpt",
            "band",
            "targ_max_offset",
            "offset",
            "max_offset",
            "mean_offset",
            "sigma_offset",
            "err_offset",
            "sigma1_mean",
            "frac",
            "targ_frac",
            "gs_mag",
            "crowdfld",
        ]

    @property
    def level3_types(self):
        return self._level3_types()

    def _level3_types(self):
        """Exposure types included in Level 3 data processing.
        Returns
        -------
        list
            Level 3 exposure types
        """
        return [
            "FGS_IMAGE",
            "MIR_IMAGE",  # (TSO & Non-TSO)
            "NRC_IMAGE",
            "MIR_LRS-FIXEDSLIT",
            "MIR_MRS",
            "MIR_LYOT",  # coron
            "MIR_4QPM",  # coron
            "MIR_LRS-SLITLESS",  # (only IF TSO)
            "MIR_WFSS", # expected in future release (cycle 5)
            "NRC_CORON",  # coron
            "NRC_WFSS",
            "NRC_TSIMAGE",  # TSO always
            "NRC_TSGRISM",  # TSO always
            "NIS_IMAGE",
            "NIS_AMI",  # AMI
            "NIS_WFSS",
            "NIS_SOSS",  # (TSO & Non-TSO)
            "NRS_FIXEDSLIT",
            "NRS_IFU",
            "NRS_MSASPEC",
            "NRS_BRIGHTOBJ",  # TSO always
        ]

    @property
    def tso_ami_coron(self):
        return self._tso_ami_coron()

    def _tso_ami_coron(self):
        return [
            "MIR_4QPM",
            "MIR_LYOT",
            "NRC_CORON",
            "NIS_AMI",
            "NRS_BRIGHTOBJ",
            "NRC_TSGRISM",
            "NRC_TSIMAGE"
        ]

    @property
    def source_based(self):
        return self._source_based()

    def _source_based(self):
        return ["MIR_WFSS", "NRC_WFSS", "NIS_WFSS", "NRS_MSASPEC", "NRS_FIXEDSLIT"]

    @property
    def expdata(self):
        return self._expdata()

    def _expdata(self):
        return dict(
            IMAGE=self.img_products,
            SPEC=self.spec_products,
            TAC=self.tac_products,
            FGS=self.fgs_products
        )

    def scrape_inputs(self):
        """Scrape input exposure header metadata from fits files on local disk located at `self.input_path`.
        """
        self.scraper = JwstFitsScraper(
            self.input_path, data=self.df, pfx=self.pfx, sfx=self.sfx
        )
        self.fpaths = self.scraper.fpaths
        if self.mode == 'df':
            self.exp_headers = self.scraper.scrape_dataframe()
        else:
            self.exp_headers = self.scraper.scrape_fits()

    def pixel_offsets(self):
        """Generate the pixel offset between exposure reference pixels and the estimated L3 fiducial.
        """
        sky = SkyTransformer("JWST")
        self.imgpix = sky.calculate_offsets(self.img_products)
        self.products.update(self.imgpix)
        sky.set_keys(ra="RA_REF", dec="DEC_REF")
        self.specpix = sky.calculate_offsets(self.spec_products)
        self.products.update(self.specpix)
        if self.mode != 'df': # use fits data
            sky.count_exposures = False
        self.tacpix = sky.calculate_offsets(self.tac_products)
        self.products.update(self.tacpix)
        self.update_fgs()

    def make_image_product_name(self, k, v, tnum):
        """Parse through exposure metadata to create expected L3 image products.
        Parameters
        ----------
        k : str
            exposure header key (L1 exposure name)
        v : dict
            exposure header data
        tnum : str
            number assigned to each unique target (targ_ra) within a program
        """
        pupil = f"{v['PUPIL']}" if v["PUPIL"] not in NANVALS else ""
        fltr = f"{v['FILTER']}" if v["FILTER"] not in NANVALS else ""
        subarray = f"-{v['SUBARRAY']}" if v["SUBARRAY"] not in SUBNAN else ""
        if not pupil:
            optelem = fltr
        elif pupil in ["CLEAR", "CLEARP", "F405N"]:
            optelem = f"{pupil}-{fltr}"
        else:
            optelem =  f"{fltr}-{pupil}"

        if 'WFSC' in v['VISITYPE']:
            if not subarray:
                p = f"jw{v['PROGRAM']}-o{v['OBSERVTN']}_{tnum}_{v['INSTRUME']}_{optelem}-{v['DETECTOR']}".lower()
            else: # ignore nrca3 target acquisition exposures (subarray != "FULL")
                return
        else:
            p = f"jw{v['PROGRAM']}-o{v['OBSERVTN']}_{tnum}_{v['INSTRUME']}_{optelem}{subarray}".lower()

        if v['EXP_TYPE'] in self.tso_ami_coron or v["TSOVISIT"] in TRUEVALS:
            self.make_tac_product_name(k, v, p)
            return
        elif v["INSTRUME"] == "FGS":
            self.make_fgs_product_name(k, v, p)
            return
        else:
            del v["NEXPOSUR"]
            if p in self.img_products:
                self.img_products[p][k] = v
            else:
                self.img_products[p] = {k: v}

    def make_spec_product_name(self, k, v, tnum):
        """Parse through exposure metadata to create expected L3 spectroscopy products. 
        NOTE: Although the pipeline would create multiple products for either source-based exposures
        or (channel-based) MIRI IFU exposures, only one product name will be created since the model is
        concerned with RAM, i.e. how large the memory footprint is to calibrate a set of input exposures.
        Source-based products use "s000000001" for the source; MIR_MRS exposures default to "ch1" or "ch3" for channel.
        Subchannel ("BAND") is ignored other than for determining if exposures are MIRI IFU.

        Parameters
        ----------
        k : str
            exposure header key (L1 exposure name)
        v : dict
            exposure header data
        tnum : str
            number assigned to each unique target (targ_ra) within a program
        """
        exptype = v["EXP_TYPE"]
        if exptype == "MIR_LRS-SLITLESS" and v["TSOVISIT"] in FALSEVALS:
            # L3 product only if TSO
            return
        if exptype in self.source_based:
            # 12/3/24 (jwst>=1.16) source ID is 9 digits
            # tnum = "s00001" # source-based exposure naming convention
            tnum = "s000000001"
        pupil = f"{v['PUPIL']}" if v["PUPIL"] not in NANVALS else ""
        fltr = f"{v['FILTER']}" if v["FILTER"] not in NANVALS else ""
        grating = (
            f"{v['GRATING']}" if v["GRATING"] not in NANVALS else ""
        )
        if pupil:
            optelem = f"{fltr}-{pupil}" if exptype in ["NRC_WFSS", "NIS_SOSS", "NRC_TSGRISM"] else f"{pupil}-{fltr}"
        elif grating: 
            optelem = f"{grating}-{fltr}" if exptype == "NRS_IFU" else f"{fltr}-{grating}"
        else:
            optelem = fltr # mir_mrs: fltr = ""

        slit = f"-{v['FXD_SLIT']}" if v["FXD_SLIT"] not in NANVALS else ""
        subarray = f"-{v['SUBARRAY']}" if v["SUBARRAY"] not in SUBNAN else ""
        # MIRI IFU only
        band = f"ch{v['CHANNEL'][0]}-{v['BAND']}" if v["BAND"] not in NANVALS else ""

        p = f"jw{v['PROGRAM']}-o{v['OBSERVTN']}_{tnum}_{v['INSTRUME']}_{optelem}{slit}{subarray}{band}".lower()

        if exptype in self.tso_ami_coron or v["TSOVISIT"] in TRUEVALS:
            if fltr == 'CLEAR' and grating == 'PRISM':
                # drop fxd slit from product name
                p = f"jw{v['PROGRAM']}-o{v['OBSERVTN']}_{tnum}_{v['INSTRUME']}_{optelem}{subarray}".lower()
            self.make_tac_product_name(k, v, p)
            return
        else:
            del v["NEXPOSUR"]
            if p in self.spec_products:
                self.spec_products[p][k] = v
            else:
                self.spec_products[p] = {k: v}

    def make_tac_product_name(self, k, v, p):
        """If an image or spec product meets the required conditions, it is added
        instead to the TAC products dictionary (Time-series, AMI, Coronagraph).
        Parameters
        ----------
        k : str
            exposure header key (L1 exposure name)
        v : dict
            exposure header data
        p : str
            product name
        """
        if self.mode != 'fits':
            del v["NEXPOSUR"]
        if v['EXP_TYPE'] == 'NRC_CORON':
            p += '-image3'
        if p in self.tac_products:
            self.tac_products[p][k] = v
        else:
            self.tac_products[p] = {k: v}

    def make_fgs_product_name(self, k, v, p):
        if self.mode != 'fits':
            del v["NEXPOSUR"]
        if p in self.fgs_products:
            self.fgs_products[p][k] = v
        else:
            self.fgs_products[p] = {k: v}

    def fake_target_ids(self):
        """Assigns a fake target ID using TARGNAME, TARG_RA or GS_MAG. These IDs are fake in that
        they're unlikely to match actual target IDs assigned later in the pipeline. For source-based exposures, 
        the id is always "s00001". (jwst>=1.16, after 12/3/24: source Id will be 9 digits: s000000001)

        Grouping logic:
        - TARG_RA (rounded to 6 decimals): VISITYPE=PRIME_TARGETED_FIXED, TARGNAME=NaN
        - TARGNAME: VISITYPE != PRIME_TARGETED_FIXED, TARGNAME != NaN
        - GS_MAG :  TARGNAME=NaN, GSMAG != NaN, VISITYPE != "PRIME_TARGETED_FIXED", "PARALLEL_PURE"

        Remaining groups not matching above parameters default to 't0' (typically 'parallel_pure' visitypes).
        """
        targ_exptypes = [t for t in self.level3_types if t not in self.source_based]
        targra = list(set([
                np.round(v['TARG_RA'], 6) for v in self.exp_headers.values() \
                    if v['EXP_TYPE'] in targ_exptypes and \
                        v['VISITYPE'] == "PRIME_TARGETED_FIXED"
            ]
        ))
        rnums = [f"t{i+1}" for i, _ in enumerate(targra)]
        rn = dict(zip(targra, rnums))

        targetnames = list(set(
            [
                v['TARGNAME'] for v in self.exp_headers.values() \
                    if v['EXP_TYPE'] in targ_exptypes and \
                        v['TARGNAME'] not in NANVALS and \
                            v['VISITYPE'] != "PRIME_TARGETED_FIXED"
            ]
        ))
        tnums = [f"t{i+1}" for i, _ in enumerate(targetnames)]
        tn = dict(zip(targetnames, tnums))

        gstargs = list(set(
            [
                v['GS_MAG'] for v in self.exp_headers.values() \
                    if v['GS_MAG'] not in NANVALS and \
                        v['EXP_TYPE'] in targ_exptypes and \
                            v['TARGNAME'] not in targetnames and \
                                v['VISITYPE'] not in ["PRIME_TARGETED_FIXED", "PARALLEL_PURE"]
            ]
        ))
        gnums = [f"t{i+1}" for i, _ in enumerate(gstargs)]
        gn = dict(zip(gstargs, gnums))
        self.targetnames = targetnames
        self.tn = tn
        return tn, rn, gn

    def get_level3_products(self):
        """Determines potential L3 products based on groups of input exposures 
        with matching Fits keywords prog+obs+optelem+fxd_slit+subarray. These groups
        are further subdivided and assigned a fake target ID by TARGNAME, GS_MAG or TARG_RA.
        """
        tn, rn, gn = self.fake_target_ids()
        for k, v in self.exp_headers.items():
            exp_type = v["EXP_TYPE"]
            if exp_type in self.level3_types:
                if exp_type in self.source_based:
                    # 12/3/24 jwst>=1.16: 's00001' -> 's000000001'
                    tnum = 's000000001'
                else:
                    tnum = tn.get(v['TARGNAME'], rn.get(np.round(v['TARG_RA'], 6), gn.get(v['GS_MAG'], 't0')))
                if "IMAGE" in exp_type.split("_")[-1]:
                    self.make_image_product_name(k, v, tnum)
                else:
                    self.make_spec_product_name(k, v, tnum)
        self.verify_target_groups()

    def verify_target_groups(self):
        """Certain L3 products need to be further defined by their L1 input TARG_RA
        values in addition to all other parameters. This only affects PRIME_TARGETED_FIXED
        visit types where TARGNAME != NaN. If multiple unique TARG_RA/DEC values (rounded to 6 digits) 
        are identified within the group of exposures, we can assume each TARG grouping is a unique L3 product.
        """
        revised = dict()
        for expmode, data in self.expdata.items():
            multitra = [
                k for k, v in data.items() \
                    if np.unique([np.round(j['TARG_RA'], 6) for j in v.values()]).size > 1 and \
                        np.unique([np.round(j['TARG_DEC'], 6) for j in v.values()]).size > 1 and \
                            list(v.values())[0]['VISITYPE'] == 'PRIME_TARGETED_FIXED' and \
                                list(v.values())[0]['TARGNAME'] in self.targetnames
            ]
            if multitra:
                revised[expmode] = multitra

        tgroups = {k:{} for k in list(revised.keys())}
        for expmode, products in revised.items():
            for k in products:
                tgroups[expmode][k] = dict()
                v = self.expdata[expmode][k]
                tname = list(v.values())[0]['TARGNAME']
                targras = np.unique([np.round(j['TARG_RA'], 6) for j in v.values()])
                tnum = self.tn.get(tname)
                for i, t in enumerate(targras):
                    exposures = [x for x, y in v.items() if np.round(y['TARG_RA'], 6) == t]
                    k2 = k.replace(tnum, tnum+f"x{i}")
                    tgroups[expmode][k][k2] = {e:v[e] for e in exposures}

        for expmode in tgroups.keys():
            expdata = self.expdata[expmode]
            ks = list(tgroups[expmode].keys())
            for k in ks:
                for k2, grp in tgroups[expmode][k].items():
                    expdata.update({k2:grp})
                del expdata[k]

    def update_fgs(self):
        self.fgspix = dict()
        if len(self.fgs_products) == 0:
            return
        for product, exp_data in self.fgs_products.items():
            first_key = list(exp_data.keys())[0]
            self.fgspix[product] = exp_data[first_key]
            self.fgspix[product].update(dict(NEXPOSUR=len(list(exp_data.keys()))))
            if product not in self.products:
                self.products[product] = exp_data

    @property
    def input_data(self):
        """Preprocessed input data grouped by exposure type
        Returns
        -------
        dict
            input data grouped by exp_type (IMAGE, SPEC, FGS, TAC)
        """
        return dict(
            IMAGE=self.imgpix,
            SPEC=self.specpix,
            TAC=self.tacpix,
            FGS=self.fgspix,
        )

    def scrub_inputs(self, exp_type="IMAGE"):
        """Main calling function for preprocessing input exposures of a given exposure type.
        Parameters
        ----------
        exp_type : str, optional
            Exposure type, by default "IMAGE"
        Returns
        -------
        pd.DataFrame
            preprocessed data with renamed columns, NaNs scrubbed and categorical data encoded
        """
        data = self.input_data[exp_type]
        if not data:
            return None
        self.df = pd.DataFrame.from_dict(data, orient="index")
        super().rename_cols(new=[c.lower() for c in self.df.columns])
        super().rename_cols(old=["instrume"], new=["instr"])
        xcols = [x for x in self.xcols if x in self.df]
        self.df = self.df[xcols]
        dtype_keys = self.get_dtype_keys()
        nandler = NaNdler(self.df, dtype_keys, allow_neg=False, verbose=False)
        self.df = nandler.apply_nandlers()
        self.group_nircam_detectors()
        self.group_subarrays()
        self.log.info(f"Encoding categorical features [{exp_type}]")
        encoder = JwstEncoder(
            self.df, fkeys=dtype_keys["categorical"], encoding_pairs=self.encoding_pairs
        )
        encoder.encode_features()
        self.df = encoder.df[xcols]
        return self.df

    def group_nircam_detectors(self):
        detectors = list(self.df['detector'].unique())
        nrca = [d for d in detectors if 'NRCA' in d and 'NRCB' not in d]
        nrcb = [d for d in detectors if 'NRCB' in d and 'NRCA' not in d]
        nrcs = [d for d in nrca + nrcb if '|' not in d] # single (any)
        multi_nrca = [d for d in nrca if '|' in d] # multiple A
        multi_nrcb = [d for d in nrcb if '|' in d] # multiple B
        multi_nrcab = [d for d in detectors if 'NRCA' in d and 'NRCB' in d] # A + B
        self.df.loc[self.df['detector'].isin(nrcs), 'detector'] = 'NRC-S'
        self.df.loc[self.df['detector'].isin(multi_nrca), 'detector'] = 'NRCA-M'
        self.df.loc[self.df['detector'].isin(multi_nrcb), 'detector'] = 'NRCB-M'
        self.df.loc[self.df['detector'].isin(multi_nrcab), 'detector'] = 'NRC-M'

    def group_subarrays(self):
        for key in ['MASK', 'SUB', 'WFSS']:
            self.df.loc[self.df['subarray'].str.startswith(key), 'subarray'] = key

    def rename_miri_mrs(self):
        """DEPRECATED: Default behavior of JWST Pipeline >=1.17.0 now generates a separate L3 Product for each sub-channel (band). 
        This class method will be removed in the next upcoming release."""
        mirmrs = {}
        for k, v in self.specpix.items():
            if k[-1] == '_':
                bands = ''.join(v['BAND'].split('|'))
                if bands:
                    mirmrs[k] = k + f'ch1-{bands.lower()}'
        for k, v in mirmrs.items():
            self.specpix[v] = self.specpix.pop(k)
            self.spec_products[v] = self.spec_products.pop(k)

    def get_dtype_keys(self):
        """Group input metadata into pre-set data types before applying NaNdlers.
        Returns
        -------
        dict
            key-value pairs of data type and exposure header / column name
        """
        return dict(
            continuous=[
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
                "targ_frac",
                "gs_mag",
            ],
            boolean=["bkgdtarg", "tsovisit", "is_imprt", "crowdfld"],
            categorical=[
                "instr",
                "detector",
                "filter",
                "pupil",
                "grating",
                "exp_type",
                "channel",
                "band",
                "subarray",
                "visitype",
            ],
        )


class NaNdler:
    def __init__(self, df, dtype_cols, allow_neg=False, verbose=False):
        self.df = df
        self.dtype_cols = dtype_cols
        self.allow_neg = allow_neg
        self.verbose = verbose
        self.continuous = self.dtype_cols.get("continuous", None)
        self.discrete = self.dtype_cols.get("discrete", None)
        self.boolean = self.dtype_cols.get("boolean", None)
        self.special_bools = self.dtype_cols.get("special_bools", None)
        self.categorical = self.dtype_cols.get("categorical", None)

    def continuous_nandler(self):
        if self.continuous:
            cols = [n for n in self.continuous if n in self.df.columns]
            if self.verbose:
                print(f"\nNaNs to be NaNdled:\n{self.df[cols].isna().sum()}\n")
            for n in cols:
                self.df.loc[self.df[n].isna(), n] = 0.0

    def discrete_nandler(self, nanval=0.0):
        if self.discrete:
            cols = [n for n in self.discrete if n in self.df.columns]
            if self.verbose:
                print(f"\nNaNs to be NaNdled:\n{self.df[cols].isna().sum()}\n")
            for n in cols:
                if self.allow_neg is True and 0.0 in self.df[n].value_counts().index:
                    nanval = -1.0
                self.df.loc[self.df[n].isna(), n] = nanval

    @staticmethod
    def nandle_cats(x, truevals):
        if x in truevals:
            return x
        else:
            return "NONE"

    def categorical_nandler(self):
        if self.categorical:
            cols = [c for c in self.categorical if c in self.df.columns]
            if self.verbose:
                print(f"\nNaNs to be NaNdled:\n{self.df[cols].isna().sum()}\n")
            df_cat = self.df[cols].copy()
            for col in cols:
                df_cat.loc[df_cat[col].isin(["N/A", "NaN", "NAN", "nan"]), col] = np.nan
                if df_cat[col].isna().sum() > 0:
                    truevals = list(df_cat[col].value_counts().index)
                    df_cat[col] = df_cat[col].apply(
                        lambda x: self.nandle_cats(x, truevals)
                    )
            self.df[cols] = df_cat[cols]

    def boolean_nandler(self, replace=True):
        if self.boolean:
            self.df = encode_booleans(self.df, self.boolean, replace=replace)
        if self.special_bools:
            self.df = encode_booleans(
                self.df, self.special_bools, special=True, replace=replace
            )

    def apply_nandlers(self):
        self.continuous_nandler()
        self.discrete_nandler()
        self.categorical_nandler()
        self.boolean_nandler()
        if self.verbose:
            for k, v in self.dtype_cols.items():
                v = [c for c in v if c in self.df.columns]
                print(f"\n{k}\n" + "---" * 3)
                print(f"\nNaNs remaining:\n{self.df[v].isna().sum()}")
        return self.df
