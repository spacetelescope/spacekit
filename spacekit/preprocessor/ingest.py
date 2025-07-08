import os
import sys
import glob
import shutil
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from spacekit.logger.log import Logger
from spacekit.extractor.scrape import JsonScraper
from spacekit.preprocessor import FALSEVALS
from spacekit.preprocessor.scrub import HstSvmScrubber, JwstCalScrubber
from spacekit.generator.draw import DrawMosaics
from spacekit.skopes.jwst.cal.config import KEYPAIR_DATA, L3_TYPES
from spacekit.analyzer.track import timer, record_metrics, xtimer


class SvmAlignmentIngest:
    """Class for ingesting and preprocessing HST single visit mosaic alignment classifier datasets

    Parameters
    ----------
    input_path : str or Path, optional
        path on local disk to the input data, by default None
    outpath : str or Path, optional
        path on local disk to save outputs, by default None
    """
    def __init__(self, input_path=None, outpath=None):
        self.input_path = os.getcwd() if input_path is None else input_path
        self.batch_out = os.getcwd() if outpath is None else outpath
        self.log_dir = None
        self.clean = True
        self.visit_data = []
        self.data_paths = []
        self.json_pattern = "*_total*_svm_*.json"
        self.crpt = 0
        self.draw = 1
        self.img_outputs = os.path.join(self.batch_out, "img")

    def start(self, func, ps="prep", visit=None, **args):
        t0, start = timer()
        func.__call__(**args)
        wall, clock = timer(t0=t0, clock=start)
        record_metrics(self.log_dir, visit, wall, clock, ps=ps)

    def prep_svm_batch(self, batch_name="drz", drz_ver="3.4.1"):
        if drz_ver:
            batch_name += f"_{''.join(drz_ver.split('.'))}"
        self.run_preprocessing(self, fname=batch_name, output_path=self.batch_out)

    def prep_single_visit(self, visit_path):
        visit = str(os.path.basename(visit_path))
        drz_file = glob.glob(f"{visit_path}/*total*.fits")
        if len(drz_file) > 0:
            dets = [drz.split("/")[-1].split("_")[4] for drz in drz_file]
            try:
                input_path = os.path.dirname(visit_path)
                for det in dets:
                    _, _ = self.run_preprocessing(
                        input_path,
                        fname=f"{visit}_{det.lower()}_data",
                        output_path=self.batch_out,
                        visit=visit,
                    )
                if self.clean is True:
                    shutil.rmtree(visit_path)
            except Exception as e:
                print(e)
                sys.exit(1)

    def run_preprocessing(
        self,
        h5=None,
        fname="svm_data",
        visit=None,
    ):
        """Scrapes SVM data from raw files, preprocesses dataframe for MLP classifier and generates png images for image CNN.
        #TODO: if no JSON files found, look for results_*.csv file instead and preprocess via alternative method

        Parameters
        ----------
        input_path : str
            path to SVM dataset directory
        h5 : str, optional
            load from existing hdf5 file, by default None
        fname : str, optional
            base filename to give the output files, by default "svm_data"
        output_path : str, optional
            where to save output files. Defaults to current working directory, by default None
        json_pattern : str, optional
            glob-based search pattern, by default "*_total*_svm_*.json"
        visit: str, optional
            single visit name (e.g. "id8f34") matching subdirectory of input_path; will search and preprocess this visit only
            (rather than all visits contained in the input_path), by default None
        crpt : int, optional
            set to 1 if using synthetic corruption data, by default 0
        draw : int, optional
            generate png images from dataset, by default 1

        Returns
        -------
        dataframe
            preprocessed Pandas dataframe
        """
        os.makedirs(self.batch_out, exist_ok=True)
        fname = os.path.basename(fname).split(".")[0]
        # 1: SCRAPE JSON FILES and make dataframe
        if h5 is None:
            search_path = (
                os.path.join(self.input_path, visit) if visit else self.input_path
            )
            patterns = self.json_pattern.split(",")
            jsc = JsonScraper(
                search_path=search_path,
                search_patterns=patterns,
                file_basename=fname,
                crpt=self.crpt,
                output_path=self.batch_out,
            )
            jsc.json_harvester()
        else:
            jsc = JsonScraper(h5_file=h5).load_h5_file()
        # 2: Scrape Fits Files and SCRUB DATAFRAME
        scrub = HstSvmScrubber(
            self.input_path,
            data=jsc.data,
            output_path=self.batch_out,
            output_file=fname,
            crpt=self.crpt,
        )
        scrub.preprocess_data()
        # 3:  DRAW IMAGES
        if self.draw:
            img_outputs = os.path.join(self.batch_out, "img")
            mos = DrawMosaics(
                self.input_path,
                output_path=img_outputs,
                fname=scrub.data_path,
                pattern="",
                gen=3,
                size=(24, 24),
                crpt=self.crpt,
            )
            mos.generate_total_images()

        self.visit_data.append(scrub.df)
        self.data_paths.append(scrub.data_path)
        print(f"DATA PATH: {scrub.data_path}\n")
        print(scrub.df_visit)
        return scrub.df, scrub.data_path

    def concat_prepped(dpath):
        datafiles = glob.glob(f"{dpath}/??????_*data.csv")
        for i, fpath in enumerate(datafiles):
            df_visit = pd.read_csv(fpath, index_col="index")
            if i == 0:
                df = df_visit
            else:
                df = pd.concat([df, df_visit], axis=0)
        if "index" not in df.columns:
            df["index"] = df.index
        df.to_csv(f"{dpath}/preprocessed.csv", index=False)

    def concat_raw(dpath):
        rawfiles = glob.glob(f"{dpath}/raw_*_data.csv")
        for i, raw in enumerate(rawfiles):
            df_raw = pd.read_csv(raw, index_col="index")
            if i == 0:
                df = df_raw
            else:
                df = pd.concat([df, df_raw], axis=0)
        if "index" not in df.columns:
            df["index"] = df.index
        df.to_csv(f"{dpath}/raw_combined.csv", index=False)

    def final_cleanup(df, dpath):
        print("Cleaning up...")
        csvfiles = glob.glob(f"{dpath}/*_data.csv")
        h5files = glob.glob(f"{dpath}/*_data.h5")
        rawfiles = glob.glob(f"{dpath}/raw_*_data.csv")
        filegroups = [csvfiles, h5files, rawfiles]
        for grp in filegroups:
            if len(df) == len(grp):
                for f in grp:
                    os.remove(f)
                print(f"Cleaned up {len(grp)} files")
            else:
                print(
                    f"{len(df)} in DF does not match {len(grp)} in filegroup. Skipping cleanup"
                )


class JwstCalIngest:
    """Loads raw JWST Calibration Pipeline metadata from local disk (`input_path`)
    and runs initial ML preprocessing steps necessary prior to model training. The resulting 
    dataframes will be "ingested" into any pre-existing training sets located in `outpath`. 
    This outpath acts as the primary database containing several "tables" (dataframes stored
    in .csv files). This class is designed to run on single or multiple files at a time 
    (limit specificity using 'pfx`). 
    
    Input file naming convention: YYYY-MM-DD_%d.csv (%d = day of year) ex: 2024-02-21_052.csv
    Alternate formats currently not supported because filenames are used to store date info.
    Examples: 
    
    - To ingest multiple files from November 2023, set `pfx="2023-11"`. 
    - To ingest only one file from January 3, 2024, set `pfx="2024-01-03"`. 
    - You can also pass in a wildcard: `pfx="*_3"` would search for all data collected on days 300-365 of any year, while `pfx="2023*_3"` would do the same but only for the year 2023.

    The contents of raw metadata files are expected to contain:

        - 1) columns consistent with Fits header keyword-values used in JWST Cal model training (see `spacekit.skopes.jwst.cal.config`) 
        - 2) rows of Level 1/1b exposures (inputs/features) along with Level 3 products
        - 3) imagesize (memory footprint) for each L3 product (outputs/target)

    Parameters
    ----------
    input_path : str (path), optional
        directory path to csv files on local disk, by default None (current working directory)
    pfx : str, optional
        filename start pattern (e.g. "2023" or "*-12-), by default ""
    outpath : str (path), optional
        directory path to save (and/or update) preprocessed files on local disk, by default None (current working directory)
    save_l1 : bool, optional
        save matched level 1 input data to separate file, by default True
    """
    def __init__(self, input_path=None, pfx="", outpath=None, save_l1=False, **log_kws):
        self.input_path = input_path.rstrip("/") if input_path is not None else os.getcwd()
        self.pfx = "" if pfx is None else pfx
        self.save_l1 = save_l1
        self.set_outpath(value=outpath)
        self.exp_types = ["IMAGE", "SPEC", "TAC", "FGS"]
        self.files = []
        self.idxcol = "Dataset"
        self.dag = "DagNodeName"
        self.df = None
        self.l1_dags = []
        self.l3_dags = []
        self.data = {}
        self.raw = {}
        self.product_matches = None
        self.exmatches = {}
        self.rem = {}
        self.param_cols = ['pid', 'OBSERVTN', 'FILTER', 'GRATING', 'PUPIL', 'SUBARRAY', 'FXD_SLIT', 'EXP_TYPE', 'BAND']
        self.scrb = None
        self.__name__ = "JwstCalIngest"
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()

    @property
    def float_cols(self):
        return self._float_cols()

    def _float_cols(self):
        return [
            'CRVAL1',
            'CRVAL2',
            'RA_REF',
            'DEC_REF',
            'GS_RA',
            'GS_DEC',
            'GS_MAG',
            'TARG_RA',
            'TARG_DEC'
        ]

    def set_outpath(self, value=None):
        """Initialize class variables relating to file paths on local disk where ingested data will be stored.
        If nothing is passed into the `value` kwarg, the default base path for outputs will be the same as inputs.

        Parameters
        ----------
        value : str or Path, optional
            custom path to a directory where output files will be saved, by default None
        """
        if value is None:
            value = str(self.input_path)
        self.outpath = value.rstrip("/")
        os.makedirs(self.outpath, exist_ok=True)
        self.ingest_file = os.path.join(self.outpath, "ingest.csv")
        self.trainpath = self.outpath + "/train-{}.csv"
        self.rempath =  self.outpath + "/rem-{}.csv"
        self.rawpath = self.outpath + "/raw-{}.csv"

    @xtimer
    def run_ingest(self):
        """Main calling function to run the entire ingest script.
        """
        self.ingest_data()
        if len(self.files) == 0:
            return
        self.initial_scrub()
        self.load_priors()
        self.scrub_exposures()
        self.extrapolate()
        self.save_ingest_data()
        self.save_training_sets()
        if self.l3 is not None:
            self.log.error(f"Houston, we have problem: {len(self.l3)} disconnected L3 product(s) floating in space")
            sys.exit(1)

    def ingest_data(self):
        """Loads all relevant files to be ingested into a single dataframe, adding columns for date, year and day of year (`doy`)
        based on the file names to demarcate the file from which each dataset originated. Additionally, only observations relating to
        jwst calibration levels 1 and 3 are kept, while the rest are dropped.
        """
        if len(self.files) == 0:
            self.read_files()
        for f in self.files:
            df = pd.read_csv(f, index_col=self.idxcol)
            df = self.drop_level2(df)
            filedate, day = os.path.basename(f).split('_')
            df['date'] = filedate
            df['year'] = filedate.split('-')[0]
            df['doy'] = int(day.split('.')[0])
            if self.df is None:
                self.df = df
            else:
                self.df = pd.concat([self.df, df], axis=0)
        if self.df is not None:
            self.log.info(f"{len(self.df)} datasets loaded from {len(self.files)} file(s)")

    def read_files(self):
        """Collects a list of filenames to be ingested from local disk according to the glob pattern
        combining `input_path` and `pfx` ending with `.csv`. A warning is issued if no files matching the pattern
        are found. The list of files are stored in the class attribute `files`.
        """
        pattern = f"{self.input_path}/{self.pfx}*.csv"
        files = sorted(glob.glob(pattern))
        self.files = [f for f in files if f not in self.files]
        if len(self.files) < 1:
            self.log.warning(f"No files found using pattern: {pattern}")
        else:
            self.log.debug(f"Files ready for ingest: {self.files}")

    def drop_level2(self, df):
        """Determines which `dag` column values relate to Level 1 and Level 3 according to their names,
        then drops any rows from the DataFrame that do not match these values. Note: starting on 6/13/2025, 
        a change in the data collection process added a new `dag` value 'ESTIMATE_LEVEL_3_MEMORY' which
        is unrelated to the actual processing of a dataset on its designated server node and therefore rows matching
        this value are also removed.
 
        Parameters
        ----------
        df : pandas.DataFrame
            dataframe to search and modify

        Returns
        -------
        pandas.DataFrame
            dataframe with only L1 and L3 datasets
        """
        alldags = sorted(list(df[self.dag].value_counts().index))
        l1_dags = [d for d in alldags if '1' in d]
        l3_dags = [d for d in alldags if '3' in d and 'MEMORY' not in d]
        dags_l1_l3 = l1_dags + l3_dags
        df = df.loc[df[self.dag].isin(dags_l1_l3)]
        self.l1_dags.extend([l for l in l1_dags if l not in self.l1_dags])
        self.l3_dags.extend([l for l in l3_dags if l not in self.l3_dags])
        return df

    def initial_scrub(self):
        """Initial preprocessing renames and adds several columns, sets the df index to Dataset, recasts datatypes, 
        and drops the following:
        - older duplicates and exposure types known to be unrelated to Level 3 processing 
        - redundant MIRI IFU products (only 1 channel per dataset is kept) 
        - mosaics (estimates for L3 datasets used to create a mosaic accurately reflect compute requirements)
        """
        if self.df is None:
            return
        self.df['dname'] = self.df.index
        self.df['dname'] = self.df['dname'].apply(lambda x: self.strip_file_suffix(x))
        self.df.rename({'ImageSize':'imagesize', self.dag: 'dag'}, axis=1, inplace=True)
        self.dag = 'dag'
        self.df['pid'] = self.df['dname'].apply(lambda x: self.extract_pid(x))
        self.df = self.recast_dtypes(self.df)
        n0 = len(self.df)
        self.df = self.df.sort_values(by=['date', 'imagesize']).drop_duplicates(subset='dname', keep='last')
        self.log.info(f"Dropped {n0 - len(self.df)} duplicates.")
        nonsci = self.df.loc[~self.df['EXP_TYPE'].isin(L3_TYPES)]
        self.df.drop(nonsci.index, axis=0, inplace=True)
        self.log.info(f"Dropped {len(nonsci)} non-L3 exposure types")
        self.set_params()
        self.reduce_mirifu_channels()
        self.drop_mosaics()
        self.df['Dataset'] = self.df['dname']
        self.df.set_index('Dataset', inplace=True)

    def reduce_mirifu_channels(self):
        """Append channel info to `params` string; drop MIRI IFU L3 products from channels 2,4 (keep only 1, 3).
        Channels 1-2 use the same input exposures, and the same goes for channels 3-4. 
        NOTE: The memory footprint for each L3 product is the same regardless of channel or subchannel ('band'), 
        so the inclusion of L3 products from both channels 1 and 3 is likely to be redundant for ML training purposes.
        The resulting metadata features will show some variability between ch1/2 and 3/4 L3 products because the input exposures
        are distinct. Further analysis is needed to determine if such variability simply adds noise to the training set, and a decision should be made at training time whether or not to include both channels. Adjustments to inference preprocessing may need to be made so that the model simply ignores channel/subchannel altogether and treats the entire group of inputs as pertaining to a single L3 product (jw_PID_OBS_TRG_miri_). The memory footprint estimates for each individual channel/subchannel combination can be inferred from a single inference output and applied to all relevant 'subproducts'.
        """
        for d, c in dict(zip(['MIRIFUSHORT', 'MIRIFULONG'],['-12', '-34'])).items():
            self.df.loc[self.df['DETECTOR'] == d, 'params'] = self.df.loc[self.df['DETECTOR'] == d].params.values + c
        self.mm = self.df.loc[
            (
                self.df['EXP_TYPE'] == "MIR_MRS"
            ) & (
                self.df[self.dag].isin(self.l3_dags)
            ) & (
                self.df['CHANNEL'].isin(['2','4'])
            )
        ]
        if len(self.mm) > 0:
            drops = self.mm.index
            self.log.info(f"Ignoring MIRI IFU channels 2,4 for {len(drops)/2} L3 products")
            self.df.drop(drops, axis=0, inplace=True)

    def load_and_recast(self, dpath, idxcol=None):
        """Loads in a dataframe from file on local disk generated by a prior ingest and recasts data types as needed
        for certain columns where that information is lost during a save.

        Parameters
        ----------
        dpath : str or Path
            path on local disk where file is stored
        idxcol : str, optional
            custom index column name, by default None

        Returns
        -------
        pandas.DataFrame
            df loaded with columns recasted as necessary
        """
        if not os.path.exists(dpath):
            self.log.warning(f"File does not exist at specified path: {dpath}")
            return
        idxcol = self.idxcol if idxcol is None else idxcol
        df = pd.read_csv(dpath, index_col=idxcol)
        return self.recast_dtypes(df)

    def recast_dtypes(self, df):
        """When loading a saved dataframe, some datatypes need to be recast appropriately
        in order to be able to edit existing / insert new values.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe to be recast

        Returns
        -------
        pandas.DataFrame
            recasted dataframe
        """
        df['OBSERVTN'] = df['OBSERVTN'].apply(lambda x: self.validate_obs(x))
        df['PROGRAM'] = df['PROGRAM'].apply(lambda x: '{:0>5}'.format(x))
        for col in self.float_cols:
            df[col] = df[col].apply(lambda x: self.convert_to_float(x))
        df['year'] = df['year'].astype('int64')
        df['date'] = pd.to_datetime(df['date'], yearfirst=True)
        df['TARG_RA'] = df['TARG_RA'].apply(lambda x: np.round(x, 8))
        # fine-grained matching param without affecting original values
        df['targra'] = df['TARG_RA'].apply(lambda x: np.round(x, 6))
        return df

    def load_priors(self):
        """Loads previously ingested but unmatched datasets from 'ingest.csv' file located in
        `output_path` on local disk. Checks the `params` column and extracts any that match
        the current ingest dataframe in order to attempt a new match. This is necessary for some
        datasets which take multiple days to complete processing. 
        """
        if not os.path.exists(self.ingest_file):
            self.log.debug("Prior data not found -- skipping.")
            return
        l3params = self.df.loc[self.df.dag.isin(self.l3_dags)].params.unique()
        self.log.info("Checking prior data")
        di = self.load_and_recast(self.ingest_file)
        di = di.sort_values(by='date').drop_duplicates(subset='dname', keep='last')
        ds = di.loc[di.params.isin(l3params)].copy()
        if len(ds) > 0:
            self.log.info(f"Prior data loaded successfully: {len(ds)} exposures added.")
            try:
                self.df = pd.concat([self.df, ds], axis=0)   
            except Exception as e:
                self.log.error(str(e))
            self.update_dags()
            self.df = self.df.sort_values(by='date').drop_duplicates(subset='dname', keep='last')
            di.drop(ds.index, axis=0, inplace=True)
            di[self.idxcol] = di.index
            di.to_csv(self.ingest_file, index=False)

    def update_dags(self):
        """Update the lists of l1 and l3 dag values once a collection of datasets from multiple files are ingested 
        (including priors loaded from ingest.csv).
        """
        alldags = sorted(list(self.df[self.dag].value_counts().index))
        self.l1_dags = [d for d in alldags if '1' in d]
        self.l3_dags = [d for d in alldags if '3' in d and 'MEMORY' not in d]

    def set_params(self):
        """Creates a new dataframe column containing a concatenated string of keywords that uniquely identify a group of 
        related L1 inputs and their L3 output. This is used (in combination with other columns such as targ_ra/dec to match
        L1 exposures with their L3 product). WFSC params are generated separately.
        """
        wftypes= ['PRIME_WFSC_SENSING_ONLY', 'PRIME_WFSC_ROUTINE', 'PRIME_WFSC_SENSING_CONTROL']
        wfcols = ['pid', 'OBSERVTN', 'FILTER', 'PUPIL', 'DETECTOR']
        wfsc = self.df.loc[self.df['VISITYPE'].isin(wftypes)].copy()
        self.df.drop(wfsc.index, axis=0, inplace=True)
        if len(self.df) > 0:
            params = list(
                map(
                    lambda x: '-'.join([str(y) for y in x if str(y) not in  FALSEVALS]),  
                    self.df[self.param_cols].values
                )
            )
            self.df['params'] = pd.DataFrame(params, index=self.df.index)
        if len(wfsc) > 0:
            wfparams = list(
                map(
                    lambda x: '-'.join([str(y) for y in x if str(y) not in  FALSEVALS]),  
                    wfsc[wfcols].values
                )
            )
            wfsc['params'] = pd.DataFrame(wfparams, index=wfsc.index)
            self.df = pd.concat([self.df, wfsc], axis=0)

    @staticmethod
    def save_kwargs(path):
        if not os.path.exists(path):
            return dict(index=False)
        return dict(mode='a', index=False, header=False)

    @staticmethod
    def strip_file_suffix(x):
        if x.endswith("fits"):
            x = '_'.join(x.split('_')[:-1])
        return x

    @staticmethod
    def extract_pid(x):
        if not isinstance(x, str):
            return x
        pid = x[2:7]
        if pid[0] == '0':
            pid = pid[1:]
        return int(pid)

    @staticmethod
    def validate_obs(x):
        return '{:0>3}'.format(x)

    @staticmethod
    def convert_to_float(x):
        if x != "NONE":
            return float(x)
        else:
            return np.nan

    @staticmethod
    def mark_mosaics(x):
        """Identify mosaic L3 products based on the dataset's name format.

        Parameters
        ----------
        x : str
            Dataset name

        Returns
        -------
        bool
            True if the dataset name is a mosiac otherwise False
        """
        if len(x.split('-')) < 2:
            return False
        elif x.split('-')[1][0] != 'c':
            return False
        return True

    def drop_mosaics(self):
        """Separate mosaic L3 products and save to `mosaics.csv` on local disk.
        """
        self.df['mosaic'] = self.df['dname'].apply(lambda x: self.mark_mosaics(x))
        mosaics = self.df.loc[self.df['mosaic']].copy()
        if len(mosaics) > 0:
            mosaics.drop('mosaic', axis=1, inplace=True)
            mpath = f"{self.outpath}/mosaics.csv"
            mosaics[self.idxcol] = mosaics.index
            mosaics.to_csv(mpath, **self.save_kwargs(mpath))
            self.log.info(f"Mosaic data saved to: {mpath}")
            self.log.info(f"Dropping {len(mosaics.index)} mosaics from ingest data")
            self.df.drop(mosaics.index, axis=0, inplace=True)
        self.df.drop('mosaic', axis=1, inplace=True)

    def scrub_exposures(self):
        """Preprocess the L1 input exposures through the JWST Scrubber. See JwstCalScrubber for details.
        """
        self.scrb = JwstCalScrubber(
                self.input_path,
                data=self.df.loc[self.df[self.dag].isin(self.l1_dags)],
                encoding_pairs=KEYPAIR_DATA,
                mode='df'
        )
        for exp_type in self.exp_types:
            inputs = self.scrb.scrub_inputs(exp_type=exp_type)
            if inputs is not None:
                inputs['dname'] = inputs.index
                self.data[exp_type] = inputs
        (self.img, self.spec, self.tac, self.fgs) = self.get_unencoded()
        self.raw = dict(zip(
            ["IMAGE", "SPEC", "TAC", "FGS"], 
            [self.img, self.spec, self.tac, self.fgs]
        ))

    def get_unencoded(self):
        """Retrieve the raw (unencoded) L3 products generated by the JWST Scrubber using preprocessed L1 exposure groups.

        Returns
        -------
        dict
            Dictionary of each exp_type's dataframe of raw (unencoded) L3 products generated based on groups of L1 input exposures run through the JWST Scubber. 
        """
        data = [self.scrb.imgpix, self.scrb.specpix, self.scrb.tacpix, self.scrb.fgspix]
        return map(lambda x: pd.DataFrame.from_dict(x, orient='index'), data)

    def extrapolate(self):
        """Match each group of L1 input exposures to a single L3 product, then separate unmatched exposures from the dataframe and convert imagesize to gigabytes. If any L3 products remain unmatched, the preliminary assumption is that these datasets were reprocessed and an attempt is made to update the relevant features for this product within the existing training file stored on local disk at `training.csv` if it exists. Warnings are reported by the log if multiple L3 products match a particular group of L1 inputs and/or L3 products remain that could not be matched with any input exposures or a previous L3 product in the existing training set. In both cases, these products are stored as a list in the `self.l3` attribute for further analysis and debugging since either occurrence indicates an error in the way data is being ingested (often as a result of unexpected changes made within the JWST pipeline after a given release).
        """
        for exp in self.data.keys():
            self.match_product_groups(exp)
            if len(self.exmatches[exp]) > 0:
                self.log.warning(f"Multiple matches in {exp}: {len(self.exmatches[exp])}")
        self.drop_unmatched()
        self.convert_imagesize_units()
        if 'pname' not in self.df.columns:
            self.log.debug("No L3 candidates to match")
            self.l3 = None
            return
        self.update_repro()
        self.l3 = self.df.loc[(self.df.pname.isna()) & (self.df.dag.isin(self.l3_dags))]
        if len(self.l3) > 0:
            self.log.warning(f"Unmatched L3 products: {len(self.l3)}")
            self.log.warning([d for d in list(self.l3.dname.values)])
        else:
            self.l3 = None

    def match_query(self, info, extra_param=None):
        """Queries the dataframe for L3 products matching the shared metadata attributes for a group of L1 input exposures. If a value is passed into the `extra_param` kwarg, the query is further restricted to include products with a value matching this additional parameter. If this initial query returns 0 results, a second broader query without the additional param is automatically run. By default, the query attempts to find L3 products within the dataframe whose `params` column value matches that of the L1 inputs' `params` column.

        Parameters
        ----------
        info : dict
            Key-value pairs of metadata pertaining to all L1 input exposures associated with a single L3 product.
        extra_param : str, optional
            Column name to match against an additional parameter value within the dataframe, by default None

        Returns
        -------
        list
            L3 products matching the specified metadata (and query parameters if requested). 
        """
        if extra_param:
            l3 = self.df.loc[
                (
                    self.df['params'] == info['params']
                ) & (
                    self.df[self.dag].isin(self.l3_dags)
                ) & (
                    self.df[extra_param] == info[extra_param]
                )
            ]
            if len(l3) == 0: # drop extra search param
                l3 = self.match_query(info)
        else:
            l3 = self.df.loc[
                (
                    self.df['params'] == info['params']
                ) & (
                    self.df[self.dag].isin(self.l3_dags)
                )
            ]
        return l3

    def match_product_groups(self, exp_type):
        """Matching L3 product with its associated L1 input exposures.
        1. If TARGNAME: match using params (PID-OBS-OPTELEM-SUBARRAY-EXP_TYPE) + TARGNAME
        2. Elif fixed target: match using params + targra (TARG_RA rounded to 6 sig. digits) 
        3. Else: match params + gs_mag

        Parameters
        ----------
        exp_type : str
            model-based 'exp_type' grouping: IMAGE, SPEC, TAC, or FGS
        """
        self.exmatches[exp_type] = {}
        for k, v in self.scrb.expdata[exp_type].items():
            exposures = list(v.keys())
            self.df.loc[self.df.dname.isin(exposures), 'expmode'] = exp_type
            info = self.df.loc[exposures[0]]
            qp = 'TARGNAME'
            if info[qp] == "NONE" or isinstance(info[qp], float):
                # TARG_RA rounded to 6 decimals for PTF
                qp = 'targra' if info['VISITYPE'] == "PRIME_TARGETED_FIXED" else 'GS_MAG'
            l3 = self.match_query(info, extra_param=qp)
            if len(l3) == 0:
                self.log.debug(f"No matching products identified: {k}")
                continue
            else:
                if len(l3) > 1:
                    if qp == 'TARGNAME' and info['VISITYPE'] == 'PRIME_TARGETED_FIXED':
                        l3 = self.match_query(info, extra_param='targra')
                    if len(l3) > 1: # FALLBACK
                        # check if miri ifu (l3 products identical for each band)
                        self.log.warning(f"MULTI MATCH ELIMINATION: {k}")
                        pnames = sorted(list(l3.index))
                        self.exmatches[exp_type][info['params']] = pnames
                        self.df.loc[self.df.dname.isin(pnames), 'expmode'] = exp_type
                        l3 = l3.loc[l3['dname'] == pnames[0]]
                pname = l3.iloc[0]['dname']
                imagesize = l3.iloc[0]['imagesize']
                self.data[exp_type].loc[k, 'pname'] = pname
                self.data[exp_type].loc[k, 'imagesize'] = imagesize
                self.data[exp_type].loc[k, 'date'] = l3.iloc[0]['date']
                self.df.loc[pname, 'pname'] = pname
                self.df.loc[self.df.dname.isin(exposures), 'pname'] = pname
                self.df.loc[self.df.pname == pname, 'expmode'] = exp_type

    def drop_unmatched(self):
        """Store any unmatched inputs into the `self.raw` attribute then remove them from the training set. Reports a log of the percentage of L3 products successfully matched during this ingest run (anything less than 100% indicates an error).
        """
        for exp in list(self.data.keys()):
            extracols = [c for c in ['imagesize','date','pname'] if c in self.data[exp].columns]
            self.raw[exp] = pd.concat([self.raw[exp], self.data[exp][extracols]], axis=1)
            try:
                if 'imagesize' in self.raw[exp].columns:
                    self.rem[exp] = self.raw[exp].loc[self.raw[exp]['imagesize'].isna()].copy()
                else:
                    self.rem[exp] = self.raw[exp].copy()
                n = self.df.loc[(self.df.dag.isin(self.l3_dags)) & (self.df.expmode == exp)]['expmode'].size
                self.data[exp].drop(self.rem[exp].index, axis=0, inplace=True)
                self.raw[exp].drop(self.rem[exp].index, axis=0, inplace=True)
                if n > 0:
                    self.log.info(f"[{exp}] L3 matched: {len(self.data[exp])} | {np.round((len(self.data[exp])/n)*100)}%")
            except KeyError:
                continue

    def convert_imagesize_units(self, data=None):
        """Converts the `imagesize` (memory footprint) column to Gigabyte units and stores the values in a new column named `imgsize_gb` for each exp_type in the `self.data` attribute (image, spec, etc). If the `data` kwarg is None, this change is also applied to the raw (unencoded) versions (`self.raw`). Otherwise the conversion is made to the dataframe passed into the `data` kwarg.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Apply the unit conversion to a particular dataframe instead of the default `self.data`, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe with additional column 'imgsize_gb` containing the GB values converted from `imagesize` column.
        """
        if data is not None:
            data['imgsize_gb'] = data['imagesize'].apply(lambda x: x / 10**6)
            return data
        for exp in self.exp_types:
            try:
                if 'imagesize' in self.data[exp].columns:
                    self.data[exp]['imgsize_gb'] = self.data[exp]['imagesize'].apply(lambda x: x / 10**6)
                    self.raw[exp]['imgsize_gb'] = self.data[exp]['imgsize_gb']
            except KeyError:
                continue

    def update_repro(self):
        """Sometimes an L3 product is reprocessed and will not have any matching L1 inputs.
        Updates the imagesize and date attributes of the previous record (if found) with that of the new one.
        """
        l3 = self.df.loc[(self.df.pname.isna()) & (self.df.dag.isin(self.l3_dags))]
        if len(l3) == 0:
            return
        self.log.info(f"Identified {len(l3)} potential reprocessed products eligible for update")
        dp = self.load_and_recast(f"{self.outpath}/training.csv")
        if dp is None:
            self.log.warning("Could not update repro data - file not found.")
            return
        dp = dp.sort_values(by='date').drop_duplicates(subset='pname', keep='last')
        updates = {}
        notrepro = []
        pnames = list(l3.index)
        for pname in pnames:
            try:
                expmode = dp.loc[pname]['expmode']
                if expmode not in updates:
                    updates[expmode] = dict()
                updates[expmode][pname] = dict(
                    imagesize=l3.loc[pname].imagesize,
                    doy=l3.loc[pname].doy,
                    date=l3.loc[pname].date,
                    year=l3.loc[pname].year
                )
            except KeyError:
                notrepro.append(pname)
                continue
        for exp_type, repro_data in updates.items():
            data = pd.read_csv(self.trainpath.format(exp_type.lower()), index_col=self.idxcol)
            for name, revised in repro_data.items():
                for k, v in revised.items():
                    dp.loc[name, k] = v
                data.loc[data.pname == name, 'imagesize'] = revised['imagesize']
                data.loc[data.pname == name, 'date'] = revised['date']
            data = self.convert_imagesize_units(data=data)
            data[self.idxcol] = data.index
            data.to_csv(self.trainpath.format(exp_type.lower()), index=False)
            self.log.info(f"Updated {len(repro_data)} reprocessed {exp_type} products.")
        dp[self.idxcol] = dp.index
        dp.to_csv(f"{self.outpath}/training.csv", index=False)
        l3 = l3.loc[~l3.dname.isin(notrepro)]
        if len(l3) > 0:
            self.df.drop(l3.index, axis=0, inplace=True)
            self.log.info(f"Training file updated and {len(l3)} L3 repro products removed from dataframe.")
        else:
            self.log.warning("0 repro candidates matched.")

    def save_training_sets(self):
        """Adds preprocessed ML training data for each model type to its respective file on local disk: `train-{exp_type}.csv`. 
        The raw (unencoded) versions are also saved to local disk as `raw-{exp_type}.csv`.
        Any remaining L1 inputs that did not have a matching L3 product are saved to `rem-{exp-type}.csv` primarily for debugging purposes.
        """
        for exp in self.exp_types:
            if exp in self.data.keys() and len(self.data[exp]) > 0:
                fpath = self.trainpath.format(exp.lower())
                self.data[exp][self.idxcol] = self.data[exp]['pname']
                self.data[exp].to_csv(fpath, **self.save_kwargs(fpath))
                self.log.info(f"{exp} training data saved to: {fpath}")
                wpath = self.rawpath.format(exp.lower())
                self.raw[exp][self.idxcol] = self.raw[exp]['pname']
                self.raw[exp].to_csv(wpath, **self.save_kwargs(wpath))
            if exp in self.rem.keys() and len(self.rem[exp]) > 0:
                rpath = self.rempath.format(exp.lower())
                self.rem[exp][self.idxcol] = self.rem[exp].index
                self.rem[exp].to_csv(rpath, index=False)
                self.log.info(f"Remaining {exp} data saved to: {rpath}")

    def save_ingest_data(self):
        """Adds unmatched L1 inputs into 'ingest.csv', matched L3 products to 'training.csv'. 
        If `save_l1` attribute is True, matched L1 input exposures are saved to a separate file 'level1.csv'.
        """
        self.df[self.idxcol] = self.df.index
        if 'pname' not in self.df.columns:
            di = self.df.loc[self.df.dag.isin(self.l1_dags)]
        else:
            di = self.df.loc[self.df.pname.isna()].copy()
            di.drop(['pname'], axis=1, inplace=True)
        di.to_csv(self.ingest_file, **self.save_kwargs(self.ingest_file))
        self.log.info(f"Remaining Ingest data saved to: {self.ingest_file}")
        dp = self.df.drop(di.index, axis=0)
        if len(dp) > 0:
            if self.save_l1 is True:
                l1 = dp.loc[dp.dag.isin(self.l1_dags)]
                l1_path = f"{self.outpath}/level1.csv"
                l1.to_csv(l1_path, **self.save_kwargs(l1_path))
                self.log.info(f"{len(l1)} L1 products added to: {l1_path}")
            dp = dp.loc[dp.dag.isin(self.l3_dags)]
            ppath = f"{self.outpath}/training.csv"
            dp.to_csv(ppath, **self.save_kwargs(ppath))
            self.log.info(f"{len(dp)} L3 products added to: {ppath}")


def hst_svm_ingest(**kwargs):
    """Main calling function for runnning HST SVM Alignment Data Ingest.
    """
    visit_path = kwargs.pop('visit_path', None)
    batch_name = kwargs.pop('batch_name', None)
    drz_ver = kwargs.pop('drz_ver', None)
    svi = SvmAlignmentIngest(**kwargs)
    if visit_path is not None:
        svi.prep_single_visit(visit_path)
    else:
        svi.prep_svm_batch(batch_name=batch_name, drz_ver=drz_ver)


def jwst_cal_ingest(**kwargs):
    """Main calling function for running JWST Calibration Data Ingest.
    """
    jc = JwstCalIngest(**kwargs)
    jc.run_ingest()


if __name__ == "__main__":
    parser = ArgumentParser(prog="spacekit.preprocessor.ingest")
    subparsers = parser.add_subparsers(title="skope", help="application skope")
    parser_jcal = subparsers.add_parser(
        "jcal",
        add_help=False, 
        parents=[parser],
        help="jwst calibration training data",
        usage="spacekit.preprocessor.ingest skope [options]"
    )
    parser_hsvm = subparsers.add_parser(
        "hsvm", 
        add_help=False, 
        parents=[parser], 
        help="hst svm alignment training data",
        usage="spacekit.preprocessor.ingest skope [options]"
    )
    for subparser in [parser_jcal, parser_hsvm]:
        subparser.add_argument("-i", "--input_path", default=os.getcwd(), type=str, help="data filepath to be ingested")
        subparser.add_argument("-o", "--outpath", type=str, default=None, help="path to save ingested data on local disk")

    parser_jcal.add_argument("-p","--pfx", type=str, default=None, help="file name prefix to limit search on local disk")
    parser_jcal.add_argument("-s", "--save_l1", type=bool, default=False, help="save matched level 1 input data to separate file")
    parser_jcal.set_defaults(func=jwst_cal_ingest)

    parser_hsvm.add_argument("-b", "--batch_name", type=str, default=None)
    parser_hsvm.add_argument("-d", "--drz_ver", type=str, default="")
    parser_hsvm.add_argument("-z", "--visit_path", type=str, default=None)
    parser_hsvm.set_defaults(func=hst_svm_ingest)

    kwargs = {**vars(parser.parse_args())}
    func = kwargs.pop('func')
    func(**kwargs)
