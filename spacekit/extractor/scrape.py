import os
from keras.utils.data_utils import get_file
import boto3
import numpy as np
import pandas as pd
import collections
import glob
import sys
import json
from stsci.tools import logutil
from zipfile import ZipFile

# from astropy.io import fits
from astroquery.mast import Observations
from progressbar import ProgressBar

# from botocore import Config
# retry_config = Config(retries={"max_attempts": 3})
client = boto3.client("s3")  # , config=retry_config)


def extract_archives(zipfiles, extract_to="data", delete_archive=False):
    fpaths = []
    os.makedirs(extract_to, exist_ok=True)
    for z in zipfiles:
        fname = os.path.basename(z).split(".")[0]
        fpath = os.path.join(extract_to, fname)
        with ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        # check just in case
        if os.path.exists(fpath):
            fpaths.append(fpath)
            if delete_archive is True:
                os.remove(z)
    return fpaths


# def unzip_images(zip_file):
#     basedir = os.path.dirname(zip_file)
#     key = os.path.basename(zip_file).split(".")[0]
#     image_folder = os.path.join(basedir, key + "/")
#     os.makedirs(image_folder, exist_ok=True)
#     with ZipFile(zip_file, "r") as zip_ref:
#         zip_ref.extractall(basedir)
#     print(len(os.listdir(image_folder)))
#     return image_folder


def scrape_web(key, uri):
    fname = key["fname"]
    origin = f"{uri}/{fname}"
    hash = key["hash"]
    fpath = get_file(
        origin=origin,
        file_hash=hash,
        hash_algorithm="sha256",  # auto
        cache_dir="~",
        cache_subdir="data",
        extract=True,
        archive_format="zip",
    )
    if os.path.exists(fpath):
        os.remove(f"data/{fname}")
    return fpath


def get_training_data(dataset=None, uri=None):
    if uri is None:
        print("Please enter a uri.")
        return None
    keys = list(dataset.keys())
    fpaths = []
    for key in keys:
        fpath = scrape_web(key, uri)
        fpaths.append(fpath)
    return fpaths


# TODO: delete (use s3scraper class below for new results file structure)
def scrape_s3(bucket, results=[]):
    res_keys = {}
    for r in results:
        os.makedirs(f"./data/{r}")
        res_keys[r] = [
            "data/latest.csv",
            "models/models.zip",
            "results/mem_bin",
            "results/memory",
            "results/wallclock",
        ]
    err = None
    for prefix, key in res_keys.items():
        obj = f"{bucket}/{prefix}/{key}"
        print("s3://", obj)
        try:
            keypath = f"./data/{prefix}/{key}"
            with open(keypath, "wb") as f:
                client.download_fileobj(bucket, obj, f)
        except Exception as e:
            err = e
            continue
    if err is not None:
        print(err)


class Scraper:
    def __init__(self, cache_dir="~", cache_subdir="data", format="zip", extract=True):
        self.cache_dir = cache_dir  # root path for downloads (home)
        self.cache_subdir = cache_subdir  # subfolder
        self.format = format
        self.extract = extract  # extract if zip/tar archive
        self.fpaths = None


class WebScraper(Scraper):
    def __init__(
        self,
        uri,
        dataset,
        hash_algorithm="sha256",
        cache_dir="~",
        cache_subdir="data",
        format="zip",
        extract=True,
    ):
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
        )
        self.uri = uri
        self.dataset = dataset
        self.hash_algorithm = hash_algorithm

    def scrape_repo(self):
        keys = list(self.dataset.keys())
        for key in keys:
            fname = key["fname"]
            origin = f"{self.uri}/{fname}"
            fpath = get_file(
                origin=origin,
                file_hash=key["hash"],
                hash_algorithm=self.hash_algorithm,
                cache_dir=self.cache_dir,
                cache_subdir=self.cache_subdir,
                extract=self.extract,
                archive_format=self.format,
            )
            self.fpaths.append(fpath)
            if os.path.exists(fpath):  # delete archive
                os.remove(f"{self.cache_subdir}/{fname}")
        return self.fpaths


# TODO
class S3Scraper(Scraper):
    def __init__(
        self,
        bucket,
        pfx="archive",
        dataset=None,
        cache_dir="~",
        cache_subdir="data",
        format="zip",
        extract=True,
    ):
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
        )
        self.bucket = bucket
        self.pfx = pfx
        self.dataset = dataset

    def make_s3_keys(
        self,
        fnames=[
            "2021-11-04-1636048291.zip",
            "2021-10-28-1635457222.zip",
            "2021-08-22-1629663047.zip",
        ],
    ):
        self.dataset = {}
        for key, fname in enumerate(fnames):
            self.dataset[str(key)] = {"fname": fname, "pfx": self.pfx}
        return self.dataset

    def extract_archive(self):
        extracted_fpaths = []
        for fpath in self.fpaths:
            with ZipFile(fpath, "r") as zip_ref:
                zip_ref.extractall()
            fname = fpath.split(".")[0]
            if os.path.exists(fname):  # delete archive
                os.remove(fpath)
                extracted_fpaths.append(fname)
        self.fpaths = extracted_fpaths
        return self.fpaths

    def scrape_s3(self):
        err = None
        dataset_keys = list(self.dataset.keys())
        for k in dataset_keys:
            fname = k["fname"]
            obj = f"{self.pfx}/{fname}"
            print(f"s3://{self.bucket}/{self.obj}")
            fpath = f"{self.cache_dir}/{self.cache_subdir}/{fname}"
            try:
                with open(fpath, "wb") as f:
                    client.download_fileobj(self.bucket, obj, f)
                self.fpaths.append(fpath)
            except Exception as e:
                err = e
                continue
        if err is not None:
            print(err)
        elif self.extract is True:
            if self.format == "zip":
                self.fpaths = self.extract_archive()
        return self.fpaths


class MastScraper:
    def __init__(self, df, trg_col="targname", ra_col="ra_targ", dec_col="dec_targ"):
        self.df = df
        self.trg_col = trg_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.targets = self.df[self.trg_col].unique()
        self.targ_any = self.df.loc[df[self.trg_col] == "ANY"][
            [self.ra_col, self.dec_col]
        ]
        self.target_categories = {}
        self.other_cat = {}
        self.categories = {}

    def scrape_mast(self):
        self.target_categories = self.scrape_target_categories()
        self.other_cat = self.scrape_other_targets()
        self.df = self.combine_categories()
        return self.df

    def scrape_target_categories(self):
        print("\n*** Assigning target name categories ***")
        print(f"\nUnique Target Names: {len(self.targets)}")
        bar = ProgressBar().start()
        for x, targ in zip(bar(range(len(self.targets))), self.targets):
            if targ != "ANY":
                obs = Observations.query_criteria(target_name=targ)
                cat = obs[np.where(obs["target_classification"])][
                    "target_classification"
                ]
                if len(cat) > 0:
                    self.target_categories[targ] = cat[0]
                else:
                    self.target_categories[targ] = "None"
            bar.update(x + 1)
        bar.finish()
        return self.target_categories

    def scrape_other_targets(self):
        self.other_cat = {}
        if len(self.targ_any) > 0:
            print(f"Other targets (ANY): {len(self.targ_any)}")
            bar = ProgressBar().start()
            for x, (idx, row) in zip(
                bar(range(len(self.targ_any))), self.targ_any.iterrows()
            ):
                self.other_cat[idx] = {}
                propid = str(idx).split("_")[1]
                ra, dec = row[self.ra_col], row[self.dec_col]
                obs = Observations.query_criteria(
                    proposal_id=propid, s_ra=ra, s_dec=dec
                )
                cat = obs[np.where(obs["target_classification"])][
                    "target_classification"
                ]
                if len(cat) > 0:
                    self.other_cat[idx] = cat[0]
                else:
                    self.other_cat[idx] = "None"
                bar.update(x)
            bar.finish()
        return self.other_cat

    def combine_categories(self):
        for k, v in self.target_categories.items():
            idx = self.df.loc[self.df[self.trg_col] == k].index
            for i in idx:
                self.categories[i] = v
        self.categories.update(self.other_cat)
        cat = pd.DataFrame.from_dict(
            self.categories, orient="index", columns={"category"}
        )
        print("\nTarget Categories Assigned.")
        print(cat["category"].value_counts())
        self.df = self.df.join(cat, how="left")
        return self.df


# TODO
class ImageScraper(Scraper):
    def __init__(self):
        super().__init__(self)


class JsonScraper:
    """Searches local files using glob pattern(s) to scrape JSON file data. Optionally can store data in h5 file (default) and/or CSV file; The JSON harvester method returns a Pandas dataframe. This class can also be used to load an h5 file.

    Parameters
    ----------
    search_path : str, optional
        The full path of the directory that will be searched for json files to process. If not explicitly
        specified, the current working directory will be used.
    search_patterns : list, optional
        list of glob patterns to use for search
    log_level : int, optional
        The desired level of verboseness in the log statements displayed on the screen and written to the
        .log file. Default value is 'INFO'.
    file_basename : str, optional
        Name of the output file basename (filename without the extension) for the Hierarchical Data Format
        version 5 (HDF5) .h5 file that the Pandas DataFrame will be written to. If not explicitly specified,
        the default filename basename that will be used is "svm_qa_dataframe". The default location that the
        output file will be written to is the current working directory
    crpt: bool, optional
        Uses extended dataframe index name to differentiate from normal svm data
    data : Pandas DataFrame
            Pandas DataFrame
    """

    def __init__(
        self,
        search_path=os.getcwd(),
        search_patterns=["*_total_*_svm_*.json"],
        file_basename="svm_data",
        crpt=0,
        save_csv=False,
        store_h5=True,
        output_path=None,
    ):
        self.search_path = search_path
        self.search_patterns = search_patterns
        self.file_basename = file_basename
        self.crpt = crpt
        self.save_csv = save_csv
        self.store_h5 = store_h5
        self.output_path = output_path
        self.__name__ = "diagnostic_json_harvester"
        self.msg_datefmt = "%Y%j%H%M%S"
        self.splunk_msg_fmt = "%(asctime)s %(levelname)s src=%(name)s- %(message)s"
        self.log_level = logutil.logging.INFO
        self.keyword_shortlist = [
            "TARGNAME",
            "DEC_TARG",
            "RA_TARG",
            "NUMEXP",
            "imgname",
            "Number of GAIA sources.Number of GAIA sources",
            "number_of_sources.point",
            "number_of_sources.segment",
        ]
        self.log = None
        self.json_dict = None
        self.data = None  # self.json_harvester()
        self.h5_file = None  # self.h5store()

    def start_logging(self):
        self.log = logutil.create_logger(
            self.__name__,
            level=self.log_level,
            stream=sys.stdout,
            format=self.splunk_msg_fmt,
            datefmt=self.msg_datefmt,
        )
        return self.log

    def flatten_dict(self, dd, separator=".", prefix=""):
        """Recursive subroutine to flatten nested dictionaries down into a single-layer dictionary.
        Borrowed from https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/

        Parameters
        ----------
        dd : dict
            dictionary to flatten
        separator : str, optional
            separator character used in constructing flattened dictionary key names from multiple recursive
            elements. Default value is '.'
        prefix : str, optional
            flattened dictionary key prefix. Default value is an empty string ('').
        Returns
        -------
        a version of input dictionary *dd* that has been flattened by one layer
        """
        return (
            {
                prefix + separator + k if prefix else k: v
                for kk, vv in dd.items()
                for k, v in self.flatten_dict(vv, separator, kk).items()
            }
            if isinstance(dd, dict)
            else {prefix: dd}
        )

    def read_json_file(self, json_filename):
        """extracts header and data sections from specified json file and returns the header and data (in it's original
        pre-json format) as a nested ordered dictionary

        Supported output data types:

        - all basic single-value python data types (float, int, string, Boolean, etc.)
        - lists
        - simple key-value dictionaries and ordered dictionaries
        - multi-layer nested dictionaries and ordered dictionaries
        - tuples
        - numpy arrays
        - astropy tables

        Parameters
        ----------
        json_filename : str
            Name of the json file to extract data from

        Returns
        -------
        out_dict : dictionary
            dictionary structured similarly to self.out_dict with separate 'header' and 'data' keys. The
            information stored in the 'data' section will be in the same format that it was in before it was serialized
            and stored as a json file.
        """
        if os.path.exists(json_filename):
            out_dict = collections.OrderedDict()
            with open(json_filename) as f:
                json_data = json.load(f)

            out_dict["header"] = json_data[
                "header"
            ]  # copy over the 'header' section directly.
            out_dict["general information"] = json_data["general information"]
            out_dict["data"] = collections.OrderedDict()  # set up blank data section
            for datakey in json_data["data"].keys():
                if (
                    json_data["data"][datakey]["original format"]
                    == "<class 'numpy.ndarray'>"
                ):  # Extract numpy array
                    self.log.info(
                        "Converting dataset '{}' back to format '{}', dtype = {}".format(
                            datakey,
                            json_data["data"][datakey]["original format"],
                            json_data["data"][datakey]["dtype"],
                        )
                    )
                    out_dict["data"][datakey] = np.asarray(
                        json_data["data"][datakey]["data"],
                        dtype=json_data["data"][datakey]["dtype"],
                    )
                elif (
                    json_data["data"][datakey]["original format"] == "<class 'tuple'>"
                ):  # Extract tuples
                    out_dict["data"][datakey] = tuple(
                        json_data["data"][datakey]["data"]
                    )
                else:  # Catchall for everything else
                    out_dict["data"][datakey] = json_data["data"][datakey]["data"]

        else:
            errmsg = "json file {} not found!".format(json_filename)
            self.log.error(errmsg)
            raise Exception(errmsg)
        return out_dict

    def get_json_files(self):
        """use glob to create a list of json files to harvest
        This function looks for all the json files containing qa test results generated
        by `runastrodriz` and `runsinglehap`.  The search starts in the directory
        specified in the `search_path` parameter, but will look in immediate
        sub-directories as well if no json files are located in the directory
        specified by `search_path`.

        Returns
        -------
        out_json_dict : ordered dictionary
            dictionary containing lists of all identified json files, grouped by and  keyed by Pandas DataFrame
            index value
        """
        # set up search string and use glob to get list of files
        json_list = []
        for search_pattern in self.search_patterns:
            search_string = os.path.join(self.search_path, search_pattern)
            search_results = glob.glob(search_string)
            if len(search_results) == 0:
                search_string = os.path.join(self.search_path, "*", search_pattern)
                search_results = glob.glob(search_string)

            self.log.info(
                "{} files found: {}".format(search_pattern, len(search_results))
            )
            if len(search_results) > 0:
                json_list += search_results

        # store json filenames in a dictionary keyed by Pandas DataFrame index value
        if json_list:
            self.json_dict = collections.OrderedDict()
            for json_filename in sorted(json_list):
                json_data = self.read_json_file(json_filename)
                dataframe_idx = json_data["general information"]["dataframe_index"]
                """***ADAPTED FOR MACHINE LEARNING ARTIFICIAL CORRUPTION FILES***"""
                if self.crpt == 1:
                    mm = "_".join(os.path.dirname(json_filename).split("_")[1:])
                    idx = f"{dataframe_idx}_{mm}"
                else:
                    idx = dataframe_idx
                if idx in self.json_dict.keys():
                    self.json_dict[idx].append(json_filename)
                else:
                    self.json_dict[idx] = [json_filename]
                del json_data  # Housekeeping!

        # Fail gracefully if no .json files were found
        else:
            err_msg = "No .json files were found!"
            self.log.error(err_msg)
            raise Exception(err_msg)
        return self.json_dict

    def h5store(self, **kwargs):
        """Write a pandas Dataframe and metadata to a HDF5 file.

        Returns
        -------
        h5 filepath
        """
        if self.output_path is None:
            self.output_path = os.getcwd()
        fname = self.file_basename.split(".")[0] + ".h5"
        self.h5_file = os.path.join(self.output_path, fname)
        if os.path.exists(self.h5_file):
            os.remove(self.h5_file)
        if self.data is not None:
            store = pd.HDFStore(self.h5_file)
            store.put("mydata", self.data)
            store.get_storer("mydata").attrs.metadata = kwargs
            store.close()
            self.log.info(
                "Wrote dataframe and metadata to HDF5 file {}".format(self.h5_file)
            )
        else:
            print("Data unavailable - run `json_scraper` to collect json data.")
        return self.h5_file

    def load_h5_file(self):
        if self.h5_file is None:
            if self.output_path is None:
                self.output_path = os.getcwd()
            self.h5_file = os.path.join(self.output_path, self.file_basename + ".h5")
        elif not self.h5_file.endswith(".h5"):
            self.h5_file += ".h5"
        if os.path.exists(self.h5_file):
            with pd.HDFStore(self.h5_file) as store:
                self.data = store["mydata"]
                print(f"Dataframe created: {self.data.shape}")
        else:
            errmsg = "HDF5 file {} not found!".format(self.h5_file)
            print(errmsg)
            raise Exception(errmsg)
        return self.data

    def json_harvester(self):
        """Main calling function"""
        self.log = self.start_logging()
        self.log.setLevel(self.log_level)
        # Get sorted list of json files
        self.data = None
        # extract all information from all json files related to a specific Pandas DataFrame index value into a
        # single line in the master dataframe
        self.json_dict = self.get_json_files()
        num_json = len(self.json_dict)
        for n, idx in enumerate(self.json_dict.keys()):
            if ((n / num_json) % 0.1) == 0:
                self.log.info(f"Harvested {num_json} of the JSON files")
            ingest_dict = self.make_dataframe_line(self.json_dict[idx])
            if ingest_dict:
                if self.data is not None:
                    self.log.debug("APPENDED DATAFRAME")
                    self.data = self.data.append(
                        pd.DataFrame(ingest_dict["data"], index=[idx])
                    )
                else:
                    self.log.debug("CREATED DATAFRAME")
                    self.data = pd.DataFrame(ingest_dict["data"], index=[idx])
        if self.save_csv:
            self.write_to_csv()
        return self.data

    def write_to_csv(self):
        """optionally also write dataframe out to .csv file for human inspection"""
        output_csv_filename = self.h5_filename.replace(".h5", ".csv")
        if os.path.exists(output_csv_filename):
            os.remove(output_csv_filename)
        self.data.to_csv(output_csv_filename)
        self.log.info("Wrote dataframe to csv file {}".format(output_csv_filename))

    def make_dataframe_line(self, json_filename_list):
        """extracts information from the json files specified by the input list *json_filename_list*.
        Parameters
        ----------
        json_filename_list : list
            list of json files to process
        Returns
        -------
        ingest_dict : collections.OrderedDict
            ordered dictionary containing all information extracted from json files specified by the input list
            *json_filename_list*.
        """
        self.log.setLevel(self.log_level)
        header_ingested = False
        gen_info_ingested = False
        ingest_dict = collections.OrderedDict()
        ingest_dict["data"] = collections.OrderedDict()
        # ingest_dict["descriptions"] = collections.OrderedDict()
        # ingest_dict["units"] = collections.OrderedDict()
        for json_filename in json_filename_list:
            # This is to differentiate point catalog compare_sourcelists columns from segment catalog
            # compare_sourcelists columns in the dataframe
            if json_filename.endswith("_point-cat_svm_compare_sourcelists.json"):
                title_suffix = "hap_vs_hla_point_"
            elif json_filename.endswith("_segment-cat_svm_compare_sourcelists.json"):
                title_suffix = "hap_vs_hla_segment_"
            else:
                title_suffix = ""
            json_data = self.read_json_file(json_filename)
            # add information from "header" section to ingest_dict just once
            keyword_shortlist = ["TARGNAME", "DEC_TARG", "RA_TARG", "NUMEXP", "imgname"]
            if not header_ingested:
                # filter out ALL header keywords not included in 'keyword_shortlist'
                for header_item in json_data["header"].keys():
                    if header_item in keyword_shortlist:
                        # if header_item in header_keywords_to_keep:
                        ingest_dict["data"]["header." + header_item] = json_data[
                            "header"
                        ][header_item]
                header_ingested = True

            # add information from "general information" section to ingest_dict just once
            if not gen_info_ingested:
                for gi_item in json_data["general information"].keys():
                    if gi_item in keyword_shortlist:
                        ingest_dict["data"]["gen_info." + gi_item] = json_data[
                            "general information"
                        ][gi_item]
                gen_info_ingested = True

            # recursively flatten nested "data" section dictionaries and build ingest_dict
            flattened_data = self.flatten_dict(json_data["data"])
            # flattened_descriptions = self.flatten_dict(json_data["descriptions"])
            # flattened_units = self.flatten_dict(json_data["units"])
            for fd_key in flattened_data.keys():
                json_data_item = flattened_data[fd_key]
                ingest_key = fd_key.replace(" ", "_")
                key_suffix = ingest_key.split(".")[-1]
                if key_suffix not in ["data", "unit", "format", "dtype"]:
                    if (
                        str(type(json_data_item))
                        == "<class 'astropy.table.table.Table'>"
                    ):
                        for coltitle in json_data_item.colnames:
                            ingest_value = json_data_item[coltitle].tolist()
                            id_key = title_suffix + ingest_key + "." + coltitle
                            ingest_dict["data"][id_key] = [ingest_value]
                    else:
                        ingest_value = json_data_item
                        id_key = title_suffix + ingest_key
                        if str(type(ingest_value)) == "<class 'list'>":
                            ingest_dict["data"][id_key] = [ingest_value]
                        else:
                            ingest_dict["data"][id_key] = ingest_value
        return ingest_dict
