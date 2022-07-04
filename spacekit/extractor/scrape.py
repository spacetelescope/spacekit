import os
from keras.utils.data_utils import get_file
import boto3
import numpy as np
import pandas as pd
import collections
import glob
import sys
import json
import csv
from stsci.tools import logutil
from zipfile import ZipFile
from astropy.io import fits, ascii
from astroquery.mast import Observations
from progressbar import ProgressBar
from botocore.config import Config
from decimal import Decimal
from boto3.dynamodb.conditions import Attr

retry_config = Config(retries={"max_attempts": 3})
client = boto3.client("s3", config=retry_config)
dynamodb = boto3.resource("dynamodb", config=retry_config, region_name="us-east-1")


SPACEKIT_DATA = os.environ.get("SPACEKIT_DATA", "~/spacekit_data")


def home_data_base(data_home=None):
    """Borrowed from ``sklearn.datasets._base.get_data_home`` function: Return the path of the spacekit data dir, and create one if not existing. Folder path can be set explicitly using ``data_home`` kwarg, otherwise it looks for the 'SPACEKIT_DATA' environment variable, or defaults to 'spacekit_data' in the user home directory (the '~' symbol is expanded to the user's home folder).

    Parameters
    ----------
    data_home : str, optional
        The path to spacekit data directory, by default `None` (will return `~/spacekit_data`)

    Returns
    -------
    data_home: str
        The path to spacekit data directory, defaults to `~/spacekit_data`
    """
    SPACEKIT_DATA = os.environ.get("SPACEKIT_DATA", "~/spacekit_data")
    if SPACEKIT_DATA == "":
        SPACEKIT_DATA = "~/spacekit_data"
    if data_home is None:
        data_home = os.environ.get(SPACEKIT_DATA, os.path.expanduser("~/spacekit_data"))
    else:
        data_home = os.path.abspath(data_home)
    try:
        os.makedirs(data_home, exist_ok=True)
    except Exception as e:
        print(e)
    return data_home


def scrape_catalogs(input_path, name, sfx="point"):
    if sfx != "ref":
        cfiles = glob.glob(f"{input_path}/{name}_{sfx}-cat.ecsv")
        if len(cfiles) > 0 and os.path.exists(cfiles[0]):
            cat = ascii.read(cfiles[0]).to_pandas()
            if len(cat) > 0:
                flagcols = [c for c in cat.columns if "Flags" in c]
                if len(flagcols) > 0:
                    flags = cat.loc[:, flagcols]
                    return flags[flags.values <= 5].shape[0]
        else:
            return 0
    else:
        cfiles = glob.glob(f"{input_path}/ref_cat.ecsv")
        if len(cfiles) > 0 and os.path.exists(cfiles[0]):
            cat = ascii.read(cfiles[0]).to_pandas()
            return len(cat)


def format_hst_cal_row_item(row):
    row["timestamp"] = int(row["timestamp"])
    row["x_files"] = float(row["x_files"])
    row["x_size"] = float(row["x_size"])
    row["bin_pred"] = float(row["bin_pred"])
    row["mem_pred"] = float(row["mem_pred"])
    row["wall_pred"] = float(row["wall_pred"])
    row["wc_mean"] = float(row["wc_mean"])
    row["wc_std"] = float(row["wc_std"])
    row["wc_err"] = float(row["wc_err"])
    return row


class Scraper:
    """Parent Class for various data scraping subclasses. Instantiating the appropriate subclass is preferred."""

    def __init__(
        self, cache_dir="~", cache_subdir="data", format="zip", extract=True, clean=True
    ):
        """Instantiates a spacekit.extractor.scrape.Scraper object.

        Parameters
        ----------
        cache_dir : str, optional
            parent folder to save data, by default "~"
        cache_subdir : str, optional
            save data in a subfolder one directory below `cache_dir`, by default "data"
        format : str, optional
            archive format type, by default "zip"
        extract : bool, optional
            extract the contents of the compressed archive file, by default True
        """
        self.cache_dir = self.check_cache(cache_dir)  # root path for downloads (home)
        self.cache_subdir = cache_subdir  # subfolder
        self.format = format
        self.extract = extract  # extract if zip/tar archive
        self.outpath = os.path.join(self.cache_dir, self.cache_subdir)
        self.clean = clean  # delete archive if extract successful
        self.source = None
        self.fpaths = []

    def check_cache(self, cache):
        if cache == "~":
            return os.path.expanduser(cache)
        elif cache == ".":
            return os.path.abspath(".")
        elif cache is None:
            return home_data_base()
        else:
            return os.path.abspath(cache)

    def extract_archives(self):
        """Extract the contents of the compreseed archive file(s).

        TODO: extract other archive types (.tar, .tgz)

        Returns
        -------
        list
            paths to downloaded and extracted dataset files
        """
        extracted_fpaths = []
        if not self.fpaths:
            return
        elif str(self.fpaths[0]).split(".")[-1] != "zip":
            return self.fpaths
        os.makedirs(self.outpath, exist_ok=True)
        for z in self.fpaths:
            with ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(self.outpath)
            # check successful extraction before deleting archive
            fname = str(z).split(".")[0]
            extracted = os.path.join(self.outpath, fname)
            if os.path.exists(extracted):
                extracted_fpaths.append(extracted)
                if self.clean is True:
                    os.remove(z)
        self.fpaths = extracted_fpaths
        return self.fpaths


class FileScraper(Scraper):
    """Scraper subclass used to search and extract files on local disk that match regex/glob pattern(s).

    Parameters
    ----------
    Scraper : spacekit.extractor.scrape.Scraper object
        parent Scraper class
    """

    def __init__(
        self,
        patterns=["*.zip"],
        cache_dir="~",
        cache_subdir="data",
        format="zip",
        extract=True,
        clean=False,
    ):
        """Instantiates a spacekit.extractor.scrape.FileScraper object.

        Parameters
        ----------
        patterns : list, optional
            glob pattern strings, by default ``["*.zip"]``
        """
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
            clean=clean,
        )
        self.patterns = patterns
        self.fpaths = []
        self.source = "file"

    def scrape(self):
        """Search local disk for files matching glob regex pattern(s)

        Parameters
        ----------
        patterns : list, optional

        Returns
        -------
        list
            paths to dataset files found in glob pattern search
        """
        for p in self.patterns:
            results = glob.glob(p)
            if len(results) > 0:
                for r in results:
                    self.fpaths.append(r)
        if self.extract is True:
            self.fpaths = super().extract_archives
        return self.fpaths


class WebScraper(Scraper):
    """Scraper subclass for extracting publicly available data off the web.

    Parameters
    ----------
    Scraper : class
        spacekit.extractor.scrape.Scraper object
    """

    def __init__(
        self,
        uri,
        dataset,
        hash_algorithm="sha256",
        cache_dir="~",
        cache_subdir="data",
        format="zip",
        extract=True,
        clean=True,
    ):
        """Uses dictionary of uri, filename and hash key-value pairs to download data securely from a website such as Github.

        Parameters
        ----------
        uri : string
            root uri (web address)
        dataset : dictionary
            key-pair values of each dataset's filenames and hash keys
        hash_algorithm : str, optional
            type of hash key algorithm used, by default "sha256"
        """
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
            clean=clean,
        )
        self.uri = uri
        self.dataset = dataset
        self.hash_algorithm = hash_algorithm
        self.source = "web"
        self.fpaths = []

    def scrape(self):
        """Using the key-pair values in `dataset` dictionary attribute, download the files from a github repo and check the hash keys match before extracting. Extraction and hash-key checking is handled externally by the `keras.utils.data_utils.get_file` method. If extraction is successful, the archive file will be deleted.

        Returns
        -------
        list
            paths to downloaded and extracted files
        """
        for _, data in self.dataset.items():
            fname = data["fname"]
            origin = f"{self.uri}/{fname}"
            chksum = data["hash"]
            fpath = get_file(
                origin=origin,
                file_hash=chksum,
                hash_algorithm=self.hash_algorithm,
                cache_dir=self.cache_dir,
                cache_subdir=self.cache_subdir,
                extract=self.extract,
                archive_format=self.format,
            )
            extracted = str(os.path.relpath(fpath)).split(".")[0]
            self.fpaths.append(extracted)
            if self.clean is True and os.path.exists(
                extracted
            ):  # deletes archive if extraction was successful
                os.remove(fpath)
                # os.remove(f"{os.path.join(self.outpath, fname)}")
        return self.fpaths


class S3Scraper(Scraper):
    """Scraper subclass for extracting data from an AWS s3 bucket (requires AWS credentials with permissions to access the bucket.)

    Parameters
    ----------
    Scraper : class
        spacekit.extractor.scrape.Scraper object
    """

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
        """Instantiates a spacekit.extractor.scrape.S3Scraper object

        Parameters
        ----------
        bucket : string
            s3 bucket name
        pfx : str, optional
            aws bucket prefix (subfolder uri path), by default "archive"
        dataset : dictionary, optional
            key-value pairs of dataset filenames and prefixes, by default None
        """
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
        )
        self.bucket = bucket
        self.pfx = pfx
        self.dataset = dataset
        self.fpaths = []
        self.source = "s3"

    def make_s3_keys(
        self,
        fnames=[
            "2022-02-14-1644848448.zip",
            "2021-11-04-1636048291.zip",
            "2021-10-28-1635457222.zip",
        ],
    ):
        """Generates a `dataset` dictionary attribute containing the filename-uriprefix key-value pairs.

        Parameters
        ----------
        fnames : list, optional
            dataset archive file names typically consisting of a hyphenated date and timestamp string when the data was generated (automatically the case for saved spacekit.analyzer.compute.Computer objects), by default [ "2021-10-28-1635457222.zip""2021-11-04-1636048291.zip", "2021-10-28-1635457222.zip" ]

        Returns
        -------
        dict
            key-value pairs of dataset archive filenames and their parent folder prefix name
        """
        self.dataset = {}
        for fname in fnames:
            key = fname.split(".")[0]
            fname = key + ".zip"
            self.dataset[key] = {"fname": fname, "pfx": self.pfx}
        return self.dataset

    def scrape(self):
        """Downloads files from s3 using the configured boto3 client. Calls the `extract_archive` method for automatic extraction of file contents if object's `extract` attribute is set to True.

        Returns
        -------
        list
            paths to downloaded and extracted files
        """
        err = None
        for _, d in self.dataset.items():
            fname = d["fname"]
            obj = f"{self.pfx}/{fname}"
            print(f"s3://{self.bucket}/{obj}")
            fpath = f"{self.cache_dir}/{self.cache_subdir}/{fname}"
            print(fpath)
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
                self.fpaths = super().extract_archives()
        return self.fpaths

    @staticmethod
    def s3_upload(keys, bucket_name, prefix):
        err = None
        for key in keys:
            obj = f"{prefix}/{key}"  # training/date-timestamp/filename
            try:
                with open(f"{key}", "rb") as f:
                    client.upload_fileobj(f, bucket_name, obj)
                    print(f"Uploaded: {obj}")
            except Exception as e:
                err = e
                continue
        if err is not None:
            print(err)

    @staticmethod
    def s3_download(keys, bucket_name, prefix):
        err = None
        for key in keys:
            obj = f"{prefix}/{key}"  # latest/master.csv
            print("s3 key: ", obj)
            try:
                with open(f"{key}", "wb") as f:
                    client.download_fileobj(bucket_name, obj, f)
            except Exception as e:
                err = e
                continue
        if err is not None:
            print(err)


class DynamoDBScraper(Scraper):
    def __init__(
        self,
        table_name,
        attr=None,
        fname="batch.csv",
        formatter=format_hst_cal_row_item,
        cache_dir="~",
        cache_subdir="data",
        format="zip",
        extract=True,
        clean=True,
    ):
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
            clean=clean,
        )
        self.table_name = table_name
        self.attr = attr
        self.ddb_data
        self.fname = fname
        self.formatter = formatter

    def get_keys(self, items):
        keys = set([])
        for item in items:
            keys = keys.union(set(item.keys()))
        return keys

    def make_fxp(self):
        """
        Generates filter expression based on attributes dict to retrieve a subset of the database using conditional operators and keyword pairs. Returns dict containing filter expression which can be passed into the dynamodb table.scan() method.
        Args:
        `name` : one of db column names ('timestamp', 'mem_bin', etc.)
        `method`: begins_with, between, eq, gt, gte, lt, lte
        `value`: str, int, float or low/high list of values if using 'between' method
        Ex: to retrieve a subset of data with 'timestamp' col greater than 1620740441:
        setting attr={'name':'timestamp', 'method': 'gt', 'value': 1620740441}
        returns dict: {'FilterExpression': Attr('timestamp').gt(0)}
        """
        # table.scan(FilterExpression=Attr('mem_bin').gt(2))
        n = self.attr["name"]
        m = self.attr["method"]

        if self.attr["type"] == "int":
            v = [int(a.strip()) for a in self.attr["value"].split(",")]
        elif self.attr["type"] == "float":
            v = [float(a.strip()) for a in self.attr["value"].split(",")]
        else:
            v = [str(a.strip()) for a in self.attr["value"].split(",")]

        print(f"DDB Subset: {n} - {m} - {v}")

        if m == "eq":
            fxp = Attr(n).eq(v[0])
        elif m == "gt":
            fxp = Attr(n).gt(v[0])
        elif m == "gte":
            fxp = Attr(n).gte(v[0])
        elif m == "lt":
            fxp = Attr(n).lt(v[0])
        elif m == "lte":
            fxp = Attr(n).lte(v[0])
        elif m == "begins_with":
            fxp = Attr(n).begins_with(v[0])
        elif m == "between":
            fxp = Attr(n).between(np.min(v), np.max(v))

        return {"FilterExpression": fxp}

    def ddb_download(self, attr=None):
        """retrieves data from dynamodb
        Args:
        table_name: dynamodb table name
        p_key: (default is 'ipst') primary key in dynamodb table
        attr: (optional) retrieve a subset using an attribute dictionary
        If attr is none, returns all items in database.
        """
        table = dynamodb.Table(self.table_name)
        key_set = ["ipst"]  # primary key
        if attr:
            scan_kwargs = self.make_fxp(attr)
            raw_data = table.scan(**scan_kwargs)
        else:
            raw_data = table.scan()
        if raw_data is None:
            return None
        items = raw_data["Items"]
        fieldnames = set([]).union(self.get_keys(items))

        while raw_data.get("LastEvaluatedKey"):
            print("Downloading ", end="")
            if attr:
                raw_data = table.scan(
                    ExclusiveStartKey=raw_data["LastEvaluatedKey"], **scan_kwargs
                )
            else:
                raw_data = table.scan(ExclusiveStartKey=raw_data["LastEvaluatedKey"])
            items.extend(raw_data["Items"])
            fieldnames - fieldnames.union(self.get_keys(items))

        print("\nTotal downloaded records: {}".format(len(items)))
        for f in fieldnames:
            if f not in key_set:
                key_set.append(f)
        self.ddb_data = {"items": items, "keys": key_set}
        return self.ddb_data

    def write_to_csv(self):
        with open(self.fname, "w") as csvfile:
            writer = csv.DictWriter(
                csvfile, delimiter=",", fieldnames=self.ddb_data["keys"], quotechar='"'
            )
            writer.writeheader()
            writer.writerows(self.ddb_data["items"])
        print(f"DDB data saved to: {self.fname}")

    def format_row_item(self, row):
        row = self.formatter(row)
        return json.loads(
            json.dumps(row, allow_nan=True), parse_int=Decimal, parse_float=Decimal
        )

    def write_to_dynamo(self, rows):
        try:
            table = dynamodb.Table(self.table_name)
        except Exception as e:
            print(
                "Error loading DynamoDB table. Check if table was created correctly and environment variable."
            )
            print(e)
        try:
            with table.batch_writer() as batch:
                for i in range(len(rows)):
                    batch.put_item(Item=rows[i])
        except Exception as e:
            print("Error executing batch_writer")
            print(e)

    def batch_ddb_writer(self, key):
        input_file = csv.DictReader(open(key))

        batch_size = 100
        batch = []

        for row in input_file:
            item = self.format_row_item(row)

            if len(batch) >= batch_size:
                self.write_to_dynamo(batch)
                batch.clear()

            batch.append(item)
        if batch:
            self.write_to_dynamo(batch)
        return {"statusCode": 200, "body": json.dumps("Uploaded to DynamoDB Table")}


class FitsScraper:
    def __init__(self, data, input_path):
        self.df = data.copy()
        self.input_path = input_path
        self.fits_keys = ["rms_ra", "rms_dec", "nmatches", "wcstype"]
        self.drz_paths = self.find_drz_paths()
        # self.data = self.scrape_fits()

    def find_drz_paths(self):
        self.drz_paths = {}
        for idx, row in self.df.iterrows():
            self.drz_paths[idx] = ""
            dname = row["dataset"]
            drz = row["imgname"]
            path = os.path.join(self.input_path, dname, drz)
            self.drz_paths[idx] = path
        return self.drz_paths

    def scrape_fits(self):
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


class MastScraper:
    """Class for scraping metadata from MAST (Mikulsky Archive for Space Telescopes) via ``astroquery``. Current functionality for this class is limited to extracting the `target_classification` values of HAP targets from the archive. An example of a target classification is "GALAXY" - an alphanumeric categorization of an image product/.fits file. Note - the files themselves are not downloaded, just this specific metadata listed in the online archive database. For downloading MAST science files, use the ``spacekit.extractor.radio`` module. The search parameter values needed for locating a HAP product on MAST can be extracted from the fits science extension headers using the ``astropy`` library. See the ``spacekit.preprocessor.scrub`` api for an example (or the astropy documentation)."""

    def __init__(self, df, trg_col="targname", ra_col="ra_targ", dec_col="dec_targ"):
        """Instantiates a spacekit.extractor.scrape.MastScraper object.

        Parameters
        ----------
        df : dataframe
            dataset containing the requisite search parameter values (kwargs for this class)
        trg_col : str, optional
            name of the column containing the image target names, by default "targname"
        ra_col : str, optional
            name of the column containing the target's right ascension values, by default "ra_targ"
        dec_col : str, optional
            name of the column containing the target's right ascension values, by default "dec_targ"
        """
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
        """Main calling function to scrape MAST

        Returns
        -------
        dataframe
            updated dataset with target classification categorical data added for each observation.
        """
        self.target_categories = self.scrape_target_categories()
        self.other_cat = self.scrape_other_targets()
        self.df = self.combine_categories()
        return self.df

    def scrape_target_categories(self):
        """Scrapes the "target_classification" for each observation (dataframe rows) from MAST using ``astroquery`` and the target name. For observations where the target classification is not found (or is blank), the ``scrape_other_targets`` method will be called using a broader set of search parameters (``ra_targ`` and ``dec_targ``).

        Returns
        -------
        dictionary
            target name and category key-value pairs
        """
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

    def backup_search(self, targ):
        self.targ_any[targ] = self.df.loc[self.df[self.trg_col] == targ][
            [self.ra_col, self.dec_col]
        ]

    def scrape_other_targets(self):
        """Scrapes MAST for remaining target classifications that could not be identified using target name. This method instead uses a broader set of query parameters: the ``ra_targ`` and ``dec_targ`` coordinates along with the dataset's proposal ID. If multiple datasets are found to match, the first of these containing a target_classification value will be used.

        Returns
        -------
        dict
            secondary set of remaining key-value pairs (target names and scraped categories)
        """
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
                    proposal_id=propid, s_ra=[ra, ra + 0.1], s_dec=[dec, dec + 0.1]
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
        """Combines the two dictionaries (``target_categories`` and ``other_cat``) and inserts back into the original dataframe as a new column named ``category``.

        Returns
        -------
        dataframe
            copy of original dataset with new "category" column data appended
        """
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

    def search_by_obs_id(self, index):
        # obs_id = 'hst_10403_29_acs_sbc_total_j96029'
        obs_id = "_".join(index.split("_")[:6])
        prop_id = index.split("_")[1]
        obs = Observations.query_criteria(proposal_id=prop_id, obs_id=obs_id)
        s_ra = obs[np.where(obs["obs_id"] == obs_id)]["s_ra"]
        s_dec = obs[np.where(obs["obs_id"] == obs_id)]["s_dec"]
        if len(s_ra) > 0:
            ra = s_ra[0]
        elif len(obs) > 0:
            ra = obs[0]["s_ra"][0]
        else:
            ra = 0
        if len(s_dec) > 0:
            dec = s_dec[0]
        elif len(obs) > 0:
            dec = obs[0]["s_dec"][0]
        else:
            dec = 0
        if ra != 0:
            obs = Observations.query_criteria(
                proposal_id=prop_id, s_ra=[ra, ra + 0.1], s_dec=[dec, dec + 0.1]
            )
            targname = obs[np.where(obs["target_name"])]["target_name"]
            if len(targname) > 0:
                targ = targname[0]
            else:
                targ = "ANY"
            category = obs[np.where(obs["target_classification"])][
                "target_classification"
            ]
            if len(category) > 0:
                cat = category[0]
            else:
                cat = "None"
        return ra, dec, targ, cat


class JsonScraper:
    """Searches local files using glob pattern(s) to scrape JSON file data. Optionally can store data in h5 file (default) and/or CSV file; The JSON harvester method returns a Pandas dataframe. This class can also be used to load an h5 file. CREDIT: Majority of the code here was repurposed into a class object from ``Drizzlepac.hap_utils.json_harvester`` - multiple customizations were needed for specific machine learning preprocessing that would be outside the scope of Drizzlepac's primary intended use-cases, hence why the code is now here in a stripped down version instead of submitted as a PR to the original repo. That, and the need to avoid including Drizzlepac as a dependency for spacekit, since spacekit is meant to be used for testing Drizzlepac's SVM processing...

    Parameters
    ----------
    search_path : str, optional
        The full path of the directory that will be searched for json files to process. If not explicitly
        specified, the current working directory will be used.
    search_patterns : list, optional
        list of glob patterns to use for search
    file_basename : str, optional
        Name of the output file basename (filename without the extension) for the Hierarchical Data Format version 5 (HDF5) .h5 file that the Pandas DataFrame will be written to. If not explicitly specified, the default filename basename that will be used is "svm_qa_dataframe". The default location that the
        output file will be written to is the current working directory
    crpt: bool, optional
        Uses extended dataframe index name to differentiate from normal svm data, by default 0 (False)
    save_csv: bool, optional
        store h5 data into a CSV file, by default False
    h5_file: str, optional
        load from a saved hdf5 file on local disk, by default None
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
        h5_file=None,
        output_path=None,
    ):
        self.search_path = search_path
        self.search_patterns = search_patterns
        self.file_basename = file_basename
        self.crpt = crpt
        self.save_csv = save_csv
        self.store_h5 = store_h5
        self.h5_file = h5_file
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
        """Initializes a logging object which logs process info to sys.stdout

        Returns
        -------
        logutil.log object
            logs process info to sys.stdout
        """
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
        Borrowed from Drizzlepac, which borrowed it from: https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/

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
        dictionary
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
        dictionary
            out_dict structured similarly to self.out_dict with separate 'header' and 'data' keys. The information stored in the 'data' section will be in the same format that it was in before it was serialized and stored as a json file.
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
        """Uses glob to create a list of json files to harvest. This function looks for all the json files containing qa test results generated by `runastrodriz` and `runsinglehap`.  The search starts in the directory
        specified in the `search_path` parameter, but will look in immediate
        sub-directories as well if no json files are located in the directory
        specified by `search_path`.

        Returns
        -------
        ordered dictionary
            out_json_dict containing lists of all identified json files, grouped by and keyed by Pandas DataFrame index value.
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
        """Store pandas Dataframe to an HDF5 file on local disk.

        Returns
        -------
        string
            path to stored h5 file
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
        """Loads dataframe from an H5 on local disk

        Returns
        -------
        dataframe
            data loaded from an H5 file and stored in a dataframe object attribute.

        Raises
        ------
        Exception
            Requested file not found
        """
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
        """Main calling function to harvest json files matching the search pattern and store in dictionaries which are then combined into a single dataframe.

        Returns
        -------
        dataframe
            dataset created by scraping data from json files on local disk.
        """
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
        """optionally write dataframe out to .csv file."""
        output_csv_filename = self.h5_filename.replace(".h5", ".csv")
        if os.path.exists(output_csv_filename):
            os.remove(output_csv_filename)
        self.data.to_csv(output_csv_filename)
        self.log.info("Wrote dataframe to csv file {}".format(output_csv_filename))

    def make_dataframe_line(self, json_filename_list):
        """Extracts information from the json files specified by the input list *json_filename_list*. Main difference between this and the original Drizzlepac source code is a much more limited collection of data: descriptions and units are not collected; only a handful of specific keyword values are scraped from general information and header extensions.

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
            flattened_data = self.flatten_dict(json_data["data"])
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


# TODO
class ImageScraper(Scraper):
    def __init__(self):
        super().__init__(self)


# def extract_archives(zipfiles, extract_to="data", delete_archive=False):
#     fpaths = []
#     os.makedirs(extract_to, exist_ok=True)
#     for z in zipfiles:
#         fname = os.path.basename(z).split(".")[0]
#         fpath = os.path.join(extract_to, fname)
#         with ZipFile(z, "r") as zip_ref:
#             zip_ref.extractall(extract_to)
#         # check just in case
#         if os.path.exists(fpath):
#             fpaths.append(fpath)
#             if delete_archive is True:
#                 os.remove(z)
#     return fpaths


# def unzip_images(zip_file):
#     basedir = os.path.dirname(zip_file)
#     key = os.path.basename(zip_file).split(".")[0]
#     image_folder = os.path.join(basedir, key + "/")
#     os.makedirs(image_folder, exist_ok=True)
#     with ZipFile(zip_file, "r") as zip_ref:
#         zip_ref.extractall(basedir)
#     print(len(os.listdir(image_folder)))
#     return image_folder


# def scrape_web(key, uri):
#     fname = key["fname"]
#     origin = f"{uri}/{fname}"
#     hash = key["hash"]
#     fpath = get_file(
#         origin=origin,
#         file_hash=hash,
#         hash_algorithm="sha256",  # auto
#         cache_dir="~",
#         cache_subdir="data",
#         extract=True,
#         archive_format="zip",
#     )
#     if os.path.exists(fpath):
#         os.remove(f"data/{fname}")
#     return fpath


# def get_training_data(dataset=None, uri=None):
#     if uri is None:
#         print("Please enter a uri.")
#         return None
#     keys = list(dataset.keys())
#     fpaths = []
#     for key in keys:
#         fpath = scrape_web(key, uri)
#         fpaths.append(fpath)
#     return fpaths
