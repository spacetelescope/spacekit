import os
import boto3
import numpy as np
import pandas as pd
import collections
import glob
import sys
import json
import csv
from zipfile import ZipFile
from astropy.io import fits, ascii
from astropy.table import Table
from botocore.config import Config
from decimal import Decimal
from boto3.dynamodb.conditions import Attr

try:
    from keras.utils import get_file
except ImportError:
    try:
        from keras.utils.data_utils import get_file
    except ImportError:
        get_file = None

from spacekit.logger.log import Logger


retry_config = Config(retries={"max_attempts": 3})
client = boto3.client("s3", config=retry_config)
dynamodb = boto3.resource("dynamodb", config=retry_config, region_name="us-east-1")
# below are maintained for backwards compatibility with static methods
s3 = boto3.resource("s3", config=retry_config)
client = boto3.client("s3", config=retry_config)


SPACEKIT_DATA = os.environ.get("SPACEKIT_DATA", "~/spacekit_data")


def home_data_base(data_home=None):
    """Borrowed from ``sklearn.datasets._base.get_data_home`` function: Return the path of the spacekit
    data dir, and create one if not existing. Folder path can be set explicitly using ``data_home`` kwarg,
    otherwise it looks for the 'SPACEKIT_DATA' environment variable, or defaults to 'spacekit_data' in the
    user home directory (the '~' symbol is expanded to the user's home folder).

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
    """Parent Class for various data scraping subclasses. Instantiating the appropriate subclass is preferred.

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

    def __init__(
        self,
        cache_dir="~",
        cache_subdir="data",
        format="zip",
        extract=True,
        clean=True,
        name="Scraper",
        **log_kws,
    ):
        self.cache_dir = self.check_cache(cache_dir)  # root path for downloads (home)
        self.cache_subdir = cache_subdir  # subfolder
        self.format = format
        self.extract = extract  # extract if zip/tar archive
        self.outpath = os.path.join(self.cache_dir, self.cache_subdir)
        self.clean = clean  # delete archive if extract successful
        self.source = None
        self.fpaths = []
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()

    def check_cache(self, cache):
        if cache == "~":
            return os.path.expanduser(cache)
        elif cache == ".":
            return os.path.abspath(".")
        elif cache is None:
            return home_data_base()
        else:
            return os.path.abspath(cache)

    def extract_archives(self, name_keys=[]):
        """Extract the contents of the compressed archive file(s).

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
        for i, z in enumerate(self.fpaths):
            with ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(self.outpath)
            # check successful extraction before deleting archive
            if isinstance(name_keys, list) and len(name_keys) == len(self.fpaths):
                extracted = os.path.join(self.outpath, name_keys[i])
            else:
                extracted = os.path.join(self.outpath, str(z).split(".")[0])
            if os.path.exists(extracted):
                extracted_fpaths.append(extracted)
                if self.clean is True:
                    os.remove(z)
                    if os.path.exists(z+'_archive'):
                        os.remove(z+'_archive')
        self.fpaths = extracted_fpaths

    def compress_files(self, target_folder, fname=None, compression="zip"):
        if fname is None:
            fname = os.path.basename(target_folder) + f".{compression}"
        else:
            fname = os.path.basename(fname).split(".")[0] + f".{compression}"
        archive_path = os.path.join(self.cache_dir, fname)
        file_paths = []
        for root, _, files in os.walk(target_folder):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
        print("Zipping files:")
        with ZipFile(archive_path, "w") as zip_ref:
            for file in file_paths:
                zip_ref.write(file)
                self.log.info(file)
        return


class FileScraper(Scraper):
    """Scraper subclass used to search and extract files on local disk that match regex/glob pattern(s).

    Parameters
    ----------
    search_path : str, optional
        top-level path to search through, by default ""
    search_patterns : list, optional
        glob pattern strings, by default ``["*.zip"]``
    cache_dir : str, optional
        parent folder to save data, by default "~"
    cache_subdir : str, optional
        save data in a subfolder one directory below `cache_dir`, by default "data"
    format : str, optional
        archive format type, by default "zip"
    extract : bool, optional
        extract the contents of the compressed archive file, by default True
    clean : bool, optional
        remove compressed file after extraction, by default False
    name : str, optional
        logging name, by default "FileScraper"
    """

    def __init__(
        self,
        search_path="",
        search_patterns=["*.zip"],
        cache_dir="~",
        cache_subdir="data",
        format="zip",
        extract=True,
        clean=False,
        name="FileScraper",
        **log_kws,
    ):
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
            clean=clean,
            name=name,
            **log_kws,
        )
        self.search_path = search_path
        self.search_patterns = search_patterns
        self.fpaths = []
        self.source = "file"

    def scrape(self):
        """Search local disk for files matching glob regex pattern(s)

        Returns
        -------
        list
            paths to dataset files found in glob pattern search
        """
        for p in self.search_patterns:
            results = glob.glob(os.path.join(self.search_path), p)
            if len(results) > 0:
                for r in results:
                    self.fpaths.append(r)
        if self.extract is True:
            super().extract_archives
        return self.fpaths


class WebScraper(Scraper):
    """Scraper subclass for extracting publicly available data off the web. 
    Uses dictionary of uri, filename and hash key-value pairs to download data securely from a website such as Github.

    Parameters
    ----------
    uri : string
        root uri (web address)
    dataset : dictionary
        key-pair values of each dataset's filenames and hash keys
    hash_algorithm : str, optional
        type of hash key algorithm used, by default "sha256"
    cache_dir : str, optional
        parent folder to save data, by default "~"
    cache_subdir : str, optional
        save data in a subfolder one directory below `cache_dir`, by default "data"
    format : str, optional
        archive format type, by default "zip"
    extract : bool, optional
        extract the contents of the compressed archive file, by default True
    clean : bool, optional
        remove compressed file after extraction
    """
    def __init__(
        self,
        uri,
        dataset,
        hash_algorithm="md5",
        cache_dir="~",
        cache_subdir="data",
        format="zip",
        extract=True,
        clean=True,
        **log_kws,
    ):
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
            clean=clean,
            name="WebScraper",
            **log_kws,
        )
        self.uri = uri
        self.dataset = dataset
        self.hash_algorithm = hash_algorithm
        self.source = "web"
        self.fpaths = []

    def scrape(self):
        """Using the key-pair values in `dataset` dictionary attribute, download the files from a website
        (such as zenodo) and check the hash keys match before extracting. Extraction and hash-key checking 
        is handled externally by the `keras.utils.data_utils.get_file` method. If extraction is successful, 
        the archive file will be deleted. See spacekit.datasets.meta for dictionary formatting examples.

        Returns
        -------
        list
            paths to downloaded and extracted files
        """
        name_keys = []
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
                extract=False,
                archive_format=self.format,
            )
            self.fpaths.append(fpath)
            name_keys.append(os.path.join(self.outpath, data['key']))
        if self.extract is True:
            self.extract_archives(name_keys=name_keys)
        return self.fpaths


class S3Scraper(Scraper):
    """Scraper subclass for extracting data from an AWS s3 bucket (requires AWS credentials with
    permissions to access the bucket.)

    Parameters
    ----------
    bucket : string
        s3 bucket name
    pfx : str, optional
        aws bucket prefix (subfolder uri path), by default "archive"
    dataset : dictionary, optional
        key-value pairs of dataset filenames and prefixes, by default None
    cache_dir : str, optional
        parent folder to save data, by default "~"
    cache_subdir : str, optional
        save data in a subfolder one directory below `cache_dir`, by default "data"
    format : str, optional
        archive format type, by default "zip"
    extract : bool, optional
        extract the contents of the compressed archive file, by default True
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
        **log_kws,
    ):
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
            name="S3Scraper",
            **log_kws,
        )
        self.bucket = bucket
        self.pfx = pfx
        self.dataset = dataset
        self.fpaths = []
        self.source = "s3"
        self.s3 = boto3.resource("s3", config=retry_config)
        self.client = boto3.client("s3", config=retry_config)
        self.aws_kwargs = self.authorize_aws()

    def authorize_aws(self):
        self.aws_kwargs = dict(
            region_name=os.environ.get("AWS_REGION_NAME", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN", ""),
        )
        self.s3 = boto3.resource("s3", config=retry_config, **self.aws_kwargs)
        self.client = boto3.client("s3", config=retry_config, **self.aws_kwargs)

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
            dataset archive file names typically consisting of a hyphenated date and timestamp string when
            the data was generated (automatically the case for saved spacekit.analyzer.compute.Computer
            objects), by default [ "2021-10-28-1635457222.zip""2021-11-04-1636048291.zip",
            "2021-10-28-1635457222.zip" ]

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

    def scrape_s3_file(self, fpath, obj):
        with open(fpath, "wb") as f:
            self.client.download_fileobj(self.bucket, obj, f)
            self.fpaths.append(fpath)

    def scrape(self):
        """Downloads files from s3 using the configured boto3 client. Calls the `extract_archive` method
        for automatic extraction of file contents if object's `extract` attribute is set to True.

        Returns
        -------
        list
            paths to downloaded and extracted files
        """
        err = None
        for _, d in self.dataset.items():
            fname = d["fname"]
            obj = f"{self.pfx}/{fname}"
            self.log.info(f"s3://{self.bucket}/{obj}")
            fpath = f"{self.cache_dir}/{self.cache_subdir}/{fname}"
            self.log.info(fpath)
            try:
                self.scrape_s3_file(fpath, obj)
            except Exception as e:
                err = e
                continue
        if err is not None:
            self.log.error(err)
        elif self.extract is True:
            if self.format == "zip":
                super().extract_archives()
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

    def import_dataset(self):
        """import job metadata file from s3 bucket"""
        bucket = self.s3.Bucket(self.bucket)
        obj = bucket.Object(self.pfx)
        input_data = {}
        body = None
        self.log.debug(f"Streaming from s3://{self.bucket}/{self.pfx}")
        try:
            body = obj.get()["Body"].read().splitlines()
            for line in body:
                k, v = str(line).strip("b'").split("=")
                input_data[k] = v
        except Exception as e:
            self.log.error(e)
            sys.exit(3)
        self.log.debug("Input data scraped successfully.")
        return input_data


class DynamoDBScraper(Scraper):
    """Scraper subclass for extracting data from an AWS DynamoDB table (requires AWS credentials with
    permissions to access the table.)

    Parameters
    ----------
    table_name : str
        name of the DynamoDB table
    attr : dict, optional
        used for building a filter expression (see ``make_fxp``), by default None
    fname : str, optional
        path or string of filename to save data, by default "batch.csv"
    formatter : function, optional
        formatting function to use, by default format_hst_cal_row_item
    cache_dir : str, optional
        parent folder to save data, by default "~"
    cache_subdir : str, optional
        save data in a subfolder one directory below `cache_dir`, by default "data"
    format : str, optional
        archive format type, by default "zip"
    extract : bool, optional
        extract the contents of the compressed archive file, by default True
    clean : bool, optional
        remove compressed file after extraction
    """

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
        **log_kws,
    ):
        super().__init__(
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
            format=format,
            extract=extract,
            clean=clean,
            name="DynamoDBScraper",
            **log_kws,
        )
        self.table_name = table_name
        self.attr = attr
        self.fname = fname
        self.formatter = formatter
        self.ddb_data = None
        self.fpath = None

    def get_keys(self, items):
        keys = set([])
        for item in items:
            keys = keys.union(set(item.keys()))
        return keys

    def make_fxp(self):
        """
        Generates filter expression based on attributes dict to retrieve a subset of the database using
        conditional operators and keyword pairs. Returns dict containing filter expression which can be
        passed into the dynamodb table.scan() method.
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
        self.fpath = os.path.join(self.cache_dir, self.cache_subdir, self.fname)
        with open(self.fpath, "w") as csvfile:
            writer = csv.DictWriter(
                csvfile, delimiter=",", fieldnames=self.ddb_data["keys"], quotechar='"'
            )
            writer.writeheader()
            writer.writerows(self.ddb_data["items"])
        print(f"DDB data saved to: {self.fpath}")

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


class FitsScraper(FileScraper):
    """FileScraper subclass used to search and extract FITS files on local disk

    Parameters
    ----------
    data : pd.DataFrame
        dataframe of visits, datasets, exposures, etc.
    input_path : str
        directory path containing fits files
    genkeys : list, optional
        general header keys to scrape, by default []
    scikeys : list, optional
        science header keys to scrape, by default []
    name : str, optional
        logging name, by default "FitsScraper"
    """

    def __init__(
        self, data, input_path, genkeys=[], scikeys=[], name="FitsScraper", **log_kws
    ):
        super().__init__(name=name, **log_kws)
        self.df = data.copy()
        self.input_path = input_path
        self.genkeys = genkeys
        self.scikeys = scikeys
        self.fpaths = None

    def get_input_exposures(self, pfx="", sfx="_uncal.fits"):
        """create list of local paths to L1B exposure files for a given program

        Parameters
        ----------
        input_path : path or str
            directory path containing input exposure files
        pfx : str, optional
            filename prefix to search for, by default ""
        sfx : str, optional
            file suffix to search for, by default "uncal.fits"

        Returns
        -------
        list
            Paths to (typically uncalibrated) input exposure .fits files in this program/visit
        """
        fpaths = glob.glob(f"{os.path.expanduser(self.input_path)}/{pfx}*{sfx}")
        if not fpaths:
            fpaths = glob.glob(f"{os.path.expanduser(self.input_path)}/*/{pfx}*{sfx}")
        return fpaths

    def scrape_fits_headers(self, fpaths=None, **kwargs):
        """scrape values from ext=0 general info header (genkeys) and ext=1 science header (scikeys)

        Parameters
        ----------
        fpaths : list, optional
            list of fits file paths

        Returns
        -------
        dict
            exposure header info scraped from fits files
        """
        self.log.info("Extracting fits data...")
        if fpaths is None:
            fpaths = self.get_input_exposures(**kwargs)
        exp_headers = {}
        for fpath in fpaths:
            try:
                fname = str(os.path.basename(fpath))
                sfx = fname.split("_")[-1]  # _uncal.fits
                name = fname.replace(f"_{sfx}", "")
                exp_headers[name] = dict()
                if self.genkeys:
                    genhdr = fits.getheader(fpath, ext=0)
                    for g in self.genkeys:
                        exp_headers[name][g] = genhdr[g] if g in genhdr else "NaN"
                if self.scikeys:
                    scihdr = fits.getheader(fpath, ext=1)
                    for s in self.scikeys:
                        exp_headers[name][s] = scihdr[s] if s in scihdr else "NaN"
            except Exception:
                del exp_headers[name]
                continue
        return exp_headers

    def scrape_dataframe(self, dnames=None, dname_col="dname"):
        if dnames is None:
            dnames = list(self.df[dname_col])
        exp_headers = {}
        for name in dnames:
            try:
                data = self.df.loc[name]
                exp_headers[name] = dict()
                if self.genkeys:
                    for g in self.genkeys:
                        exp_headers[name][g] = data[g] if g in self.df.columns else "NaN"
                if self.scikeys:
                    for s in self.scikeys:
                        exp_headers[name][s] = data[s] if s in self.df.columns else "NaN"
            except Exception:
                del exp_headers[name]
                continue
        return exp_headers

    def find_drz_paths(self, dname_col="dataset", drzimg_col="imgname"):
        """Looks for SVM input files based on information contained in the ``self.df`` attribute.
        Input paths for files are constructed using the ``dname_col`` and ``drzimg_col`` along with
        the ``self.input_path`` attribute.

        Parameters
        ----------
        dname_col : str, optional
            name of the column containing dataset names, by default "dataset"
        drzimg_col : str, optional
            name of the column containing image filenames, by default "imgname"

        Returns
        -------
        list
            absolute paths to all SVM fits files located.
        """
        if not self.fpaths:
            self.fpaths = dict()
        try:
            for idx, row in self.df.iterrows():
                self.fpaths[idx] = ""
                dname = row[dname_col]
                drz = row[drzimg_col]
                path = os.path.join(self.input_path, dname, drz)
                self.fpaths[idx] = path
        except Exception:
            self.log.error("Unable to locate drizzle files from dataframe.")
        return self.fpaths

    def scrape_drizzle_fits(self):
        """Scrape sciheaders of SVM input exposures located using ``self.find_drz_paths``.
        Specific sciheader keys extracted are set in the ``self.scikeys`` attribute.

        Returns
        -------
        pd.DataFrame
            dataframe with extracted fits header information for each dataset
        """
        if not self.fpaths:
            self.fpaths = self.find_drz_paths()
        self.log.info("*** Extracting fits data ***")
        fits_dct = {}
        for key, path in self.fpaths.items():
            fits_dct[key] = {}
            scihdr = fits.getheader(path, ext=1)
            for k in self.scikeys:
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


class JwstFitsScraper(FitsScraper):
    """FitsScraper subclass used to search and extract JWST Fits files on local disk

    Parameters
    ----------
    input_path : str or path
        directory path containing input exposure files
    data : pd.DataFrame, optional
        dataframe of visits, datasets, exposures, etc., by default None
    pfx : str, optional
        filename prefix to search for, by default ""
    sfx : str, optional
        file suffix to search for, by default "uncal.fits"
    """

    def __init__(self, input_path, data=None, pfx="", sfx="_uncal.fits", **log_kws):
        self.genkeys = self.general_header_keys()
        self.scikeys = self.science_header_keys()
        if data is None:
            data = pd.DataFrame()
        super().__init__(
            data,
            input_path,
            genkeys=self.genkeys,
            scikeys=self.scikeys,
            name="JwstFitsScraper",
            **log_kws,
        )
        self.pfx = pfx
        self.sfx = sfx
        self.fpaths = super().get_input_exposures(pfx=self.pfx, sfx=self.sfx)
        self.exp_headers = None

    def general_header_keys(self):
        """General header key names to scrape from input exposure fits files.
        Returns
        -------
        list
            list of key names to scrape from fits header extension 0.
        """
        return [
            "PROGRAM",  # Program number
            "OBSERVTN",  # Observation number
            "NEXPOSUR",  # number of exposures
            "BKGDTARG",  # Background target
            "IS_IMPRT",  # NIRSpec imprint exposure
            "VISITYPE",  # Visit type
            "TSOVISIT",  # Time Series Observation visit indicator
            "TARGNAME",  # Standard astronomical catalog name for target
            "TARG_RA",  # Target RA at mid time of exposure
            "TARG_DEC",  # Target Dec at mid time of exposure
            "INSTRUME",  # Instrument used to acquire the data
            "DETECTOR",  # Name of detector used to acquire the data
            "FILTER",  # Name of the filter element used
            "PUPIL",  # Name of the pupil element used
            "GRATING",  # Name of the grating element used (SPEC)
            "FXD_SLIT", # Name of fixed slit aperture used
            "EXP_TYPE",  # Type of data in the exposure
            "CHANNEL",  # Instrument channel
            "BAND", # MRS wavelength band
            "SUBARRAY",  # Subarray used
            "NUMDTHPT",  # Total number of points in pattern
            "GS_RA",  # guide star right ascension
            "GS_DEC",  # guide star declination
            "GS_MAG",  # guide star magnitude in FGS detector
            "CROWDFLD",  # Are the FGSes in a crowded field?
        ]

    def science_header_keys(self):
        """Science header key names to scrape from input exposure fits files science headers.
        Returns
        -------
        list
            list of key names to scrape from fits header science extension headers.
        """
        return [
            "RA_REF",
            "DEC_REF",
            "CRVAL1",
            "CRVAL2",
        ]

    def scrape_fits(self):
        """invokes parent class method ``scrape_fits_headers`` using pre-set JWST attributes.

        Returns
        -------
        dict
            exposure header metadata scraped from fits files on local disk
        """
        self.exp_headers = super().scrape_fits_headers(fpaths=self.fpaths)
        return self.exp_headers


class SvmFitsScraper(FitsScraper):
    """FitsScraper subclass used to search and extract HST SVM Fits files on local disk

    Parameters
    ----------
    data : pd.DataFrame
        data containing visit or dataset names
    input_path : str or path
        input path containing fits files to scrape
    """
    def __init__(self, data, input_path, **log_kws):


        self.scikeys = ["rms_ra", "rms_dec", "nmatches", "wcstype"]
        super().__init__(
            data, input_path, scikeys=self.scikeys, name="SvmFitsScraper", **log_kws
        )
        self.fpaths = self.find_drz_paths(dname_col="dataset", drzimg_col="imgname")

    def scrape_fits(self):
        """Invokes parent class method ``scrape_drizzle_fits`` using pre-set attributes specific to HST SVM data.

        Returns
        -------
        pd.DataFrame
            dataframe with extracted fits header information for each dataset
        """
        return self.scrape_drizzle_fits()


class JsonScraper(FileScraper):
    """Searches local files using glob pattern(s) to scrape JSON file data. Optionally can store data in h5
    file (default) and/or CSV file; The JSON harvester method returns a Pandas dataframe. This class can
    also be used to load an h5 file. CREDIT: Majority of the code here was repurposed into a class object
    from ``Drizzlepac.hap_utils.json_harvester`` - multiple customizations were needed for specific machine
    learning preprocessing that would be outside the scope of Drizzlepac's primary intended use-cases,
    hence why the code is now here in a stripped down version instead of submitted as a PR to the original
    repo. That, and the need to avoid including Drizzlepac as a dependency for spacekit, since spacekit is
    meant to be used for testing Drizzlepac's SVM processing...

    Parameters
    ----------
    search_path : _type_, optional
        The full path of the directory that will be searched for json files to process, by default os.getcwd()
    search_patterns : list, optional
        list of glob patterns to use for search, by default ["*_total_*_svm_*.json"]
    file_basename : str, optional
        Name of the output file basename (filename without the extension) for the Hierarchical Data
        Format version 5 (HDF5) .h5 file that the DataFrame will be written to, by default "svm_data"
    crpt : int, optional
        Uses extended dataframe index name to differentiate from normal svm data, by default 0
    save_csv : bool, optional
        store h5 data into a CSV file, by default False
    store_h5 : bool, optional
        save data in hdf5 format, by default True
    h5_file : str or path, optional
        load from a saved hdf5 file on local disk, by default None
    output_path : str or path, optional
        where to save the data, by default None
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
        **log_kws,
    ):
        super().__init__(
            search_path=search_path,
            search_patterns=search_patterns,
            name="JsonScraper",
            **log_kws,
        )
        self.file_basename = file_basename
        self.crpt = crpt
        self.save_csv = save_csv
        self.store_h5 = store_h5
        self.h5_file = h5_file
        self.output_path = os.getcwd() if output_path is None else output_path
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
        self.json_dict = None
        self.data = None  # self.json_harvester()
        # self.h5_file = None  # self.h5store()

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
        """extracts header and data sections from specified json file and returns the header and data (in its
        original pre-json format) as a nested ordered dictionary

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
            out_dict structured similarly to self.out_dict with separate 'header' and 'data' keys. The information
            stored in the 'data' section will be in the same format that it was in before it was serialized and
            stored as a json file.
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
        """Uses glob to create a list of json files to harvest. This function looks for all the json files containing
        qa test results generated by `runastrodriz` and `runsinglehap`.  The search starts in the directory
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
        if self.store_h5 is False:
            return
        fname = self.file_basename.split(".")[0] + ".h5"
        self.h5_file = os.path.join(self.output_path, fname)

        if self.data is not None:
            if os.path.exists(self.h5_file):
                self.log.warning("Overwriting existing h5 file.")
                os.remove(self.h5_file)
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
            self.h5_file = os.path.join(self.output_path, self.file_basename + ".h5")
        elif not self.h5_file.endswith(".h5"):
            self.h5_file += ".h5"
        if not os.path.exists(self.h5_file):
            h5_path = os.path.join(self.output_path, self.h5_file)
            if os.path.exists(h5_path):
                self.h5_file = h5_path
        try:
            with pd.HDFStore(self.h5_file) as store:
                self.data = store["mydata"]
                self.log.info(f"Dataframe created: {self.data.shape}")
        except Exception as e:
            print(e)
            errmsg = "HDF5 file {} not found!".format(self.h5_file)
            self.log.error(errmsg)
            raise Exception(errmsg)
        return self.data

    def json_harvester(self):
        """Main calling function to harvest json files matching the search pattern and store in dictionaries which
        are then combined into a single dataframe.

        Returns
        -------
        dataframe
            dataset created by scraping data from json files on local disk.
        """
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

        self.write_to_csv()
        self.h5store()
        return self.data

    def write_to_csv(self):
        """optionally write dataframe out to .csv file."""
        if not self.save_csv:
            return
        output_csv_filename = self.h5_filename.replace(".h5", ".csv")
        if os.path.exists(output_csv_filename):
            self.log.warning("Overwriting existing CSV")
            os.remove(output_csv_filename)
        self.data.to_csv(output_csv_filename)
        self.log.info("Wrote dataframe to csv file {}".format(output_csv_filename))

    def make_dataframe_line(self, json_filename_list):
        """Extracts information from the json files specified by the input list *json_filename_list*. Main difference
        between this and the original Drizzlepac source code is a much more limited collection of data: descriptions
        and units are not collected; only a handful of specific keyword values are scraped from general information
        and header extensions.

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
        # self.log.setLevel(self.log_level)
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
            if not header_ingested:
                # filter out ALL header keywords not included in 'keyword_shortlist'
                for header_item in json_data["header"].keys():
                    if header_item in self.keyword_shortlist:
                        # if header_item in header_keywords_to_keep:
                        ingest_dict["data"]["header." + header_item] = json_data[
                            "header"
                        ][header_item]
                header_ingested = True
            # add information from "general information" section to ingest_dict just once
            if not gen_info_ingested:
                for gi_item in json_data["general information"].keys():
                    if gi_item in self.keyword_shortlist:
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
                    if isinstance(json_data_item, Table):
                        for coltitle in json_data_item.colnames:
                            ingest_value = json_data_item[coltitle].tolist()
                            id_key = title_suffix + ingest_key + "." + coltitle
                            ingest_dict["data"][id_key] = [ingest_value]
                    else:
                        ingest_value = json_data_item
                        id_key = title_suffix + ingest_key
                        if isinstance(ingest_value, list):
                            ingest_dict["data"][id_key] = [ingest_value]
                        else:
                            ingest_dict["data"][id_key] = ingest_value
        return ingest_dict


# TODO
class ImageScraper(Scraper):
    def __init__(self):
        super().__init__()
