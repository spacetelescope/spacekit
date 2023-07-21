import os
import shutil
import glob
import boto3
import numpy as np
import pandas as pd
from spacekit.logger.log import Logger

try:
    from astroquery.mast import Observations
except ImportError:
    Observations = None
try:
    from progressbar import ProgressBar
except ImportError:
    ProgressBar = None


def check_astroquery():
    return Observations is not None


def check_progressbar():
    return ProgressBar is not None


def check_imports():
    if not check_astroquery():
        return False
    elif not check_progressbar():
        return False
    else:
        return True


class Radio:
    """Class for querying and downloading .fits files from a MAST s3 bucket on AWS. Note this was originally created for K2 LLC data and is in the process of being revised for other data types/telescopes..."""

    def __init__(self, config="disable", name="Radio", **log_kws):
        """Instantiates a spacekit.extractor.Radio object.

        Parameters
        ----------
        config : str, optional
            enable or disable aws cloud access (disable uses MAST only), by default "disable"
        """
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        self.config = config
        self.region = "us-east-1"
        self.s3 = boto3.resource("s3", region_name=self.region)
        self.bucket = self.s3.Bucket("stpubdata")
        self.location = {"LocationConstraint": self.region}
        self.target_list = None
        self.proposal_id = None  # '13926'
        self.collection = None # "K2" "HST" "HLA" "JWST"
        self.filters = None  # 'F657N'
        self.obsid = None  # 'ICK90[5678]*'
        self.subgroup = None  # ['FLC', 'SPT'], ["LLC"], ["CAL"], ["I2D"]
        self.radius = "0s"
        self.exptime = 1800.0
        self.s3_uris = []
        self.errors = []
        self.science_files = []
        if not check_imports():
            self.log.error("astroquery and/or progressbar not installed.")
            raise ImportError(
                "You must install astroquery (`pip install astroquery`) "
                "and progressbar (`pip install progressbar`) for the "
                "radio module to work. \n\nInstall extra deps via "
                "`pip install spacekit[x]`"
            )


        self.configure_aws()
        

    def configure_aws(self):
        """Sets cloud (AWS) configuration On or Off."""
        # configure aws settings
        if self.config == "enable":
            self.log.info("Configuring for AWS cloud data retrieval...")
            Observations.enable_cloud_dataset(provider="AWS", profile="default")
        elif self.config == "disable":
            Observations.disable_cloud_dataset()

    def set_query_params(self, **kwargs):
        """keyword arguments can be any valid MAST search params, e.g.
        proposal_id, filters, obsid
        target, radius, s_ra, s_dec
        """
        query_params = dict()
        for k, v in kwargs.items():
            if v != None:
                query_params[k]=v
        return query_params

    def set_product_params(self, obs, obsid=None):
        """keyword arguments can be any valid MAST data product params, e.g.
        obs_collection, t_exptime, target_classification
        """
        if self.collection and self.exptime:
            want = (obs["obs_collection"] == self.collection) & (
                    obs["t_exptime"] == self.exptime
            )
        elif obsid:
            want = (obs['obsid'] == obsid)

        return want


    def prop_search(self, proposal_id, filters, obsid, subgroup):
        """Sets parameters for prop search as object attributes: proposal ID, filters, obsid and subgroup.

        Parameters
        ----------
        proposal_id : string
            match proposal id, e.g. '13926'
        filters : string
            match filters 'F657N'
        obsid : string
            match obsid or regex pattern 'ICK90[5678]*'
        subgroup : list
            data file types ['FLC', 'SPT']

        Returns
        -------
        self
            class object with attributes updated
        """
        self.proposal_id = proposal_id
        self.filters = filters
        self.obsid = obsid
        self.subgroup = subgroup

    def cone_search(self, radius, collection, exptime, subgroup):
        """Sets parameters for a cone search as object attributes: radius, collection, exptime, subgroup.

        Parameters
        ----------
        radius : string
            radius for the cone search e.g. 0s
        collection : string
            observatory collection name e.g. "K2"
        exptime : float
            exposure time e.g. 1800.0
        subgroup : list
            # data file type e.g. ["LLC"]

        Returns
        -------
        self
            class object with attributes updated
        """
        self.radius = radius
        self.collection = collection
        self.exptime = exptime
        self.subgroup = subgroup

    def get_object_uris(self):
        """Run observation query via cone search and return list of product uris.

        Returns
        -------
        self
            class object with attributes updated
        """
        if self.target_list is None:
            self.log.error("target_list (IDs) must be set first.")
            return
        # Do a cone search and find the K2 long cadence data for each target
        for target in self.target_list:
            obs = Observations.query_object(target, radius=self.radius)
            # want = (obs["obs_collection"] == self.collection) & (
            #     obs["t_exptime"] == self.exptime
            # )
            want = self.set_product_params(obs)
            data_prod = Observations.get_product_list(obs[want])
            filt_prod = Observations.filter_products(
                data_prod, productSubGroupDescription=self.subgroup
            )
            try:
                uri = Observations.get_cloud_uris(filt_prod)
                self.s3_uris.append(uri)
                if uri in self.errors:
                    self.errors.remove(uri)
            except Exception:  # ResolverError:
                self.log.error(f"Could not resolve {target} to a sky position.")
                self.errors.append(target)
                continue
        return self

    def s3_download(self):
        """Download datasets in list of uris from AWS s3 bucket (public access via STScI)

        Returns
        -------
        self
            class object with attributes updated
        """
        self.log.info(f"Downloading {len(self.s3_uris)} from AWS")
        count = 0
        for uri in self.s3_uris:
            U = uri[0]
            key = U.replace("s3://stpubdata/", "")
            root = U.split("/")[-1]
            try:
                self.bucket.download_file(
                    key, root, ExtraArgs={"RequestPayer": "requester"}
                )
                count += 1
                self.science_files.append(root)
            except FileExistsError:
                continue
        self.log.info(f"Download Complete: {count} files")
        return self

    def mast_download(self):
        """Download datasets from MAST"""
        if self.obsid is None:
            query_params = self.set_query_params(
                proposal_id=self.proposal_id, filters=self.filters
            )
        else:
            query_params = self.set_query_params(
                proposal_id=self.proposal_id, filters=self.filters, obsid=self.obsid
            )
        obs = Observations.query_criteria(**query_params)
        Observations.download_products(
            obs["obsid"],
            mrp_only=False,
            download_dir="./science",
            productSubGroupDescription=self.subgroup,
        )

        files = glob.glob(
            os.path.join(os.curdir, "science", "mastDownload", "HST", "*", "*fits")
        )

        for im in files:
            root = "./" + im.split("/")[-1]
            os.rename(im, root)
            self.science_files.append(root)
        shutil.rmtree("science/")

    def search_targets_by_obs_id(self, obs_id, prop_id):
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

    def search_by_targname(self, targets, datacol="target_classification"):
        """Scrapes the "target_classification" for each observation (dataframe rows) from MAST using ``astroquery`` and the target name. For observations where the target classification is not found (or is blank), the ``scrape_other_targets`` method will be called using a broader set of search parameters (``ra_targ`` and ``dec_targ``).

        Returns
        -------
        dictionary
            target name and category key-value pairs
        """
        target_categories = {}
        self.log.info("\n*** Assigning target name categories ***")
        self.log.info(f"\nUnique Target Names: {len(targets)}")
        bar = ProgressBar().start()
        for x, targ in zip(bar(range(len(targets))), targets):
            if targ != "ANY":
                obs = Observations.query_criteria(target_name=targ)
                cat = obs[np.where(obs[datacol])][datacol]
                if len(cat) > 0:
                    target_categories[targ] = cat[0]
                else:
                    target_categories[targ] = "None"
            bar.update(x + 1)
        bar.finish()
        return target_categories

    def search_by_radec(self, data, propid="proposal_id", ra="ra_targ", dec="dec_targ", datacol="target_classification"):
        """Scrapes MAST for remaining target classifications that could not be identified using target name. This method instead uses a broader set of query parameters: the ``ra_targ`` and ``dec_targ`` coordinates along with the dataset's proposal ID. If multiple datasets are found to match, the first of these containing a target_classification value will be used.

        Returns
        -------
        dict
            secondary set of remaining key-value pairs (target names and scraped categories)
        """
        other_cat = {}
        if len(data) > 0:
            bar = ProgressBar().start()
            for x, (k, v) in zip(
                bar(range(len(data))), data.items()
            ):
                other_cat[k] = {}
                propid, ra, dec = v[propid], v[ra], v[dec]
                obs = Observations.query_criteria(
                    proposal_id=propid, s_ra=[ra, ra + 0.1], s_dec=[dec, dec + 0.1]
                )
                cat = obs[np.where(obs[datacol])][datacol]
                if len(cat) > 0:
                    other_cat[k] = cat[0]
                else:
                    other_cat[k] = "None"
                bar.update(x)
            bar.finish()
        return other_cat


class HstSvmRadio(Radio):
    """Class for scraping metadata from MAST (Mikulsky Archive for Space Telescopes) via ``astroquery``. Current functionality for this class is limited to extracting the `target_classification` values of HAP targets from the archive. An example of a target classification is "GALAXY" - an alphanumeric categorization of an image product/.fits file. Note - the files themselves are not downloaded, just this specific metadata listed in the online archive database. For downloading MAST science files, use the ``spacekit.extractor.radio`` module. The search parameter values needed for locating a HAP product on MAST can be extracted from the fits science extension headers using the ``astropy`` library. See the ``spacekit.preprocessor.scrub`` api for an example (or the astropy documentation)."""

    def __init__(
        self, df, trg_col="targname", ra_col="ra_targ", dec_col="dec_targ", **log_kws
    ):
        """Instantiates a spacekit.extractor.radio.HstSvmRadio object.

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
        super().__init__(name="HstSvmRadio", **log_kws)
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
        self.target_categories = self.search_by_targname(self.targets)
        params = self.prop_radec_dict()
        self.other_cat = self.search_by_radec(params)
        self.df = self.combine_categories()
        return self.df

    def backup_search(self, targ):
        self.targ_any[targ] = self.df.loc[self.df[self.trg_col] == targ][
            [self.ra_col, self.dec_col]
        ]

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
            self.categories, orient="index", columns=["category"]
        )
        self.log.info("\nTarget Categories Assigned.")
        self.log.info(cat["category"].value_counts())
        self.df = self.df.join(cat, how="left")
        return self.df

    def extract_params_from_index(self, index):
        # obs_id = 'hst_10403_29_acs_sbc_total_j96029'
        obs_id = "_".join(index.split("_")[:6])
        prop_id = str(index).split("_")[1]
        return obs_id, prop_id

    def prop_radec_dict(self):
        params = dict()
        for idx, row in self.targ_any.iterrows():
            params[idx] = dict()
            obs_id, prop_id = self.extract_params_from_index(idx)
            params[idx]['obs_id'] = obs_id
            params[idx]['proposal_id'] = prop_id
            params[idx]['ra_targ'] = row[self.ra_col]
            params[idx]['dec_targ'] = row[self.dec_targ]
        self.log.info(f"Other targets (ANY): {len(params)}")
