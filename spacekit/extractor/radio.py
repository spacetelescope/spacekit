import os
import shutil
import glob
import boto3
from astroquery.mast import Observations

# from astroquery.mast import Catalogs


class Radio:
    """Class for querying and downloading .fits files from a MAST s3 bucket on AWS. Note this was originally created for K2 LLC data and is in the process of being revised for other data types/telescopes..."""

    def __init__(self, config="disable"):
        """Instantiates a spacekit.extractor.Radio object.

        Parameters
        ----------
        config : str, optional
            enable or disable aws cloud access (disable uses MAST only), by default "disable"
        """
        self.config = config
        self.region = "us-east-1"
        self.s3 = boto3.resource("s3", region_name=self.region)
        self.bucket = self.s3.Bucket("stpubdata")
        self.location = {"LocationConstraint": self.region}
        self.target_list = None
        self.proposal_id = None  # '13926'
        self.collection = "K2"  # "HST" "HLA"
        self.filters = None  # 'F657N'
        self.obsid = None  # 'ICK90[5678]*'
        self.subgroup = None  # ['FLC', 'SPT'] or ["LLC"]
        self.radius = "0s"
        self.exptime = 1800.0
        self.s3_uris = []
        self.errors = []
        self.science_files = []

    def configure_aws(self):
        """Sets cloud (AWS) configuration On or Off."""
        # configure aws settings
        if self.config == "enable":
            Observations.enable_cloud_dataset(provider="AWS", profile="default")
        elif self.config == "disable":
            Observations.disable_cloud_dataset()

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
        return self

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
        return self

    def get_object_uris(self):
        """Run observation query via cone search and return list of product uris.

        Returns
        -------
        self
            class object with attributes updated
        """
        if self.target_list is None:
            print("Error: target_list (IDs) must be set first.")
            return
        # Do a cone search and find the K2 long cadence data for each target
        for target in self.target_list:
            obs = Observations.query_object(target, radius=self.radius)
            want = (obs["obs_collection"] == self.collection) & (
                obs["t_exptime"] == self.exptime
            )
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
                print(f"Could not resolve {target} to a sky position.")
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
        print(f"Downloading {len(self.s3_uris)} from AWS")
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
        print(f"Download Complete: {count} files")
        return self

    def mast_download(self):
        """Download datasets from MAST"""
        if self.obsid is None:
            search_params = dict(proposal_id=self.proposal_id, filters=self.filters)
        else:
            search_params = dict(
                proposal_id=self.proposal_id, filters=self.filters, obsid=self.obsid
            )
        obs = Observations.query_criteria(**search_params)
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
