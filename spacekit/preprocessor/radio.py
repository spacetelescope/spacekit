"""
# class for querying and downloading .fits files from MAST s3 bucket on AWS
"""
import os
import shutil
import glob
import boto3
from astroquery.mast import Observations

# from astroquery.mast import Catalogs


class Radio:
    def __init__(self, config="disable"):
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
        # configure aws settings
        if self.config == "enable":
            Observations.enable_cloud_dataset(provider="AWS", profile="default")
        elif self.config == "disable":
            Observations.disable_cloud_dataset()

    def prop_search(self, proposal_id, filters, obsid, subgroup):
        self.proposal_id = proposal_id  # '13926'
        self.filters = filters  # 'F657N'
        self.obsid = obsid  # 'ICK90[5678]*'
        self.subgroup = subgroup  # ['FLC', 'SPT']
        return self

    def cone_search(self, radius, collection, exptime, subgroup):
        self.radius = radius  # 0s
        self.collection = collection  # "K2"
        self.exptime = exptime  # 1800.0
        self.subgroup = subgroup["LLC"]
        return self

    def get_object_uris(self):
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
