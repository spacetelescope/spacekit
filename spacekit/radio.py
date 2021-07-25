"""
# class for querying and downloading .fits files from MAST s3 bucket on AWS
"""

import boto3
from astroquery.mast import Observations
from astroquery.mast import Catalogs

class Radio:
    def __init__(self, target_list, config='enable'):
        self.target_list = target_list
        self.config = config
        self.region = 'us-east-1'
        self.s3 = boto3.resource('s3', region_name=self.region)
        self.bucket = self.s3.Bucket('stpubdata')
        self.location = {'LocationConstraint': self.region}
        self.s3_uris = []
        self.file_list = []

    def configure_aws(self):
    # configure aws settings
        if self.config == 'enable':
            Observations.enable_cloud_dataset(provider='AWS', profile='default')
        elif self.config == 'disable':
            Observations.disable_cloud_dataset()

    def get_uris(self):
        #Do a cone search and find the K2 long cadence data for each target
        for target in self.target_list:
            obs = Observations.query_object(target,radius="0s")
            want = (obs['obs_collection'] == "K2") & (obs['t_exptime'] ==1800.0)
            data_prod = Observations.get_product_list(obs[want])
            filt_prod = Observations.filter_products(data_prod, productSubGroupDescription="LLC")
            try:
                uri = Observations.get_cloud_uris(filt_prod)
                self.s3_uris.append(uri)
            except Exception: #ResolverError: 
                print(f"Could not resolve {target} to a sky position.")
                continue
        return self

    def download_fits(self):
        count = 0
        print(f"Downloading {len(self.s3_uris)} from AWS (MAST))")
        for uri in self.s3_uris:
            U = uri[0]
            key = U.replace("s3://stpubdata/", "")
            root = U.split('/')[-1]
            try:
                self.bucket.download_file(key, root, ExtraArgs={"RequestPayer": "requester"})
                count+=1
                self.file_list.append(root)
            except FileExistsError:
                continue
        print(f"Download Complete: {count} files")
        return self
