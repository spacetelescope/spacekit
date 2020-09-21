
class Radio:

    # function for downloading data from MAST s3 bucket on AWS
    def mast_aws(target_list):
        import boto3
        from astroquery.mast import Observations
        from astroquery.mast import Catalogs
        # configure aws settings
        region = 'us-east-1'
        s3 = boto3.resource('s3', region_name=region)
        bucket = s3.Bucket('stpubdata')
        location = {'LocationConstraint': region}
        Observations.enable_cloud_dataset(provider='AWS', profile='default')
        
        for target in target_list:
        #Do a cone search and find the K2 long cadence data for target
            obs = Observations.query_object(target,radius="0s")
            want = (obs['obs_collection'] == "K2") & (obs['t_exptime'] ==1800.0)
            data_prod = Observations.get_product_list(obs[want])
            filt_prod = Observations.filter_products(data_prod, productSubGroupDescription="LLC")
            s3_uris = Observations.get_cloud_uris(filt_prod)
            for url in s3_uris:
            # Extract the S3 key from the S3 URL
                fits_s3_key = url.replace("s3://stpubdata/", "")
                root = url.split('/')[-1]
                bucket.download_file(fits_s3_key, root, ExtraArgs={"RequestPayer": "requester"})
        Observations.disable_cloud_dataset()
        return print('Download Complete')