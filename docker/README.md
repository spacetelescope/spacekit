# Spacekit Dashboard

Pull the most recent calcloud training data and model results (default):

`docker pull alphasentaurii/spacekit:dash-cal-latest`


## Configuring Custom Datasets via Environment file `.env`

The variables below are used by spacekit.datasets.beam to find specific datasets. Using the defaults will pull in the 3 most recent dataset iterations and model training results. To configure the dashboard to use other datasets, you'll need to set some configuration options. Copy variables into the `.env` file located in `docker/images/dashboard_image` - feel free to use one of the templates (`spacekit/docker/images/dashboard_image/templates`) and customize further as desired.

```bash
 # pkg, s3, web, file
SRC="web"
# collection, bucketname, repo url, or local path
COLLECTION="calcloud" # e.g. "svm", "calcloud", "myS3bucket"
# used by spacekit.datasets as dictionary keys
DATASETS="2022-02-14,2021-11-04,2021-10-28"
# for s3 use the names of the .zip files 
DATASETS="2022-02-14-1644848448,2021-11-04-1636048291,2021-10-28-1635457222"
# for s3 this is the folder prefix
PFX="archive"
```

### Import data from S3 (aws)

```bash
SRC=s3
COLLECTION=mybucket
PFX=somefolder
```

### Mount from local path

You can also have your data in a local directory, and just bind mount the folder when you go to launch the container, or set container mode to "-it" and use spacekit.datasets to get the data before launching the dashboard. 

```bash
CONTAINER_MODE="-d" # -d for detached, -it for interactive
MOUNTS=1 # >0 will bind mount the below source and dest paths
SOURCEDATA="/path/to/datasets"
DESTDATA="/home/developer/data"
```


## Build the image

Once you have variables set in the .env file, build the image

```bash
$ cd spacekit
$ sh scripts/build.sh
```

# Run the container

Launch a container with your brand new image then fire it up in a browser: `http://0.0.0.0:8050/`

```bash
$ sh scripts/launch.sh
# you should see a SHA like: "6cb2bee87fbef53f44686" etc
```