# Spacekit Dashboard

Before launching the dashboard, you'll need to set some configuration options for the project/datasets you want to load. Copy variables into the empty `.env` file located in `docker/images/dashboard_image` - use one of the templates (`spacekit/docker/images/dashboard_image/templates`) and customize further as desired.


## Configuring Environment Variables

- `CFG`: tells the build script which Dockerfile to use; setting it to "dev" points the builder to the one in `templates/dev` which installs a dev version of `spacekit` from a github repo branch.

- `APP`: "cal" for CALCLOUD Dashboard and "svm" for Single Visit Mosaics Dashboard. Note- These may get merged into a single dashboard app that configures on the fly according to some setting, but for now they are distinct.

- `VERSION`: used for meaningful image tagging / container names

```bash
# change these to your heart's content
CFG="dev"
APP="cal"
VERSION="-dev"
TAG="dashboard-${APP}${VERSION}"
DOCKER_IMAGE="alphasentaurii/spacekit:${TAG}"

# only used if building spacekit from source; must set CFG="dev")
SPACEKIT_REPO="alphasentaurii/spacekit"
SPACEKIT_BRANCH="rev-0.3.2"
```

Data import settings 

These are used by spacekit.datasets.beam to find specific datasets. Using the below defaults will pull in up to 3 of the most recent collection of trained models and datasets iterations. Although, if you're using the defaults, what are you even doing here? Just pull the image from docker hub `docker pull alphasentaurii/spacekit:dashboard-cal-latest`.

```bash
 # pkg, s3, git, file
SRC="pkg"
# collection, bucketname, repo url, or local path
COLLECTION="calcloud" # e.g. "svm" or "calcloud"
# used by spacekit.datasets as dictionary keys
DATASETS="2022-02-14,2021-11-04,2021-10-28"
# typically the names of the actual dataset directories (or .zip files)
DATES=('2022-02-14-1644848448' '2021-11-04-1636048291' '2021-10-28-1635457222')
# for s3 this is the folder prefix
PFX="archive"
```

Alternatively, you could import your own data from S3, for example:

```bash
SRC=s3
COLLECTION=mybucket
PFX=somefolder
```

Launch settings (for starting a container)

```bash
CONTAINER_MODE="-d" # -d for detached, -it for interactive
MOUNTS=0 # >0 will bind mount the below source and dest paths
SOURCEDATA=$(pwd)
DESTDATA="/home/developer/spacekit"
```

Expert settings

***don't change these unless you know what you're doing***

```bash
BASE_IMG="alphasentaurii/spacekit:base"
SPACEKIT_DATA=/home/developer
TF_CPP_MIN_LOG_LEVEL=2
HOSTNAME="localhost"
IPADDRESS=0.0.0.0
NAME="spkt-dash-${APP}${VERSION}"
EPCOMMAND="python -m spacekit.dashboard.${APP}.index"
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