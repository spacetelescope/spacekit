# Spacekit Dashboard

Before launching the dashboard, you'll need to set some configuration options for the project/datasets you want to load. The simplest route is to use the default configuration, which loads the 3 latest datasets from the spacekit package. There are some templates in `spacekit/docker/images/dashboard_image/envs`. Feel free to add a new folder in "envs" directory and create custom settings.


## Configuration options

1. "APP"

There are currently only two options for "APP": "cal" for CALCLOUD and "svm" for Single Visit Mosaics.

2. "SRC","COLLECTION", and "PFX"

The default source:collection is the spacekit package ("pkg") and the "calcloud" datasets. 

```bash
SRC=pkg
COLLECTION=calcloud  # (or "svm", "k2")
PFX=archive
```

Another option is to point to "s3". In this case, your bucketname is the "collection" and the prefix folder in that bucket is "PFX".

```bash
SRC=s3
COLLECTION=mybucket
PFX
```

## Building image locally

The primary environment file `dashboard.env` installs the latest spacekit release using pip, while the other env files are setup to install spacekit from a github repo branch. Use these as a template or edit them directly.

```bash
# building from the dashboard.env file
$ cd spacekit/docker
$ sh scripts/build.sh
```

If you create your own .env file (assuming it's saved in the `envs` directory) you would input the name of this env as the first command line argument, for example:

```bash
# building from one of the default templates (dev, testing, nightly)
$ cd spacekit/docker
$ sh scripts/build.sh dev

# building from a customized config named "custom.env"
$ cd spacekit/docker
$ sh scripts/build.sh custom
```
