# Spacekit Dashboard

Before launching the dashboard, you'll need to set some configuration options for the project/datasets you want to load. Copy variables into the empty `.env` file located in `docker/images/dashboard_image` - use one of the templates (`spacekit/docker/images/dashboard_image/templates`) and customize further as desired.


## Configuration options

"APP": There are currently only two options for "APP": "cal" for CALCLOUD and "svm" for Single Visit Mosaics.

"SRC","COLLECTION", and "PFX"

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

Once you have variables set in the .env file, build the image

```bash
# building from the dashboard.env file
$ cd spacekit
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
