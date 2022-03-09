# A - pull from dockerhub using latest datasets

```bash
export DOCKER_IMAGE=alphasentaurii/spacekit:dash
docker pull $DOCKER_IMAGE
docker run -d -p 8050:8050 $DOCKER_IMAGE
```

---

# B - From Source

1. Configure environment vars using .env-template -> put these in the .env file

SRC: where to get the datasets
-pkg: default from spacekit archive, 3 latest versions
-s3: bucket on aws. Also need to set:
    - COLLECTION={bucketname}
    - PFX={parent directory in bucket}
    - DATES={subfolder directory or zipfile names}
-file: local directory path to data (if you already have it, it will get copied over to image)

HINT: You can also use the default ('pkg') source data, then bind-mount your own data to the target destination when you go to run the container.

DATASETS={comma separate string of each dataset name, usually a date "2021-02-14}

*Note: If you already have datasets locally, you can use the default `pkg` variable to build the image then bind-mount your own source data to the home `data` directory when you go to run the container. See example in Step 3c below*


2. build the image locally

```bash
sh scripts/build_image.sh
```

3. run container:
    -a: dash app from local image
    -b: interactive from source
    -c: dash app with local data bind-mounted at run time

a) run dashboard app in browser

```bash
sh scripts/run-dashboard.sh
```

The default entrypoint for running the app in a web browser is: 
`python -m spacekit.dashboard.cal.index`

Developer version (includes debug info) is run using command-line option: `--env dev`

b) Run dashboard interactively from source

```bash
sh scripts/run-interactive.sh
```

c) Mount data from local source

```bash
# the 1 indicates you want to mount whatever path $src_data is set to
sh scripts/run-dashboard.sh cal $src_data 1
```

