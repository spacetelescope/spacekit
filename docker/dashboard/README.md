# A - pull from dockerhub using latest datasets

```bash
docker pull alphasentaurii/spacekit:dashboard
docker run -d -p 8050:8050 $DOCKER_IMAGE
```

---

# B - Get your own datasets then Build and run image locally

1. download/copy datasets into the `spacekit_data` folder in this directory
2. build the image locally
3. run dash app from local image

```bash
# scrape data from web or s3
python -m spacekit.datasets.beam -s=git:calcloud
```

*Alternatively, use the shell script in this directory to access s3 via awscli:*

```bash
export DATES=('2021-11-04-1636048291' '2022-02-14-1644848448' '2021-10-28-1635457222')
sh scrape.sh $BUCKET $DATES
```

2A) Build image locally

```bash
cd spacekit/docker/dashboard
export DOCKER_IMAGE=alphasentaurii/spacekit:dash
export CAL_BASE_IMAGE="stsci/hst-pipeline:latest"
docker build -f Dockerfile -t ${DOCKER_IMAGE} --build-arg CAL_BASE_IMAGE="${CAL_BASE_IMAGE}" .
```

# 3 - Run container (mount local data source folder)

The default entrypoint for running the app in a web browser is: `python -m spacekit.dashboard.cal.index`

```bash
docker run -d -p 8050:8050 $DOCKER_IMAGE
```
