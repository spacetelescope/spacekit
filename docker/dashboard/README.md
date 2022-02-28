# A - pull from dockerhub using latest datasets

```bash
export DOCKER_IMAGE=alphasentaurii/spacekit:dash
docker pull $DOCKER_IMAGE
docker run -d -p 8050:8050 $DOCKER_IMAGE
```

---

# B - Get latest datasets from github or s3

1. download/copy datasets into the `data` folder in this directory
2. build the image locally
3. run dash app from local image

The default entrypoint for running the app in a web browser is: 
`python -m spacekit.dashboard.cal.index`

```bash
sh scripts/scrape-web.sh # OR: sh scripts/scrape-s3.sh $BUCKET $DATES
sh scripts/build_image.sh
sh run-dashboard.sh
```

---

# C - Run dashboard interactively from source

```bash
sh scripts/scrape-web.sh # OR: sh scripts/scrape-s3.sh $BUCKET $DATES
sh scripts/build_image.sh
sh scripts/run-interactive.sh
```
