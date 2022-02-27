#! /bin/bash -eu
#export DATES=('2021-11-04-1636048291' '2022-02-14-1644848448' '2021-10-28-1635457222')
#sh scrape.sh $BUCKET $DATES

# A : connect with your own datasets

# 1. download/copy datasets into the `spacekit_data` folder in this directory
# 2. build the image locally
# 3. run dash app from local image


# B : mount local data folder and run image interactively

src_data=${1:-"$(pwd)"/spacekit_data}
docker run \
-it \
--name spacekitdashboard \
--mount type=bind,source=${src_data},target=/home/developer/data \
alphasentaurii/spacekit:dashboard
