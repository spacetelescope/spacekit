#! /bin/bash -eu
src_data=${1:-"${HOME}/data"} # "$(pwd)"/${svm_data}
docker run \
-it \
--name spacekitdashboard \
--mount type=bind,source=${src_data},target=/home/developer/data \
alphasentaurii/spacekit:dashboard

## ENTRYPOINTS
## Get these 3 specific model training timestamps from s3
# export r0=2021-11-04-1636048291
# export r1=2021-10-28-1635457222
# export r2=2021-08-22-1629663047
# export bucketname=mybucketname
# python -m spacekit.dashboard.cal.app -s=s3 -u=$bucketname -r0=$r0 -r1=$r1 -r2=$r2

## Load from mounted local directory
# export src_data=${HOME}/path/to/data 
## or: 
# src_data="$(pwd)"/${svm_data}
## then:
# python -m spacekit.dashboard.cal.app

## Download 3 latest datasets, results, models from github
# python -m spacekit.dashboard.cal.app -s=git