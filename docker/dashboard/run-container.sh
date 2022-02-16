#! /bin/bash -eu
#export DATES=('2021-11-04-1636048291' '2022-02-14-1644848448' '2021-10-28-1635457222')
#sh scrape.sh $BUCKET $DATES

src_data=${1:-"${HOME}/data"} # "$(pwd)"/${svm_data}
docker run \
-it \
--name spacekitdashboard \
--mount type=bind,source=${src_data},target=/home/developer/data \
alphasentaurii/spacekit:dashboard

## ENTRYPOINTS

## S3 ##
## Get these 3 specific model training timestamps from an s3 bucket
# export r0=2021-11-04-1636048291
# export r1=2021-10-28-1635457222
# export r2=2021-08-22-1629663047
# export bucketname=mybucketname
# python -m spacekit.dashboard.cal.app -s=s3 -u=$bucketname -r0=$r0 -r1=$r1 -r2=$r2

## Local (mounted) directory ##
## Manually download or create zip archives
# export src_data=${HOME}/path/to/data
## or: 
# src_data="$(pwd)"/${svm_data}
## then:
# python -m spacekit.dashboard.cal.app
## (uses default options: -s=file -u=data)

## Git ##
## Download 3 latest datasets, results, models from github
# python -m spacekit.dashboard.cal.index -s=git