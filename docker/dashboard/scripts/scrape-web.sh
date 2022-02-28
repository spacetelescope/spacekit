#!/bin/bash -xu

# DATASETS=${1-:"2022-02-14-1644848448,2021-11-04-1636048291,2021-10-28-1635457222"}
DATASETS=${1-:"2022-02-14,2021-11-04,2021-10-28"}

export SPACEKIT_DATA=`pwd`/data
count=`ls $SPACEKIT_DATA | wc -l`
echo $count
if [ "${count}" -ne "3" ]; then
    echo "Scraping datasets"
    python -m spacekit.datasets.beam -s=git:calcloud -d=$DATASETS -o="."
else
    echo "Copying datasets to container"
fi