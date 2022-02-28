#!/bin/bash -xu

# Ex1: $ sh scrape.sh $BUCKET $DATES

BUCKET=$1
DATES=${2:-('2022-02-14-1644848448' '2021-11-04-1636048291' '2021-10-28-1635457222')}
ZIPS=${3:-"1"}
PFX=${4:-"archive"}

data_tmp="${HOME}/spacekit_data"
local_data=`pwd`/data

if [ -z "${ZIPS}" ]; then
	for d in "${DATES[@]}"
		do
			datapath=`echo ${data_tmp}/${d}`
			mkdir -p $datapath
			aws s3 cp s3://${BUCKET}/{d} ${datapath}/ --recursive
		done
	cp -R ${datapath}/ ${local_data}/
else
	datapath=$data_tmp
	if [ ! -d "${datapath}" ]; then
		mkdir -p $datapath
	fi
	for d in "${DATES[@]}"
		do
			aws s3api get-object --bucket $BUCKET --key ${PFX}/{d}.zip ${datapath}/${d}.zip
			unzip ${datapath}/${d}.zip ${local_data}/${d}/
		done
fi
