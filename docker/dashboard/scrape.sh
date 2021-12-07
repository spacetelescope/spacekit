#!/bin/bash -xu

# Ex1: $ sh scrape.sh calcloud-modeling-sb ('2021-08-22-1629663047' '2021-10-28-1635457222' '2021-10-28-1635462798')
# Ex2: $ export KFOLD=1 && sh scrape.sh calcloud-modeling-sb ('2021-08-22-1629663047')

# DATES=('2021-11-04-1636048291')
# DATES=('2021-08-22-1629663047' '2021-10-28-1635457222' '2021-10-28-1635462798')
# DATES=('2021-05-11-1620740441' '2021-05-13-1620929899' '2021-05-15-1621096666')

BUCKET=$1
DATES=$2
KFOLD=${KFOLD:-""}

if [ -z "${KFOLD}" ]
then
    RESCLF=('duration' 'history' 'matrix' 'preds' 'proba' 'scores' 'y_pred' 'y_true')
    RESREG=('duration' 'history' 'predictions' 'residuals' 'scores')
else
    RESCLF=('duration' 'history' 'kfold' 'matrix' 'preds' 'proba' 'scores' 'y_pred' 'y_true')
    RESREG=('duration' 'history' 'kfold' 'predictions' 'residuals' 'scores')
fi

DATA=('latest.csv' 'pt_transform')

for d in "${DATES[@]}"
do
	datapath=`echo ${d}/data`
	mkdir -p $datapath
	for f in "${DATA[@]}"
	do
		aws s3api get-object --bucket $BUCKET --key ${datapath}/${f} ${datapath}/${f}
	done

	clfpath=`echo ${d}/results/mem_bin`
	mkdir -p $clfpath
	for r in "${RESCLF[@]}"

	do
		aws s3api get-object --bucket $BUCKET --key ${clfpath}/${r} ${clfpath}/${r}
	done

	mempath=`echo ${d}/results/memory`
	wallpath=`echo ${d}/results/wallclock`
	mkdir -p $mempath && mkdir -p $wallpath

	for r in "${RESREG[@]}"

	do
		aws s3api get-object --bucket $BUCKET --key ${mempath}/${r} ${mempath}/${r}
		aws s3api get-object --bucket $BUCKET --key ${wallpath}/${r} ${wallpath}/${r}
	done
	
	model_path=`echo ${d}`
	aws s3api get-object --bucket $BUCKET --key ${d}/models/models.zip ${model_path}/models.zip
done
