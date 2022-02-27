#!/bin/bash -xu

# Ex1: $ sh scrape.sh $BUCKET $DATES

DATES=('2022-02-14-1644848448' '2021-11-04-1636048291' '2021-10-28-1635457222')
# DATES=('2021-08-22-1629663047' '2021-10-28-1635457222' '2021-11-04-1636048291')
# DATES=('2021-05-11-1620740441' '2021-05-13-1620929899' '2021-05-15-1621096666')

BUCKET=$1
DATES=$2
ZIPS=${ZIPS:=""}

if [ -z "${ZIPS}" ]; then
	for d in "${DATES[@]}"
		do
			datapath=`echo ${HOME}/data/${d}`
			mkdir -p $datapath
			aws s3 cp s3://${BUCKET}/{d} ${datapath}/ --recursive
		done
else
	datapath=`echo ${HOME}/data`
	mkdir -p $datapath
	for d in "${DATES[@]}"
		do
			aws s3api get-object --bucket $BUCKET --key ${archive}/{d}.zip ${datapath}/${d}.zip
		done
fi

mv ${HOME}/data/ ./spacekit_data/

# KFOLD=${KFOLD:-""}

# if [ -z "${KFOLD}" ]
# then
#     RESCLF=('duration' 'history' 'matrix' 'preds' 'proba' 'scores' 'y_pred' 'y_true')
#     RESREG=('duration' 'history' 'predictions' 'residuals' 'scores')
# else
#     RESCLF=('duration' 'history' 'kfold' 'matrix' 'preds' 'proba' 'scores' 'y_pred' 'y_true')
#     RESREG=('duration' 'history' 'kfold' 'predictions' 'residuals' 'scores')
# fi

# DATA=('latest.csv' 'pt_transform')
	# datapath=`echo data/${d}`
	#mkdir -p $datapath
	# for f in "${DATA[@]}"
	# do
	# 	aws s3api get-object --bucket $BUCKET --key ${datapath}/${f} ${datapath}/${f}
	# done

	# clfpath=`echo ${d}/results/mem_bin`
	# mkdir -p $clfpath
	# for r in "${RESCLF[@]}"

	# do
	# 	aws s3api get-object --bucket $BUCKET --key ${clfpath}/${r} ${clfpath}/${r}
	# done

	# mempath=`echo ${d}/results/memory`
	# wallpath=`echo ${d}/results/wallclock`
	# mkdir -p $mempath && mkdir -p $wallpath

	# for r in "${RESREG[@]}"

	# do
	# 	aws s3api get-object --bucket $BUCKET --key ${mempath}/${r} ${mempath}/${r}
	# 	aws s3api get-object --bucket $BUCKET --key ${wallpath}/${r} ${wallpath}/${r}
	# done
	
	# model_path=`echo ${d}`
	#aws s3api get-object --bucket $BUCKET --key ${d}/models/models.zip ${model_path}/models.zip
#done
