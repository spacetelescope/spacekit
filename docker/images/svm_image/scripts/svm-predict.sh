#!/bin/bash -xu
# export SVM_QUALITY_TESTING=on
export TF_CPP_MIN_LOG_LEVEL=2
SRCPATH=${1:-"./data/2021-10-06"} # singlevisits/results_2021-10-06
OUT=${2:-"."} # data/svm/2021-10-06
# MODELPATH=${3:-"./ensembleSVM"}

img_path=${OUT}/img
results=${OUT}/results
mkdir -p $img_path && mkdir -p $results

unlabeled_data=${OUT}/svm_unlabeled.csv
predictions=${OUT}/svm_predictions.csv

python -m spacekit.extractor.frame_data $SRCPATH -o=$unlabeled_data
python -m spacekit.extractor.draw_mosaics $SRCPATH -o=$img_path
python mosaic_ml.mosaic_predict.py $unlabeled_data $img_path -o=$predictions 
python -m spacekit.skopes.hst.mosaic.svm_predict $unlabeled_data $img_path -o=./results
#-m=$MODELPATH
