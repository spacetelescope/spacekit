#!/bin/bash -xu
export TF_CPP_MIN_LOG_LEVEL=2
REG=${1:-"./data/train_mosaic.csv"}
IMG=${2:-"./data/img"}
SYNTH=${3:-""} # train with synthetic dataset also (if stored in separate csv file)
MOVEPOS=${4:-""} # created automatically when images are generated from a labeled dataframe (ex: "data/pos.txt")

neg=${IMG}/0
pos=${IMG}/1
echo "Misaligned training image count:"
ls $pos | wc -l
echo "Aligned training image count:"
ls $neg | wc -l

# move misaligned images to '1' image label subdirectory
if [[ ${MOVEPOS} -ne "" ]]; then
    
    misaligned=`cat $MOVEPOS`
    if [[ ${misaligned} -ne "" ]]; then
        echo "Moving images to assigned directories"
        for m in ${misaligned}
        do 
            mv ${neg}/"${m}" ${pos}/. 
        done
        echo "Misaligned training image count:"
        ls $pos | wc -l
        echo "Aligned training image count:"
        ls $neg | wc -l
    fi
fi

if [[ -z $SYNTH ]]; then
    python -m spacekit.skopes.hst.mosaic.svm_train $REG $IMG
else
    python -m spacekit.skopes.hst.mosaic.svm_train $REG $IMG -c=$SYNTH
fi
