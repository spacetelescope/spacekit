#!/bin/bash -xu
export TF_CPP_MIN_LOG_LEVEL=2
SRCPATH=${1:-""}  # data/singlevisits/results_2021-07-28
DATA=${2:-""}
DRAW=${3:-""}

# The only required argument for this script is SRCPATH. Env vars will use defaults if not set explicitly and create directories if they don't already exist.

OUT=${OUT:-"./data"}
IMG=${IMG:-"./data/img"}
H5=${H5:-"./data/train_mosaic"}
CSV=${CSV:-"data/train_mosaic.csv"}

if [[ -z ${SRCPATH} ]]; then
    echo "SRCPATH argument is required."
    exit 1
fi

if [[ -z ${DATASETS} ]]; then
    echo "DATASETS not set explicitly - using all datasets in ${SRCPATH}"
    DATASETS=`ls "${SRCPATH}"`
fi

neg=${IMG}/0
pos=${IMG}/1
mkdir -p $neg && mkdir -p $pos

if [[ ${DATA} -ne "" ]]; then
    python -m spacekit.extractor.frame_data $SRCPATH -o=$CSV -d=$H5
fi

if [[ ${DRAW} -ne "" ]]; then
    python -m spacekit.extractor.draw_mosaics $SRCPATH -o=$neg
fi


# # Only if using filter similarity embeddings

#FILTERS=${filters:-""}
# if [[ ${FILTERS} -ne "" ]]; then
#     IMG_FLTR=${OUT}/img/total
#     mkdir $IMG_FLTR
#     python make_images.py $SVMCRPT -o=$FLTR -c=1 -t=filter

# fi


