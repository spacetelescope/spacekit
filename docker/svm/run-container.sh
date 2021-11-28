#! /bin/bash -eu
src_data=${1:-"${HOME}/svm-data"} # "$(pwd)"/${svm_data}
docker run \
-it \
-m=10000m \
--name spacekit_svm \
--mount type=bind,source=${src_data},target=/home/developer/data \
alphasentaurii/spacekit:svm
