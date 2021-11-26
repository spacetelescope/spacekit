#! /bin/bash -eu
src_data=${1:-"../../svm-data"}
docker run \
-it \
-m=8000m \
--name spacekit_svm \
--mount type=bind,source="$(pwd)"/${svm_data},target=/home/developer/data \
alphasentaurii/spacekit:svm