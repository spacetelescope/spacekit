#! /bin/bash -eu
export DOCKER_IMAGE=alphasentaurii/spacekit:dash
src_data=${1:-"$(pwd)"/data}

docker run \
-it \
--name spacekitdash \
--mount type=bind,source=${src_data},target=/home/developer/data \
$DOCKER_IMAGE
