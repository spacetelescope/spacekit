#!/bin/bash -xu
cfg=${1:-"cal"}
src_data=${2:-"$(pwd)"/data}
mounts=${3:-""}
source ./scripts/envs/${cfg}.env

if [ -ne $3]; then
    docker run -d -p 8050:8050 $DOCKER_IMAGE $EPCOMMAND
else
    docker run -d -p 8050:8050 $DOCKER_IMAGE $EPCOMMAND
    --mount type=bind,source=${src_data},target=/home/developer/data \
fi