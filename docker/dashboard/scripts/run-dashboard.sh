#!/bin/bash -xu
src_data=${1:-"$(pwd)"/docker/dashboard/data}
mounts=${2:-""}

envfile="./docker/dashboard/.env"
source $envfile

if [ -ne $2]; then
    docker run -d -p 8050:8050 $DOCKER_IMAGE $EPCOMMAND
else
    docker run -d -p 8050:8050 $DOCKER_IMAGE $EPCOMMAND
    --mount type=bind,source=${src_data},target=/home/developer/data
fi