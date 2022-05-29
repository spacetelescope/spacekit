#! /bin/bash -eu
cfg=${1:-"cal"}
src_data=${2:-"$(pwd)"/data}
source ./docker/dashboard/scripts/envs/${cfg}.env

docker run \
-it \
--name dashboard \
--mount type=bind,source=${src_data},target=/home/developer/data \
$DOCKER_IMAGE /bin/bash
