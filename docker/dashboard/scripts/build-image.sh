#!/bin/bash -xu
cfg=${1:-"cal"}
#envfile="./docker/dashboard/scripts/envs/${cfg}.env"
envfile="./docker/dashboard/.env"
source $envfile

docker build -f Dockerfile -t ${DOCKER_IMAGE} \
    --build-arg BASE_IMAGE="${BASE_IMG}" \
    --build-arg CFG="${cfg}" \
    --build-arg SOURCEDATA="${SOURCEDATA}" \
    --build-arg SRC="${SRC}" \
    --build-arg COLLECTION="${COLLECTION}" \
    --build-arg DATASETS=$DATASETS .
