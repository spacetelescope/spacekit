#!/bin/bash -xu
envfile=".env"
source $envfile

docker build -f Dockerfile -t ${DOCKER_IMAGE} \
    --build-arg BASE_IMAGE="${BASE_IMG}" \
    --build-arg SRC="${SRC}" \
    --build-arg COLLECTION="${COLLECTION}" \
    --build-arg DATASETS=${DATASETS} .
