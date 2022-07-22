#!/bin/bash -xu
source "./docker/images/dashboard_image/.env"

if [[ "${CFG}" != "dev" ]]; then
    DOCKER_FILE=./docker/images/dashboard_image/Dockerfile
    docker build -f ${DOCKER_FILE} -t ${DOCKER_IMAGE} \
    --build-arg BASE_IMAGE="${BASE_IMG}" \
    --build-arg SRC="${SRC}" \
    --build-arg COLLECTION="${COLLECTION}" \
    --build-arg PFX="${PFX}" \
    --build-arg DATASETS=${DATASETS} .
else
    DOCKER_FILE=./docker/images/dashboard_image/templates/dev/Dockerfile
    docker build -f ${DOCKER_FILE} -t ${DOCKER_IMAGE} \
    --build-arg BASE_IMAGE="${BASE_IMG}" \
    --build-arg SPACEKIT_REF="${SPACEKIT_BRANCH}" \
    --build-arg PFX="${PFX}" \
    --build-arg SRC="${SRC}" \
    --build-arg COLLECTION="${COLLECTION}" \
    --build-arg DATASETS=${DATASETS} .
fi
