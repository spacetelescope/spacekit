#!/bin/bash -xu
source "./docker/images/dashboard_image/.env"

DOCKER_FILE=./docker/images/dashboard_image/Dockerfile

docker build -f ${DOCKER_FILE} -t ${DOCKER_IMAGE} \
--build-arg BASE_IMAGE="${BASE_IMG}" \
--build-arg SPACEKIT_VERSION="${SPACEKIT_VERSION}"
--build-arg SRC="${SRC}" \
--build-arg COLLECTION="${COLLECTION}" \
--build-arg PFX="${PFX}" \
--build-arg DATASETS=${DATASETS} .
