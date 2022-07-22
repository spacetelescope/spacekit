#!/bin/bash -xu
# This should only be built if you made your own customizations to the base image (rare).
# Typically you only would need to customize the dashboard layer image built on top of this.
# If you do customize and build the base image, be sure to specify it in your .env config
# The base image is pulled from docker hub automatically if/when you build the dashboard image
ENV=$1

if [[ $ENV == "dev" ]]; then
    reqs="./docker/images/base_image/requirements-dev.txt"
    echo "ENV=${ENV}: building deps from ${reqs}"
else
    reqs="./docker/images/base_image/requirements.txt"
    echo "ENV=prod: building deps from ${reqs}"
fi

export DOCKER_IMAGE=alphasentaurii/spacekit:base
export BASE_IMG="debian:bullseye-slim"
export BASE_DOCKERFILE="./docker/images/base_image/Dockerfile"

docker build -f ${BASE_DOCKERFILE} -t ${DOCKER_IMAGE} --build-arg BASE_IMAGE="${BASE_IMG}" --build-arg REQ_FILE=$reqs .