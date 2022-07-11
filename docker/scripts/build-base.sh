#!/bin/bash -xu
# This should only be built if you made your own customizations to the base image (rare).
# Typically you only would need to customize the second layer image built on top of this.
# If you do customize and build the base image, be sure to specify it in your .env config
# The base image is pulled from docker hub automatically if/when you build the dashboard image

export DOCKER_IMAGE=alphasentaurii/spacekit:base
export BASE_IMG="debian:bullseye-slim"
export BASE_DOCKERFILE="images/base_image/Dockerfile"

docker build -f ${BASE_DOCKERFILE} -t ${DOCKER_IMAGE} --build-arg BASE_IMAGE="${BASE_IMG}" .