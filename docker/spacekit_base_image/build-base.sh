#!/bin/bash -xu
# envfile="./docker/spacekit_base_image/.env"
# source $envfile

export DOCKER_IMAGE=alphasentaurii/spacekit:base
export BASE_IMG="debian:bullseye-slim"

docker build -f Dockerfile -t ${DOCKER_IMAGE} --build-arg BASE_IMAGE="${BASE_IMG}" .