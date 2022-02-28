#!/bin/bash -xu
export DOCKER_IMAGE=alphasentaurii/spacekit:dash
export CAL_BASE_IMAGE="stsci/hst-pipeline:latest"
docker build -f Dockerfile -t ${DOCKER_IMAGE} --build-arg CAL_BASE_IMAGE="${CAL_BASE_IMAGE}" .