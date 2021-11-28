#! /bin/bash -eu
export SVM_DOCKER_IMAGE=alphasentaurii/spacekit:svm
# export CAL_BASE_IMAGE="stsci/hst-pipeline:CALDP_drizzlecats_CAL_rc6"
export CAL_BASE_IMAGE="stsci/hst-pipeline:latest"
docker build -f Dockerfile -t ${SVM_DOCKER_IMAGE} --build-arg CAL_BASE_IMAGE="${CAL_BASE_IMAGE}" .
