#!/bin/bash -xu
export DOCKER_IMAGE=alphasentaurii/spacekit:dash
docker run -d -p 8050:8050 $DOCKER_IMAGE