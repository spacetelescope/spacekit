#!/bin/bash -xu
## dev, testing, nightly or custom name with matching .env file inside envs directory
environment=${1:-"dashboard"}
envfile="images/dashboard_image/envs/${environment}.env"
source $envfile

docker build -f images/dashboard_image/Dockerfile -t ${DOCKER_IMAGE} \
    --build-arg BASE_IMAGE="${BASE_IMG}" \
    --build-arg ENVIRONMENT=$environment \
    --build-arg SRC="${SRC}" \
    --build-arg COLLECTION="${COLLECTION}" \
    --build-arg DATASETS=${DATASETS} .
