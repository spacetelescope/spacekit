#!/bin/bash -xu
cfg=${1:-"dashboard"} # dev, test, or none (defaults to "dashboard")
envfile="dashboard_image/envs/${cfg}/.env"
source $envfile

if [ "${CONTAINER_MODE}" == "-it" ]; then
    EPCOMMAND="/bin/bash"
fi

if [ -ne $MOUNTS]; then
    docker run ${CONTAINER_MODE} --name $NAME \
    --ip $IPADDRESS --hostname $HOSTNAME -p 8050:8050 \
    $DOCKER_IMAGE $EPCOMMAND
else
    docker run ${CONTAINER_MODE} --name $NAME \
    --ip $IPADDRESS --hostname $HOSTNAME -p 8050:8050 \
    --mount type=bind,source=${SOURCEDATA},target=${DESTDATA} \
    $DOCKER_IMAGE $EPCOMMAND
fi