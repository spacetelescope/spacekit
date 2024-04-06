#!/bin/bash -xu
envfile="./docker/images/dashboard_image/.env"
source $envfile

if [ "${CONTAINER_MODE}" == "-it" ]; then
    EPCOMMAND="/bin/bash"
fi

if [ $MOUNTS -ne 0 ]; then
    docker run ${CONTAINER_MODE} --name $NAME \
    --mount type=bind,source=${SOURCEDATA},target=${DESTDATA} \
    $DOCKER_IMAGE $EPCOMMAND
else
    docker run ${CONTAINER_MODE} --name $NAME \
    $DOCKER_IMAGE $EPCOMMAND
fi
