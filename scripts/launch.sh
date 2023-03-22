#!/bin/bash -xu
envfile="./docker/images/dashboard_image/.env"
source $envfile

if [ "${CONTAINER_MODE}" == "-it" ]; then
    EPCOMMAND="/bin/bash"
fi

if [ $MOUNTS -ne 0 ]; then
    docker run ${CONTAINER_MODE} --name $NAME \
    --ip $IPADDRESS --hostname $HOSTNAME -p 8050:8050 $* \
    --mount type=bind,source=${SOURCEDATA},target=${DESTDATA} \
    $DOCKER_IMAGE $EPCOMMAND
else
    docker run ${CONTAINER_MODE} --name $NAME \
    --ip $IPADDRESS --hostname $HOSTNAME -p 8050:8050 $* \
    $DOCKER_IMAGE $EPCOMMAND
fi
