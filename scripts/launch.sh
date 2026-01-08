#!/bin/bash -e
# To view notebooks in -it mode: jupyter-lab --ip 0.0.0.0

envfile="./docker/images/dashboard_image/.env"
source $envfile

if [ "${CONTAINER_MODE}" == "-it" ]; then
    EPCOMMAND="/bin/bash"
fi

existing=`docker ps -aqf "name=${NAME}"`

if [[ -z $existing ]]; then
    if [ $MOUNTS -ne 0 ]; then
        docker run ${CONTAINER_MODE} \
        -p 8080:8050 -p 8888:8888 \
        --name $NAME \
        --mount type=bind,source=${SOURCEDATA},target=${DESTDATA} \
        $DOCKER_IMAGE $EPCOMMAND
    else
        docker run ${CONTAINER_MODE} \
        -p 8080:8050 -p 8888:8888 \
        --name $NAME \
        $DOCKER_IMAGE $EPCOMMAND
    fi
else
    echo "Restarting existing container: ${NAME} (ID=${existing})"
    docker container start $NAME
    docker container exec ${CONTAINER_MODE} $NAME $EPCOMMAND
fi
