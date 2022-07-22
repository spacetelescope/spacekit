
DOCKER_FILE=./docker/images/test_image/Dockerfile
docker build -f ${DOCKER_FILE} -t ${DOCKER_IMAGE} \
--build-arg SPACEKIT_VERSION="${SPACEKIT_BRANCH}" \
--build-arg PFX="${PFX}" \
--build-arg SRC="${SRC}" \
--build-arg COLLECTION="${COLLECTION}" \
--build-arg DATASETS=${DATASETS} .