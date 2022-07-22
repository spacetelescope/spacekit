ARG BASE_IMAGE=alphasentaurii/spacekit:base
FROM ${BASE_IMAGE}

# Import data
USER developer
WORKDIR /home/developer
COPY ./docker/images/dashboard_image/.env /home/developer/.env
ENV SPACEKIT_DATA=/home/developer
ENV TF_CPP_MIN_LOG_LEVEL=2
ARG PFX=archive
ENV PFX=${PFX}
ARG SRC
ARG COLLECTION
ARG DATASETS
RUN python -m spacekit.datasets.beam -s="${SRC}:${COLLECTION}" -d=$DATASETS -o=$SPACEKIT_DATA
EXPOSE 8050
CMD /bin/bash
