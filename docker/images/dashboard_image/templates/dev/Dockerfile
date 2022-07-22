ARG BASE_IMAGE=alphasentaurii/spacekit:base
FROM ${BASE_IMAGE}

USER developer
WORKDIR /home/developer

## Install editable spacekit source code
WORKDIR /home/developer
ARG SPACEKIT_REPO=alphasentaurii/spacekit
RUN git clone https://github.com/${SPACEKIT_REPO}.git

WORKDIR /home/developer/spacekit
ARG SPACEKIT_BRANCH=main
RUN git checkout $SPACEKIT_BRANCH && pip install -e .

# Import data
WORKDIR /home/developer
ENV SPACEKIT_DATA=/home/developer
ENV TF_CPP_MIN_LOG_LEVEL=2
ARG SRC
ARG COLLECTION
ARG DATASETS
RUN python -m spacekit.datasets.beam -s="${SRC}:${COLLECTION}" -d=$DATASETS -o=$SPACEKIT_DATA
EXPOSE 8050
CMD /bin/bash