ARG BASE_IMAGE=alphasentaurii/spacekit:base
FROM ${BASE_IMAGE}

USER developer
WORKDIR /home/developer
# temp fix for problematic dash dep (downgrade werzeug)
RUN pip uninstall werkzeug -y && pip install werkzeug==2.0.3

## Install dev branch (editable)
ARG SPACEKIT_REPO=alphasentaurii/spacekit
ARG SPACEKIT_BRANCH=main
RUN git clone --branch $SPACEKIT_BRANCH https://github.com/${SPACEKIT_REPO}.git
WORKDIR $DEV_HOME/spacekit
RUN pip install -e .

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