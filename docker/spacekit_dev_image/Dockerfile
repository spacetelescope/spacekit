ARG BASE_IMAGE=alphasentaurii/spacekit:base
FROM ${BASE_IMAGE}

# Set env vars
ENV $DEV_HOME=/home/developer
COPY .env $DEV_HOME/.
USER developer
RUN source $DEV_HOME/.env

## Install dev branch (editable)
USER developer
WORKDIR /home/developer
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
ENTRYPOINT ["python","-m","spacekit.dashboard.cal.index"]