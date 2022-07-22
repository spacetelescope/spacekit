ARG BASE_IMAGE=debian:bullseye-slim
FROM ${BASE_IMAGE}

RUN apt update && \
    apt upgrade --assume-yes && \
    ACCEPT_EULA=Y DEBIAN_FRONTEND=noninteractive && \
    apt install --assume-yes \
        git \
        python3 \
        python3-pip \
        python3-venv \
        sudo \
        make \
        gcc \
        htop \
        wget \
        tar \
        curl \
        time \
        vim

RUN useradd --create-home --shell /bin/bash developer
ENV DEV_HOME=/home/developer
USER developer
WORKDIR /home/developer
RUN python3 -m venv /home/developer/venv
ENV PATH=/home/developer/venv/bin:${PATH}
ENV LD_LIBRARY_PATH=/home/developer/venv/LD_LIBRARY_PATH

## Install latest spacekit release
WORKDIR /home/developer
COPY ./docker/images/test_image/requirements.txt $DEV_HOME/requirements.txt
ARG SPACEKIT_VERSION=rev-0.3.2
ENV SPACEKIT_REF=${SPACEKIT_VERSION}
RUN git clone https://github.com/alphasentaurii/spacekit.git --branch $SPACEKIT_REF && \
  cd /home/developer/spacekit && \
  pip install --upgrade pip && \
  pip install -e . && \
  pip install -r $DEV_HOME/requirements.txt


# WORKDIR /home/developer
# COPY ./docker/images/dashboard_image/requirements.txt $DEV_HOME/requirements.txt
# ARG SPACEKIT_VERSION=0.3.2
# ENV SPACEKIT_REF=${SPACEKIT_VERSION}
# RUN git clone https://github.com/alphasentaurii/spacekit.git && \
#   cd /home/developer/spacekit && \
#   git fetch --all --tags && \
#   git checkout tags/${SPACEKIT_REF} && \
#   pip install --upgrade pip && \
#   pip install -e . && \
#   pip install -r $DEV_HOME/requirements.txt

# temp fix for problematic dash dep (downgrade werzeug)
RUN pip uninstall werkzeug -y && pip install werkzeug==2.0.3

WORKDIR /home/developer
USER root
RUN chown -R developer:developer /home/developer
USER developer
WORKDIR /home/developer
RUN mkdir /home/developer/data


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
