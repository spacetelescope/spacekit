FROM debian:bullseye-slim

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

WORKDIR /home/developer
ARG SPACEKIT_VERSION=0.3.2
ENV SPACEKIT_VERSION=${SPACEKIT_VERSION}

## Install spacekit dev branch
# RUN git clone https://github.com/alphasentaurii/spacekit.git --branch ${SPACEKIT_VERSION} && \
#   cd /home/developer/spacekit && \
#   pip install --upgrade pip && \
#   pip install -e . && \
#   pip install \
#     sklearn \
#     dash~=2.0.0 \
#     dash-cytoscape \
#     dash-daq

# # Install and test new spacekit release from git
# RUN git clone https://github.com/alphasentaurii/spacekit.git && \
#   cd /home/developer/spacekit && \
#   git fetch --all --tags && \
#   git checkout tags/${SPACEKIT_VERSION} && \
#   pip install -e . && \
#   pip install \
#     sklearn \
#     dash~=2.0.0 \
#     dash-cytoscape \
#     dash-daq

## Install dependencies and latest spacekit release from pip
RUN pip install --upgrade pip && \
  pip install \
    spacekit \
    sklearn \
    dash~=2.0.0 \
    dash-cytoscape \
    dash-daq

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