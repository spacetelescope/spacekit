FROM debian:bullseye-slim
ENV DEV_HOME=/home/developer

RUN apt update && \
    apt upgrade --assume-yes && \
    ACCEPT_EULA=Y DEBIAN_FRONTEND=noninteractive && \
    apt install --assume-yes \
        sudo \
        git \
        vim \
        python3 \
        python3-pip \
        python3-venv \
        gcc \
        tar \
        curl \
        time && \
    apt autoremove && \
    apt clean && \
    useradd --create-home --shell /bin/bash developer && \
    chown -R developer:developer /home/developer

USER developer
ENV PATH=/home/developer/venv/bin:${PATH}
ENV LD_LIBRARY_PATH=/home/developer/venv/LD_LIBRARY_PATH
ENV SPACEKIT_DATA=/home/developer
ENV TF_CPP_MIN_LOG_LEVEL=2
ARG PFX=archive
ENV PFX=${PFX}
ARG SPACEKIT_VERSION=1.0.0
ENV SPACEKIT_VERSION=${SPACEKIT_VERSION}
ARG SRC
ARG COLLECTION
ARG DATASETS
WORKDIR /home/developer
RUN python3 -m venv /home/developer/venv && \
  pip install --upgrade pip && \
  pip install \
    spacekit[x] \
    dash~=2.0.0 \
    dash-cytoscape \
    dash-daq && \
  mkdir /home/developer/data && \
  python -m spacekit.datasets.beam -s="${SRC}:${COLLECTION}" -d=$DATASETS -o=$SPACEKIT_DATA

# Import data
USER developer
WORKDIR /home/developer
COPY ./docker/images/dashboard_image/.env /home/developer/.env
RUN python -m spacekit.datasets.beam -s="${SRC}:${COLLECTION}" -d=$DATASETS -o=$SPACEKIT_DATA
EXPOSE 8050
CMD /bin/bash
