FROM debian:bookworm-slim
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
        libhdf5-dev \
        hdf5-tools \
        gcc \
        tar \
        curl \
        pkg-config \
        time && \
    apt autoremove && \
    apt clean && \
    useradd --create-home --shell /bin/bash developer && \
    chown -R developer:developer /home/developer
COPY ./docker/images/dashboard_image/.env /home/developer/.env

USER developer
ENV PATH=/home/developer/venv/bin:${PATH}
ENV LD_LIBRARY_PATH=/home/developer/venv/LD_LIBRARY_PATH
ENV SPACEKIT_DATA=data
ENV TF_CPP_MIN_LOG_LEVEL=2
ARG PFX=archive
ENV PFX=${PFX}
ARG SPACEKIT_VERSION=1.1.1
ENV SPACEKIT_VERSION=${SPACEKIT_VERSION}
ARG SRC
ARG COLLECTION
ARG DATASETS
WORKDIR /home/developer
RUN python3 -m venv /home/developer/venv && \
  pip install --upgrade pip && \
  pip install \
    spacekit[x]~="${SPACEKIT_VERSION}" \
    dash \
    dash-cytoscape \
    dash-daq && \
  mkdir $SPACEKIT_DATA && \
  python -m spacekit.datasets.beam -s="${SRC}:${COLLECTION}" -d=$DATASETS -o=$SPACEKIT_DATA

EXPOSE 8050
CMD /bin/bash
