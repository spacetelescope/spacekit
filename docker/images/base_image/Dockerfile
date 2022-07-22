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
COPY ./docker/images/base_image/requirements.txt $DEV_HOME/requirements.txt
ARG SPACEKIT_VERSION=0.3.2
ENV SPACEKIT_VERSION=${SPACEKIT_VERSION}
RUN git clone https://github.com/alphasentaurii/spacekit.git --branch $SPACEKIT_VERSION && \
  cd /home/developer/spacekit && \
  pip install --upgrade pip && \
  pip install -e . && \
  pip install -r $DEV_HOME/requirements.txt
# RUN git clone https://github.com/alphasentaurii/spacekit.git && \
#   cd /home/developer/spacekit && \
#   git fetch --all --tags && \
#   git checkout tags/${SPACEKIT_VERSION} && \
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
CMD ["/bin/bash"]