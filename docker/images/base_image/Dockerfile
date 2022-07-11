# syntax = docker/dockerfile:1.0-experimental

ARG BASE_IMAGE=debian:bullseye-slim
FROM ${BASE_IMAGE}

RUN apt update && \
    apt upgrade --assume-yes && \
    ACCEPT_EULA=Y DEBIAN_FRONTEND=noninteractive && \
    apt install --assume-yes \
        csh \
        fitsverify \
        git \
        python3 \
        python3-pip \
        python3-venv \
        rsync \
        sudo \
        tcsh \
        make \
        gcc \
        htop \
        wget \
        git \
        tar \
        curl \
        time \
        vim

RUN useradd --create-home --shell /bin/bash developer
ENV DEV_HOME=/home/developer

# RUN wget --quiet --output-document - http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-3.49.tar.gz | tar -xz -C $DEV_HOME && cd $DEV_HOME/cfitsio-3.49 && ./configure --prefix=/usr/local --enable-reentrant --disable-curl && make shared && make install

# RUN wget --quiet --output-document - https://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/fitsverify-4.20.tar.gz | tar -xz -C $DEV_HOME && cd $DEV_HOME/fitsverify-4.20 && gcc -o fitsverify ftverify.c fvrf_data.c fvrf_file.c fvrf_head.c fvrf_key.c fvrf_misc.c -DSTANDALONE -lcfitsio && cp fitsverify /usr/local/bin

USER developer
WORKDIR /home/developer
RUN python3 -m venv /home/developer/venv
ENV PATH=/home/developer/venv/bin:${PATH}
ENV LD_LIBRARY_PATH=/home/developer/venv/LD_LIBRARY_PATH

## Install latest release
WORKDIR /home/developer
COPY requirements.txt $DEV_HOME/.
RUN pip install --upgrade pip && \
  pip install -r $DEV_HOME/requirements.txt

# temp fix for problematic dash dep (downgrade werzeug)
RUN pip uninstall werkzeug -y && pip install werkzeug==2.0.3

WORKDIR /home/developer
USER root
RUN chown -R developer:developer /home/developer
USER developer
WORKDIR /home/developer
RUN mkdir /home/developer/data
CMD ["/bin/bash"]