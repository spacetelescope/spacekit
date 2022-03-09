ARG BASE_IMAGE=library/ubuntu:20.04
FROM ${BASE_IMAGE}
ARG CFG="cal"
ENV CFG=$CFG
#SSL/TLS cert setup for STScI AWS firewalling
USER root
ENV REQUESTS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
ENV CURL_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
RUN mkdir -p /etc/ssl/certs/ && mkdir -p /etc/pki/ca-trust/extracted/pem/
COPY certs/tls-ca-bundle.pem /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
RUN ln -sf /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem /etc/ssl/certs/ca-bundle.crt \
&& ln -sf /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem /etc/ssl/certs/ca-certificates.crt \
&& mkdir -p /etc/pki/ca-trust/extracted/openssl

RUN DEBIAN_FRONTEND=noninteractive && apt update && apt upgrade -y && apt install -y \
build-essential \
curl \
libbz2-dev \
libffi-dev \
libgdbm-dev \
liblzma-dev \
libncurses5-dev \
libnss3-dev \
libreadline-dev \
libsqlite3-dev \
libssl-dev \
unzip \
zlib1g-dev

RUN DEBIAN_FRONTEND=noninteractive \
&& curl -o Python-3.9.5.tgz https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tgz \
&& tar -xf Python-3.9.5.tgz \
&& cd Python-3.9.5 \
&& ./configure --enable-optimizations \
&& make altinstall \
&& update-alternatives --install /usr/local/bin/python python /usr/local/bin/python3.9 10 

# RUN DEBIAN_FRONTEND=noninteractive \
# && cd ../ && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
# && unzip awscliv2.zip \
# && ./aws/install

RUN useradd --create-home --shell /bin/bash developer
USER developer
WORKDIR /home/developer/
RUN python -m pip install --upgrade pip && python -m pip install virtualenv
RUN python -m virtualenv /home/developer/venv
ENV PATH=/home/developer/venv/bin:${PATH}
ENV LD_LIBRARY_PATH=/home/developer/venv/lib
RUN mkdir /home/developer/data
ARG SOURCEDATA=./data
ADD $SOURCEDATA /home/developer/data
# TEMP
ARG SPACEKIT_REPO=alphasentaurii/spacekit
ARG SPACEKIT_REF=main
# RUN git clone --branch $SPACEKIT_REF https://github.com/${SPACEKIT_REPO}.git
# WORKDIR /home/developer/spacekit
# RUN pip install -e .
WORKDIR /home/developer
RUN pip install spacekit
# Get datasets

ARG SRC
ARG COLLECTION
ARG DATASETS
ENV SPACEKIT_DATA=/home/developer
ENV TF_CPP_MIN_LOG_LEVEL=2
RUN python -m spacekit.datasets.beam -s="${SRC}:${COLLECTION}" -d=$DATASETS -o=/home/developer
EXPOSE 8050
ENTRYPOINT ["python","-m","spacekit.dashboard.cal.index"]
