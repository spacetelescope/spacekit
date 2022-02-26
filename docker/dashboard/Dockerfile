FROM library/ubuntu:20.04
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

RUN DEBIAN_FRONTEND=noninteractive \
&& cd ../ && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
&& unzip awscliv2.zip \
&& ./aws/install

RUN useradd --create-home --shell /bin/bash developer
USER developer
COPY scrape.sh /home/developer/
COPY requirements.txt ./home/developer/
WORKDIR /home/developer/
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
RUN mkdir /home/developer/data
ADD spacekit_data /home/developer/data
ENV SPACEKIT_DATA=/home/developer/data
ENV TF_CPP_MIN_LOG_LEVEL=2
ENTRYPOINT ["python","-m","spacekit.dashboard.cal.index"]
# ["python","-m","spacekit.dashboard.svm.app"]