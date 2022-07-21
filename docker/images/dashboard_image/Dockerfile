ARG BASE_IMAGE=alphasentaurii/spacekit:base
FROM ${BASE_IMAGE}

USER developer
WORKDIR /home/developer
## temp fix for problematic dash dep (downgrade werzeug)
RUN pip uninstall werkzeug -y && pip install werkzeug==2.0.3

## Install spacekit
WORKDIR /home/developer
COPY ./images/dashboard_image/.env /home/developer/.env
# COPY ./scripts/installer.sh /home/developer/installer.sh
# RUN /home/developer/installer.sh
RUN pip install spacekit

# Import data
USER developer
WORKDIR /home/developer
ENV SPACEKIT_DATA=/home/developer
ENV TF_CPP_MIN_LOG_LEVEL=2
ARG SRC
ARG COLLECTION
ARG DATASETS
RUN python -m spacekit.datasets.beam -s="${SRC}:${COLLECTION}" -d=$DATASETS -o=$SPACEKIT_DATA
EXPOSE 8050
CMD /bin/bash
