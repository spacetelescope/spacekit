ARG BASE_IMAGE=alphasentaurii/spacekit:base
FROM ${BASE_IMAGE}

USER developer
WORKDIR /home/developer
## temp fix for problematic dash dep (downgrade werzeug)
RUN pip uninstall werkzeug -y && pip install werkzeug==2.0.3

## Install latest release
WORKDIR /home/developer
COPY ./scripts/installer.sh /home/developer/installer.sh
ARG ENVIRONMENT=dashboard
ENV ENVIRONMENT=$ENVIRONMENT
ENV ENVFILE="${ENVIRONMENT}.env"
COPY ./images/dashboard_image/envs/$ENVFILE /home/developer/.
RUN sh /home/developer/installer.sh /home/developer/$ENVFILE

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
