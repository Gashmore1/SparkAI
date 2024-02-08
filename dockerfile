FROM spark:python3
USER root
RUN /bin/sh -c set -ex; apt-get update; apt-get install -y python3-opencv; rm -rf /var/lib/apt/lists/*
USER spark