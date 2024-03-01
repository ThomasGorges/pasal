# syntax=docker/dockerfile:1.4
FROM docker.io/tensorflow/tensorflow@sha256:3aeb6a5489ad8221d79ab50ec09e0b09afc483dfdb4b868ea38cfb9335269049
# Image corresponds to docker.io/tensorflow/tensorflow:2.10.0-gpu
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y install git openjdk-17-jre python3.9 python3.9-dev && \
    apt-get clean && useradd --uid 1000 -m pasal
USER pasal
RUN pip3 install --user pipenv==2023.9.8
WORKDIR /home/pasal/paper
ENV PATH="/home/pasal/.venv/bin:$PATH"
COPY Pipfile Pipfile.lock /home/pasal/
CMD cd /home/pasal/ && PIPENV_VENV_IN_PROJECT=1 python3 -m pipenv install --python 3.9 && sed -i 's/from fractions import gcd/from math import gcd/g' /home/pasal/.venv/lib/python3.9/site-packages/networkx/algorithms/dag.py && . /home/pasal/.venv/bin/activate && bash

