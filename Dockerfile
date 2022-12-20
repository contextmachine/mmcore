FROM continuumio/anaconda-pkg-build:latest
MAINTAINER "CONTEXTMACHINE"
WORKDIR /party

COPY environment.yml /party/environment.yml
RUN /bin/bash conda env update --name base --file==environment.yml && act

# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
ENV MY_FAVORIT_TRANSPORT="CONTEXTMACHINE"
COPY mmcore /party/mmcore
COPY setup.py /party/mmcore
RUN conda build mmcore
