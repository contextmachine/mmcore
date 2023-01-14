FROM mambaorg/micromamba:latest
MAINTAINER "CONTEXTMACHINE"
USER root
ENV MAMBA_USER=root
# Copy and install package

RUN micromamba env update --file
WORKDIR mmodel
COPY . .
# Install extra packages
RUN python -m pip install git+https://github.com/contextmachine/cxmdata.git
RUN pip install git+https://github.com/tpaviot/pythonocc-utils


# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
ENV MY_FAVORIT_TRANSPORT="CONTEXTMACHINE"
ENTRYPOINT ["/bin/bash"]