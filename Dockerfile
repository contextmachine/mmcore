FROM mambaorg/micromamba:latest
MAINTAINER "CONTEXTMACHINE"
USER root
ENV MAMBA_USER=root
# Copy and install package
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n "base" -f /tmp/env.yaml && \
    micromamba clean --all --yes
WORKDIR mmodel
COPY . .
# Install extra packages
RUN python -m pip install git+https://github.com/contextmachine/cxmdata.git
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
ENV MY_FAVORIT_TRANSPORT="CONTEXTMACHINE"
ENTRYPOINT ["/bin/bash"]